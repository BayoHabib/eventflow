"""Spatial pipeline steps with registry integration.

Each step:
- Inherits from Step base class
- Registers inputs/outputs via FeatureProvenance
- Updates EventSchema properly
- Supports Pydantic config models for validation
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np
import polars as pl
from pydantic import BaseModel, Field, field_validator
from shapely import STRtree, geometry
from shapely.wkt import loads as wkt_loads

from eventflow.core.grid import GridSpec
from eventflow.core.pipeline import Step
from eventflow.core.schema import FeatureProvenance
from eventflow.core.spatial import require_metric_crs, transform_crs
from eventflow.core.utils import get_logger
from eventflow.core.weights import SpatialWeights

if TYPE_CHECKING:
    from eventflow.core.event_frame import EventFrame

logger = get_logger(__name__)


def _validate_rings(rings: object) -> list[int]:
    """Validate a ring list: non-empty, unique, positive integers."""
    if isinstance(rings, (str, bytes)):
        raise ValueError("rings must be a sequence of positive integers")
    try:
        ring_list = list(rings)  # type: ignore[call-overload]
    except TypeError:
        raise ValueError("rings must be a sequence of positive integers") from None
    validated: list[int] = []
    for ring in ring_list:
        if isinstance(ring, bool) or not isinstance(ring, (int, np.integer)) or int(ring) < 1:
            raise ValueError(f"rings must contain positive integers, got {ring!r}")
        validated.append(int(ring))
    if not validated:
        raise ValueError("rings must not be empty")
    if len(set(validated)) != len(validated):
        raise ValueError("rings must not contain duplicates")
    return validated


# -----------------------------------------------------------------------------
# Pydantic Config Models for Step Parameters
# -----------------------------------------------------------------------------


class TransformCRSConfig(BaseModel):
    """Configuration for CRS transformation step."""

    target_crs: str = Field(..., description="Target CRS (e.g., 'EPSG:26971')")


class AssignToGridConfig(BaseModel):
    """Configuration for grid assignment step."""

    grid_size_m: float = Field(default=500.0, gt=0, description="Grid cell size in meters")
    origin: tuple[float, float] | None = Field(
        default=None, description="Grid origin (minx, miny); auto-calculated if None"
    )


class AssignToZonesConfig(BaseModel):
    """Configuration for zone assignment step."""

    zone_id_col: str = Field(default="zone_id", description="Zone ID column in zones DataFrame")
    zone_geom_col: str = Field(
        default="geometry", description="Geometry column (WKT) in zones DataFrame"
    )


class ComputeDistancesConfig(BaseModel):
    """Configuration for distance computation step."""

    poi_lon_col: str = Field(default="longitude", description="POI longitude column")
    poi_lat_col: str = Field(default="latitude", description="POI latitude column")
    poi_name_col: str = Field(default="name", description="POI name column for feature naming")


class GridAggregationConfig(BaseModel):
    """Configuration for grid-level aggregation step."""

    grid_col: str = Field(default="grid_id", description="Grid ID column to aggregate by")
    agg_funcs: list[Literal["count", "sum", "mean", "std", "min", "max"]] = Field(
        default=["count"], description="Aggregation functions to apply"
    )
    value_cols: list[str] = Field(
        default_factory=list, description="Columns to aggregate (empty = count only)"
    )
    time_col: str | None = Field(
        default=None, description="Optional time column for temporal grouping"
    )


class KDEConfig(BaseModel):
    """Configuration for Kernel Density Estimation step."""

    bandwidth: float = Field(default=500.0, gt=0, description="KDE bandwidth in coordinate units")
    kernel: Literal["gaussian", "epanechnikov", "uniform"] = Field(
        default="gaussian", description="Kernel function type"
    )
    output_col: str = Field(default="kde_density", description="Output column name")


class SpatialLagConfig(BaseModel):
    """Configuration for spatial lag computation step."""

    neighbor_col: str = Field(default="grid_id", description="Spatial unit identifier column")
    value_cols: list[str] = Field(..., description="Columns to compute spatial lag for")
    weights: Literal["queen", "rook", "knn", "distance"] = Field(
        default="queen", description="Spatial weights type"
    )
    group_cols: list[str] = Field(default_factory=list)
    k: int = Field(default=4, ge=1, description="Number of neighbors for KNN weights")
    distance_threshold: float | None = Field(
        default=None, description="Distance threshold for distance-based weights"
    )


class MultiScaleSpatialLagConfig(BaseModel):
    """Configuration for multi-scale spatial lag computation step."""

    neighbor_col: str = Field(default="grid_id", description="Spatial unit identifier column")
    value_cols: list[str] = Field(..., description="Columns to compute multi-scale spatial lag for")
    rings: list[int] = Field(
        default=[1, 2, 3],
        description="Queen rings (Chebyshev distances) to lag over; ring 1 is queen contiguity",
    )
    group_cols: list[str] = Field(default_factory=list)

    @field_validator("rings", mode="before")
    @classmethod
    def _check_rings(cls, rings: object) -> object:
        _validate_rings(rings)
        return rings


class LocalAutocorrelationConfig(BaseModel):
    """Configuration for local spatial autocorrelation step."""

    value_col: str = Field(..., description="Column to compute local autocorrelation for")
    spatial_unit_col: str = Field(default="grid_id", description="Spatial unit identifier")
    weights: Literal["queen", "rook", "knn"] = Field(
        default="queen", description="Spatial weights type"
    )
    permutations: int = Field(default=99, ge=0, description="Number of conditional permutations")
    group_cols: list[str] = Field(default_factory=list)
    k: int = Field(default=4, ge=1)
    seed: int = 0
    significance: float = Field(default=0.05, gt=0, lt=1)


class GetisOrdConfig(BaseModel):
    """Configuration for Getis-Ord Gi* step."""

    group_cols: list[str] = Field(default_factory=list)
    value_col: str = Field(..., description="Column to compute hotspot statistics for")
    spatial_unit_col: str = Field(default="grid_id", description="Spatial unit identifier")
    distance_threshold: float | None = Field(
        default=None, description="Distance threshold for neighbor weights"
    )


# -----------------------------------------------------------------------------
# Spatial Steps
# -----------------------------------------------------------------------------


class TransformCRSStep(Step):
    """Transform event coordinates to a different CRS.

    Inputs:
        - lat_col, lon_col from EventSchema

    Outputs:
        - {lon_col}_proj, {lat_col}_proj columns
        - Updated CRS in metadata
    """

    def __init__(self, target_crs: str) -> None:
        self.target_crs = target_crs

    def run(self, event_frame: EventFrame) -> EventFrame:
        """Execute CRS transformation."""
        if event_frame.metadata.crs == self.target_crs:
            logger.debug(f"CRS already matches target: {self.target_crs}")
            return event_frame

        logger.info(f"Transforming CRS from {event_frame.metadata.crs} to {self.target_crs}")
        schema = event_frame.schema

        if schema.lat_col is None or schema.lon_col is None:
            raise ValueError("EventFrame must have lat/lon columns for CRS transformation")

        lon_col = schema.lon_col
        lat_col = schema.lat_col

        provenance = FeatureProvenance(
            produced_by="TransformCRSStep",
            inputs=[lon_col, lat_col],
            tags={"spatial"},
            description=f"CRS transformation to {self.target_crs}",
        )

        result = transform_crs(event_frame, self.target_crs)

        result = result.register_feature(
            f"{lon_col}_proj",
            {"source_step": "TransformCRSStep", "inputs": [lon_col]},
            provenance=provenance,
        )
        result = result.register_feature(
            f"{lat_col}_proj",
            {"source_step": "TransformCRSStep", "inputs": [lat_col]},
            provenance=provenance,
        )

        return result.require_context(spatial_crs=self.target_crs)


class AssignToGridStep(Step):
    """Assign events to spatial grid cells.

    Inputs:
        - lat_col, lon_col from EventSchema

    Outputs:
        - grid_id column
    """

    def __init__(
        self,
        grid_size_m: float = 500.0,
        origin: tuple[float, float] | None = None,
    ) -> None:
        if not 0 < grid_size_m < float("inf"):
            raise ValueError("Grid cell size must be positive and finite")
        self.grid_size_m = grid_size_m
        self.origin = origin

    def run(self, event_frame: EventFrame) -> EventFrame:
        """Execute grid assignment."""
        require_metric_crs(event_frame.metadata.crs)
        schema = event_frame.schema

        if schema.lat_col is None or schema.lon_col is None:
            raise ValueError("EventFrame must have lat/lon columns for grid assignment")

        lon_col = schema.lon_col
        lat_col = schema.lat_col

        lf = event_frame.lazy_frame

        # Calculate origin if not provided (requires collecting stats)
        if self.origin is not None:
            min_x, min_y = self.origin
        else:
            # Use streaming min to get origin
            stats = (
                lf.select(
                    [pl.col(lon_col).min().alias("min_x"), pl.col(lat_col).min().alias("min_y")]
                )
                .collect()
                .row(0)
            )
            min_x, min_y = stats[0], stats[1]

        if min_x is None or min_y is None:
            raise ValueError(
                "Cannot infer a grid origin from empty coordinates; supply origin explicitly"
            )
        grid_size = self.grid_size_m
        spec = GridSpec(grid_size, (min_x, min_y), event_frame.metadata.crs)
        lf = spec.assign(lf, lon_col, lat_col)

        provenance = FeatureProvenance(
            produced_by="AssignToGridStep",
            inputs=[lon_col, lat_col],
            tags={"spatial"},
            description=f"Grid assignment at {grid_size}m resolution",
            metadata=spec.metadata(),
        )

        result = event_frame.with_lazy_frame(lf).with_metadata(
            grid_size_m=grid_size,
            custom={**event_frame.metadata.custom, "grid_spec": spec.metadata()},
        )
        result = result.register_feature(
            "grid_id",
            {
                "source_step": "AssignToGridStep",
                "inputs": [lon_col, lat_col],
                "grid_size_m": grid_size,
            },
            provenance=provenance,
        )

        return result


class AssignToZonesStep(Step):
    """Assign events to predefined zones (e.g., neighborhoods, census tracts).

    Inputs:
        - lat_col, lon_col from EventSchema
        - zones DataFrame with geometry

    Outputs:
        - zone_id column
    """

    def __init__(
        self,
        zones: pl.DataFrame,
        zone_id_col: str = "zone_id",
        zone_geom_col: str = "geometry",
    ) -> None:
        self.zones = zones
        self.zone_id_col = zone_id_col
        self.zone_geom_col = zone_geom_col

    def run(self, event_frame: EventFrame) -> EventFrame:
        """Execute zone assignment."""
        schema = event_frame.schema

        if schema.lat_col is None or schema.lon_col is None:
            raise ValueError("EventFrame must have lat/lon columns for zone assignment")

        lon_col = schema.lon_col
        lat_col = schema.lat_col

        # Build spatial index
        zone_geoms = [wkt_loads(wkt) for wkt in self.zones[self.zone_geom_col]]
        zone_ids_raw = self.zones[self.zone_id_col].to_list()
        zone_ids: list[int | None] = []
        for value in zone_ids_raw:
            try:
                zone_ids.append(int(value))
            except (TypeError, ValueError):
                logger.debug("Skipping non-integer zone id value: %s", value)
                zone_ids.append(None)
        tree = STRtree(zone_geoms)

        def find_zone(lon: float, lat: float) -> int | None:
            """Find zone containing point."""
            point = geometry.Point(lon, lat)
            idx = tree.query(point, predicate="within")
            if len(idx) > 0:
                first_idx = int(idx[0])
                return zone_ids[first_idx]
            return None

        lf = event_frame.lazy_frame.with_columns(
            [
                pl.struct([lon_col, lat_col])
                .map_elements(
                    lambda coord: find_zone(coord[lon_col], coord[lat_col]),
                    return_dtype=pl.Int32,
                )
                .alias("zone_id")
            ]
        )

        provenance = FeatureProvenance(
            produced_by="AssignToZonesStep",
            inputs=[lon_col, lat_col],
            tags={"spatial"},
            description="Zone assignment via spatial join",
        )

        result = event_frame.with_lazy_frame(lf)
        return result.register_feature(
            "zone_id",
            {"source_step": "AssignToZonesStep", "inputs": [lon_col, lat_col]},
            provenance=provenance,
        )


class ComputeDistancesStep(Step):
    """Compute distances from events to points of interest.

    Inputs:
        - lat_col, lon_col from EventSchema
        - points_of_interest DataFrame

    Outputs:
        - dist_to_{poi_name} columns for each POI
    """

    def __init__(
        self,
        points_of_interest: pl.DataFrame,
        poi_lon_col: str = "longitude",
        poi_lat_col: str = "latitude",
        poi_name_col: str = "name",
    ) -> None:
        self.points_of_interest = points_of_interest
        self.poi_lon_col = poi_lon_col
        self.poi_lat_col = poi_lat_col
        self.poi_name_col = poi_name_col

    def run(self, event_frame: EventFrame) -> EventFrame:
        """Execute distance computation."""
        schema = event_frame.schema

        if schema.lat_col is None or schema.lon_col is None:
            raise ValueError("EventFrame must have lat/lon columns for distance computation")

        lon_col = schema.lon_col
        lat_col = schema.lat_col
        lf = event_frame.lazy_frame
        result = event_frame

        for poi in self.points_of_interest.iter_rows(named=True):
            poi_name = poi[self.poi_name_col]
            poi_lon = poi[self.poi_lon_col]
            poi_lat = poi[self.poi_lat_col]
            col_name = f"dist_to_{poi_name}"

            lf = lf.with_columns(
                [
                    (
                        ((pl.col(lon_col) - poi_lon) ** 2 + (pl.col(lat_col) - poi_lat) ** 2) ** 0.5
                    ).alias(col_name)
                ]
            )

            provenance = FeatureProvenance(
                produced_by="ComputeDistancesStep",
                inputs=[lon_col, lat_col],
                tags={"spatial"},
                description=f"Euclidean distance to {poi_name}",
            )
            result = result.register_feature(
                col_name,
                {"source_step": "ComputeDistancesStep", "poi_name": poi_name},
                provenance=provenance,
            )

        return result.with_lazy_frame(lf)


class GridAggregationStep(Step):
    """Aggregate event data at grid cell level.

    Computes counts, sums, means, etc. at spatial grid level.

    Inputs:
        - grid_id column
        - value_cols for aggregation

    Outputs:
        - {grid_col}_{agg_func}_{value_col} columns
    """

    def __init__(
        self,
        grid_col: str = "grid_id",
        agg_funcs: Sequence[str] | None = None,
        value_cols: Sequence[str] | None = None,
        time_col: str | None = None,
    ) -> None:
        self.grid_col = grid_col
        self.agg_funcs = list(agg_funcs) if agg_funcs else ["count"]
        self.value_cols = list(value_cols) if value_cols else []
        self.time_col = time_col

    def run(self, event_frame: EventFrame) -> EventFrame:
        """Execute grid aggregation."""
        lf = event_frame.lazy_frame

        # Determine grouping columns
        group_cols = [self.grid_col]
        if self.time_col:
            group_cols.append(self.time_col)

        # Build aggregation expressions
        agg_exprs: list[pl.Expr] = []
        feature_names: list[str] = []

        if "count" in self.agg_funcs:
            agg_exprs.append(pl.len().alias("event_count"))
            feature_names.append("event_count")

        for value_col in self.value_cols:
            for agg_func in self.agg_funcs:
                if agg_func == "count":
                    continue  # Already handled above

                col_name = f"{value_col}_{agg_func}"
                if agg_func == "sum":
                    agg_exprs.append(pl.col(value_col).sum().alias(col_name))
                elif agg_func == "mean":
                    agg_exprs.append(pl.col(value_col).mean().alias(col_name))
                elif agg_func == "std":
                    agg_exprs.append(pl.col(value_col).std().alias(col_name))
                elif agg_func == "min":
                    agg_exprs.append(pl.col(value_col).min().alias(col_name))
                elif agg_func == "max":
                    agg_exprs.append(pl.col(value_col).max().alias(col_name))
                feature_names.append(col_name)

        # Perform aggregation
        agg_lf = lf.group_by(group_cols).agg(agg_exprs)

        # Join back to original frame
        result_lf = lf.join(agg_lf, on=group_cols, how="left")

        result = event_frame.with_lazy_frame(result_lf)

        # Register features
        for feat_name in feature_names:
            provenance = FeatureProvenance(
                produced_by="GridAggregationStep",
                inputs=[self.grid_col] + self.value_cols,
                tags={"spatial"},
                description=f"Grid-level aggregation: {feat_name}",
            )
            result = result.register_feature(
                feat_name,
                {"source_step": "GridAggregationStep", "grid_col": self.grid_col},
                provenance=provenance,
            )

        return result


class KDEStep(Step):
    """Approximate local density for event locations (not a kernel density).

    Counts neighbours inside the bandwidth and divides by the buffer area;
    no kernel function is applied, so this is a uniform-kernel count
    density, not a Gaussian (or other) KDE. For a proper Gaussian KDE on
    the completed panel use ``KernelDensityStep``; for a space-time KDE use
    ``SpaceTimeKernelDensityStep``. Kept registered for backwards
    compatibility.

    Inputs:
        - lat_col, lon_col from EventSchema

    Outputs:
        - kde_density column (or custom output_col)
    """

    def __init__(
        self,
        bandwidth: float = 500.0,
        kernel: str = "gaussian",
        output_col: str = "kde_density",
    ) -> None:
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.output_col = output_col

    def run(self, event_frame: EventFrame) -> EventFrame:
        """Execute KDE computation."""
        schema = event_frame.schema

        if schema.lat_col is None or schema.lon_col is None:
            raise ValueError("EventFrame must have lat/lon columns for KDE")

        lon_col = schema.lon_col
        lat_col = schema.lat_col

        # Collect coordinates for KDE calculation
        coords_df = event_frame.lazy_frame.select([lon_col, lat_col]).collect()
        coords = coords_df.to_numpy()

        if len(coords) == 0:
            # No points - return zeros
            lf = event_frame.lazy_frame.with_columns(pl.lit(0.0).alias(self.output_col))
        else:
            # Compute KDE using spatial index for efficiency
            # For simplicity, compute approximate local density using neighbor count within bandwidth
            # A full KDE would require O(n²) computation or more sophisticated spatial indexing
            lf = event_frame.lazy_frame

            # Build spatial index for efficiency
            points = [geometry.Point(lon, lat) for lon, lat in coords]
            tree = STRtree(points)

            def compute_density(lon: float, lat: float) -> float:
                """Compute local density at a point."""
                point = geometry.Point(lon, lat)
                buffer = point.buffer(self.bandwidth)
                neighbors = tree.query(buffer)
                count = len(neighbors)
                # Normalize by area of buffer
                area = 3.14159265359 * self.bandwidth**2
                return count / area if area > 0 else 0.0

            lf = lf.with_columns(
                [
                    pl.struct([lon_col, lat_col])
                    .map_elements(
                        lambda coord: compute_density(coord[lon_col], coord[lat_col]),
                        return_dtype=pl.Float64,
                    )
                    .alias(self.output_col)
                ]
            )

        provenance = FeatureProvenance(
            produced_by="KDEStep",
            inputs=[lon_col, lat_col],
            tags={"spatial"},
            description=f"Kernel density estimation (bandwidth={self.bandwidth}, kernel={self.kernel})",
            metadata={"bandwidth": self.bandwidth, "kernel": self.kernel},
        )

        result = event_frame.with_lazy_frame(lf)
        return result.register_feature(
            self.output_col,
            {"source_step": "KDEStep", "bandwidth": self.bandwidth, "kernel": self.kernel},
            provenance=provenance,
        )


# -----------------------------------------------------------------------------
# Vectorized spatial lag helpers (shared by SpatialLagStep and
# MultiScaleSpatialLagStep). LocalMoranStep / GetisOrdStep keep using the
# per-snapshot helper _spatial_snapshots below.
# -----------------------------------------------------------------------------


def _require_spatial_columns(
    df: pl.DataFrame, unit_col: str, value_cols: list[str], group_cols: list[str]
) -> None:
    missing = sorted({unit_col, *value_cols, *group_cols} - set(df.columns))
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")


def _validate_spatial_values(
    df: pl.DataFrame, unit_col: str, value_cols: list[str], group_cols: list[str]
) -> None:
    if df[unit_col].null_count() or any(df[c].null_count() for c in group_cols):
        raise ValueError("Spatial unit and snapshot keys cannot contain nulls")
    if any(not np.isfinite(df[c].to_numpy().astype(float)).all() for c in value_cols):
        raise ValueError(
            "Spatial statistics require finite observed values; choose a missing-data policy first"
        )


def _aggregate_stats(
    df: pl.DataFrame, unit_col: str, value_cols: list[str], group_cols: list[str]
) -> pl.DataFrame:
    """Mean of each value column per (snapshot key, unit)."""
    return df.group_by([*group_cols, unit_col]).agg([pl.col(c).mean().alias(c) for c in value_cols])


def _lag_means(
    stats: pl.DataFrame,
    edges: pl.DataFrame,
    group_cols: list[str],
    unit_col: str,
    value_cols: list[str],
) -> pl.DataFrame:
    """Row-standardized binary lag: one join + one ``group_by`` mean over edges.

    ``edges`` carries (unit, neighbour) pairs. Joining on the neighbour key
    alone replicates each edge across snapshots, so the lag stays
    per-snapshot with no pooling across time. Units with no edges (islands)
    get no rows here and therefore null lags after the join. The neighbour
    key is dropped right after the join: it only widens the intermediate
    (edges x snapshots) that the ``group_by`` must scan. The join + group_by
    run through the lazy engine: on Chicago-size panels it is roughly twice
    as fast as the eager equivalent (the optimizer pipelines the aggregation
    instead of materializing the full intermediate first).
    """
    return (
        edges.lazy()
        .join(stats.lazy().rename({unit_col: "__nbr"}), on="__nbr", how="inner")
        .drop("__nbr")
        .group_by([*group_cols, unit_col])
        .agg([pl.col(c).mean().alias(c) for c in value_cols])
        .collect()
    )


def _edges_from_weights(weights: SpatialWeights, unit_col: str) -> pl.DataFrame:
    """One (unit, neighbour) row per binary edge; islands get no rows."""
    unit_ids = list(weights.unit_ids)
    edges = pl.DataFrame(
        {unit_col: unit_ids, "__nbr_idx": [list(row) for row in weights.neighbors]},
        schema_overrides={"__nbr_idx": pl.List(pl.Int64)},
    ).explode("__nbr_idx")
    index = pl.DataFrame({"__nbr_idx": list(range(len(unit_ids))), "__nbr": unit_ids})
    return (
        edges.drop_nulls(subset=["__nbr_idx"])
        .join(index, on="__nbr_idx", how="left")
        .drop("__nbr_idx")
        .select(unit_col, "__nbr")
    )


def _reference_weights(
    df: pl.DataFrame,
    event_frame: EventFrame,
    unit_col: str,
    method: str,
    k: int,
    threshold: float | None,
    spatial_units: pl.DataFrame | None,
    supplied: SpatialWeights | None,
    data_ids: list,
) -> SpatialWeights:
    """Build the neighbour graph once from the panel's distinct units."""
    if supplied is not None:
        if set(supplied.unit_ids) != set(data_ids):
            raise ValueError(
                "Spatial weights and data must contain the same units; "
                "complete the observation panel first"
            )
        return supplied
    geometry_col = event_frame.schema.geometry_col
    lon_col = event_frame.schema.lon_col
    lat_col = event_frame.schema.lat_col
    if spatial_units is not None:
        reference = spatial_units
    elif {"grid_row", "grid_col"} <= set(df.columns) and method in {"queen", "rook"}:
        reference = df.select(unit_col, "grid_row", "grid_col").unique(maintain_order=True)
    elif geometry_col and geometry_col in df.columns and method in {"queen", "rook"}:
        reference = df.select(unit_col, pl.col(geometry_col).alias("geometry")).unique(
            maintain_order=True
        )
    elif lon_col and lat_col and lon_col in df.columns and lat_col in df.columns:
        reference = df.select(
            unit_col, pl.col(lon_col).alias("x"), pl.col(lat_col).alias("y")
        ).unique(maintain_order=True)
    else:
        raise ValueError("Supply a spatial reference table or SpatialWeights")
    weights = SpatialWeights.from_units(
        reference,
        unit_col,
        method=method,
        crs=event_frame.metadata.crs,
        k=k,
        distance_threshold=threshold,
    )
    if set(weights.unit_ids) != set(data_ids):
        raise ValueError(
            "Spatial weights and data must contain the same units; "
            "complete the observation panel first"
        )
    return weights


def _grid_reference(
    df: pl.DataFrame,
    unit_col: str,
    spatial_units: pl.DataFrame | None,
    data_ids: list,
) -> pl.DataFrame:
    """Distinct (unit, grid_row, grid_col) table for queen-ring edges."""
    if spatial_units is not None:
        reference = spatial_units
    elif {"grid_row", "grid_col"} <= set(df.columns):
        reference = df.select(unit_col, "grid_row", "grid_col").unique(maintain_order=True)
    else:
        raise ValueError(
            "Multi-scale spatial lag requires grid_row/grid_col in the panel " "or in spatial_units"
        )
    if unit_col not in reference.columns:
        raise ValueError(f"Spatial reference must carry the unit column {unit_col!r}")
    if {"grid_row", "grid_col"} - set(reference.columns):
        raise ValueError("spatial_units must carry grid_row/grid_col for multi-scale spatial lag")
    if set(reference[unit_col].to_list()) != set(data_ids):
        raise ValueError(
            "Spatial weights and data must contain the same units; "
            "complete the observation panel first"
        )
    rows = reference["grid_row"].to_list()
    cols = reference["grid_col"].to_list()
    if any(
        r is None or c is None or int(r) != r or int(c) != c
        for r, c in zip(rows, cols, strict=True)
    ):
        raise ValueError("Grid indices must be non-null integers")
    if len(set(zip(rows, cols, strict=True))) != len(rows):
        raise ValueError("Each spatial unit requires a distinct grid location")
    return reference.select(unit_col, "grid_row", "grid_col")


def _queen_ring_edges(units: pl.DataFrame, unit_col: str, rings: list[int]) -> pl.DataFrame:
    """(unit, neighbour, ring) rows for disjoint queen rings.

    Ring ``r`` is the set of cells at Chebyshev distance exactly ``r`` from
    the unit, so rings are disjoint and ring 1 is queen contiguity.
    """
    offsets = [
        (ring, dr, dc)
        for ring in rings
        for dr in range(-ring, ring + 1)
        for dc in range(-ring, ring + 1)
        if max(abs(dr), abs(dc)) == ring
    ]
    offsets_df = pl.DataFrame(
        {
            "ring": [offset[0] for offset in offsets],
            "__dr": [offset[1] for offset in offsets],
            "__dc": [offset[2] for offset in offsets],
        },
        schema={"ring": pl.Int64, "__dr": pl.Int64, "__dc": pl.Int64},
    )
    candidates = units.join(offsets_df, how="cross").with_columns(
        (pl.col("grid_row") + pl.col("__dr")).alias("__nbr_row"),
        (pl.col("grid_col") + pl.col("__dc")).alias("__nbr_col"),
    )
    lookup = units.rename({unit_col: "__nbr", "grid_row": "__nbr_row", "grid_col": "__nbr_col"})
    return candidates.join(lookup, on=["__nbr_row", "__nbr_col"], how="inner").select(
        unit_col, "__nbr", "ring"
    )


def _empty_spatial_result(event_frame: EventFrame, columns: list[str]) -> EventFrame:
    """An empty panel gains the output columns as null Float64, nothing else."""
    lf = event_frame.lazy_frame.with_columns(
        [pl.lit(None, dtype=pl.Float64).alias(name) for name in columns]
    )
    return event_frame.with_lazy_frame(lf)


class SpatialLagStep(Step):
    """Row-standardized binary spatial lag, computed with one edge join.

    The neighbour edge list is built once from the panel's distinct units
    and the lag is a single join + single ``group_by`` mean in pure Polars:
    no per-bin snapshots and no Python loop over cells.

    Inputs:
        - value_cols on the (completed) panel; finite non-null values required
        - neighbor_col unit identifiers; grid_row/grid_col, polygon geometry,
          or lon/lat supply the spatial reference (or spatial_units /
          spatial_weights)

    Outputs:
        - {value_col}_spatial_lag columns (Float64)

    Window policy: contemporaneous within a snapshot — the lag at bin ``t``
    reads neighbours' values from the same snapshot only (group_cols, e.g.
    time_bin, are never pooled). The recipe feeds it lagged counts, so the
    feature at ``t`` describes neighbours' counts strictly before ``t``.
    Islands (units with no neighbours) yield null.
    """

    def __init__(
        self,
        value_cols: Sequence[str],
        neighbor_col: str = "grid_id",
        weights: str = "queen",
        k: int = 4,
        distance_threshold: float | None = None,
        group_cols: Sequence[str] | None = None,
        spatial_units: pl.DataFrame | None = None,
        spatial_weights: SpatialWeights | None = None,
    ) -> None:
        self.value_cols = list(value_cols)
        self.neighbor_col = neighbor_col
        self.weights_type = weights
        self.k = k
        self.distance_threshold = distance_threshold
        self.group_cols = list(group_cols or [])
        self.spatial_units = spatial_units
        self.spatial_weights = spatial_weights

    def run(self, event_frame: EventFrame) -> EventFrame:
        df = event_frame.collect()
        _require_spatial_columns(df, self.neighbor_col, self.value_cols, self.group_cols)
        _validate_spatial_values(df, self.neighbor_col, self.value_cols, self.group_cols)
        lag_cols = [f"{col}_spatial_lag" for col in self.value_cols]
        if df.is_empty():
            return self._register_features(_empty_spatial_result(event_frame, lag_cols))
        stats = _aggregate_stats(df, self.neighbor_col, self.value_cols, self.group_cols)
        weights = _reference_weights(
            df,
            event_frame,
            self.neighbor_col,
            self.weights_type,
            self.k,
            self.distance_threshold,
            self.spatial_units,
            self.spatial_weights,
            stats[self.neighbor_col].to_list(),
        )
        edges = _edges_from_weights(weights, self.neighbor_col)
        lag = _lag_means(stats, edges, self.group_cols, self.neighbor_col, self.value_cols)
        lag = lag.rename({col: f"{col}_spatial_lag" for col in self.value_cols})
        # maintain_order="left" keeps the panel's row order without a sort.
        out = df.join(
            lag,
            on=[*self.group_cols, self.neighbor_col],
            how="left",
            validate="m:1",
            maintain_order="left",
        )
        return self._register_features(event_frame.with_lazy_frame(out.lazy()))

    def _register_features(self, result: EventFrame) -> EventFrame:
        for col in self.value_cols:
            result = result.register_feature(
                f"{col}_spatial_lag",
                {"source_step": "SpatialLagStep"},
                provenance=FeatureProvenance(
                    produced_by="SpatialLagStep",
                    inputs=[col, self.neighbor_col],
                    tags={"spatial"},
                    metadata={
                        "weights": (
                            "supplied_binary"
                            if self.spatial_weights is not None
                            else self.weights_type
                        ),
                        "normalization": "row",
                        "group_cols": self.group_cols,
                        "islands": "null",
                    },
                ),
            )
        return result


class MultiScaleSpatialLagStep(Step):
    """Row-standardized binary spatial lag over disjoint queen rings.

    Ring ``r`` is the set of cells at Chebyshev distance exactly ``r`` from
    the unit, so rings are disjoint and ring 1 reproduces ``SpatialLagStep``
    exactly. Edges carry the ring number and each ring's lag is one join +
    one ``group_by`` mean in pure Polars, as in ``SpatialLagStep`` (rings are
    computed one at a time so the edges x snapshots intermediate stays
    bounded by the widest ring).

    Inputs:
        - value_cols on the (completed) panel; finite non-null values required
        - neighbor_col unit identifiers; grid_row/grid_col in the panel or in
          spatial_units supply the ring geometry (a supplied SpatialWeights
          object cannot define rings 2+ and is rejected at construction)

    Outputs:
        - {value_col}_spatial_lag_r{r} columns (Float64), one per value and ring

    Window policy: contemporaneous within a snapshot, as in ``SpatialLagStep``
    — the lag at bin ``t`` reads neighbours' values from the same snapshot
    only (group_cols are never pooled). The recipe feeds it lagged counts.
    Islands in a ring (units with no neighbours at that Chebyshev distance)
    yield null for that ring.
    """

    def __init__(
        self,
        value_cols: Sequence[str],
        neighbor_col: str = "grid_id",
        rings: Sequence[int] = (1, 2, 3),
        group_cols: Sequence[str] | None = None,
        spatial_units: pl.DataFrame | None = None,
        spatial_weights: SpatialWeights | None = None,
    ) -> None:
        if spatial_weights is not None:
            raise ValueError(
                "MultiScaleSpatialLagStep cannot derive queen rings from a supplied "
                "SpatialWeights object; supply grid_row/grid_col (in the panel or "
                "via spatial_units) instead"
            )
        self.value_cols = list(value_cols)
        self.neighbor_col = neighbor_col
        self.rings = _validate_rings(rings)
        self.group_cols = list(group_cols or [])
        self.spatial_units = spatial_units
        self.spatial_weights = spatial_weights

    def run(self, event_frame: EventFrame) -> EventFrame:
        df = event_frame.collect()
        _require_spatial_columns(df, self.neighbor_col, self.value_cols, self.group_cols)
        _validate_spatial_values(df, self.neighbor_col, self.value_cols, self.group_cols)
        lag_cols = [f"{col}_spatial_lag_r{ring}" for col in self.value_cols for ring in self.rings]
        if df.is_empty():
            return self._register_features(_empty_spatial_result(event_frame, lag_cols))
        stats = _aggregate_stats(df, self.neighbor_col, self.value_cols, self.group_cols)
        units = _grid_reference(
            df, self.neighbor_col, self.spatial_units, stats[self.neighbor_col].to_list()
        )
        edges = _queen_ring_edges(units, self.neighbor_col, self.rings)
        # One join + group_by per ring, not one over all rings at once: the
        # intermediate is edges x snapshots, and the widest ring alone is
        # already ~half of all edges, so rings are joined back one at a time
        # to bound peak memory on Chicago-size panels.
        out = df
        for ring in self.rings:
            ring_lag = _lag_means(
                stats,
                edges.filter(pl.col("ring") == ring).drop("ring"),
                self.group_cols,
                self.neighbor_col,
                self.value_cols,
            )
            ring_lag = ring_lag.rename(
                {col: f"{col}_spatial_lag_r{ring}" for col in self.value_cols}
            )
            out = out.join(
                ring_lag,
                on=[*self.group_cols, self.neighbor_col],
                how="left",
                validate="m:1",
                maintain_order="left",
            )
        return self._register_features(event_frame.with_lazy_frame(out.lazy()))

    def _register_features(self, result: EventFrame) -> EventFrame:
        for col in self.value_cols:
            for ring in self.rings:
                out_col = f"{col}_spatial_lag_r{ring}"
                result = result.register_feature(
                    out_col,
                    {"source_step": "MultiScaleSpatialLagStep"},
                    provenance=FeatureProvenance(
                        produced_by="MultiScaleSpatialLagStep",
                        inputs=[col, self.neighbor_col],
                        tags={"spatial"},
                        metadata={
                            "weights": "queen",
                            "normalization": "row",
                            "rings": self.rings,
                            "ring": ring,
                            "group_cols": self.group_cols,
                            "islands": "null",
                        },
                    ),
                )
        return result


class LocalMoranStep(Step):
    """Anselin local I with population second moment and conditional randomization.

    Binary neighbors are row-standardized, excluding self. P-values use a
    two-sided deviation from the conditional expectation, with +1 correction.
    No cluster significance is claimed when permutations=0 or variance is zero.
    """

    def __init__(
        self,
        value_col: str,
        spatial_unit_col: str = "grid_id",
        weights: str = "queen",
        permutations: int = 99,
        group_cols: Sequence[str] | None = None,
        k: int = 4,
        seed: int = 0,
        significance: float = 0.05,
        spatial_units: pl.DataFrame | None = None,
        spatial_weights: SpatialWeights | None = None,
    ) -> None:
        if not isinstance(permutations, int) or permutations < 0 or not 0 < significance < 1:
            raise ValueError("permutations must be non-negative and significance in (0, 1)")
        self.value_col, self.spatial_unit_col = value_col, spatial_unit_col
        self.weights_type, self.permutations = weights, permutations
        self.group_cols, self.k = list(group_cols or []), k
        self.seed, self.significance = seed, significance
        self.spatial_units, self.spatial_weights = spatial_units, spatial_weights

    def run(self, event_frame: EventFrame) -> EventFrame:
        i_col = f"{self.value_col}_local_moran_i"
        p_col = f"{self.value_col}_local_moran_p"
        cluster_col = f"{self.value_col}_moran_cluster"
        outputs = []
        for part, stats, rows in _spatial_snapshots(
            event_frame,
            self.spatial_unit_col,
            [self.value_col],
            self.group_cols,
            self.weights_type,
            self.k,
            None,
            self.spatial_units,
            self.spatial_weights,
        ):
            values = stats[self.value_col].to_numpy().astype(float)
            centered = values - values.mean()
            m2 = float(np.mean(centered**2))
            n = len(values)
            observed: list[float | None] = []
            pvalues: list[float | None] = []
            clusters = []
            rng = np.random.default_rng(self.seed)
            for i, row in enumerate(rows):
                statistic = pvalue = None
                cluster = "NS"
                if row and m2 > 0 and n > 1:
                    lag = float(centered[list(row)].mean())
                    statistic = float(centered[i] * lag / m2)
                    if self.permutations:
                        pool = np.delete(centered, i)
                        expected = -float(centered[i] ** 2) / ((n - 1) * m2)
                        simulated = [
                            float(
                                centered[i] * rng.choice(pool, len(row), replace=False).mean() / m2
                            )
                            for _ in range(self.permutations)
                        ]
                        extreme = sum(
                            abs(x - expected) >= abs(statistic - expected) - 1e-12
                            for x in simulated
                        )
                        pvalue = (extreme + 1) / (self.permutations + 1)
                        if pvalue <= self.significance and centered[i] != 0 and lag != 0:
                            cluster = ("H" if centered[i] > 0 else "L") + ("H" if lag > 0 else "L")
                observed.append(statistic)
                pvalues.append(pvalue)
                clusters.append(cluster)
            lookup = stats.select(self.spatial_unit_col).with_columns(
                pl.Series(i_col, observed, dtype=pl.Float64),
                pl.Series(p_col, pvalues, dtype=pl.Float64),
                pl.Series(cluster_col, clusters, dtype=pl.String),
            )
            outputs.append(
                part.join(
                    lookup,
                    on=self.spatial_unit_col,
                    how="left",
                    validate="m:1",
                    maintain_order="left",
                )
            )
        result = _spatial_result(
            event_frame, outputs, {i_col: pl.Float64, p_col: pl.Float64, cluster_col: pl.String}
        )
        for name in (i_col, p_col, cluster_col):
            result = result.register_feature(
                name,
                {"source_step": "LocalMoranStep"},
                provenance=FeatureProvenance(
                    produced_by="LocalMoranStep",
                    inputs=[self.value_col, self.spatial_unit_col],
                    tags={"spatial"},
                    metadata={
                        "weights": (
                            "supplied_binary"
                            if self.spatial_weights is not None
                            else self.weights_type
                        ),
                        "permutations": self.permutations,
                        "seed": self.seed,
                        "variance_divisor": "n",
                        "group_cols": self.group_cols,
                        "alternative": "two-sided conditional deviation",
                        "significance": self.significance,
                    },
                ),
            )
        return result


class GetisOrdStep(Step):
    """Binary Gi* normal z-score using neighbors plus self and population variance.

    Default adjacency is queen contiguity; a threshold selects metric distance
    bands instead. Degenerate variance/denominator produces null, never evidence
    of a hotspot. Labels use unadjusted two-sided normal cutoff 1.96.
    """

    def __init__(
        self,
        value_col: str,
        spatial_unit_col: str = "grid_id",
        distance_threshold: float | None = None,
        group_cols: Sequence[str] | None = None,
        spatial_units: pl.DataFrame | None = None,
        spatial_weights: SpatialWeights | None = None,
    ) -> None:
        self.value_col, self.spatial_unit_col = value_col, spatial_unit_col
        self.distance_threshold = distance_threshold
        self.group_cols = list(group_cols or [])
        self.spatial_units, self.spatial_weights = spatial_units, spatial_weights

    def run(self, event_frame: EventFrame) -> EventFrame:
        gi_col, hotspot_col = f"{self.value_col}_gi_star", f"{self.value_col}_hotspot"
        outputs = []
        for part, stats, rows in _spatial_snapshots(
            event_frame,
            self.spatial_unit_col,
            [self.value_col],
            self.group_cols,
            "distance" if self.distance_threshold is not None else "queen",
            4,
            self.distance_threshold,
            self.spatial_units,
            self.spatial_weights,
            aggregation="sum",
        ):
            values = stats[self.value_col].to_numpy().astype(float)
            n, mean, std = len(values), float(values.mean()), float(values.std())
            scores: list[float | None] = []
            labels = []
            for i, row in enumerate(rows):
                indices = list(row) + [i]
                w = len(indices)
                denom = std * np.sqrt((n * w - w**2) / (n - 1)) if n > 1 else 0.0
                score = float((values[indices].sum() - mean * w) / denom) if denom > 0 else None
                scores.append(score)
                labels.append(
                    "hot"
                    if score is not None and score > 1.96
                    else "cold" if score is not None and score < -1.96 else "not_significant"
                )
            lookup = stats.select(self.spatial_unit_col).with_columns(
                pl.Series(gi_col, scores, dtype=pl.Float64),
                pl.Series(hotspot_col, labels, dtype=pl.String),
            )
            outputs.append(
                part.join(
                    lookup,
                    on=self.spatial_unit_col,
                    how="left",
                    validate="m:1",
                    maintain_order="left",
                )
            )
        result = _spatial_result(event_frame, outputs, {gi_col: pl.Float64, hotspot_col: pl.String})
        for name in (gi_col, hotspot_col):
            result = result.register_feature(
                name,
                {"source_step": "GetisOrdStep"},
                provenance=FeatureProvenance(
                    produced_by="GetisOrdStep",
                    inputs=[self.value_col, self.spatial_unit_col],
                    tags={"spatial"},
                    metadata={
                        "self_weight": 1,
                        "weights": "binary",
                        "group_cols": self.group_cols,
                        "inference": "unadjusted normal z-score",
                    },
                ),
            )
        return result


def _spatial_snapshots(
    event_frame: EventFrame,
    unit_col: str,
    value_cols: list[str],
    group_cols: list[str],
    method: str,
    k: int,
    threshold: float | None,
    spatial_units: pl.DataFrame | None,
    supplied: SpatialWeights | None,
    *,
    aggregation: str = "mean",
) -> Iterator[tuple[pl.DataFrame, pl.DataFrame, tuple[tuple[int, ...], ...]]]:
    df = event_frame.collect()
    if df[unit_col].null_count() or any(df[c].null_count() for c in group_cols):
        raise ValueError("Spatial unit and snapshot keys cannot contain nulls")
    if any(not np.isfinite(df[c].to_numpy().astype(float)).all() for c in value_cols):
        raise ValueError(
            "Spatial statistics require finite observed values; choose a missing-data policy first"
        )
    for part in (df.partition_by(group_cols, maintain_order=True) if group_cols else [df]):
        if part.is_empty():
            continue
        aggs = [pl.col(c).sum() if aggregation == "sum" else pl.col(c).mean() for c in value_cols]
        stats = part.group_by(unit_col).agg(aggs).sort(unit_col)
        ids = stats[unit_col].to_list()
        if supplied is not None:
            weights = supplied
        else:
            if spatial_units is not None:
                reference = spatial_units
            elif {"grid_row", "grid_col"} <= set(part.columns) and method in {"queen", "rook"}:
                reference = part.select(unit_col, "grid_row", "grid_col").unique(
                    maintain_order=True
                )
            elif event_frame.schema.geometry_col and method in {"queen", "rook"}:
                reference = part.select(
                    unit_col, pl.col(event_frame.schema.geometry_col).alias("geometry")
                ).unique(maintain_order=True)
            elif event_frame.schema.lon_col and event_frame.schema.lat_col:
                reference = part.select(
                    unit_col,
                    pl.col(event_frame.schema.lon_col).alias("x"),
                    pl.col(event_frame.schema.lat_col).alias("y"),
                ).unique(maintain_order=True)
            else:
                raise ValueError("Supply a spatial reference table or SpatialWeights")
            weights = SpatialWeights.from_units(
                reference,
                unit_col,
                method=method,
                crs=event_frame.metadata.crs,
                k=k,
                distance_threshold=threshold,
            )
        yield part, stats, weights.aligned(ids)


def _spatial_result(
    event_frame: EventFrame,
    outputs: list[pl.DataFrame],
    columns: dict[str, pl.DataType | type[pl.DataType]],
) -> EventFrame:
    lf = (
        pl.concat(outputs).lazy()
        if outputs
        else event_frame.lazy_frame.with_columns(
            [pl.lit(None, dtype=dtype).alias(name) for name, dtype in columns.items()]
        )
    )
    return event_frame.with_lazy_frame(lf)
