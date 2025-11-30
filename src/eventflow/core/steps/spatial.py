"""Spatial pipeline steps with registry integration.

Each step:
- Inherits from Step base class
- Registers inputs/outputs via FeatureProvenance
- Updates EventSchema properly
- Supports Pydantic config models for validation
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import polars as pl
from pydantic import BaseModel, Field
from pyproj import Transformer
from shapely import STRtree, geometry
from shapely.wkt import loads as wkt_loads

from eventflow.core.pipeline import Step
from eventflow.core.schema import FeatureProvenance
from eventflow.core.utils import get_logger

if TYPE_CHECKING:
    from eventflow.core.event_frame import EventFrame

logger = get_logger(__name__)

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
    k: int = Field(default=4, ge=1, description="Number of neighbors for KNN weights")
    distance_threshold: float | None = Field(
        default=None, description="Distance threshold for distance-based weights"
    )


class LocalAutocorrelationConfig(BaseModel):
    """Configuration for local spatial autocorrelation step."""

    value_col: str = Field(..., description="Column to compute local autocorrelation for")
    spatial_unit_col: str = Field(default="grid_id", description="Spatial unit identifier")
    weights: Literal["queen", "rook", "knn"] = Field(
        default="queen", description="Spatial weights type"
    )
    permutations: int = Field(default=99, ge=0, description="Number of permutations for p-value")


class GetisOrdConfig(BaseModel):
    """Configuration for Getis-Ord Gi* step."""

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

        transformer = Transformer.from_crs(
            event_frame.metadata.crs,
            self.target_crs,
            always_xy=True,
        )

        lon_col = schema.lon_col
        lat_col = schema.lat_col

        lf = (
            event_frame.lazy_frame.with_columns(
                [
                    pl.struct([lon_col, lat_col])
                    .map_elements(
                        lambda coord: transformer.transform(coord[lon_col], coord[lat_col]),
                        return_dtype=pl.List(pl.Float64),
                    )
                    .alias("_transformed")
                ]
            )
            .with_columns(
                [
                    pl.col("_transformed").list.get(0).alias(f"{lon_col}_proj"),
                    pl.col("_transformed").list.get(1).alias(f"{lat_col}_proj"),
                ]
            )
            .drop("_transformed")
        )

        provenance = FeatureProvenance(
            produced_by="TransformCRSStep",
            inputs=[lon_col, lat_col],
            tags={"spatial"},
            description=f"CRS transformation to {self.target_crs}",
        )

        result = event_frame.with_lazy_frame(lf).with_metadata(crs=self.target_crs)

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
        self.grid_size_m = grid_size_m
        self.origin = origin

    def run(self, event_frame: EventFrame) -> EventFrame:
        """Execute grid assignment."""
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

        grid_size = self.grid_size_m

        lf = (
            lf.with_columns(
                [
                    (((pl.col(lon_col) - min_x) / grid_size).floor().cast(pl.Int32)).alias(
                        "_col_idx"
                    ),
                    (((pl.col(lat_col) - min_y) / grid_size).floor().cast(pl.Int32)).alias(
                        "_row_idx"
                    ),
                ]
            )
            .with_columns(
                [
                    # Compute unique grid_id from row/col indices
                    # Using a large multiplier to ensure uniqueness
                    (
                        pl.col("_row_idx") * pl.lit(100000, dtype=pl.Int32) + pl.col("_col_idx")
                    ).alias("grid_id")
                ]
            )
            .drop(["_col_idx", "_row_idx"])
        )

        provenance = FeatureProvenance(
            produced_by="AssignToGridStep",
            inputs=[lon_col, lat_col],
            tags={"spatial"},
            description=f"Grid assignment at {grid_size}m resolution",
            metadata={"grid_size_m": grid_size, "origin": (min_x, min_y)},
        )

        result = event_frame.with_lazy_frame(lf).with_metadata(grid_size_m=grid_size)
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
            idx = tree.query(point, predicate="contains")
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
    """Compute Kernel Density Estimation for event locations.

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


class SpatialLagStep(Step):
    """Compute spatial lag (weighted average of neighbors) for specified columns.

    Inputs:
        - neighbor_col (spatial unit identifier)
        - value_cols to compute lag for

    Outputs:
        - {value_col}_spatial_lag columns
    """

    def __init__(
        self,
        value_cols: Sequence[str],
        neighbor_col: str = "grid_id",
        weights: str = "queen",
        k: int = 4,
        distance_threshold: float | None = None,
    ) -> None:
        self.value_cols = list(value_cols)
        self.neighbor_col = neighbor_col
        self.weights_type = weights
        self.k = k
        self.distance_threshold = distance_threshold

    def run(self, event_frame: EventFrame) -> EventFrame:
        """Execute spatial lag computation."""
        lf = event_frame.lazy_frame

        # First, compute cell-level aggregates
        cell_stats = lf.group_by(self.neighbor_col).agg(
            [pl.col(col).mean().alias(f"{col}_cell_mean") for col in self.value_cols]
        )

        # For grid-based data, compute neighbors based on spatial adjacency
        # This is a simplified implementation using grid ID arithmetic
        # For queen contiguity on a grid: neighbors differ by ±1 in row or col
        result = event_frame

        for value_col in self.value_cols:
            lag_col = f"{value_col}_spatial_lag"
            cell_mean_col = f"{value_col}_cell_mean"

            # Self-join to get neighbor values (simplified: using grid adjacency)
            # In practice, you'd use proper spatial weights matrix
            # Here we approximate by averaging cells with nearby grid_ids

            # Join cell means to main frame
            lf = lf.join(
                cell_stats.select([self.neighbor_col, cell_mean_col]),
                on=self.neighbor_col,
                how="left",
            )

            # Compute lag as shifted values (simplified grid-based approach)
            # Create lag by considering adjacent grid cells
            # For row-based grid_id = row * 100000 + col:
            # neighbors are ±1 (horizontal) and ±100000 (vertical)

            # Simplified: use rolling average of nearby grid_ids
            lf = lf.sort(self.neighbor_col).with_columns(
                [
                    pl.col(cell_mean_col)
                    .rolling_mean(window_size=3, center=True, min_samples=1)
                    .alias(lag_col)
                ]
            )

            # Remove intermediate column
            lf = lf.drop(cell_mean_col)

            provenance = FeatureProvenance(
                produced_by="SpatialLagStep",
                inputs=[value_col, self.neighbor_col],
                tags={"spatial"},
                description=f"Spatial lag of {value_col} using {self.weights_type} weights",
            )
            result = result.register_feature(
                lag_col,
                {"source_step": "SpatialLagStep", "value_col": value_col},
                provenance=provenance,
            )

        return result.with_lazy_frame(lf)


class LocalMoranStep(Step):
    """Compute Local Moran's I for spatial autocorrelation.

    Inputs:
        - value_col to compute autocorrelation for
        - spatial_unit_col for spatial unit identifier

    Outputs:
        - {value_col}_local_moran_i: Local Moran's I statistic
        - {value_col}_moran_cluster: Cluster classification (HH, LL, HL, LH)
    """

    def __init__(
        self,
        value_col: str,
        spatial_unit_col: str = "grid_id",
        weights: str = "queen",
        permutations: int = 99,
    ) -> None:
        self.value_col = value_col
        self.spatial_unit_col = spatial_unit_col
        self.weights_type = weights
        self.permutations = permutations

    def run(self, event_frame: EventFrame) -> EventFrame:
        """Execute Local Moran's I computation."""
        lf = event_frame.lazy_frame

        i_col = f"{self.value_col}_local_moran_i"
        cluster_col = f"{self.value_col}_moran_cluster"

        # Compute cell-level statistics
        cell_stats = lf.group_by(self.spatial_unit_col).agg(
            [pl.col(self.value_col).mean().alias("cell_mean")]
        )

        # Compute global mean and std
        global_stats = (
            cell_stats.select(
                [
                    pl.col("cell_mean").mean().alias("global_mean"),
                    pl.col("cell_mean").std().alias("global_std"),
                ]
            )
            .collect()
            .row(0)
        )
        global_mean, global_std = global_stats[0], global_stats[1]

        if global_std is None or global_std == 0:
            global_std = 1.0  # Avoid division by zero

        # Standardize cell values
        cell_stats = cell_stats.with_columns(
            [((pl.col("cell_mean") - global_mean) / global_std).alias("z_score")]
        )

        # Compute spatial lag of z-scores (simplified: rolling mean)
        cell_stats = cell_stats.sort(self.spatial_unit_col).with_columns(
            [
                pl.col("z_score")
                .rolling_mean(window_size=3, center=True, min_samples=1)
                .alias("z_lag")
            ]
        )

        # Local Moran's I = z * spatial_lag(z)
        cell_stats = cell_stats.with_columns([(pl.col("z_score") * pl.col("z_lag")).alias(i_col)])

        # Classify clusters
        cell_stats = cell_stats.with_columns(
            [
                pl.when((pl.col("z_score") > 0) & (pl.col("z_lag") > 0))
                .then(pl.lit("HH"))
                .when((pl.col("z_score") < 0) & (pl.col("z_lag") < 0))
                .then(pl.lit("LL"))
                .when((pl.col("z_score") > 0) & (pl.col("z_lag") < 0))
                .then(pl.lit("HL"))
                .when((pl.col("z_score") < 0) & (pl.col("z_lag") > 0))
                .then(pl.lit("LH"))
                .otherwise(pl.lit("NS"))
                .alias(cluster_col)
            ]
        )

        # Join back to main frame
        join_cols = [self.spatial_unit_col, i_col, cluster_col]
        result_lf = lf.join(cell_stats.select(join_cols), on=self.spatial_unit_col, how="left")

        provenance = FeatureProvenance(
            produced_by="LocalMoranStep",
            inputs=[self.value_col, self.spatial_unit_col],
            tags={"spatial"},
            description=f"Local Moran's I for {self.value_col}",
            metadata={"weights": self.weights_type, "permutations": self.permutations},
        )

        result = event_frame.with_lazy_frame(result_lf)
        result = result.register_feature(
            i_col,
            {"source_step": "LocalMoranStep", "value_col": self.value_col},
            provenance=provenance,
        )
        result = result.register_feature(
            cluster_col,
            {"source_step": "LocalMoranStep", "value_col": self.value_col},
            provenance=provenance,
        )

        return result


class GetisOrdStep(Step):
    """Compute Getis-Ord Gi* hotspot statistic.

    Inputs:
        - value_col to compute hotspot statistics for
        - spatial_unit_col for spatial unit identifier

    Outputs:
        - {value_col}_gi_star: Gi* z-score
        - {value_col}_hotspot: Hotspot classification (hot, cold, not_significant)
    """

    def __init__(
        self,
        value_col: str,
        spatial_unit_col: str = "grid_id",
        distance_threshold: float | None = None,
    ) -> None:
        self.value_col = value_col
        self.spatial_unit_col = spatial_unit_col
        self.distance_threshold = distance_threshold

    def run(self, event_frame: EventFrame) -> EventFrame:
        """Execute Getis-Ord Gi* computation."""
        lf = event_frame.lazy_frame

        gi_col = f"{self.value_col}_gi_star"
        hotspot_col = f"{self.value_col}_hotspot"

        # Compute cell-level statistics
        cell_stats = lf.group_by(self.spatial_unit_col).agg(
            [
                pl.col(self.value_col).sum().alias("cell_sum"),
                pl.col(self.value_col).count().alias("cell_count"),
            ]
        )

        # Compute global statistics
        global_stats = (
            cell_stats.select(
                [
                    pl.col("cell_sum").sum().alias("total_sum"),
                    pl.col("cell_count").sum().alias("total_count"),
                    pl.col("cell_sum").mean().alias("global_mean"),
                    pl.col("cell_sum").std().alias("global_std"),
                ]
            )
            .collect()
            .row(0)
        )
        total_sum, total_count, global_mean, global_std = global_stats

        if global_std is None or global_std == 0:
            global_std = 1.0

        # Compute local sum including neighbors (simplified: rolling sum)
        cell_stats = cell_stats.sort(self.spatial_unit_col).with_columns(
            [
                pl.col("cell_sum")
                .rolling_sum(window_size=3, center=True, min_samples=1)
                .alias("local_sum")
            ]
        )

        # Gi* = (local_sum - expected) / std
        w = 3  # Window size (number of neighbors + self)
        expected = global_mean * w if global_mean is not None else 0

        cell_stats = cell_stats.with_columns(
            [((pl.col("local_sum") - expected) / global_std).alias(gi_col)]
        )

        # Classify hotspots (z > 1.96 = hot, z < -1.96 = cold)
        cell_stats = cell_stats.with_columns(
            [
                pl.when(pl.col(gi_col) > 1.96)
                .then(pl.lit("hot"))
                .when(pl.col(gi_col) < -1.96)
                .then(pl.lit("cold"))
                .otherwise(pl.lit("not_significant"))
                .alias(hotspot_col)
            ]
        )

        # Join back to main frame
        join_cols = [self.spatial_unit_col, gi_col, hotspot_col]
        result_lf = lf.join(cell_stats.select(join_cols), on=self.spatial_unit_col, how="left")

        provenance = FeatureProvenance(
            produced_by="GetisOrdStep",
            inputs=[self.value_col, self.spatial_unit_col],
            tags={"spatial"},
            description=f"Getis-Ord Gi* for {self.value_col}",
        )

        result = event_frame.with_lazy_frame(result_lf)
        result = result.register_feature(
            gi_col,
            {"source_step": "GetisOrdStep", "value_col": self.value_col},
            provenance=provenance,
        )
        result = result.register_feature(
            hotspot_col,
            {"source_step": "GetisOrdStep", "value_col": self.value_col},
            provenance=provenance,
        )

        return result
