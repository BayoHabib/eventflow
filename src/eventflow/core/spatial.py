"""Spatial operations for event data."""

import math
from typing import cast

import polars as pl
from pyproj import CRS, Transformer
from shapely import STRtree, from_wkt, geometry

from eventflow.core.event_frame import EventFrame
from eventflow.core.grid import GridSpec, grid_cell_id, grid_id_expr, require_metric_crs
from eventflow.core.utils import get_logger

logger = get_logger(__name__)


def transform_crs(
    event_frame: EventFrame,
    target_crs: str,
) -> EventFrame:
    """
    Transform event coordinates to a different CRS.

    Args:
        event_frame: Input EventFrame
        target_crs: Target CRS (e.g., "EPSG:26971")

    Returns:
        EventFrame with transformed coordinates
    """
    if event_frame.metadata.crs == target_crs:
        logger.debug(f"CRS already matches target: {target_crs}")
        return event_frame

    logger.info(f"Transforming CRS from {event_frame.metadata.crs} to {target_crs}")
    schema = event_frame.schema
    if schema.lat_col is None or schema.lon_col is None:
        raise ValueError("EventFrame must have lat/lon columns for CRS transformation")

    source_crs = event_frame.metadata.crs

    # A projection is a materialization boundary. Native PROJ objects must not
    # be created in expression callbacks that the query optimizer can duplicate
    # onto different short-lived engine threads. Create/use the handle in this
    # step's owning Python thread; async callers already offload the entire step.
    df = event_frame.collect()
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    x, y = transformer.transform(df[schema.lon_col].to_numpy(), df[schema.lat_col].to_numpy())
    lf = df.with_columns(
        pl.Series(f"{schema.lon_col}_proj", x, dtype=pl.Float64),
        pl.Series(f"{schema.lat_col}_proj", y, dtype=pl.Float64),
    ).lazy()

    return (
        event_frame.with_lazy_frame(lf)
        .with_metadata(crs=target_crs)
        .with_schema(lon_col=f"{schema.lon_col}_proj", lat_col=f"{schema.lat_col}_proj")
        .require_context(spatial_crs=target_crs)
    )


def create_grid(
    bounds: tuple[float, float, float, float],
    size_m: float,
    crs: str = "EPSG:4326",
    *,
    origin: tuple[float, float] | None = None,
) -> pl.DataFrame:
    """Create canonical half-open metric cells covering bounds.

    Supply the same origin across calls to keep identities stable as bounds change.
    The generated reference includes its lattice parameters and signed cell indices.
    IDs use signed_cantor_v1; legacy row-major/100000-stride tables must be migrated.
    """
    require_metric_crs(crs)
    minx, miny, maxx, maxy = bounds
    if not all(math.isfinite(v) for v in bounds) or minx > maxx or miny > maxy:
        raise ValueError("Grid bounds must be finite and ordered")
    spec = GridSpec(size_m, origin if origin is not None else (minx, miny), crs)
    first_col, last_col = [math.floor((v - spec.origin[0]) / size_m) for v in (minx, maxx)]
    first_row, last_row = [math.floor((v - spec.origin[1]) / size_m) for v in (miny, maxy)]
    grid_cell_id(first_row, first_col)
    grid_cell_id(last_row, last_col)
    cells = []
    for row in range(first_row, last_row + 1):
        for col in range(first_col, last_col + 1):
            x, y = spec.origin[0] + col * size_m, spec.origin[1] + row * size_m
            cells.append(
                {
                    "grid_id": grid_cell_id(row, col),
                    "grid_row": row,
                    "grid_col": col,
                    "minx": x,
                    "miny": y,
                    "maxx": x + size_m,
                    "maxy": y + size_m,
                    "geometry": geometry.box(x, y, x + size_m, y + size_m).wkt,
                    "grid_origin_x": spec.origin[0],
                    "grid_origin_y": spec.origin[1],
                    "grid_crs": spec.crs,
                }
            )
    return pl.DataFrame(cells)


def assign_to_grid(event_frame: EventFrame, grid: pl.DataFrame) -> EventFrame:
    """Assign canonical IDs present in a validated square-cell reference.

    Rows outside the supplied reference receive null grid_id, never a fabricated
    ID that happens to alias an in-domain cell. Row order and grid indices survive.
    """
    schema = event_frame.schema
    if schema.lon_col is None or schema.lat_col is None:
        raise ValueError("EventFrame must have lat/lon columns for grid assignment")
    require_metric_crs(event_frame.metadata.crs)
    if grid.is_empty():
        raise ValueError("Grid must contain at least one cell")
    size = float(grid["maxx"][0] - grid["minx"][0])
    origin_cols = {"grid_origin_x", "grid_origin_y", "grid_crs"}
    if origin_cols & set(grid.columns) and not origin_cols <= set(grid.columns):
        raise ValueError("Grid reference must include all origin/CRS fields or none")
    if origin_cols <= set(grid.columns):
        if any(grid[c].null_count() or grid[c].n_unique() != 1 for c in origin_cols):
            raise ValueError("Grid reference must have one non-null origin and CRS")
        if CRS.from_user_input(grid["grid_crs"][0]) != CRS.from_user_input(
            event_frame.metadata.crs
        ):
            raise ValueError("Grid and event CRS must agree")
        origin = (float(grid["grid_origin_x"][0]), float(grid["grid_origin_y"][0]))
    else:
        origin = (float(cast(float, grid["minx"].min())), float(cast(float, grid["miny"].min())))
    spec = GridSpec(size, origin, event_frame.metadata.crs)
    # A sparse or shuffled reference is allowed, but its cell identity must agree.
    reference = (
        grid.lazy()
        .with_columns(
            ((pl.col("minx") - origin[0]) / size).round(0).cast(pl.Int64).alias("grid_col"),
            ((pl.col("miny") - origin[1]) / size).round(0).cast(pl.Int64).alias("grid_row"),
        )
        .with_columns(grid_id_expr())
        .collect()
    )
    for col in ("minx", "miny", "maxx", "maxy"):
        if grid[col].null_count() or not grid[col].is_finite().all():
            raise ValueError("Grid bounds must be finite and non-null")
    for row in reference.iter_rows(named=True):
        x, y = origin[0] + row["grid_col"] * size, origin[1] + row["grid_row"] * size
        if not all(
            math.isclose(row[c], value, rel_tol=0, abs_tol=max(1e-8, size * 1e-10))
            for c, value in [("minx", x), ("miny", y), ("maxx", x + size), ("maxy", y + size)]
        ):
            raise ValueError("Grid cells must be aligned squares with one common size")
    if grid["grid_id"].null_count() or grid["grid_id"].n_unique() != grid.height:
        raise ValueError("Grid IDs must be unique and non-null")
    if grid["grid_id"].to_list() != reference["grid_id"].to_list():
        raise ValueError(
            "Legacy or incompatible grid IDs; regenerate with create_grid and migrate by cell geometry"
        )
    if {"grid_row", "grid_col"} <= set(grid.columns):
        if any(grid[c].to_list() != reference[c].to_list() for c in ("grid_row", "grid_col")):
            raise ValueError("Grid indices disagree with cell geometry")
    lf = spec.assign(event_frame.lazy_frame, schema.lon_col, schema.lat_col).drop("grid_id")
    lf = lf.join(
        reference.lazy().select("grid_row", "grid_col", "grid_id"),
        on=["grid_row", "grid_col"],
        how="left",
        maintain_order="left",
    )
    return event_frame.with_lazy_frame(lf).with_metadata(
        grid_size_m=size, custom={**event_frame.metadata.custom, "grid_spec": spec.metadata()}
    )


def assign_to_zones(
    event_frame: EventFrame,
    zones: pl.DataFrame,
    zone_id_col: str = "zone_id",
    zone_geom_col: str = "geometry",
) -> EventFrame:
    """
    Assign events to predefined zones (e.g., neighborhoods, census tracts).

    Args:
        event_frame: Input EventFrame with coordinates
        zones: DataFrame with zone geometries
        zone_id_col: Name of zone ID column
        zone_geom_col: Name of geometry column (WKT format)

    Returns:
        EventFrame with zone_id column added
    """
    schema = event_frame.schema

    if schema.lat_col is None or schema.lon_col is None:
        raise ValueError("EventFrame must have lat/lon columns for zone assignment")

    # Build spatial index
    zone_geoms = [from_wkt(wkt) for wkt in zones[zone_geom_col]]
    zone_ids_raw = zones[zone_id_col].to_list()
    zone_ids: list[int | None] = []
    for value in zone_ids_raw:
        try:
            zone_ids.append(int(value))
        except (TypeError, ValueError):
            logger.debug("Skipping non-integer zone id value: %s", value)
            zone_ids.append(None)
    tree = STRtree(zone_geoms)

    # Assign events to zones
    def find_zone(lon: float, lat: float) -> int | None:
        """Find zone containing point."""
        point = geometry.Point(lon, lat)
        idx = tree.query(point, predicate="within")
        if len(idx) > 0:
            first_idx = int(idx[0])
            zone_id = zone_ids[first_idx]
            return zone_id
        return None

    lf = event_frame.lazy_frame.with_columns(
        [
            pl.struct([schema.lon_col, schema.lat_col])
            .map_elements(
                lambda coord: find_zone(coord[schema.lon_col], coord[schema.lat_col]),
                return_dtype=pl.Int32,
            )
            .alias("zone_id")
        ]
    )

    return event_frame.with_lazy_frame(lf)


def compute_distances(
    event_frame: EventFrame,
    points_of_interest: pl.DataFrame,
    poi_lon_col: str = "longitude",
    poi_lat_col: str = "latitude",
    poi_name_col: str = "name",
) -> EventFrame:
    """
    Compute distances from events to points of interest.

    Args:
        event_frame: Input EventFrame
        points_of_interest: DataFrame with POI coordinates
        poi_lon_col: POI longitude column
        poi_lat_col: POI latitude column
        poi_name_col: POI name column

    Returns:
        EventFrame with distance columns added
    """
    schema = event_frame.schema

    if schema.lat_col is None or schema.lon_col is None:
        raise ValueError("EventFrame must have lat/lon columns for distance computation")

    # Compute Euclidean distances (assumes projected coordinates)
    lf = event_frame.lazy_frame

    for poi in points_of_interest.iter_rows(named=True):
        poi_name = poi[poi_name_col]
        poi_lon = poi[poi_lon_col]
        poi_lat = poi[poi_lat_col]

        lf = lf.with_columns(
            [
                (
                    (
                        (pl.col(schema.lon_col) - poi_lon) ** 2
                        + (pl.col(schema.lat_col) - poi_lat) ** 2
                    )
                    ** 0.5
                ).alias(f"dist_to_{poi_name}")
            ]
        )

    return event_frame.with_lazy_frame(lf)
