"""Spatial operations for event data."""

import polars as pl
from pyproj import Transformer
from shapely import STRtree, geometry

from eventflow.core.event_frame import EventFrame
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

    transformer = Transformer.from_crs(
        event_frame.metadata.crs,
        target_crs,
        always_xy=True,
    )

    # Transform coordinates
    lf = event_frame.lazy_frame.with_columns([
        pl.struct([schema.lon_col, schema.lat_col])
        .map_elements(
            lambda coord: transformer.transform(coord[schema.lon_col], coord[schema.lat_col]),
            return_dtype=pl.List(pl.Float64),
        )
        .alias("_transformed")
    ]).with_columns([
        pl.col("_transformed").list.get(0).alias(f"{schema.lon_col}_proj"),
        pl.col("_transformed").list.get(1).alias(f"{schema.lat_col}_proj"),
    ]).drop("_transformed")

    return event_frame.with_lazy_frame(lf).with_metadata(crs=target_crs)


def create_grid(
    bounds: tuple[float, float, float, float],
    size_m: float,
    crs: str = "EPSG:4326",
) -> pl.DataFrame:
    """
    Create a spatial grid covering the given bounds.

    Args:
        bounds: Bounding box (minx, miny, maxx, maxy)
        size_m: Grid cell size in meters (assumes projected CRS)
        crs: Coordinate reference system

    Returns:
        DataFrame with grid cells (grid_id, minx, miny, maxx, maxy, geometry)
    """
    minx, miny, maxx, maxy = bounds

    # Calculate number of cells
    n_cols = int((maxx - minx) / size_m) + 1
    n_rows = int((maxy - miny) / size_m) + 1

    logger.info(f"Creating spatial grid: {n_cols}x{n_rows} cells ({n_cols * n_rows} total) at {size_m}m resolution")

    # Generate grid cells
    cells = []
    grid_id = 0

    for i in range(n_rows):
        for j in range(n_cols):
            cell_minx = minx + j * size_m
            cell_miny = miny + i * size_m
            cell_maxx = cell_minx + size_m
            cell_maxy = cell_miny + size_m

            cells.append({
                "grid_id": grid_id,
                "minx": cell_minx,
                "miny": cell_miny,
                "maxx": cell_maxx,
                "maxy": cell_maxy,
                "geometry": geometry.box(cell_minx, cell_miny, cell_maxx, cell_maxy).wkt,
            })
            grid_id += 1

    return pl.DataFrame(cells)


def assign_to_grid(
    event_frame: EventFrame,
    grid: pl.DataFrame,
) -> EventFrame:
    """
    Assign events to grid cells.

    Args:
        event_frame: Input EventFrame with coordinates
        grid: Grid DataFrame from create_grid()

    Returns:
        EventFrame with grid_id column added
    """
    schema = event_frame.schema

    if schema.lat_col is None or schema.lon_col is None:
        raise ValueError("EventFrame must have lat/lon columns for grid assignment")

    # Get grid parameters
    grid_data = grid.to_dict(as_series=False)

    # Assign events to grid cells using spatial join logic
    # This is a simplified version - in production, use spatial index for efficiency
    lf = event_frame.lazy_frame

    # For each event, find the grid cell it falls into
    # This implementation assumes projected coordinates
    grid_size = grid_data["maxx"][0] - grid_data["minx"][0]
    if grid_size <= 0:
        raise ValueError("Grid definition must contain positive cell size")

    min_x = min(grid_data["minx"])
    min_y = min(grid_data["miny"])
    unique_minx = {value for value in grid_data["minx"]}
    n_cols = max(len(unique_minx), 1)

    lf = lf.with_columns([
        (
            ((pl.col(schema.lon_col) - min_x) / grid_size).floor().cast(pl.Int32)
        ).alias("_col_idx"),
        (
            ((pl.col(schema.lat_col) - min_y) / grid_size).floor().cast(pl.Int32)
        ).alias("_row_idx"),
    ]).with_columns([
        (
            pl.col("_row_idx") * pl.lit(n_cols, dtype=pl.Int32) + pl.col("_col_idx")
        ).alias("grid_id")
    ]).drop(["_col_idx", "_row_idx"])

    return event_frame.with_lazy_frame(lf)


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
    zone_geoms = [geometry.from_wkt(wkt) for wkt in zones[zone_geom_col]]
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
        idx = tree.query(point, predicate="contains")
        if len(idx) > 0:
            first_idx = int(idx[0])
            zone_id = zone_ids[first_idx]
            return zone_id
        return None

    lf = event_frame.lazy_frame.with_columns([
        pl.struct([schema.lon_col, schema.lat_col])
        .map_elements(
            lambda coord: find_zone(coord[schema.lon_col], coord[schema.lat_col]),
            return_dtype=pl.Int32,
        )
        .alias("zone_id")
    ])

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

        lf = lf.with_columns([
            (
                ((pl.col(schema.lon_col) - poi_lon) ** 2
                 + (pl.col(schema.lat_col) - poi_lat) ** 2) ** 0.5
            ).alias(f"dist_to_{poi_name}")
        ])

    return event_frame.with_lazy_frame(lf)
