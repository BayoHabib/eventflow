"""Tests for spatial operations."""

import polars as pl
import pytest
from shapely import geometry

from eventflow.core.event_frame import EventFrame
from eventflow.core.schema import EventSchema, EventMetadata
from eventflow.core.spatial import (
    assign_to_grid,
    assign_to_zones,
    compute_distances,
    create_grid,
    transform_crs,
)


def _sample_event_frame() -> EventFrame:
    data = {
        "timestamp": ["2024-01-01", "2024-01-02"],
        "latitude": [0.5, 1.5],
        "longitude": [0.5, 1.5],
    }
    lf = pl.LazyFrame(data).with_columns(pl.col("timestamp").str.to_datetime())
    schema = EventSchema(timestamp_col="timestamp", lat_col="latitude", lon_col="longitude")
    metadata = EventMetadata(dataset_name="test", crs="EPSG:4326")
    return EventFrame(lf, schema, metadata)


def test_transform_crs_adds_projected_columns() -> None:
    ef = _sample_event_frame()
    transformed = transform_crs(ef, "EPSG:3857")
    cols = transformed.lazy_frame.collect_schema().names()
    assert {"longitude_proj", "latitude_proj"}.issubset(cols)
    assert transformed.metadata.crs == "EPSG:3857"


def test_transform_crs_noop_if_same() -> None:
    ef = _sample_event_frame()
    same = transform_crs(ef, ef.metadata.crs)
    assert same is ef


def test_transform_crs_requires_latlon() -> None:
    # geometry-only schema should raise
    lf = pl.LazyFrame({"timestamp": ["2024-01-01"], "geometry": [geometry.Point(0, 0).wkt]}).with_columns(
        pl.col("timestamp").str.to_datetime()
    )
    schema = EventSchema(timestamp_col="timestamp", geometry_col="geometry")
    metadata = EventMetadata(dataset_name="test")
    ef = EventFrame(lf, schema, metadata)
    with pytest.raises(ValueError):
        transform_crs(ef, "EPSG:3857")


def test_create_grid_and_assign_to_grid() -> None:
    ef = _sample_event_frame()
    grid = create_grid(bounds=(0, 0, 2, 2), size_m=1.0)
    assert "grid_id" in grid.columns

    assigned = assign_to_grid(ef, grid).collect()
    assert "grid_id" in assigned.columns
    assert assigned["grid_id"].to_list() == [0, 4]


def test_assign_to_zones() -> None:
    ef = _sample_event_frame()
    zones = pl.DataFrame(
        {
            "zone_id": [1],
            "geometry": [geometry.box(0, 0, 1, 1).wkt],
        }
    )
    assigned = assign_to_zones(ef, zones).collect()
    assert assigned["zone_id"].to_list() == [1, None]


def test_compute_distances() -> None:
    ef = _sample_event_frame()
    pois = pl.DataFrame({"name": ["origin"], "longitude": [0.0], "latitude": [0.0]})
    with_dist = compute_distances(ef, pois).collect()
    assert "dist_to_origin" in with_dist.columns
    first_dist = with_dist["dist_to_origin"].to_list()[0]
    assert first_dist == pytest.approx((0.5**2 + 0.5**2) ** 0.5)
