"""Tests for schema definitions."""

import pytest

from eventflow.core.schema import ContextSchema, EventMetadata, EventSchema


def test_event_schema_with_latlon() -> None:
    """Test EventSchema with lat/lon columns."""
    schema = EventSchema(
        timestamp_col="timestamp",
        lat_col="latitude",
        lon_col="longitude",
        categorical_cols=["type"],
        numeric_cols=["value"],
    )

    assert schema.timestamp_col == "timestamp"
    assert schema.lat_col == "latitude"
    assert schema.lon_col == "longitude"


def test_event_schema_with_geometry() -> None:
    """Test EventSchema with geometry column."""
    schema = EventSchema(
        timestamp_col="timestamp",
        geometry_col="geometry",
        categorical_cols=["type"],
    )

    assert schema.timestamp_col == "timestamp"
    assert schema.geometry_col == "geometry"


def test_event_schema_requires_spatial() -> None:
    """Test that EventSchema requires spatial information."""
    with pytest.raises(ValueError):
        EventSchema(
            timestamp_col="timestamp",
            categorical_cols=["type"],
        )


def test_context_schema_temporal() -> None:
    """Test ContextSchema with temporal dimension."""
    schema = ContextSchema(
        timestamp_col="timestamp",
        attribute_cols=["temperature", "humidity"],
    )

    assert schema.has_temporal()
    assert not schema.has_spatial()


def test_context_schema_spatial() -> None:
    """Test ContextSchema with spatial dimension."""
    schema = ContextSchema(
        spatial_col="zone_id",
        attribute_cols=["population"],
    )

    assert schema.has_spatial()
    assert not schema.has_temporal()


def test_event_metadata() -> None:
    """Test EventMetadata."""
    metadata = EventMetadata(
        dataset_name="test_dataset",
        crs="EPSG:4326",
        time_zone="UTC",
    )

    assert metadata.dataset_name == "test_dataset"
    assert metadata.crs == "EPSG:4326"
    assert metadata.time_zone == "UTC"
