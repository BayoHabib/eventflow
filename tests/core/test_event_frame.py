"""Tests for EventFrame."""

import pytest
import polars as pl
from eventflow.core.event_frame import EventFrame
from eventflow.core.schema import EventSchema, EventMetadata


@pytest.fixture
def sample_event_frame():
    """Create a sample EventFrame for testing."""
    data = {
        "timestamp": ["2024-01-01 00:00:00", "2024-01-01 01:00:00"],
        "latitude": [41.8781, 41.8800],
        "longitude": [-87.6298, -87.6300],
        "type": ["A", "B"],
        "value": [1, 2],
    }
    
    lf = pl.LazyFrame(data).with_columns([
        pl.col("timestamp").str.to_datetime()
    ])
    
    schema = EventSchema(
        timestamp_col="timestamp",
        lat_col="latitude",
        lon_col="longitude",
        categorical_cols=["type"],
        numeric_cols=["value"],
    )
    
    metadata = EventMetadata(
        dataset_name="test",
        crs="EPSG:4326",
        time_zone="UTC",
    )
    
    return EventFrame(lf, schema, metadata)


def test_event_frame_creation(sample_event_frame):
    """Test EventFrame creation."""
    assert sample_event_frame.schema.timestamp_col == "timestamp"
    assert sample_event_frame.metadata.dataset_name == "test"


def test_event_frame_collect(sample_event_frame):
    """Test collecting EventFrame to DataFrame."""
    df = sample_event_frame.collect()
    assert len(df) == 2
    assert "timestamp" in df.columns


def test_event_frame_count(sample_event_frame):
    """Test counting events."""
    count = sample_event_frame.count()
    assert count == 2


def test_event_frame_filter(sample_event_frame):
    """Test filtering EventFrame."""
    filtered = sample_event_frame.filter(pl.col("type") == "A")
    assert filtered.count() == 1


def test_event_frame_select(sample_event_frame):
    """Test selecting columns."""
    selected = sample_event_frame.select("timestamp", "type")
    df = selected.collect()
    assert len(df.columns) == 2


def test_event_frame_with_columns(sample_event_frame):
    """Test adding columns."""
    new_ef = sample_event_frame.with_columns(
        new_col=pl.lit("test")
    )
    df = new_ef.collect()
    assert "new_col" in df.columns
