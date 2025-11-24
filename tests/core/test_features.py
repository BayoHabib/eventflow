"""Tests for feature engineering utilities."""

import polars as pl
import pytest

from eventflow.core.event_frame import EventFrame
from eventflow.core.features import (
    aggregate_by_category,
    aggregate_counts,
    compute_ratios,
    encode_categorical,
    moving_window_aggregation,
    normalize_features,
    lag_features,
)
from eventflow.core.schema import EventSchema, EventMetadata


@pytest.fixture
def sample_event_frame() -> EventFrame:
    """Sample EventFrame for feature tests."""
    data = {
        "timestamp": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "latitude": [1.0, 1.0, 1.0],
        "longitude": [1.0, 1.0, 1.0],
        "type": ["A", "A", "B"],
        "value": [1, 2, 3],
    }

    lf = pl.LazyFrame(data).with_columns(pl.col("timestamp").str.to_datetime())

    schema = EventSchema(
        timestamp_col="timestamp",
        lat_col="latitude",
        lon_col="longitude",
        categorical_cols=["type"],
        numeric_cols=["value"],
    )

    metadata = EventMetadata(dataset_name="test")
    return EventFrame(lf, schema, metadata)


def test_aggregate_counts(sample_event_frame: EventFrame) -> None:
    """Counts are aggregated per group."""
    result = aggregate_counts(sample_event_frame, ["type"]).collect().sort("type")
    assert result["count"].to_list() == [2, 1]
    assert result["type"].to_list() == ["A", "B"]


def test_aggregate_by_category(sample_event_frame: EventFrame) -> None:
    """Pivoted counts per category are produced."""
    result = aggregate_by_category(sample_event_frame, ["type"], "value").collect().sort(
        "type"
    )

    row_a = result.row(0, named=True)
    row_b = result.row(1, named=True)

    assert row_a["count_1"] == 1
    assert row_a["count_2"] == 1
    assert row_b["count_3"] == 1


def test_encode_categorical(sample_event_frame: EventFrame) -> None:
    """One-hot encoding creates indicator columns."""
    encoded = encode_categorical(sample_event_frame, "type").collect().sort("timestamp")
    assert {"type_A", "type_B"}.issubset(encoded.columns)
    assert encoded.select("type_A", "type_B").to_dict(as_series=False) == {
        "type_A": [1, 1, 0],
        "type_B": [0, 0, 1],
    }


def test_compute_ratios(sample_event_frame: EventFrame) -> None:
    """Ratios are computed safely."""
    ratios = compute_ratios(sample_event_frame, "value", "value").collect()
    assert ratios["value_per_value"].to_list() == [1.0, 1.0, 1.0]


def test_lag_features(sample_event_frame: EventFrame) -> None:
    """Lagged values are added with correct offsets."""
    lagged = lag_features(sample_event_frame, "value", [1]).collect().sort("timestamp")
    assert lagged["value_lag_1"].to_list() == [None, 1, 2]


def test_moving_window_aggregation(sample_event_frame: EventFrame) -> None:
    """Rolling mean feature is added."""
    mw = moving_window_aggregation(
        sample_event_frame, window_size=2, group_by=[], agg_col="value", agg_fn="mean"
    ).collect()
    assert mw["value_mean_2"].to_list() == [None, 1.5, 2.5]


def test_encode_categorical_ordinal(sample_event_frame: EventFrame) -> None:
    """Ordinal encoding produces integer codes."""
    encoded = encode_categorical(sample_event_frame, "type", method="ordinal").collect()
    assert "type_encoded" in encoded.columns
    assert encoded["type_encoded"].to_list() == [0, 0, 1]


def test_normalize_features(sample_event_frame: EventFrame) -> None:
    """Normalization yields expected ranges."""
    normalized = normalize_features(sample_event_frame, ["value"], method="minmax").collect()
    assert normalized["value_normalized"].to_list() == [0.0, 0.5, 1.0]
