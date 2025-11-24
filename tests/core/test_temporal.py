"""Tests for temporal operations."""

import polars as pl
import pytest

from eventflow.core.event_frame import EventFrame
from eventflow.core.schema import EventSchema, EventMetadata, ContextSchema
from eventflow.core.temporal import (
    align_temporal,
    compute_time_deltas,
    create_temporal_windows,
    create_time_bins,
    extract_temporal_components,
)


def _sample_event_frame() -> EventFrame:
    data = {
        "timestamp": ["2024-01-01 00:00:00", "2024-01-01 01:00:00", "2024-01-01 02:00:00"],
        "latitude": [0.0, 0.0, 0.0],
        "longitude": [0.0, 0.0, 0.0],
    }
    lf = pl.LazyFrame(data).with_columns(pl.col("timestamp").str.to_datetime())
    schema = EventSchema(timestamp_col="timestamp", lat_col="latitude", lon_col="longitude")
    metadata = EventMetadata(dataset_name="test")
    return EventFrame(lf, schema, metadata)


def test_extract_temporal_components() -> None:
    ef = _sample_event_frame()
    enriched = extract_temporal_components(
        ef, ["hour_of_day", "day_of_week", "month", "year", "is_weekend"]
    ).collect()

    assert {"hour_of_day", "day_of_week", "month", "year", "is_weekend"}.issubset(
        enriched.columns
    )
    assert enriched["hour_of_day"].to_list() == [0, 1, 2]
    assert enriched["month"].to_list() == [1, 1, 1]


def test_create_time_bins_sets_metadata() -> None:
    ef = _sample_event_frame()
    binned = create_time_bins(ef, "1h", bin_col="bin_col")
    df = binned.collect()

    assert "bin_col" in df.columns
    assert binned.metadata.time_bin == "1h"


def test_align_temporal_nearest() -> None:
    ef = _sample_event_frame()
    context = pl.LazyFrame(
        {
            "ctx_ts": ["2024-01-01 00:30:00", "2024-01-01 02:30:00"],
            "ctx_val": [10, 20],
        }
    ).with_columns(pl.col("ctx_ts").str.to_datetime())
    schema = ContextSchema(timestamp_col="ctx_ts", attribute_cols=["ctx_val"])

    joined = align_temporal(ef, context, "ctx_ts", strategy="nearest")
    df = joined.collect().sort("timestamp")

    assert df["ctx_val"].to_list() == [10, 10, 20]


def test_align_temporal_exact_and_before() -> None:
    ef = _sample_event_frame()
    context = pl.LazyFrame(
        {"ctx_ts": ["2024-01-01 01:00:00"], "ctx_val": [99]}
    ).with_columns(pl.col("ctx_ts").str.to_datetime())
    schema = ContextSchema(timestamp_col="ctx_ts", attribute_cols=["ctx_val"])

    exact = align_temporal(ef, context, "ctx_ts", strategy="exact").collect().sort("timestamp")
    assert exact["ctx_val"].to_list() == [None, 99, None]

    before = align_temporal(ef, context, "ctx_ts", strategy="before").collect().sort("timestamp")
    assert before["ctx_val"].to_list() == [None, 99, 99]


def test_align_temporal_invalid_strategy_raises() -> None:
    ef = _sample_event_frame()
    context = pl.LazyFrame({"ctx_ts": ["2024-01-01 00:00:00"], "ctx_val": [1]}).with_columns(
        pl.col("ctx_ts").str.to_datetime()
    )
    schema = ContextSchema(timestamp_col="ctx_ts", attribute_cols=["ctx_val"])
    with pytest.raises(ValueError):
        align_temporal(ef, context, "ctx_ts", strategy="bogus")


def test_compute_time_deltas() -> None:
    ef = _sample_event_frame()
    refs = pl.LazyFrame({"ref_ts": ["2024-01-01 01:30:00"]}).with_columns(
        pl.col("ref_ts").str.to_datetime()
    )

    result = compute_time_deltas(ef, refs, "ref_ts", delta_col="delta_seconds").collect()
    assert result["delta_seconds"].to_list() == [-5400.0, -1800.0, 1800.0]


def test_create_temporal_windows() -> None:
    ef = _sample_event_frame()
    windows = create_temporal_windows(ef, ["1h", "1d"]).collect()
    assert {"window_1h", "window_1d"}.issubset(windows.columns)
