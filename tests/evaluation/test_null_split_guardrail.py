"""Regression tests for the demo-findings guardrails.

Finding 2 (grave): splitting on a nullable column silently drops rows —
on the completed Chicago panel ``time_bin`` is null on all 1,851,597
explicit-zero rows, so a temporal split on it keeps 2.9% of the frame
and reports a perfect, meaningless PR-AUC of 1.0. Every split entry
point must now refuse a nullable split column loudly.

Finding 1: passing a collected DataFrame to the evaluation API raised a
cryptic ``AttributeError``; DataFrames are now accepted (``.lazy()``)
and the return kind matches the input kind.
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from shikong.core.event_frame import EventFrame
from shikong.core.schema import EventMetadata, EventSchema
from shikong.evaluation.splits import (
    blocked_split,
    entity_split,
    partition_units,
    spatial_split,
    temporal_split,
    validate_split,
)
from shikong.evaluation.walkforward import evaluate_walk_forward, walk_forward_splits


def _nullable_frame() -> pl.LazyFrame:
    """10 rows over 5 days; ``timestamp`` has 2 nulls, ``unit`` has 1."""
    times = [dt.datetime(2024, 1, 1) + dt.timedelta(days=i) for i in range(10)]
    times[3] = None
    times[7] = None
    units = ["A", "B", "C", "D", "A", "B", "C", "D", "A", "B"]
    units[5] = None
    return pl.LazyFrame(
        {
            "timestamp": times,
            "unit": units,
            "entity": [f"e{i % 4}" for i in range(10)],
            "value": list(range(10)),
        }
    )


def _clean_frame() -> pl.LazyFrame:
    """Same shape as _nullable_frame but with no nulls anywhere."""
    times = [dt.datetime(2024, 1, 1) + dt.timedelta(days=i) for i in range(10)]
    return pl.LazyFrame(
        {
            "timestamp": times,
            "unit": ["A", "B", "C", "D", "A", "B", "C", "D", "A", "B"],
            "entity": [f"e{i % 4}" for i in range(10)],
            "value": list(range(10)),
        }
    )


def _panel_like_frame() -> pl.LazyFrame:
    """Mimics the completed Chicago panel: ``time_bin`` null on the
    explicit-zero rows, ``date`` non-null on every row."""
    n = 100
    times = [dt.datetime(2024, 1, 1) + dt.timedelta(hours=6 * i) for i in range(n)]
    time_bin = [t if i % 10 == 0 else None for i, t in enumerate(times)]
    return pl.LazyFrame(
        {
            "grid_id": [i % 5 for i in range(n)],
            "time_bin": time_bin,
            "date": times,
            "count": [1 if i % 10 == 0 else 0 for i in range(n)],
        }
    )


def _event_frame(lazy: pl.LazyFrame) -> EventFrame:
    schema = EventSchema(
        timestamp_col="timestamp",
        lat_col="latitude",
        lon_col="longitude",
        numeric_cols=["value"],
    )
    return EventFrame(lazy, schema, EventMetadata(dataset_name="guardrail-test"))


# --- Finding 2: nullable split columns are refused loudly ----------------


def test_temporal_split_rejects_null_time_column() -> None:
    with pytest.raises(ValueError, match="contains 2 null value"):
        temporal_split(_nullable_frame(), "timestamp", dt.datetime(2024, 1, 6))


def test_temporal_split_rejects_panel_time_bin_but_accepts_date() -> None:
    frame = _panel_like_frame()
    with pytest.raises(ValueError, match="contains 90 null value"):
        temporal_split(frame, "time_bin", dt.datetime(2024, 1, 11))
    train, test = temporal_split(frame, "date", dt.datetime(2024, 1, 11))
    assert train.collect().height + test.collect().height == 100


def test_spatial_split_rejects_null_unit_column() -> None:
    with pytest.raises(ValueError, match="contains 1 null value"):
        spatial_split(_nullable_frame(), "unit", ["A", "B"])


def test_entity_split_rejects_null_entity_column() -> None:
    lf = _clean_frame().with_columns(
        pl.when(pl.col("value") == 4).then(None).otherwise(pl.col("entity")).alias("entity")
    )
    with pytest.raises(ValueError, match="contains 1 null value"):
        entity_split(lf, "entity", ["e0", "e1"])


def test_blocked_split_rejects_null_time_and_unit_columns() -> None:
    with pytest.raises(ValueError, match="contains 2 null value"):
        blocked_split(
            _nullable_frame(),
            "timestamp",
            dt.datetime(2024, 1, 6),
            unit_col="unit",
            test_units=["A"],
        )
    clean = _clean_frame().with_columns(
        pl.when(pl.col("value") == 2).then(None).otherwise(pl.col("unit")).alias("unit")
    )
    with pytest.raises(ValueError, match="contains 1 null value"):
        blocked_split(
            clean,
            "timestamp",
            dt.datetime(2024, 1, 6),
            unit_col="unit",
            test_units=["A"],
        )


def test_partition_units_rejects_null_unit_column() -> None:
    with pytest.raises(ValueError, match="contains 1 null value"):
        partition_units(_nullable_frame(), "unit")


def test_validate_split_rejects_nulls_on_either_side() -> None:
    train, test = temporal_split(_clean_frame(), "timestamp", dt.datetime(2024, 1, 6))
    validate_split(train, test, time_col="timestamp")  # positive control
    train_null, _ = temporal_split(_clean_frame(), "timestamp", dt.datetime(2024, 1, 6))
    train_null = train_null.with_columns(
        pl.when(pl.col("value") == 0).then(None).otherwise(pl.col("timestamp")).alias("timestamp")
    )
    with pytest.raises(ValueError, match="contains 1 null value"):
        validate_split(train_null, test, time_col="timestamp")


def test_walk_forward_splits_rejects_null_time_column() -> None:
    with pytest.raises(ValueError, match="contains 2 null value"):
        list(
            walk_forward_splits(
                _nullable_frame(),
                "timestamp",
                dt.timedelta(days=3),
                dt.timedelta(days=1),
            )
        )


# --- Finding 1: collected DataFrames are accepted --------------------------


def test_dataframe_input_accepted_and_returned() -> None:
    df = _clean_frame().collect()
    train, test = temporal_split(df, "timestamp", dt.datetime(2024, 1, 6))
    assert isinstance(train, pl.DataFrame) and isinstance(test, pl.DataFrame)
    assert train.height + test.height == 10
    train, test = spatial_split(df, "unit", ["A"])
    assert isinstance(train, pl.DataFrame) and isinstance(test, pl.DataFrame)
    train, test = blocked_split(
        df, "timestamp", dt.datetime(2024, 1, 6), unit_col="unit", test_units=["A"]
    )
    assert isinstance(train, pl.DataFrame) and isinstance(test, pl.DataFrame)
    validate_split(train, test, time_col="timestamp", unit_col="unit")
    folds = list(walk_forward_splits(df, "timestamp", dt.timedelta(days=3), dt.timedelta(days=1)))
    assert folds
    assert all(isinstance(f.train, pl.DataFrame) for f in folds)


def test_lazy_and_eventframe_kinds_preserved() -> None:
    lf = _clean_frame()
    train, test = temporal_split(lf, "timestamp", dt.datetime(2024, 1, 6))
    assert isinstance(train, pl.LazyFrame) and isinstance(test, pl.LazyFrame)
    ef = _event_frame(
        _clean_frame().with_columns(
            pl.lit(41.0).alias("latitude"), pl.lit(-87.0).alias("longitude")
        )
    )
    train, test = temporal_split(ef, "timestamp", dt.datetime(2024, 1, 6))
    assert isinstance(train, EventFrame) and isinstance(test, EventFrame)


def test_evaluate_walk_forward_rejects_non_eventframe() -> None:
    df = _clean_frame().collect()
    with pytest.raises(TypeError, match="must be an EventFrame"):
        evaluate_walk_forward(
            [],
            df,  # type: ignore[arg-type]
            "timestamp",
            dt.timedelta(days=3),
            dt.timedelta(days=1),
            lambda a, b: {},
        )


def test_leaking_split_still_rejected_after_guardrail() -> None:
    lf = _clean_frame()
    with pytest.raises(ValueError, match="share"):
        validate_split(lf, lf, time_col="timestamp")
