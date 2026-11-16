"""Tests for blocked train/test splits."""

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


@pytest.fixture()
def frame() -> EventFrame:
    """Hourly events over 10 days, 4 units, 4 entities."""
    start = dt.datetime(2024, 1, 1)
    times = [start + dt.timedelta(hours=h) for h in range(10 * 24)]
    units = ["A", "B", "C", "D"]
    rows = [
        {
            "timestamp": t,
            "unit": units[i % 4],
            "entity": f"e{(i % 4) + 1}",
            "latitude": 41.0 + (i % 4) * 0.01,
            "longitude": -87.0,
            "value": i,
        }
        for i, t in enumerate(times)
    ]
    lf = pl.LazyFrame(rows)
    schema = EventSchema(
        timestamp_col="timestamp",
        lat_col="latitude",
        lon_col="longitude",
        numeric_cols=["value"],
    )
    return EventFrame(lf, schema, EventMetadata(dataset_name="split-test"))


def _collect(frame_like: EventFrame | pl.LazyFrame) -> pl.DataFrame:
    """Materialise a LazyFrame or EventFrame."""
    if isinstance(frame_like, EventFrame):
        return frame_like.collect()
    return frame_like.collect()


def test_temporal_split_disjoint_time_intervals(frame: EventFrame) -> None:
    cutoff = dt.datetime(2024, 1, 6)
    train, test = temporal_split(frame, "timestamp", cutoff)
    train_df, test_df = _collect(train), _collect(test)
    assert train_df["timestamp"].max() < cutoff
    assert test_df["timestamp"].min() >= cutoff
    assert len(train_df) + len(test_df) == len(_collect(frame))
    assert isinstance(train, EventFrame) and isinstance(test, EventFrame)


def test_temporal_split_gap_is_an_embargo(frame: EventFrame) -> None:
    cutoff = dt.datetime(2024, 1, 6)
    _, test = temporal_split(frame, "timestamp", cutoff, gap=dt.timedelta(days=1))
    assert _collect(test)["timestamp"].min() >= cutoff + dt.timedelta(days=1)


def test_temporal_split_rejects_bad_inputs(frame: EventFrame) -> None:
    with pytest.raises(ValueError, match="not present"):
        temporal_split(frame, "nope", dt.datetime(2024, 1, 6))
    with pytest.raises(ValueError, match="must be a datetime"):
        temporal_split(frame, "timestamp", dt.date(2024, 1, 6))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must not be negative"):
        temporal_split(frame, "timestamp", dt.datetime(2024, 1, 6), gap=dt.timedelta(days=-1))
    with pytest.raises(ValueError, match="empty"):
        temporal_split(frame, "timestamp", dt.datetime(2024, 1, 1))
    with pytest.raises(ValueError, match="empty"):
        temporal_split(frame, "timestamp", dt.datetime(2024, 1, 11))
    with pytest.raises(ValueError, match="Datetime or Date"):
        temporal_split(frame, "value", dt.datetime(2024, 1, 6))


def test_temporal_split_works_on_lazy_frames(frame: EventFrame) -> None:
    train, test = temporal_split(frame.lazy_frame, "timestamp", dt.datetime(2024, 1, 6))
    assert isinstance(train, pl.LazyFrame) and isinstance(test, pl.LazyFrame)
    assert _collect(train)["timestamp"].max() < dt.datetime(2024, 1, 6)


def test_spatial_split_disjoint_units(frame: EventFrame) -> None:
    train, test = spatial_split(frame, "unit", ["C", "D"])
    assert set(_collect(train)["unit"].unique()) == {"A", "B"}
    assert set(_collect(test)["unit"].unique()) == {"C", "D"}


def test_entity_split_disjoint_entities(frame: EventFrame) -> None:
    train, test = entity_split(frame, "entity", ["e3"])
    assert set(_collect(train)["entity"].unique()) == {"e1", "e2", "e4"}
    assert set(_collect(test)["entity"].unique()) == {"e3"}


def test_partition_splits_reject_bad_inputs(frame: EventFrame) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        spatial_split(frame, "unit", [])
    with pytest.raises(ValueError, match="not present"):
        spatial_split(frame, "unit", ["Z"])
    with pytest.raises(ValueError, match="every"):
        spatial_split(frame, "unit", ["A", "B", "C", "D"])
    with pytest.raises(ValueError, match="not present"):
        entity_split(frame, "nope", ["e1"])


def test_partition_units_is_deterministic_and_covers_all(frame: EventFrame) -> None:
    train_units, test_units = partition_units(frame, "unit", test_fraction=0.25, seed=3)
    assert len(test_units) == 1
    assert sorted(train_units + test_units) == ["A", "B", "C", "D"]
    assert set(train_units).isdisjoint(test_units)
    again_train, again_test = partition_units(frame, "unit", test_fraction=0.25, seed=3)
    assert (again_train, again_test) == (train_units, test_units)


def test_partition_units_rejects_bad_inputs(frame: EventFrame) -> None:
    with pytest.raises(ValueError, match="test_fraction"):
        partition_units(frame, "unit", test_fraction=1.0)
    with pytest.raises(ValueError, match="test_fraction"):
        partition_units(frame, "unit", test_fraction=0.0)
    with pytest.raises(ValueError, match="not present"):
        partition_units(frame, "nope")


def test_validate_split_accepts_clean_splits(frame: EventFrame) -> None:
    train, test = temporal_split(frame, "timestamp", dt.datetime(2024, 1, 6))
    validate_split(train, test, time_col="timestamp")
    train_u, test_u = spatial_split(frame, "unit", ["C", "D"])
    validate_split(train_u, test_u, unit_col="unit")
    train_e, test_e = entity_split(frame, "entity", ["e3"])
    validate_split(train_e, test_e, entity_col="entity")


def test_validate_split_rejects_shared_time_bin(frame: EventFrame) -> None:
    # Deliberately leaking split: test starts a day before train ends.
    lazy = frame.lazy_frame
    train = frame.with_lazy_frame(lazy.filter(pl.col("timestamp") < dt.datetime(2024, 1, 6)))
    test = frame.with_lazy_frame(lazy.filter(pl.col("timestamp") >= dt.datetime(2024, 1, 5)))
    with pytest.raises(ValueError, match="share 24 time bin"):
        validate_split(train, test, time_col="timestamp")


def test_validate_split_rejects_shared_unit_and_entity(frame: EventFrame) -> None:
    lazy = frame.lazy_frame
    train = frame.with_lazy_frame(lazy.filter(pl.col("unit").is_in(["A", "B", "C"])))
    test = frame.with_lazy_frame(lazy.filter(pl.col("unit").is_in(["C", "D"])))
    with pytest.raises(ValueError, match="share 1 unit"):
        validate_split(train, test, unit_col="unit")

    train_e = frame.with_lazy_frame(lazy.filter(pl.col("entity").is_in(["e1", "e2"])))
    test_e = frame.with_lazy_frame(lazy.filter(pl.col("entity").is_in(["e2", "e3"])))
    with pytest.raises(ValueError, match="share 1 entity"):
        validate_split(train_e, test_e, entity_col="entity")


def test_validate_split_requires_a_blocking_column(frame: EventFrame) -> None:
    train, test = temporal_split(frame, "timestamp", dt.datetime(2024, 1, 6))
    with pytest.raises(ValueError, match="at least one"):
        validate_split(train, test)
    with pytest.raises(ValueError, match="not present"):
        validate_split(train, test, time_col="nope")


def test_blocked_split_applies_both_conditions(frame: EventFrame) -> None:
    train, test = blocked_split(
        frame, "timestamp", dt.datetime(2024, 1, 6), unit_col="unit", test_units=["C", "D"]
    )
    train_df, test_df = _collect(train), _collect(test)
    assert train_df["timestamp"].max() < dt.datetime(2024, 1, 6)
    assert test_df["timestamp"].min() >= dt.datetime(2024, 1, 6)
    assert set(train_df["unit"].unique()) == {"A", "B"}
    assert set(test_df["unit"].unique()) == {"C", "D"}
    # Off-diagonal quadrants are dropped, not assigned.
    assert len(train_df) + len(test_df) < len(_collect(frame))
    validate_split(train, test, time_col="timestamp", unit_col="unit")


def test_blocked_split_with_gap_and_entity(frame: EventFrame) -> None:
    train, test = blocked_split(
        frame,
        "timestamp",
        dt.datetime(2024, 1, 6),
        entity_col="entity",
        test_entities=["e3", "e4"],
        gap=dt.timedelta(days=1),
    )
    assert _collect(test)["timestamp"].min() >= dt.datetime(2024, 1, 7)
    assert set(_collect(test)["entity"].unique()) == {"e3", "e4"}
    validate_split(train, test, time_col="timestamp", entity_col="entity")


def test_blocked_split_requires_a_blocking_dimension(frame: EventFrame) -> None:
    with pytest.raises(ValueError, match="supply unit_col"):
        blocked_split(frame, "timestamp", dt.datetime(2024, 1, 6))
    with pytest.raises(ValueError, match="without a unit column"):
        blocked_split(frame, "timestamp", dt.datetime(2024, 1, 6), test_units=["C"])
    with pytest.raises(ValueError, match="without test units"):
        blocked_split(frame, "timestamp", dt.datetime(2024, 1, 6), unit_col="unit")
