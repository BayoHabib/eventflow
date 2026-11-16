"""Regression tests for issue #39: validations must not materialize frames.

``_validate_time_keys`` and the ``CompletePanelStep`` checks used to
``collect()`` full key columns (and, for the panel, every non-key column)
only to count nulls, look for duplicates and derive a handful of scalars.
They now run as one-row aggregations. These tests fail on the old code
because the recorded ``collect()`` results are as tall as the input.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from eventflow.core.event_frame import EventFrame
from eventflow.core.schema import EventMetadata, EventSchema
from eventflow.core.steps.panel import CompletePanelStep
from eventflow.core.steps.temporal import MovingAverageStep, TemporalLagStep


def _collect_heights(monkeypatch):
    """Record the height of every frame collected while the patch is live."""
    heights: list[int] = []
    original = pl.LazyFrame.collect

    def recording(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        heights.append(result.height)
        return result

    monkeypatch.setattr(pl.LazyFrame, "collect", recording)
    return heights


def _temporal_frame(n_rows: int, *, null_time: bool = False) -> EventFrame:
    times = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_rows)]
    if null_time:
        times[0] = None  # type: ignore[list-item]
    lf = pl.LazyFrame(
        {
            "time": times,
            "x": [0.5] * n_rows,
            "y": [0.5] * n_rows,
            "value": [float(i) for i in range(n_rows)],
        }
    )
    return EventFrame(
        lf,
        EventSchema(timestamp_col="time", lon_col="x", lat_col="y", numeric_cols=["value"]),
        EventMetadata(dataset_name="test-lazy-validation", crs="EPSG:4326"),
    )


def _panel_frame(n_rows: int) -> EventFrame:
    rows = [
        (unit, datetime(2024, 1, 1) + timedelta(days=day), count, "north")
        for unit in (1, 2, 3)
        for day, count in enumerate(range(1, n_rows // 3 + 1))
    ]
    lf = pl.LazyFrame(
        {
            "grid_id": [r[0] for r in rows],
            "time_bin": [r[1] for r in rows],
            "count": [r[2] for r in rows],
            "zone": [r[3] for r in rows],
            "latitude": [41.8781] * len(rows),
            "longitude": [-87.6298] * len(rows),
        }
    )
    return EventFrame(
        lf,
        EventSchema(
            timestamp_col="time_bin",
            lat_col="latitude",
            lon_col="longitude",
            numeric_cols=["count"],
            categorical_cols=["zone"],
        ),
        EventMetadata(dataset_name="test-lazy-validation", crs="EPSG:4326", time_bin="1d"),
    )


@pytest.mark.parametrize(
    "step",
    [TemporalLagStep(["value"], [1]), MovingAverageStep(["value"], [2])],
    ids=["lag", "moving-average"],
)
def test_time_key_validation_collects_one_row(monkeypatch, step) -> None:
    """Duration-mode validation must aggregate, not materialize the keys."""
    heights = _collect_heights(monkeypatch)
    step.run(_temporal_frame(5_000))
    assert heights, "expected the validation to collect at least once"
    assert heights == [1], f"validation collected full frames: {heights}"


def test_time_key_validation_still_rejects_nulls() -> None:
    with pytest.raises(ValueError, match="Temporal keys cannot contain nulls"):
        TemporalLagStep(["value"], [1]).run(_temporal_frame(3, null_time=True))


def test_time_key_validation_still_rejects_duplicate_group_keys() -> None:
    lf = pl.LazyFrame(
        {
            "time": [datetime(2024, 1, 1), datetime(2024, 1, 1)],
            "entity": ["A", "A"],
            "x": [0.5, 0.5],
            "y": [0.5, 0.5],
            "value": [1.0, 2.0],
        }
    )
    frame = EventFrame(
        lf,
        EventSchema(timestamp_col="time", lon_col="x", lat_col="y", numeric_cols=["value"]),
        EventMetadata(dataset_name="test-lazy-validation", crs="EPSG:4326"),
    )
    with pytest.raises(ValueError, match="Duplicate temporal keys"):
        TemporalLagStep(["value"], [1], group_cols=["entity"]).run(frame)


def test_panel_checks_collect_small_results_only(monkeypatch) -> None:
    """Panel validation must aggregate; only the per-unit table is small."""
    heights = _collect_heights(monkeypatch)
    result = CompletePanelStep().run(_panel_frame(300))
    assert heights, "expected the panel checks to collect"
    # One-row key stats (row count, null counts, min/max, duplicate and
    # off-grid flags), one row per unit for the min-periods filter, one
    # row for the static/varying classification.
    assert heights == [1, 3, 1], f"panel collected full frames: {heights}"
    monkeypatch.undo()  # the result collect below is the test's own, not the step's
    assert result.collect().height == 3 * 100  # 3 units x 100 bins


def test_panel_still_rejects_off_grid_bins() -> None:
    lf = pl.LazyFrame(
        {
            "grid_id": [1, 1],
            "time_bin": [datetime(2024, 1, 1), datetime(2024, 1, 1, 12)],
            "count": [5, 3],
            "zone": ["north", "north"],
            "latitude": [41.8781, 41.8781],
            "longitude": [-87.6298, -87.6298],
        }
    )
    frame = EventFrame(
        lf,
        EventSchema(timestamp_col="time_bin", lat_col="latitude", lon_col="longitude"),
        EventMetadata(dataset_name="test-lazy-validation", crs="EPSG:4326", time_bin="1d"),
    )
    with pytest.raises(ValueError, match="not aligned to"):
        CompletePanelStep().run(frame)


def test_panel_still_rejects_null_time() -> None:
    lf = pl.LazyFrame(
        {
            "grid_id": [1, 1],
            "time_bin": [datetime(2024, 1, 1), None],
            "count": [5, 3],
            "zone": ["north", "north"],
            "latitude": [41.8781, 41.8781],
            "longitude": [-87.6298, -87.6298],
        }
    )
    frame = EventFrame(
        lf,
        EventSchema(timestamp_col="time_bin", lat_col="latitude", lon_col="longitude"),
        EventMetadata(dataset_name="test-lazy-validation", crs="EPSG:4326", time_bin="1d"),
    )
    with pytest.raises(ValueError, match="cannot contain nulls"):
        CompletePanelStep().run(frame)
