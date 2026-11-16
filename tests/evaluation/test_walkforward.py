"""Tests for the walk-forward harness, incl. a Chicago-shaped worked example.

The worked evaluation uses a small in-test panel in the shape of the
Chicago recipe output (``time_bin`` 6h bins, grid ``unit``, ``count`` plus a
lag feature ``count_lag_1``) — the full 1.9M-row replay is never re-run
inside unit tests.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from eventflow.core.event_frame import EventFrame
from eventflow.core.pipeline import Step
from eventflow.core.schema import EventMetadata, EventSchema
from eventflow.evaluation.metrics import pai_score, pr_auc_score
from eventflow.evaluation.splits import validate_split
from eventflow.evaluation.walkforward import (
    evaluate_walk_forward,
    walk_forward_splits,
)


@pytest.fixture()
def panel() -> EventFrame:
    """Small Chicago-shaped panel: 4 units x 120 six-hour bins (30 days)."""
    rng = np.random.default_rng(7)
    start = dt.datetime(2024, 1, 1)
    bins = [start + dt.timedelta(hours=6 * b) for b in range(120)]
    units = ["cell_0", "cell_1", "cell_2", "cell_3"]
    rows = [
        {
            "time_bin": t,
            "unit": unit,
            "latitude": 41.8 + j * 0.01,
            "longitude": -87.6,
            "count": int(c),
        }
        for j, unit in enumerate(units)
        for t, c in zip(bins, rng.poisson(2, size=len(bins)), strict=True)
    ]
    df = (
        pl.DataFrame(rows)
        .sort(["unit", "time_bin"])
        .with_columns(pl.col("count").shift(1).over("unit").fill_null(0).alias("count_lag_1"))
    )
    schema = EventSchema(
        timestamp_col="time_bin",
        lat_col="latitude",
        lon_col="longitude",
        numeric_cols=["count", "count_lag_1"],
    )
    metadata = EventMetadata(dataset_name="chicago-like-panel", time_bin="6h")
    return EventFrame(df.lazy(), schema, metadata)


def test_walk_forward_folds_expand_and_do_not_overlap(panel: EventFrame) -> None:
    folds = list(
        walk_forward_splits(panel, "time_bin", dt.timedelta(days=10), dt.timedelta(days=5))
    )
    assert len(folds) == 4
    assert [f.fold for f in folds] == [0, 1, 2, 3]
    for fold in folds:
        assert fold.train_start == dt.datetime(2024, 1, 1)
        validate_split(fold.train, fold.test, time_col="time_bin")
    train_ends = [f.train_end for f in folds]
    assert train_ends == sorted(train_ends) and len(set(train_ends)) == 4
    for prev, cur in zip(folds, folds[1:], strict=False):
        assert cur.test_start == prev.test_end
        assert cur.train_end == prev.test_end
    assert folds[0].train_end == dt.datetime(2024, 1, 11)
    assert folds[0].test_start == dt.datetime(2024, 1, 11)
    assert folds[0].test_end == dt.datetime(2024, 1, 16)


def test_walk_forward_gap_is_an_embargo(panel: EventFrame) -> None:
    folds = list(
        walk_forward_splits(
            panel,
            "time_bin",
            dt.timedelta(days=10),
            dt.timedelta(days=5),
            gap=dt.timedelta(days=1),
        )
    )
    assert folds[0].test_start == dt.datetime(2024, 1, 12)
    assert folds[0].test_end == dt.datetime(2024, 1, 17)


def test_walk_forward_max_folds(panel: EventFrame) -> None:
    folds = list(
        walk_forward_splits(
            panel,
            "time_bin",
            dt.timedelta(days=10),
            dt.timedelta(days=5),
            max_folds=2,
        )
    )
    assert len(folds) == 2


def test_walk_forward_rejects_bad_inputs(panel: EventFrame) -> None:
    with pytest.raises(ValueError, match="not present"):
        list(walk_forward_splits(panel, "nope", dt.timedelta(days=1), dt.timedelta(days=1)))
    with pytest.raises(ValueError, match="Datetime or Date"):
        list(walk_forward_splits(panel, "count", dt.timedelta(days=1), dt.timedelta(days=1)))
    with pytest.raises(ValueError, match="must be positive"):
        list(walk_forward_splits(panel, "time_bin", dt.timedelta(days=0), dt.timedelta(days=1)))
    with pytest.raises(ValueError, match="must be positive"):
        list(walk_forward_splits(panel, "time_bin", dt.timedelta(days=1), dt.timedelta(days=-1)))
    with pytest.raises(ValueError, match="must not be negative"):
        list(
            walk_forward_splits(
                panel,
                "time_bin",
                dt.timedelta(days=1),
                dt.timedelta(days=1),
                gap=dt.timedelta(days=-1),
            )
        )
    with pytest.raises(ValueError, match="max_folds"):
        list(
            walk_forward_splits(
                panel, "time_bin", dt.timedelta(days=1), dt.timedelta(days=1), max_folds=0
            )
        )


def _score_panel(train_out: EventFrame, test_out: EventFrame) -> dict[str, float]:
    """Score a fold: hotspot ranking of the test panel by its lag feature."""
    df = test_out.lazy_frame.select(["count", "count_lag_1"]).collect()
    y_true = (df["count"] >= 3).to_numpy()
    y_score = df["count_lag_1"].to_numpy().astype(float)
    return {
        "pai_top20": pai_score(y_true, y_score, 0.2),
        "pr_auc": pr_auc_score(y_true, y_score),
    }


def test_evaluate_walk_forward_on_chicago_like_panel(panel: EventFrame) -> None:
    results = evaluate_walk_forward(
        [],
        panel,
        "time_bin",
        dt.timedelta(days=10),
        dt.timedelta(days=5),
        _score_panel,
    )
    assert len(results) == 4
    assert [r.fold for r in results] == [0, 1, 2, 3]
    # Train windows expand; each test window holds 5 days x 4 bins x 4 units.
    assert [r.n_train for r in results] == [160, 240, 320, 400]
    assert [r.n_test for r in results] == [80, 80, 80, 80]
    for result in results:
        assert set(result.scores) == {"pai_top20", "pr_auc"}
        assert 0.0 <= result.scores["pr_auc"] <= 1.0
        assert result.scores["pai_top20"] >= 0.0


def test_evaluate_walk_forward_refits_fittable_steps_per_fold(panel: EventFrame) -> None:
    """Fittable steps are fit once per fold, on the train fold only."""
    calls: list[tuple[str, dt.datetime | None]] = []

    class RecordingStep(Step):
        def fit(self, event_frame: EventFrame) -> RecordingStep:
            max_time = event_frame.lazy_frame.select(pl.col("time_bin").max()).collect().item()
            calls.append(("fit", max_time))
            return self

        @property
        def is_fitted(self) -> bool:
            return any(kind == "fit" for kind, _ in calls)

        def run(self, event_frame: EventFrame) -> EventFrame:
            calls.append(("run", None))
            return event_frame

    results = evaluate_walk_forward(
        [RecordingStep()],
        panel,
        "time_bin",
        dt.timedelta(days=10),
        dt.timedelta(days=5),
        _score_panel,
    )
    assert len(results) == 4
    # Per fold: one fit, then one run on train and one on test, in that order.
    kinds = [kind for kind, _ in calls]
    assert kinds == ["fit", "run", "run"] * 4
    fit_times = [when for kind, when in calls if kind == "fit"]
    for result, fit_time in zip(results, fit_times, strict=True):
        assert fit_time is not None
        # The fit never sees a test time bin: it stops at the train end.
        assert fit_time < result.train_end
        assert fit_time < result.test_start


def test_evaluate_walk_forward_rejects_no_folds(panel: EventFrame) -> None:
    with pytest.raises(ValueError, match="no folds produced"):
        evaluate_walk_forward(
            [],
            panel,
            "time_bin",
            dt.timedelta(days=60),
            dt.timedelta(days=5),
            _score_panel,
        )


def test_evaluate_walk_forward_rejects_malformed_scores(panel: EventFrame) -> None:
    def bad_score(_train: EventFrame, _test: EventFrame) -> dict[str, float]:
        return {"pr_auc": float("nan")}  # type: ignore[return-value]

    with pytest.raises(ValueError, match="not finite"):
        evaluate_walk_forward(
            [],
            panel,
            "time_bin",
            dt.timedelta(days=10),
            dt.timedelta(days=5),
            bad_score,
            max_folds=1,
        )

    def not_a_dict(_train: EventFrame, _test: EventFrame) -> dict[str, float]:
        return [("pr_auc", 0.5)]  # type: ignore[return-value]

    with pytest.raises(ValueError, match="must return a dict"):
        evaluate_walk_forward(
            [],
            panel,
            "time_bin",
            dt.timedelta(days=10),
            dt.timedelta(days=5),
            not_a_dict,
            max_folds=1,
        )
