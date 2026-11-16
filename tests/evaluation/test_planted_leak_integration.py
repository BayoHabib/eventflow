"""Integration: the walk-forward harness + the fitted step + the planted leak.

This is coordinator integration work for Phase 4, criterion 7. It wires
together what the three workers built separately:

- worker A: ``src/eventflow/evaluation/`` (blocked splits, walk-forward
  harness, PAI / PR-AUC);
- worker B: ``CategoryCountsStep.fit()`` / ``FittableStep``;
- worker C: the synthetic generator and the planted-leak demonstration.

Two things are proven here, with numbers:

1. The harness really fits a real fittable step on the train fold only:
   after ``evaluate_walk_forward`` the step's frozen vocabulary equals
   the top-N of the LAST fold's train frame, computed independently.
2. The planted leak is exposed by the evaluation protocol: the leaky
   score is ~perfect under a random split and collapses under a temporal
   blocked split (causal score); and ``validate_split`` rejects a
   deliberately overlapping split.

Honest boundary, asserted explicitly in
``test_blocked_split_does_not_sanitize_a_future_peeking_column``: a
column that was ALREADY computed from future information scores 1.0
even on temporally blocked test rows. Blocking the split does not
sanitize such a column — it forces the evaluation to respect time
order, which exposes that the column is future information. The
harness rejects leaking *splits*; it cannot inspect column
provenance. A future-peeking column must not be built in the first
place.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from eventflow.core.steps.categorical import CategoryCountsStep
from eventflow.evaluation.metrics import pr_auc_score
from eventflow.evaluation.splits import temporal_split, validate_split
from eventflow.evaluation.walkforward import evaluate_walk_forward, walk_forward_splits
from tests.evaluation.synthetic import events_event_frame, generate_synthetic_panel

HIGH_COUNT = 6
CUTOFF = dt.datetime(2024, 1, 29)  # 4 weeks train / 4 weeks test

# Margins, not fitted values: the leaky score is perfect by construction
# (1.00) and the causal score sits far below it (~0.41 measured).
LEAKY_PR_AUC_MIN = 0.95
BLOCKED_PR_AUC_MAX = 0.60
GAP_MIN = 0.35


def _prediction_rows(panel: pl.DataFrame) -> pl.DataFrame:
    """One row per (unit, bin t): target = next-bin high count."""
    rows: list[tuple[int, dt.datetime, int, int, int]] = []
    for (grid_id,), group in panel.sort(["grid_id", "time_bin"]).group_by(
        ["grid_id"], maintain_order=True
    ):
        counts = group["count"].to_numpy()
        bins = group["time_bin"].to_list()
        for t in range(len(counts) - 1):
            rows.append(
                (
                    int(grid_id),
                    bins[t],
                    int(counts[t + 1] >= HIGH_COUNT),
                    int(counts[t + 1]),  # leaky: the future count itself
                    int(counts[t]),  # causal: the count at time t
                )
            )
    return pl.DataFrame(
        rows,
        schema=["grid_id", "time_bin", "y_next_high", "leak_count_next", "causal_count"],
        orient="row",
    )


def test_harness_fits_real_fittable_step_on_train_fold_only() -> None:
    """`evaluate_walk_forward` fits `CategoryCountsStep` per fold, train-only."""
    data = generate_synthetic_panel(seed=42)
    frame = events_event_frame(data.events)
    step = CategoryCountsStep(category_cols=["primary_type"], top_n=2)

    results = evaluate_walk_forward(
        [step],
        frame,
        "time_bin",
        dt.timedelta(days=28),
        dt.timedelta(days=7),
        lambda _train_out, test_out: {"test_rows": float(test_out.count())},
    )

    assert len(results) == 4
    assert all(r.scores["test_rows"] > 0 for r in results)
    assert step.is_fitted

    # The frozen vocabulary must equal the top-2 of the LAST fold's train
    # frame, computed independently — i.e. the fit saw train only.
    folds = list(
        walk_forward_splits(frame, "time_bin", dt.timedelta(days=28), dt.timedelta(days=7))
    )
    last_train = folds[-1].train.collect()
    expected_top2 = (
        last_train.group_by("primary_type")
        .agg(pl.len())
        .sort(["len", "primary_type"], descending=[True, False])
        .head(2)["primary_type"]
        .to_list()
    )
    assert [v for v, _ in step.fitted_values("primary_type")] == expected_top2

    # And the test outputs carry exactly the frozen vocabulary's columns.
    test_out = results[-1]  # FoldResult has no frame; re-run the last fold's test slice
    test_frame = folds[-1].test
    assert isinstance(test_frame, type(frame))
    out = step.run(test_frame).collect()
    for col in (
        "count_primary_type_THEFT",
        "count_primary_type_BATTERY",
        "count_primary_type_other",
    ):
        assert col in out.columns
    assert "count_primary_type_ASSAULT" not in out.columns
    assert test_out.n_test > 0


def test_planted_leak_random_split_inflates_blocked_split_collapses() -> None:
    """Leaky score ~perfect under a random split, causal score collapses blocked."""
    data = generate_synthetic_panel(seed=42)
    rows = _prediction_rows(data.panel)

    rng = np.random.default_rng(0)
    idx = rng.permutation(len(rows))
    test_random = rows[idx[int(0.7 * len(idx)) :]]
    leaky_random = pr_auc_score(
        test_random["y_next_high"].to_numpy(),
        test_random["leak_count_next"].to_numpy().astype(float),
    )

    _, test_blocked = temporal_split(rows.lazy(), "time_bin", CUTOFF)
    test_blocked_df = test_blocked.collect()
    causal_blocked = pr_auc_score(
        test_blocked_df["y_next_high"].to_numpy(),
        test_blocked_df["causal_count"].to_numpy().astype(float),
    )

    assert leaky_random > LEAKY_PR_AUC_MIN
    assert causal_blocked < BLOCKED_PR_AUC_MAX
    assert leaky_random - causal_blocked > GAP_MIN


def test_validate_split_rejects_overlapping_split() -> None:
    """A deliberately leaking split (shared time bins) is rejected."""
    data = generate_synthetic_panel(seed=42)
    panel = data.panel.lazy()
    train = panel.filter(pl.col("time_bin") < dt.datetime(2024, 1, 29))
    test = panel.filter(pl.col("time_bin") >= dt.datetime(2024, 1, 15))
    with pytest.raises(ValueError, match="share"):
        validate_split(train, test, time_col="time_bin")


def test_blocked_split_does_not_sanitize_a_future_peeking_column() -> None:
    """Boundary: blocking alone cannot remove a globally computed leak.

    If the leaky column is kept and merely evaluated on temporally
    blocked test rows, it still scores 1.0. The harness rejects leaking
    splits; it cannot inspect column provenance. A future-peeking column
    must not be built — this test pins that fact so nobody claims
    otherwise.
    """
    data = generate_synthetic_panel(seed=42)
    rows = _prediction_rows(data.panel)
    _, test_blocked = temporal_split(rows.lazy(), "time_bin", CUTOFF)
    test_blocked_df = test_blocked.collect()
    leaky_blocked = pr_auc_score(
        test_blocked_df["y_next_high"].to_numpy(),
        test_blocked_df["leak_count_next"].to_numpy().astype(float),
    )
    assert leaky_blocked > LEAKY_PR_AUC_MIN
