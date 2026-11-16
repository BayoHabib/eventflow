#!/usr/bin/env python
"""Planted-leak demonstration (Phase 4, criterion 7).

Standalone: it imports only the synthetic generator
(``tests.evaluation.synthetic``) and computes everything else with plain
Polars/NumPy in this file. It does NOT import the evaluation harness
(``src/eventflow/evaluation/``) — the harness-level detection of the
leak (``validate_split`` rejecting the leaking split) is coordinator
integration work; this script demonstrates WHY blocking matters.

Setup
-----
A synthetic panel (seed 42, known ground truth) is turned into one
prediction row per (unit, bin ``t``):

- target ``y``: 1 if the count in the NEXT bin (``t + 1``) is >= 6
  ("very high next bin"), else 0. Prevalence ~ 11%.
- leaky score: ``count_next`` — the actual count of bin ``t + 1``,
  i.e. a column computed from future information. It is the target's
  own underlying value, so as a ranking score it is perfect by
  construction.
- causal score: ``count`` at bin ``t`` (the lag-1 count), the only
  information genuinely available at prediction time ``t``.

Two evaluation arms
--------------------
1. RANDOM split: rows are split 70/30 at random, the way a sloppy
   pipeline does it (features computed on the full panel BEFORE the
   split). The leaky score is evaluated on the test rows.
2. TEMPORAL blocked split: train rows are bins strictly before
   2024-01-29, test rows are bins on/after that date. A time-respecting
   pipeline cannot produce ``count_next`` at prediction time ``t`` —
   bin ``t + 1`` has not happened yet — so the blocked arm scores with
   the causal feature instead. (What the harness adds on top of this is
   rejecting a split/feature that smuggles the future in; see below.)

Expected outcome: the leaky score is ~perfect under the random split
and the causal score on the blocked split collapses to ~0.40 PR-AUC
(still above the 0.11 prevalence baseline, because the planted AR(1)
signal is real — the point is the collapse from 1.00, not a fall to
exactly the baseline).

One subtlety, printed honestly: if the leaky column is kept and merely
evaluated on temporally blocked test rows, it still scores 1.00 —
blocking the split alone does not remove a future-peeking column that
was computed globally before the split. Blocking matters because it
forces the evaluation to respect time order, which exposes that the
column is future information; removing/rejecting the column is the
harness's job (``validate_split``), not the split's.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

# Repo root on sys.path so the test utility package is importable.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests.evaluation.synthetic import generate_synthetic_panel  # noqa: E402

# Thresholds for the demonstration (margins, not fitted values): the
# leaky score is perfect by construction (1.00) and the causal score
# sits far below it (~0.40 in the prototype run).
LEAKY_PR_AUC_MIN = 0.95
BLOCKED_PR_AUC_MAX = 0.60
GAP_MIN = 0.35

CUTOFF = datetime(2024, 1, 29)  # 4 weeks train / 4 weeks test
HIGH_COUNT = 6


def average_precision(y: np.ndarray, scores: np.ndarray) -> float:
    """Step average precision (sklearn ``average_precision_score`` formula).

    AP = sum over positives of precision@rank / n_positives, with ties
    broken by stable order (descending score). A constant score yields
    AP == prevalence (chance for a ranking metric).
    """
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="stable")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    precision = tp / np.maximum(tp + fp, 1)
    return float(np.sum(precision * y_sorted) / y_sorted.sum())


def build_rows(panel: pl.DataFrame) -> pl.DataFrame:
    """One row per (unit, bin t): target = next-bin high count."""
    rows: list[tuple[int, datetime, int, int, int]] = []
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


def main() -> None:
    """Run the planted-leak demonstration and print both arms."""
    data = generate_synthetic_panel(seed=42)
    rows = build_rows(data.panel)
    prevalence = float(rows.select(pl.col("y_next_high").mean()).item())
    print("Planted-leak demo (synthetic panel, seed 42)")
    print(f"  rows: {rows.height}, target = next-bin count >= {HIGH_COUNT}")
    print(f"  prevalence of target: {prevalence:.3f} (chance PR-AUC)")

    # --- Arm 1: random row split, leaky feature computed before the split.
    rng = np.random.default_rng(0)
    is_train = rng.random(rows.height) < 0.7
    test_random = rows.filter(~pl.Series(is_train))
    pr_random = average_precision(
        test_random["y_next_high"].to_numpy(), test_random["leak_count_next"].to_numpy()
    )
    print(f"  RANDOM split, leaky score (count of NEXT bin): PR-AUC = {pr_random:.4f}")

    # --- Arm 2: temporal blocked split, causal feature only.
    test_blocked = rows.filter(pl.col("time_bin") >= CUTOFF)
    train_blocked = rows.filter(pl.col("time_bin") < CUTOFF)
    pr_blocked = average_precision(
        test_blocked["y_next_high"].to_numpy(), test_blocked["causal_count"].to_numpy()
    )
    print(
        f"  BLOCKED split (train < {CUTOFF.date()}, test >= {CUTOFF.date()}, "
        f"{train_blocked.height} train / {test_blocked.height} test rows), "
        f"causal score (count at t): PR-AUC = {pr_blocked:.4f}"
    )

    # --- Honesty footnote: blocking alone does not remove the column.
    pr_naive_blocked = average_precision(
        test_blocked["y_next_high"].to_numpy(), test_blocked["leak_count_next"].to_numpy()
    )
    print(
        "  NOTE: the leaky column evaluated naively on the blocked test rows "
        f"still scores {pr_naive_blocked:.4f} — a temporal block alone does not "
        "remove a future-peeking column computed globally before the split. "
        "Blocking matters because it forces time order, which exposes the "
        "column as future information; rejecting it is the harness's job."
    )
    print(f"  Gap (random leaky - blocked causal): {pr_random - pr_blocked:.4f}")

    assert pr_random > LEAKY_PR_AUC_MIN, f"leaky PR-AUC {pr_random:.4f} not near-perfect"
    assert pr_blocked < BLOCKED_PR_AUC_MAX, f"blocked PR-AUC {pr_blocked:.4f} did not collapse"
    assert pr_random - pr_blocked > GAP_MIN, "gap between arms too small"
    print(
        "  DEMO OK: the leak scores ~perfect under a random split and collapses "
        "under a temporal block."
    )


if __name__ == "__main__":
    main()
