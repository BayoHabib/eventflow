#!/usr/bin/env python
"""Worked downstream evaluation example on the Chicago replay (Phase 4, criterion 8).

DOWNSTREAM EXAMPLES ONLY — these numbers are NOT universal preprocessing
acceptance criteria. They illustrate how the Phase 1–3 features behave
under two standard downstream scorings; they do not certify the
preprocessing, and no pass/fail threshold is attached to them. The
Phase 4 acceptance criteria are the blocked splits, the walk-forward
harness, the fitted-statistics freeze and the synthetic ground truth —
not these numbers.

Scoring choices (stated explicitly; no model is trained):
- The replay is the Phase 3 Chicago replay (1,908,088 rows x 90 cols,
  sum ``count`` = 60,235), read from the audit parquet when present.
- Score: ``count_lag_1`` — the previous 6 h bin's count in the same
  cell (a lag feature the library produces; no model, no fitting).
  ``count_ma_7`` (trailing 7-bin mean) is reported as a second score
  for the PR-AUC only.
- Target for PAI: the count in the current bin (i.e. the next bin
  relative to the lag). Target for PR-AUC: ``count > 0`` in the
  current bin ("any crime in this cell/bin").

Metrics (small local implementations in this file; no harness import,
no new dependency):
- PAI (Predictive Accuracy Index) at x% of area: share of target
  counts captured by the top x% of ranked units, divided by x%.
  Two views are reported: (i) cell-bin PAI — the top x% of
  (cell, bin) pairs ranked by ``count_lag_1``; (ii) cell PAI — cells
  ranked by their mean ``count_lag_1`` over the period, top x% of
  cells, share of all crimes in those cells. PAI = 1 is chance.
- PR-AUC: step average precision of the score against the binary
  target. Chance PR-AUC equals the target prevalence, printed
  alongside.

Fallback: if no replay parquet is found, the script runs the Chicago
recipe once via ``scripts/run_chicago_recipe.py`` (a few minutes) and
reads its output. Large files are never copied into the repo.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = Path.home() / "workspace" / "eventflow-eda"

# Already-computed Phase 3 / Phase 2 replay parquets, in preference order.
REPLAY_CANDIDATES = [
    WORKSPACE / "phase3" / "work" / "audit" / "phase-3" / "chicago_features.parquet",
    WORKSPACE / "phase2" / "work" / "audit" / "phase-2" / "chicago_features.parquet",
    WORKSPACE / "data" / "processed" / "chicago_features.parquet",
]
RAW_CSV = WORKSPACE / "data" / "raw" / "chicago_crime" / "crimes_q1_2024.csv"

NEEDED_COLS = ["grid_id", "time_bin", "count", "count_lag_1", "count_ma_7"]


def average_precision(y: np.ndarray, scores: np.ndarray) -> float:
    """Step average precision; a constant score yields the prevalence."""
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="stable")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    precision = tp / np.maximum(tp + fp, 1)
    return float(np.sum(precision * y_sorted) / y_sorted.sum())


def pai_at_fraction(
    scores: np.ndarray, targets: np.ndarray, fraction: float
) -> tuple[float, float]:
    """PAI at ``fraction`` of ranked units: (capture share, PAI).

    Units are ranked by score descending; the top ``fraction`` of units
    capture ``share`` of the total target; PAI = share / fraction.
    """
    order = np.argsort(-scores, kind="stable")
    n_top = max(1, int(round(fraction * len(scores))))
    captured = float(targets[order[:n_top]].sum())
    total = float(targets.sum())
    share = captured / total
    return share, share / fraction


def load_replay() -> pl.DataFrame:
    """Read the replay parquet, or run the Chicago recipe once as fallback."""
    for path in REPLAY_CANDIDATES:
        if path.is_file():
            print(f"Reading replay parquet: {path}")
            return pl.read_parquet(path, columns=NEEDED_COLS)
    print("No replay parquet found; running the Chicago recipe once ...")
    with tempfile.TemporaryDirectory(prefix="phase4-chicago-") as tmp:
        out = Path(tmp) / "chicago_features.parquet"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "run_chicago_recipe.py"),
                "--input",
                str(RAW_CSV),
                "--output",
                str(out),
            ],
            check=True,
        )
        return pl.read_parquet(out, columns=NEEDED_COLS)


def main() -> None:
    """Compute PAI and PR-AUC on the Chicago replay and print the numbers."""
    print("=" * 72)
    print("DOWNSTREAM EVALUATION EXAMPLES ONLY — NOT acceptance criteria")
    print("=" * 72)
    df = load_replay()
    print(
        f"Replay rows: {df.height:,}; bins: {df['time_bin'].n_unique()}; "
        f"cells: {df['grid_id'].n_unique()}"
    )
    print(f"Total crimes (sum count): {df['count'].sum():,}")
    t_start = str(df["time_bin"].min())
    t_end = str(df["time_bin"].max())
    print(f"Time range: {t_start} .. {t_end}")
    print("Scores: count_lag_1 (previous-bin count), count_ma_7 (trailing mean).")
    print("No model is trained; the lag features themselves are the scores.")
    print()

    df = df.drop_nulls(["count_lag_1"])
    scores = df["count_lag_1"].to_numpy().astype(float)
    targets = df["count"].to_numpy().astype(float)

    print("PAI — cell-bin view (top x% of (cell, bin) pairs by count_lag_1,")
    print("  target = count in that bin; PAI = 1 is chance):")
    for fraction in (0.01, 0.10):
        share, pai = pai_at_fraction(scores, targets, fraction)
        print(f"  PAI at {fraction:.0%}: capture share = {share:.3f}, " f"PAI = {pai:.2f}")

    print("PAI — cell view (cells ranked by mean count_lag_1, target = all")
    print("  crimes in the cell over the period; PAI = 1 is chance):")
    per_cell = df.group_by("grid_id").agg(
        pl.col("count_lag_1").mean().alias("mean_lag_1"),
        pl.col("count").sum().alias("total_count"),
    )
    for fraction in (0.01, 0.10):
        share, pai = pai_at_fraction(
            per_cell["mean_lag_1"].to_numpy().astype(float),
            per_cell["total_count"].to_numpy().astype(float),
            fraction,
        )
        print(f"  PAI at {fraction:.0%}: capture share = {share:.3f}, " f"PAI = {pai:.2f}")

    print()
    print("PR-AUC — target = any crime in the cell/bin (count > 0):")
    y = (df["count"] > 0).to_numpy().astype(int)
    print(f"  prevalence of count > 0: {y.mean():.3f} (chance PR-AUC)")
    pr_lag1 = average_precision(y, scores)
    print(f"  score = count_lag_1: PR-AUC = {pr_lag1:.4f}")
    sub = df.drop_nulls(["count_ma_7"])
    pr_ma7 = average_precision(
        (sub["count"] > 0).to_numpy().astype(int),
        sub["count_ma_7"].to_numpy().astype(float),
    )
    print(f"  score = count_ma_7:  PR-AUC = {pr_ma7:.4f} (n = {sub.height:,})")
    print()
    print("Reminder: worked examples only. No threshold on PAI or PR-AUC")
    print("accepts or rejects the preprocessing.")


if __name__ == "__main__":
    main()
