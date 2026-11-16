"""Downstream evaluation metrics, implemented on numpy only.

No sklearn/scipy: both metrics are pure functions over array-likes.
They are *downstream* evaluation examples (Phase 4 criterion 8), not
universal preprocessing acceptance criteria.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike


def _as_binary_labels(y_true: ArrayLike, *, where: str) -> np.ndarray:
    """Validate *y_true* as a non-empty 1-D array of 0/1 labels."""
    labels = np.asarray(y_true)
    if labels.ndim != 1:
        raise ValueError(f"{where}: y_true must be 1-D, got shape {labels.shape}")
    if labels.size == 0:
        raise ValueError(f"{where}: y_true must not be empty")
    if labels.dtype == bool:
        labels = labels.astype(np.int64)
    if not np.issubdtype(labels.dtype, np.integer):
        try:
            as_float = labels.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{where}: y_true must contain 0/1 labels") from exc
        if not np.all(np.isin(as_float, (0.0, 1.0))):
            raise ValueError(f"{where}: y_true must contain only 0/1 labels")
        labels = as_float.astype(np.int64)
    elif not np.all(np.isin(labels, (0, 1))):
        raise ValueError(f"{where}: y_true must contain only 0/1 labels")
    return labels.astype(np.int64)


def _as_scores(y_score: ArrayLike, *, n: int, where: str) -> np.ndarray:
    """Validate *y_score* as a finite 1-D float array of length *n*."""
    scores = np.asarray(y_score, dtype=float)
    if scores.ndim != 1:
        raise ValueError(f"{where}: y_score must be 1-D, got shape {scores.shape}")
    if scores.shape[0] != n:
        raise ValueError(
            f"{where}: y_true and y_score have different lengths: {n} vs {scores.shape[0]}"
        )
    if not np.all(np.isfinite(scores)):
        raise ValueError(f"{where}: y_score must be finite (no NaN or inf)")
    return scores


def pai_score(y_true: ArrayLike, y_score: ArrayLike, area_fraction: float) -> float:
    """Prediction Accuracy Index (PAI) for hotspot ranking.

    Rank all rows by *y_score* (descending) and take the top ``x`` fraction
    as the predicted hotspot area. With ``n`` rows, the top set has
    ``n_top = ceil(area_fraction * n)`` rows and realised area fraction
    ``a = n_top / n`` (ranks are discrete, so ``a`` can differ from the
    requested fraction by less than ``1 / n``). Then::

        PAI = (events captured in the top set / total events) / a

    PAI = 1 means the ranking is no better than chance; PAI > 1 means the
    top area concentrates events; the maximum is ``1 / a`` (all events in
    the top set). A perfect ranking of a rare event reaches that maximum.

    Tie handling: rows are ranked by score descending with a *stable* sort,
    so tied scores keep their input order. A tie straddling the cutoff is
    therefore resolved by input order, deterministically but arbitrarily —
    PAI is a ranking metric, not a threshold metric, and tied scores carry
    no ranking information.

    Args:
        y_true: Binary 0/1 labels (1 = event present).
        y_score: Ranking scores; higher means more likely an event.
        area_fraction: Requested top fraction, in (0, 1].

    Returns:
        The PAI value (>= 0).

    Raises:
        ValueError: empty input, mismatched lengths, non-binary labels,
            non-finite scores, ``area_fraction`` outside (0, 1], or no
            positive events (PAI is then 0/0, undefined).
    """
    where = "pai_score"
    labels = _as_binary_labels(y_true, where=where)
    scores = _as_scores(y_score, n=labels.shape[0], where=where)
    if not 0.0 < area_fraction <= 1.0:
        raise ValueError(f"{where}: area_fraction must be in (0, 1], got {area_fraction!r}")
    n_pos = int(labels.sum())
    if n_pos == 0:
        raise ValueError(f"{where}: no positive events; PAI is undefined (0/0)")

    n = labels.shape[0]
    n_top = max(1, math.ceil(area_fraction * n))
    order = np.argsort(-scores, kind="stable")
    captured = int(labels[order[:n_top]].sum())
    realised_fraction = n_top / n
    pai: float = (captured / n_pos) / realised_fraction
    return pai


def pr_auc_score(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Area under the precision-recall curve (trapezoid rule).

    Rows are ranked by *y_score* descending; precision and recall are
    evaluated after each distinct score value (the curve steps once per
    threshold, starting at recall 0 / precision 1) and the area under the
    resulting curve is integrated with the trapezoid rule.

    Tie handling: rows sharing a score are consumed as one group, so the
    curve — and the area — do not depend on the arbitrary order of tied
    rows. This matches the ``precision_recall_curve`` convention.

    A perfect ranking scores 1.0; a ranking that puts every positive last
    scores close to the positive rate. Unlike ROC-AUC, the chance level is
    the positive rate, not 0.5.

    Args:
        y_true: Binary 0/1 labels (1 = event present).
        y_score: Ranking scores; higher means more likely an event.

    Returns:
        The PR-AUC value in [0, 1].

    Raises:
        ValueError: empty input, mismatched lengths, non-binary labels,
            non-finite scores, or no positive events (recall is then
            undefined).
    """
    where = "pr_auc_score"
    labels = _as_binary_labels(y_true, where=where)
    scores = _as_scores(y_score, n=labels.shape[0], where=where)
    n_pos = int(labels.sum())
    if n_pos == 0:
        raise ValueError(f"{where}: no positive events; recall is undefined")

    order = np.argsort(-scores, kind="stable")
    ranked_labels = labels[order]
    ranked_scores = scores[order]

    # Group boundaries: indices where the score changes, plus the final row.
    change = np.flatnonzero(np.diff(ranked_scores) != 0.0) + 1
    boundaries = np.concatenate([change, [ranked_labels.shape[0]]])

    tp_cum = np.cumsum(ranked_labels, dtype=np.float64)
    counts = boundaries.astype(np.float64)
    tp_at = tp_cum[boundaries - 1]
    precision = tp_at / counts
    recall = tp_at / n_pos

    # The PR curve starts at (recall 0, precision 1) by convention.
    recall_pts = np.concatenate([[0.0], recall])
    precision_pts = np.concatenate([[1.0], precision])
    return float(np.trapezoid(precision_pts, recall_pts))


__all__ = ["pai_score", "pr_auc_score"]
