"""Tests for PAI and PR-AUC metrics."""

from __future__ import annotations

import numpy as np
import pytest

from eventflow.evaluation.metrics import pai_score, pr_auc_score


def test_pai_perfect_ranking() -> None:
    y_true = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    y_score = [0.9, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    # Top 20% captures both events: PAI = 1 / 0.2 = 5.
    assert pai_score(y_true, y_score, 0.2) == pytest.approx(5.0)


def test_pai_worst_ranking_is_zero() -> None:
    y_true = [0, 0, 0, 0, 1, 1]
    y_score = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    assert pai_score(y_true, y_score, 1 / 3) == pytest.approx(0.0)


def test_pai_all_positive_is_one() -> None:
    assert pai_score([1, 1, 1, 1], [0.4, 0.3, 0.2, 0.1], 0.5) == pytest.approx(1.0)


def test_pai_uses_realised_fraction_for_discrete_ranks() -> None:
    # n = 10, area 0.25 -> n_top = ceil(2.5) = 3, realised fraction 0.3.
    y_true = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    y_score = [0.9, 0.8, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    assert pai_score(y_true, y_score, 0.25) == pytest.approx(1.0 / 0.3)


def test_pai_ties_are_deterministic_stable_order() -> None:
    # All scores tied: stable sort keeps input order, top half is [1, 0].
    y_true = [1, 0, 1, 0]
    y_score = [0.5, 0.5, 0.5, 0.5]
    assert pai_score(y_true, y_score, 0.5) == pytest.approx(1.0)
    assert pai_score(y_true, y_score, 0.5) == pai_score(y_true, y_score, 0.5)


def test_pai_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        pai_score([], [], 0.2)
    with pytest.raises(ValueError, match="different lengths"):
        pai_score([1, 0], [0.5], 0.5)
    with pytest.raises(ValueError, match="only 0/1"):
        pai_score([1, 2], [0.5, 0.5], 0.5)
    with pytest.raises(ValueError, match="area_fraction"):
        pai_score([1, 0], [0.5, 0.5], 0.0)
    with pytest.raises(ValueError, match="area_fraction"):
        pai_score([1, 0], [0.5, 0.5], 1.5)
    with pytest.raises(ValueError, match="no positive events"):
        pai_score([0, 0, 0], [0.5, 0.5, 0.5], 0.5)
    with pytest.raises(ValueError, match="finite"):
        pai_score([1, 0], [float("nan"), 0.5], 0.5)


def test_pr_auc_perfect_ranking() -> None:
    assert pr_auc_score([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]) == pytest.approx(1.0)


def test_pr_auc_known_value() -> None:
    # Thresholds: (r=0.5, p=1), (r=0.5, p=0.5), (r=1, p=2/3), (r=1, p=0.5).
    # Trapezoid from (0, 1): 0.5 * 1 + 0.5 * (0.5 + 2/3) / 2 = 0.791666...
    assert pr_auc_score([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.6]) == pytest.approx(0.7916667)


def test_pr_auc_worst_ranking() -> None:
    # Positives ranked last: area = 1/12 + 5/24 = 7/24.
    assert pr_auc_score([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == pytest.approx(7 / 24)


def test_pr_auc_all_positive_is_one() -> None:
    assert pr_auc_score([1, 1, 1], [0.3, 0.2, 0.1]) == pytest.approx(1.0)


def test_pr_auc_ties_consumed_as_one_group() -> None:
    # All scores tied: single threshold, precision 0.5 at recall 1.
    # Curve (0, 1) -> (1, 0.5): area 0.75.
    assert pr_auc_score([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.75)
    # Tied positives vs tied negatives must not depend on input order.
    assert pr_auc_score([1, 0], [0.5, 0.5]) == pr_auc_score([0, 1], [0.5, 0.5])


def test_pr_auc_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        pr_auc_score([], [])
    with pytest.raises(ValueError, match="different lengths"):
        pr_auc_score([1, 0], [0.5])
    with pytest.raises(ValueError, match="only 0/1"):
        pr_auc_score([1, -1], [0.5, 0.5])
    with pytest.raises(ValueError, match="no positive events"):
        pr_auc_score([0, 0], [0.5, 0.5])
    with pytest.raises(ValueError, match="finite"):
        pr_auc_score([1, 0], [float("inf"), 0.5])


def test_metrics_accept_numpy_arrays() -> None:
    y_true = np.array([1, 0, 1, 0])
    y_score = np.array([0.9, 0.2, 0.8, 0.1])
    assert pr_auc_score(y_true, y_score) == pytest.approx(1.0)
    assert pai_score(y_true, y_score, 0.5) == pytest.approx(2.0)
