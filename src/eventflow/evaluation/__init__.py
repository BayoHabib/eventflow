"""Blocked splits, walk-forward evaluation, and downstream metrics."""

from eventflow.evaluation.metrics import pai_score, pr_auc_score
from eventflow.evaluation.splits import (
    blocked_split,
    entity_split,
    partition_units,
    spatial_split,
    temporal_split,
    validate_split,
)
from eventflow.evaluation.walkforward import (
    FoldResult,
    WalkForwardFold,
    evaluate_walk_forward,
    walk_forward_splits,
)

__all__ = [
    "FoldResult",
    "WalkForwardFold",
    "blocked_split",
    "entity_split",
    "evaluate_walk_forward",
    "pai_score",
    "partition_units",
    "pr_auc_score",
    "spatial_split",
    "temporal_split",
    "validate_split",
    "walk_forward_splits",
]
