"""Fitted-statistics inventory and recipe-path checks (Phase 4, criterion 5).

The executable form of the inventory in
``docs/design-notes/phase4-fitted-statistics.md``: among all Step classes
in the library, only ``CategoryCountsStep`` learns statistics from data
and reuses them across frames (its top-N vocabulary). Every other step
recomputes whatever it needs from the frame it is run on, so there is
nothing to freeze — and nothing to leak across a train/test split via
step state.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from eventflow.core.pipeline import FittableStep, Step
from eventflow.core.schema import RecipeConfig
from eventflow.core.steps.categorical import CategoryCountsStep
from eventflow.core.steps.registration import get_default_registry
from eventflow.datasets.chicago_crime.recipes.chicago_crime_v1 import ChicagoCountStep
from eventflow.recipes.registry import get_recipe

CONFIGS = Path(__file__).resolve().parents[2] / "configs" / "recipes"


def _all_step_classes() -> set[type[Step]]:
    """Every Step subclass reachable from the steps package and recipes."""
    import eventflow.core.steps  # noqa: F401 — imports every step module

    seen: set[type[Step]] = set()
    queue: list[type[Step]] = [Step]
    while queue:
        cls = queue.pop()
        if cls in seen:
            continue
        seen.add(cls)
        queue.extend(cls.__subclasses__())
    seen.discard(Step)
    return seen


def test_fittable_step_protocol_shape() -> None:
    """The contract worker A imports: fit() + is_fitted, nothing else."""
    assert "fit" in FittableStep.__dict__
    assert "is_fitted" in FittableStep.__dict__
    step = CategoryCountsStep(category_cols=["primary_type"], top_n=2)
    assert hasattr(step, "fit") and hasattr(step, "is_fitted")
    assert step.is_fitted is False


def test_only_category_counts_step_defines_fit() -> None:
    """No Step class other than CategoryCountsStep learns fitted statistics.

    ``fit`` is looked up in each class's own ``__dict__`` (not inherited),
    so a step that merely inherits an implementation would still be caught.
    ``ChicagoCountStep`` (dataset-local, not in the step registry) is
    covered too via the import above.
    """
    assert ChicagoCountStep in _all_step_classes()
    fittable = [cls for cls in _all_step_classes() if "fit" in cls.__dict__]
    assert fittable == [CategoryCountsStep]


def test_registry_builds_category_counts_step_unfitted() -> None:
    """The YAML/registry path builds an unfitted step (recipes run unfitted)."""
    registry = get_default_registry()
    step = registry.create(
        "category_counts",
        params={"category_cols": ["primary_type"], "top_n": 2},
    )
    assert isinstance(step, CategoryCountsStep)
    assert step.is_fitted is False
    assert step.top_n == 2


def test_chicago_recipe_category_counts_step_is_unfitted() -> None:
    """The shipped Chicago recipe carries exactly one, unfitted, CategoryCountsStep.

    The recipe YAML declares ``categorical_features: [primary_type,
    location_description]`` with ``categorical_top_n: 10``; the step built
    from it must hold no frozen vocabulary — ``fit`` is a Python-API /
    harness concern, not YAML-declarable. The full Chicago replay
    (1,908,088 rows) is the coordinator's non-regression check, not a
    unit test; the unfitted column set is pinned in
    ``tests/core/test_categorical_steps.py::TestCategoryCountsFitFreeze``.
    """
    with (CONFIGS / "chicago_crime_v1.yaml").open(encoding="utf-8") as f:
        config = RecipeConfig(**yaml.safe_load(f))
    pipeline = get_recipe("chicago_crime", "chicago_crime_v1", config=config).get_pipeline()

    category_steps = [s for s in pipeline.steps if isinstance(s, CategoryCountsStep)]
    assert len(category_steps) == 1
    step = category_steps[0]
    assert step.is_fitted is False
    assert step.category_cols == ["primary_type", "location_description"]
    assert step.top_n == 10
    assert step.unit_col == "grid_id"
    assert step.time_col == "time_bin"
