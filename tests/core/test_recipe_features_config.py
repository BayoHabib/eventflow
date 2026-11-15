"""Tests for honest recipe configuration (phase 0).

Phase-0 rule: no configuration key may be silently ignored. An unknown key
or a declared-but-unimplemented feature must raise a blocking error, with a
message that lists the supported keys.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from eventflow.core.schema import FeatureConfig, RecipeConfig
from eventflow.recipes.registry import get_recipe


def _base_recipe_kwargs() -> dict:
    return {"dataset": "chicago_crime", "recipe": "chicago_crime_v1"}


def test_unknown_features_key_rejected_with_supported_keys_listed() -> None:
    """A typo in the features section must fail loudly, not fall back."""
    with pytest.raises(ValidationError) as excinfo:
        RecipeConfig(**_base_recipe_kwargs(), features={"count_feature": True})
    message = str(excinfo.value)
    assert "count_feature" in message
    assert "count_features" in message  # the valid key is listed


def test_unknown_top_level_recipe_key_rejected() -> None:
    """An unknown top-level key must fail loudly."""
    with pytest.raises(ValidationError) as excinfo:
        RecipeConfig(**_base_recipe_kwargs(), grid_size_m=300)
    assert "grid_size_m" in str(excinfo.value)


def test_features_defaults_are_honest() -> None:
    """A bare config declares nothing unimplemented."""
    config = RecipeConfig(**_base_recipe_kwargs())
    assert config.features.count_features is True
    assert config.features.categorical_features == []
    assert config.features.aggregation_windows == []
    assert config.features.steps is None


def test_features_steps_still_accepted() -> None:
    """The generic features.steps mechanism keeps working typed."""
    config = RecipeConfig(**_base_recipe_kwargs(), features={"steps": ["double"]})
    assert config.features.steps == ["double"]


def test_categorical_features_raise_in_chicago_recipe() -> None:
    """Re-adding categorical_features must fail, not be silently dropped."""
    config = RecipeConfig(
        **_base_recipe_kwargs(),
        features={"categorical_features": ["primary_type"]},
    )
    recipe = get_recipe("chicago_crime", "chicago_crime_v1", config)
    with pytest.raises(ValueError, match="categorical_features.*not implemented"):
        recipe.build_pipeline()


def test_aggregation_windows_raise_in_chicago_recipe() -> None:
    """Re-adding aggregation_windows must fail, not be silently dropped."""
    config = RecipeConfig(
        **_base_recipe_kwargs(),
        features={"aggregation_windows": ["7d"]},
    )
    recipe = get_recipe("chicago_crime", "chicago_crime_v1", config)
    with pytest.raises(ValueError, match="aggregation_windows.*not implemented"):
        recipe.build_pipeline()


def test_count_features_false_rejected_by_chicago_recipe() -> None:
    """The Chicago recipe always counts; disabling it must fail loudly."""
    config = RecipeConfig(**_base_recipe_kwargs(), features={"count_features": False})
    recipe = get_recipe("chicago_crime", "chicago_crime_v1", config)
    with pytest.raises(ValueError, match="always counts"):
        recipe.build_pipeline()


def test_grid_buffer_rejected_by_schema() -> None:
    """grid.buffer_m is not a supported key — rejected at load, not ignored."""
    with pytest.raises(ValidationError) as excinfo:
        RecipeConfig(**_base_recipe_kwargs(), grid={"size_m": 300, "buffer_m": 100})
    assert "buffer_m" in str(excinfo.value)


def test_grid_typo_rejected_with_supported_keys_listed() -> None:
    """A typo such as grid.siz_m must fail instead of falling back to 300."""
    with pytest.raises(ValidationError) as excinfo:
        RecipeConfig(**_base_recipe_kwargs(), grid={"siz_m": 300})
    message = str(excinfo.value)
    assert "siz_m" in message
    assert "size_m" in message


def test_temporal_typo_rejected() -> None:
    """A typo in the temporal section must fail loudly."""
    with pytest.raises(ValidationError) as excinfo:
        RecipeConfig(**_base_recipe_kwargs(), temporal={"time_binn": "6h"})
    assert "time_binn" in str(excinfo.value)


def test_temporal_component_value_rejected() -> None:
    """Issue #30: an unknown temporal component value must fail loudly.

    Unknown *keys* were already rejected (phase 0); an unknown *value*
    such as ``hour_of_dya`` must be rejected the same way, with the
    supported components listed, instead of silently producing no column.
    """
    with pytest.raises(ValidationError) as excinfo:
        RecipeConfig(**_base_recipe_kwargs(), temporal={"components": ["hour_of_dya"]})
    message = str(excinfo.value)
    assert "hour_of_dya" in message
    assert "hour_of_day" in message  # the valid component is listed


def test_get_pipeline_rejects_unknown_configured_step() -> None:
    """Issue #29: get_pipeline() must resolve features.steps, not ignore it.

    This is the contract CLI validate relies on: the pipeline that
    run()/run_async() will execute must fail here for an unknown step
    instead of falling back to the recipe's built-in pipeline.
    """
    from eventflow.core.steps import get_default_registry

    config = RecipeConfig(**_base_recipe_kwargs(), features={"steps": ["not_a_real_step"]})
    recipe = get_recipe(
        "chicago_crime", "chicago_crime_v1", config, step_registry=get_default_registry()
    )
    with pytest.raises(KeyError, match="not_a_real_step"):
        recipe.get_pipeline()


def test_typed_grid_and_temporal_defaults() -> None:
    """Grid/temporal sections keep their documented defaults, typed."""
    config = RecipeConfig(**_base_recipe_kwargs())
    assert config.grid.size_m == 300.0
    assert config.grid.origin == [0.0, 0.0]
    assert config.temporal.time_bin == "6h"
    assert config.temporal.components == ["hour_of_day", "day_of_week"]


def test_enabled_context_raises_in_chicago_recipe() -> None:
    """Enabling a context source must fail until enrichment is wired."""
    config = RecipeConfig(**_base_recipe_kwargs(), context={"weather": {"enabled": True}})
    recipe = get_recipe("chicago_crime", "chicago_crime_v1", config)
    with pytest.raises(ValueError, match="context"):
        recipe.build_pipeline()


def test_shipped_chicago_yaml_loads_and_builds() -> None:
    """The shipped YAML (after phase-0 cleanup) loads and builds cleanly."""
    repo_root = Path(__file__).resolve().parents[2]
    with open(repo_root / "configs" / "recipes" / "chicago_crime_v1.yaml") as f:
        raw = yaml.safe_load(f)
    assert "categorical_features" not in raw.get("features", {})
    assert "aggregation_windows" not in raw.get("features", {})
    assert "partition_by" not in raw.get("output", {})
    assert "buffer_m" not in raw.get("grid", {})
    config = RecipeConfig(**raw)
    recipe = get_recipe("chicago_crime", "chicago_crime_v1", config)
    pipeline = recipe.build_pipeline()
    assert len(pipeline.steps) == 5


def test_require_implemented_lists_both_features() -> None:
    """FeatureConfig.require_implemented names the offending declaration."""
    config = FeatureConfig(categorical_features=["primary_type"], aggregation_windows=["1d"])
    with pytest.raises(ValueError, match="categorical_features"):
        config.require_implemented("test_recipe")
