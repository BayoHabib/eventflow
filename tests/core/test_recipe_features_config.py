"""Tests for honest recipe configuration (phases 0 and 1).

Phase-0 rule: no configuration key may be silently ignored. An unknown key
or a declared-but-unimplemented feature must raise a blocking error, with a
message that lists the supported keys.

Phase 1 implements the previously declared-but-unimplemented keys
(categorical_features, aggregation_windows) plus panel completion, EWMA,
cyclical encodings, spatial lag, trend, recency and zero features — with
cross-key rules enforced by FeatureConfig.require_implemented().
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from shikong.core.schema import FeatureConfig, RecipeConfig
from shikong.recipes.registry import get_recipe


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
    assert config.features.complete_panel is False
    assert config.features.panel_min_periods == 1
    assert config.features.categorical_top_n == 10
    assert config.features.ewma_halflife_days is None
    assert config.features.cyclical_encoding is False
    assert config.features.spatial_lag is False
    assert config.features.trend is False
    assert config.features.recency is False
    assert config.features.zero_features is False


def test_features_steps_still_accepted() -> None:
    """The generic features.steps mechanism keeps working typed."""
    config = RecipeConfig(**_base_recipe_kwargs(), features={"steps": ["double"]})
    assert config.features.steps == ["double"]


def test_categorical_features_build_in_chicago_recipe() -> None:
    """Phase 1: categorical_features are implemented (CategoryCountsStep)."""
    config = RecipeConfig(
        **_base_recipe_kwargs(),
        features={"categorical_features": ["primary_type"]},
    )
    recipe = get_recipe("chicago_crime", "chicago_crime_v1", config)
    pipeline = recipe.build_pipeline()
    step_names = [type(s).__name__ for s in pipeline.steps]
    assert "CategoryCountsStep" in step_names
    # Category counts run before the aggregation that consumes them.
    assert step_names.index("CategoryCountsStep") < step_names.index("ChicagoCountStep")


def test_aggregation_windows_build_in_chicago_recipe() -> None:
    """Phase 1: aggregation_windows drive lags and rolling means."""
    config = RecipeConfig(
        **_base_recipe_kwargs(),
        features={"aggregation_windows": ["7d"]},
    )
    recipe = get_recipe("chicago_crime", "chicago_crime_v1", config)
    pipeline = recipe.build_pipeline()
    step_names = [type(s).__name__ for s in pipeline.steps]
    assert "TemporalLagStep" in step_names
    assert "MovingAverageStep" in step_names


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
    from shikong.core.steps import get_default_registry

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
    """The shipped YAML (phase 2: full feature set) loads and builds cleanly."""
    repo_root = Path(__file__).resolve().parents[2]
    with open(repo_root / "configs" / "recipes" / "chicago_crime_v1.yaml") as f:
        raw = yaml.safe_load(f)
    features = raw.get("features", {})
    assert features["categorical_features"] == ["primary_type", "location_description"]
    assert features["aggregation_windows"] == ["1d", "7d", "30d"]
    assert features["complete_panel"] is True
    assert features["spatial_lag_rings"] == [1, 2, 3]
    assert features["kde_bandwidth_m"] == 600.0
    assert features["stkde_bandwidth_m"] == 600.0
    assert features["stkde_halflife_days"] == 7.0
    assert features["near_repeat"] is True
    assert features["hawkes_halflife_days"] == 7.0
    assert "partition_by" not in raw.get("output", {})
    assert "buffer_m" not in raw.get("grid", {})
    config = RecipeConfig(**raw)
    recipe = get_recipe("chicago_crime", "chicago_crime_v1", config)
    pipeline = recipe.build_pipeline()
    assert [type(s).__name__ for s in pipeline.steps] == [
        "TransformCRSStep",
        "AssignToGridStep",
        "TimeBinsStep",
        "CategoryCountsStep",
        "ChicagoCountStep",
        "CompletePanelStep",
        "ExtractTemporalComponentsStep",
        "CalendarEncodingStep",
        "TemporalLagStep",
        "MovingAverageStep",
        "EWMAMeanStep",
        "TrendStep",
        "FillNullStep",
        "MultiScaleSpatialLagStep",
        "KernelDensityStep",
        "SpaceTimeKernelDensityStep",
        "NearRepeatStep",
        "HawkesExcitationStep",
        "TimeSinceLastEventStep",
        "BurstinessStep",
        "ZeroFeaturesStep",
    ]


def test_require_implemented_accepts_full_phase1_config() -> None:
    """A coherent full phase-1 declaration passes require_implemented."""
    config = FeatureConfig(
        categorical_features=["primary_type"],
        aggregation_windows=["1d", "7d", "30d"],
        complete_panel=True,
        ewma_halflife_days=7.0,
        cyclical_encoding=True,
        spatial_lag=True,
        trend=True,
        recency=True,
        zero_features=True,
    )
    config.require_implemented("test_recipe")  # must not raise


def test_require_implemented_spatial_lag_needs_windows() -> None:
    """A spatial lag without windows has no lagged count to lag spatially."""
    config = FeatureConfig(spatial_lag=True)
    with pytest.raises(ValueError, match="spatial_lag needs aggregation_windows"):
        config.require_implemented("test_recipe")


def test_require_implemented_trend_needs_two_windows() -> None:
    """A trend is a ratio of two rolling means — one window is not enough."""
    config = FeatureConfig(trend=True, aggregation_windows=["7d"])
    with pytest.raises(ValueError, match="trend needs at least two aggregation_windows"):
        config.require_implemented("test_recipe")


def test_require_implemented_recency_needs_complete_panel() -> None:
    """Trailing recency statistics over a sparse panel would be misleading."""
    config = FeatureConfig(recency=True)
    with pytest.raises(ValueError, match="recency.*complete_panel"):
        config.require_implemented("test_recipe")


def test_require_implemented_zero_features_needs_complete_panel() -> None:
    """Zero features are only defined on a completed panel."""
    config = FeatureConfig(zero_features=True)
    with pytest.raises(ValueError, match="zero_features.*complete_panel"):
        config.require_implemented("test_recipe")


def test_require_implemented_ewma_needs_complete_panel() -> None:
    """An EWMA over a sparse panel would mix observed and absent bins."""
    config = FeatureConfig(ewma_halflife_days=7.0)
    with pytest.raises(ValueError, match="ewma_halflife_days.*complete_panel"):
        config.require_implemented("test_recipe")


def test_non_whole_day_window_rejected() -> None:
    """Aggregation windows must be whole days (lags/rolling use days)."""
    with pytest.raises(ValidationError, match="whole number of days"):
        FeatureConfig(aggregation_windows=["12h"])
    with pytest.raises(ValidationError, match="Unknown aggregation window"):
        FeatureConfig(aggregation_windows=["1month"])


def test_window_days_sorted_and_deduplicated() -> None:
    """window_days() returns ascending, de-duplicated whole days."""
    config = FeatureConfig(aggregation_windows=["30d", "1d", "7d", "7d"])
    assert config.window_days() == [1, 7, 30]


def test_require_implemented_accepts_full_phase2_config() -> None:
    """A coherent full phase-2 declaration passes require_implemented."""
    config = FeatureConfig(
        categorical_features=["primary_type"],
        aggregation_windows=["1d", "7d", "30d"],
        complete_panel=True,
        ewma_halflife_days=7.0,
        cyclical_encoding=True,
        spatial_lag=True,
        spatial_lag_rings=[1, 2, 3],
        kde_bandwidth_m=600.0,
        stkde_bandwidth_m=600.0,
        stkde_halflife_days=7.0,
        near_repeat=True,
        hawkes_halflife_days=7.0,
        trend=True,
        recency=True,
        zero_features=True,
    )
    config.require_implemented("test_recipe")  # must not raise


def test_require_implemented_kde_needs_windows() -> None:
    """A KDE without windows has no lagged count to smooth."""
    config = FeatureConfig(kde_bandwidth_m=600.0)
    with pytest.raises(ValueError, match="kde_bandwidth_m needs aggregation_windows"):
        config.require_implemented("test_recipe")


def test_require_implemented_stkde_needs_both_parameters() -> None:
    """A space-time KDE needs both its bandwidth and its half-life."""
    config = FeatureConfig(stkde_bandwidth_m=600.0)
    with pytest.raises(ValueError, match="must be set together"):
        config.require_implemented("test_recipe")
    config = FeatureConfig(stkde_halflife_days=7.0)
    with pytest.raises(ValueError, match="must be set together"):
        config.require_implemented("test_recipe")


def test_require_implemented_phase2_panel_steps_need_complete_panel() -> None:
    """Near-repeat, Hawkes and STKDE are only defined on a completed panel."""
    for kwargs in (
        {"near_repeat": True},
        {"hawkes_halflife_days": 7.0},
        {"stkde_bandwidth_m": 600.0, "stkde_halflife_days": 7.0},
    ):
        config = FeatureConfig(**kwargs)
        with pytest.raises(ValueError, match="complete_panel"):
            config.require_implemented("test_recipe")


def test_spatial_lag_rings_rejected_when_invalid() -> None:
    """Rings must be non-empty, unique, positive integers."""
    for rings in ([], [0], [-1], [1, 1], [1.5], [True]):
        with pytest.raises(ValidationError, match="spatial_lag_rings"):
            FeatureConfig(spatial_lag_rings=rings)  # type: ignore[arg-type]


def test_phase2_bandwidths_must_be_positive() -> None:
    """Bandwidths and half-lives must be positive (pydantic gt=0)."""
    for kwargs in (
        {"kde_bandwidth_m": 0.0},
        {"kde_bandwidth_m": -600.0},
        {"stkde_bandwidth_m": 0.0},
        {"stkde_halflife_days": -7.0},
        {"hawkes_halflife_days": 0.0},
    ):
        with pytest.raises(ValidationError):
            FeatureConfig(**kwargs)
