"""Tests for pipeline step registry."""

import polars as pl
import pytest

from eventflow.core.event_frame import EventFrame
from eventflow.core.pipeline import Step
from eventflow.core.registry import StepRegistry
from eventflow.core.schema import EventMetadata, EventSchema, RecipeConfig
from eventflow.recipes.base import BaseRecipe


class DummyStep(Step):
    """No-op step for testing."""

    def run(self, event_frame):  # pragma: no cover - trivial pass-through
        return event_frame


class AnotherStep(Step):
    """Another no-op for tagging tests."""

    def run(self, event_frame):  # pragma: no cover - trivial pass-through
        return event_frame


@pytest.fixture()
def fresh_registry():
    return StepRegistry()


@pytest.fixture()
def tiny_event_frame():
    """Minimal EventFrame for pipeline integration tests."""

    lf = pl.LazyFrame(
        {
            "timestamp": ["2024-01-01T00:00:00", "2024-01-01T01:00:00"],
            "value": [1, 2],
            "latitude": [41.0, 41.1],
            "longitude": [-87.0, -87.1],
        }
    ).with_columns(pl.col("timestamp").str.to_datetime())

    schema = EventSchema(
        timestamp_col="timestamp",
        lat_col="latitude",
        lon_col="longitude",
        categorical_cols=[],
        numeric_cols=["value"],
    )
    metadata = EventMetadata(dataset_name="test-suite")
    return EventFrame(lf, schema, metadata)


def test_step_registry_register_and_get(fresh_registry):
    fresh_registry.register(
        name="dummy",
        step_cls=DummyStep,
        tags={"spatial", "aggregation"},
        description="Dummy test step",
        config_schema={"window": "int"},
    )

    spec = fresh_registry.get("dummy")
    assert spec.name == "dummy"
    assert spec.cls is DummyStep
    assert spec.tags == {"spatial", "aggregation"}
    assert spec.description == "Dummy test step"
    assert spec.config_schema == {"window": "int"}


def test_step_registry_filters_by_tag(fresh_registry):
    fresh_registry.register("dummy", DummyStep, tags={"spatial"})
    fresh_registry.register("another", AnotherStep, tags={"temporal"})

    spatial_specs = fresh_registry.list(tag="spatial")
    names = {spec.name for spec in spatial_specs}
    assert names == {"dummy"}

    all_specs = fresh_registry.list()
    assert {spec.name for spec in all_specs} == {"dummy", "another"}


def test_step_registry_rejects_duplicate_names(fresh_registry):
    fresh_registry.register("dummy", DummyStep)
    with pytest.raises(ValueError):
        fresh_registry.register("dummy", AnotherStep)


def test_step_registry_create_instantiates_step(fresh_registry):
    class WithParamsStep(Step):
        def __init__(self, label: str):
            self.label = label

        def run(self, event_frame):  # pragma: no cover - simple pass-through
            return event_frame

    fresh_registry.register("with_params", WithParamsStep)

    instance = fresh_registry.create("with_params", params={"label": "ok"})
    assert isinstance(instance, WithParamsStep)
    assert instance.label == "ok"


def test_step_registry_build_pipeline_runs_steps(fresh_registry, tiny_event_frame):
    class AddConstantStep(Step):
        def __init__(self, column: str, value: int):
            self.column = column
            self.value = value

        def run(self, event_frame):
            return event_frame.with_columns(
                **{self.column: pl.lit(self.value)}
            )

    class DoubleValueStep(Step):
        def run(self, event_frame):
            return event_frame.with_columns(value=pl.col("value") * 2)

    fresh_registry.register("add_constant", AddConstantStep)
    fresh_registry.register("double_value", DoubleValueStep)

    pipeline = fresh_registry.build_pipeline(
        [
            {"name": "add_constant", "params": {"column": "source", "value": 7}},
            "double_value",
        ]
    )

    result = pipeline.run(tiny_event_frame)
    df = result.collect()
    assert df["source"].to_list() == [7, 7]
    assert df["value"].to_list() == [2, 4]


def test_step_registry_build_pipeline_missing_step(fresh_registry):
    with pytest.raises(KeyError):
        fresh_registry.build_pipeline(["missing"])


def test_recipe_uses_registry_when_steps_configured(fresh_registry, tiny_event_frame):
    class DoubleValueStep(Step):
        def run(self, event_frame):
            return event_frame.with_columns(value=pl.col("value") * 2)

    fresh_registry.register("double", DoubleValueStep)

    config = RecipeConfig(
        dataset="demo",
        recipe="configured",
        features={"steps": ["double"]},
    )

    class RegistryRecipe(BaseRecipe):
        def build_pipeline(self):
            raise AssertionError("Should use registry-defined steps")

    recipe = RegistryRecipe(config, step_registry=fresh_registry)
    result = recipe.run(tiny_event_frame)
    df = result.collect()
    assert df["value"].to_list() == [2, 4]


def test_recipe_requires_registry_when_steps_configured(tiny_event_frame):
    config = RecipeConfig(
        dataset="demo",
        recipe="configured",
        features={"steps": ["noop"]},
    )

    class NoRegistryRecipe(BaseRecipe):
        def build_pipeline(self):
            raise AssertionError("Should not build default pipeline")

    recipe = NoRegistryRecipe(config)

    with pytest.raises(ValueError):
        recipe.run(tiny_event_frame)
