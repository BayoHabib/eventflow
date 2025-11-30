"""Tests for pipeline step registry."""

from typing import Any

import polars as pl
import pytest
from pydantic import BaseModel

from eventflow.core.context.sources import BaseContextSource
from eventflow.core.event_frame import EventFrame
from eventflow.core.output_adapters import BaseOutputAdapter
from eventflow.core.pipeline import Pipeline, Step
from eventflow.core.registry import (
    ContextSourceRegistry,
    FeatureStepRegistry,
    OutputAdapterRegistry,
)
from eventflow.core.schema import ContextSchema, EventMetadata, EventSchema, RecipeConfig
from eventflow.recipes.base import BaseRecipe


class DummyStep(Step):
    """No-op step for testing."""

    def run(self, event_frame: EventFrame) -> EventFrame:  # pragma: no cover - trivial pass-through
        return event_frame


class AnotherStep(Step):
    """Another no-op for tagging tests."""

    def run(self, event_frame: EventFrame) -> EventFrame:  # pragma: no cover - trivial pass-through
        return event_frame


@pytest.fixture()
def fresh_registry() -> FeatureStepRegistry:
    return FeatureStepRegistry()


@pytest.fixture()
def tiny_event_frame() -> EventFrame:
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


def test_step_registry_register_and_get(fresh_registry: FeatureStepRegistry) -> None:
    class DummyConfig(BaseModel):
        window: int

    fresh_registry.register(
        name="dummy",
        step_cls=DummyStep,
        tags={"spatial", "context"},
        description="Dummy test step",
        config_model=DummyConfig,
    )

    spec = fresh_registry.get("dummy")
    assert spec.name == "dummy"
    assert spec.cls is DummyStep
    assert spec.tags == {"spatial", "context"}
    assert spec.description == "Dummy test step"
    assert spec.config_model is DummyConfig


def test_step_registry_filters_by_tag(fresh_registry: FeatureStepRegistry) -> None:
    fresh_registry.register("dummy", DummyStep, tags={"spatial"})
    fresh_registry.register("another", AnotherStep, tags={"temporal"})

    spatial_specs = fresh_registry.list(tag="spatial")
    names = {spec.name for spec in spatial_specs}
    assert names == {"dummy"}

    all_specs = fresh_registry.list()
    assert {spec.name for spec in all_specs} == {"dummy", "another"}


def test_step_registry_rejects_duplicate_names(fresh_registry: FeatureStepRegistry) -> None:
    fresh_registry.register("dummy", DummyStep)
    with pytest.raises(ValueError):
        fresh_registry.register("dummy", AnotherStep)


def test_step_registry_create_instantiates_step(fresh_registry: FeatureStepRegistry) -> None:
    class WithParamsStep(Step):
        def __init__(self, label: str) -> None:
            self.label = label

        def run(
            self, event_frame: EventFrame
        ) -> EventFrame:  # pragma: no cover - simple pass-through
            return event_frame

    fresh_registry.register("with_params", WithParamsStep)

    instance = fresh_registry.create("with_params", params={"label": "ok"})
    assert isinstance(instance, WithParamsStep)
    assert instance.label == "ok"


def test_step_registry_build_pipeline_runs_steps(
    fresh_registry: FeatureStepRegistry, tiny_event_frame: EventFrame
) -> None:
    class AddConstantStep(Step):
        def __init__(self, column: str, value: int) -> None:
            self.column = column
            self.value = value

        def run(self, event_frame: EventFrame) -> EventFrame:
            return event_frame.with_columns(**{self.column: pl.lit(self.value)})

    class DoubleValueStep(Step):
        def run(self, event_frame: EventFrame) -> EventFrame:
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


def test_step_registry_build_pipeline_missing_step(fresh_registry: FeatureStepRegistry) -> None:
    with pytest.raises(KeyError):
        fresh_registry.build_pipeline(["missing"])


def test_recipe_uses_registry_when_steps_configured(
    fresh_registry: FeatureStepRegistry, tiny_event_frame: EventFrame
) -> None:
    class DoubleValueStep(Step):
        def run(self, event_frame: EventFrame) -> EventFrame:
            return event_frame.with_columns(value=pl.col("value") * 2)

    fresh_registry.register("double", DoubleValueStep)

    config = RecipeConfig(
        dataset="demo",
        recipe="configured",
        features={"steps": ["double"]},
    )

    class RegistryRecipe(BaseRecipe):
        def build_pipeline(self) -> Pipeline:
            raise AssertionError("Should use registry-defined steps")

    recipe = RegistryRecipe(config, step_registry=fresh_registry)
    result = recipe.run(tiny_event_frame)
    df = result.collect()
    assert df["value"].to_list() == [2, 4]


def test_recipe_requires_registry_when_steps_configured(tiny_event_frame: EventFrame) -> None:
    config = RecipeConfig(
        dataset="demo",
        recipe="configured",
        features={"steps": ["noop"]},
    )

    class NoRegistryRecipe(BaseRecipe):
        def build_pipeline(self) -> Pipeline:
            raise AssertionError("Should not build default pipeline")

    recipe = NoRegistryRecipe(config)

    with pytest.raises(ValueError):
        recipe.run(tiny_event_frame)


def test_step_registry_decorator_registration() -> None:
    registry = FeatureStepRegistry()

    class DecoratedConfig(BaseModel):
        factor: float = 1.0

    @registry.decorator("decorated", tags={"temporal"}, config_model=DecoratedConfig)
    class DecoratedStep(Step):
        def __init__(self, factor: float = 1.0) -> None:
            self.factor = factor

        def run(self, event_frame: EventFrame) -> EventFrame:
            return event_frame.with_columns(value=pl.col("value") * self.factor)

    spec = registry.get("decorated")
    assert spec.config_model is DecoratedConfig
    instance = registry.create("decorated", params={"factor": 3})
    assert isinstance(instance, DecoratedStep)
    assert instance.factor == 3


def test_context_source_registry_registration() -> None:
    registry = ContextSourceRegistry()

    class Config(BaseModel):
        path: str

    class DummySource(BaseContextSource):
        def __init__(self, path: str) -> None:
            self._path = path

        def load(self) -> pl.LazyFrame:  # pragma: no cover - trivial stub
            return pl.LazyFrame({})

        @property
        def schema(self) -> ContextSchema:  # pragma: no cover - not used in assertions
            return ContextSchema(spatial_col="zone")

    registry.register("dummy", DummySource, config_model=Config)
    instance = registry.create("dummy", params={"path": "ctx.parquet"})
    assert isinstance(instance, DummySource)


def test_output_adapter_registry_registration() -> None:
    registry = OutputAdapterRegistry()

    class Config(BaseModel):
        destination: str

    class DummyAdapter(BaseOutputAdapter):
        def __init__(self, destination: str) -> None:
            self.destination = destination

        def write(self, event_frame: EventFrame, **kwargs: Any) -> None:  # pragma: no cover
            _ = event_frame

    registry.register("dummy", DummyAdapter, config_model=Config)
    adapter = registry.create("dummy", params={"destination": "s3://bucket"})
    assert isinstance(adapter, DummyAdapter)
