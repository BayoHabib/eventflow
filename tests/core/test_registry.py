"""Tests for pipeline step registry."""

import pytest

from eventflow.core.pipeline import Step
from eventflow.core.registry import StepRegistry


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
