"""Registry helpers for reusable pipeline steps."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

from .pipeline import Step
from .utils import get_logger

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from .pipeline import Pipeline

StepDefinition = str | tuple[str, Mapping[str, Any] | None] | Mapping[str, Any]

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class StepSpec:
    """Descriptor for a registered pipeline step."""

    name: str
    cls: type[Step]
    tags: frozenset[str] = field(default_factory=frozenset)
    description: str | None = None
    config_schema: Mapping[str, Any] | None = None


class StepRegistry:
    """In-memory registry for pipeline steps keyed by name."""

    def __init__(self) -> None:
        self._registry: dict[str, StepSpec] = {}

    def register(
        self,
        name: str,
        step_cls: type[Step],
        *,
        tags: Iterable[str] | None = None,
        description: str | None = None,
        config_schema: Mapping[str, Any] | None = None,
    ) -> StepSpec:
        """Register a step class under *name* and return the spec."""
        if name in self._registry:
            raise ValueError(f"Step already registered: {name}")

        tag_set = frozenset(tags or ())
        schema_map = (
            MappingProxyType(dict(config_schema)) if config_schema is not None else None
        )

        spec = StepSpec(
            name=name,
            cls=step_cls,
            tags=tag_set,
            description=description,
            config_schema=schema_map,
        )
        self._registry[name] = spec
        logger.debug("Registered step %s with tags=%s", name, sorted(tag_set))
        return spec

    def get(self, name: str) -> StepSpec:
        """Return the step specification for *name* or raise KeyError."""
        return self._registry[name]

    def list(self, *, tag: str | None = None) -> list[StepSpec]:
        """List registered step specs, optionally filtered by *tag*."""
        specs = list(self._registry.values())
        if tag is None:
            return specs
        return [spec for spec in specs if tag in spec.tags]

    def create(
        self,
        name: str,
        *,
        params: Mapping[str, Any] | None = None,
    ) -> Step:
        """Instantiate the step identified by *name* with optional params."""
        spec = self.get(name)
        kwargs = dict(params or {})
        instance = spec.cls(**kwargs)
        logger.debug("Instantiated step %s with params=%s", name, sorted(kwargs.keys()))
        return instance

    def build_pipeline(self, steps: Iterable[StepDefinition]) -> Pipeline:
        """Construct a Pipeline from an iterable of step definitions."""
        from .pipeline import Pipeline  # Local import to avoid circular reference

        instances: list[Step] = []
        for entry in steps:
            if isinstance(entry, str):
                name = entry
                params: Mapping[str, Any] | None = None
            elif isinstance(entry, tuple) and len(entry) == 2:
                name, params = entry
            elif isinstance(entry, Mapping):
                raw_name = entry.get("name")
                if not isinstance(raw_name, str):
                    raise ValueError("Step mapping must include a string 'name' key")
                name = raw_name
                raw_params = entry.get("params")
                if raw_params is None:
                    raw_params = entry.get("config")
                params = cast(Mapping[str, Any] | None, raw_params)
            else:
                raise TypeError(
                    "Step definitions must be str, mapping with 'name', or (name, params) tuple"
                )
            if params is not None and not isinstance(params, Mapping):
                raise TypeError("Step params must be a mapping when provided")
            instances.append(self.create(name, params=params))

        logger.debug("Built pipeline with steps=%s", [type(step).__name__ for step in instances])
        return Pipeline(instances)
