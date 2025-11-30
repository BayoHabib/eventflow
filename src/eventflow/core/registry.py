"""Registry helpers for reusable pipeline steps."""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from .pipeline import Step
from .utils import get_logger

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
