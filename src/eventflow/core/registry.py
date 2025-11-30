"""Registry helpers for reusable pipeline components and plugins."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from importlib import metadata
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel

from eventflow.core.context.sources import BaseContextSource
from eventflow.core.output_adapters import BaseOutputAdapter

from .pipeline import Step
from .utils import get_logger

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from .pipeline import Pipeline

StepDefinition = str | tuple[str, Mapping[str, Any] | None] | Mapping[str, Any]

logger = get_logger(__name__)

ALLOWED_STEP_TAGS = frozenset({"spatial", "temporal", "context", "continuous"})


def _ensure_base_model(model: type[BaseModel] | None) -> type[BaseModel] | None:
    if model is None:
        return None
    if not issubclass(model, BaseModel):
        raise TypeError("config_model must inherit from pydantic.BaseModel")
    return model


def _iter_entry_points(group: str) -> Iterator[metadata.EntryPoint]:
    eps = metadata.entry_points()
    if hasattr(eps, "select"):
        selected = eps.select(group=group)
        return iter(selected)

    if isinstance(eps, dict):
        legacy = eps.get(group, ())
    else:
        legacy = ()

    return iter(cast(Iterable[metadata.EntryPoint], legacy))


@dataclass(frozen=True, slots=True)
class StepSpec:
    """Descriptor for a registered pipeline step."""

    name: str
    cls: type[Step]
    tags: frozenset[str] = field(default_factory=frozenset)
    description: str | None = None
    config_model: type[BaseModel] | None = None


class FeatureStepRegistry:
    """In-memory registry for feature engineering steps."""

    entry_point_group = "eventflow.steps"

    def __init__(self) -> None:
        self._registry: dict[str, StepSpec] = {}

    def register(
        self,
        name: str,
        step_cls: type[Step],
        *,
        tags: Iterable[str] | None = None,
        description: str | None = None,
        config_model: type[BaseModel] | None = None,
    ) -> StepSpec:
        """Register a step class under *name* and return the spec."""
        if name in self._registry:
            raise ValueError(f"Step already registered: {name}")

        normalised_tags = frozenset(tags or ())
        invalid_tags = normalised_tags - ALLOWED_STEP_TAGS
        if invalid_tags:
            raise ValueError(f"Unsupported step tags: {sorted(invalid_tags)}")

        config_model = _ensure_base_model(config_model)

        spec = StepSpec(
            name=name,
            cls=step_cls,
            tags=normalised_tags,
            description=description,
            config_model=config_model,
        )
        self._registry[name] = spec
        logger.debug("Registered step %s with tags=%s", name, sorted(normalised_tags))
        return spec

    def decorator(
        self,
        name: str,
        *,
        tags: Iterable[str] | None = None,
        description: str | None = None,
        config_model: type[BaseModel] | None = None,
    ) -> Callable[[type[Step]], type[Step]]:
        """Decorator for registering a Step subclass."""

        def wrapper(step_cls: type[Step]) -> type[Step]:
            if not issubclass(step_cls, Step):  # pragma: no cover - defensive guard
                raise TypeError("Only Step subclasses can be registered")
            self.register(
                name,
                step_cls,
                tags=tags,
                description=description,
                config_model=config_model,
            )
            return step_cls

        return wrapper

    def load_entry_points(self) -> None:
        """Discover and register external steps via Python entry points."""

        for ep in _iter_entry_points(self.entry_point_group):
            try:
                loader = ep.load()
            except Exception as exc:  # pragma: no cover - defensive logging path
                logger.error("Failed to load entry point %s: %s", ep.name, exc)
                continue

            if callable(loader):
                loader(self)
            else:  # pragma: no cover - defensive logging path
                logger.warning("Entry point %s did not return a callable; skipping", ep.name)

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

        kwargs: dict[str, Any]
        if params is None:
            kwargs = {}
        elif spec.config_model is not None:
            config = spec.config_model(**dict(params))
            kwargs = config.model_dump()
        else:
            kwargs = dict(params)

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


@dataclass(frozen=True, slots=True)
class ContextSourceSpec:
    """Descriptor for registered context data sources."""

    name: str
    cls: type[BaseContextSource]
    tags: frozenset[str] = field(default_factory=frozenset)
    description: str | None = None
    config_model: type[BaseModel] | None = None


class ContextSourceRegistry:
    """Registry for context data source providers."""

    entry_point_group = "eventflow.context_sources"

    def __init__(self) -> None:
        self._registry: dict[str, ContextSourceSpec] = {}

    def register(
        self,
        name: str,
        source_cls: type[BaseContextSource],
        *,
        tags: Iterable[str] | None = None,
        description: str | None = None,
        config_model: type[BaseModel] | None = None,
    ) -> ContextSourceSpec:
        if name in self._registry:
            raise ValueError(f"Context source already registered: {name}")

        config_model = _ensure_base_model(config_model)
        spec = ContextSourceSpec(
            name=name,
            cls=source_cls,
            tags=frozenset(tags or ()),
            description=description,
            config_model=config_model,
        )
        self._registry[name] = spec
        logger.debug("Registered context source %s", name)
        return spec

    def decorator(
        self,
        name: str,
        *,
        tags: Iterable[str] | None = None,
        description: str | None = None,
        config_model: type[BaseModel] | None = None,
    ) -> Callable[[type[BaseContextSource]], type[BaseContextSource]]:
        def wrapper(source_cls: type[BaseContextSource]) -> type[BaseContextSource]:
            if not issubclass(source_cls, BaseContextSource):  # pragma: no cover
                raise TypeError("Only BaseContextSource subclasses can be registered")
            self.register(
                name,
                source_cls,
                tags=tags,
                description=description,
                config_model=config_model,
            )
            return source_cls

        return wrapper

    def load_entry_points(self) -> None:
        for ep in _iter_entry_points(self.entry_point_group):
            try:
                loader = ep.load()
            except Exception as exc:  # pragma: no cover - defensive logging path
                logger.error("Failed to load context entry point %s: %s", ep.name, exc)
                continue

            if callable(loader):
                loader(self)
            else:  # pragma: no cover
                logger.warning(
                    "Context entry point %s did not return a callable; skipping", ep.name
                )

    def get(self, name: str) -> ContextSourceSpec:
        return self._registry[name]

    def list(self) -> list[ContextSourceSpec]:
        return list(self._registry.values())

    def create(
        self,
        name: str,
        *,
        params: Mapping[str, Any] | None = None,
    ) -> BaseContextSource:
        spec = self.get(name)

        if params is None:
            kwargs: dict[str, Any] = {}
        elif spec.config_model is not None:
            config = spec.config_model(**dict(params))
            kwargs = config.model_dump()
        else:
            kwargs = dict(params)

        return spec.cls(**kwargs)


@dataclass(frozen=True, slots=True)
class OutputAdapterSpec:
    """Descriptor for output adapter registrations."""

    name: str
    cls: type[BaseOutputAdapter]
    tags: frozenset[str] = field(default_factory=frozenset)
    description: str | None = None
    config_model: type[BaseModel] | None = None


class OutputAdapterRegistry:
    """Registry for output adapters (materialisation targets)."""

    entry_point_group = "eventflow.output_adapters"

    def __init__(self) -> None:
        self._registry: dict[str, OutputAdapterSpec] = {}

    def register(
        self,
        name: str,
        adapter_cls: type[BaseOutputAdapter],
        *,
        tags: Iterable[str] | None = None,
        description: str | None = None,
        config_model: type[BaseModel] | None = None,
    ) -> OutputAdapterSpec:
        if name in self._registry:
            raise ValueError(f"Output adapter already registered: {name}")

        config_model = _ensure_base_model(config_model)
        spec = OutputAdapterSpec(
            name=name,
            cls=adapter_cls,
            tags=frozenset(tags or ()),
            description=description,
            config_model=config_model,
        )
        self._registry[name] = spec
        logger.debug("Registered output adapter %s", name)
        return spec

    def decorator(
        self,
        name: str,
        *,
        tags: Iterable[str] | None = None,
        description: str | None = None,
        config_model: type[BaseModel] | None = None,
    ) -> Callable[[type[BaseOutputAdapter]], type[BaseOutputAdapter]]:
        def wrapper(adapter_cls: type[BaseOutputAdapter]) -> type[BaseOutputAdapter]:
            if not issubclass(adapter_cls, BaseOutputAdapter):  # pragma: no cover
                raise TypeError("Only BaseOutputAdapter subclasses can be registered")
            self.register(
                name,
                adapter_cls,
                tags=tags,
                description=description,
                config_model=config_model,
            )
            return adapter_cls

        return wrapper

    def load_entry_points(self) -> None:
        for ep in _iter_entry_points(self.entry_point_group):
            try:
                loader = ep.load()
            except Exception as exc:  # pragma: no cover
                logger.error("Failed to load output adapter entry point %s: %s", ep.name, exc)
                continue

            if callable(loader):
                loader(self)
            else:  # pragma: no cover
                logger.warning(
                    "Output adapter entry point %s did not return a callable; skipping",
                    ep.name,
                )

    def get(self, name: str) -> OutputAdapterSpec:
        return self._registry[name]

    def list(self) -> list[OutputAdapterSpec]:
        return list(self._registry.values())

    def create(
        self,
        name: str,
        *,
        params: Mapping[str, Any] | None = None,
    ) -> BaseOutputAdapter:
        spec = self.get(name)

        if params is None:
            kwargs: dict[str, Any] = {}
        elif spec.config_model is not None:
            config = spec.config_model(**dict(params))
            kwargs = config.model_dump()
        else:
            kwargs = dict(params)

        return spec.cls(**kwargs)


# Backwards compatibility export for previous API
StepRegistry = FeatureStepRegistry
