"""Schema definitions for events and context sources."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from shikong.core._time import duration_seconds


class OutputModality(str, Enum):
    """Enumerates supported output modalities for derived artefacts."""

    TABLE = "table"
    SEQUENCE = "sequence"
    RASTER = "raster"
    GRAPH = "graph"
    STREAM = "stream"

    def __str__(self) -> str:  # pragma: no cover - trivial wrapper
        return self.value


class FeatureProvenance(BaseModel):
    """Record describing how a feature was produced during a pipeline run."""

    produced_by: str | None = None
    inputs: list[str] = Field(default_factory=list)
    tags: set[str] = Field(default_factory=set)
    description: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextRequirementState(BaseModel):
    """Aggregate requirements needed to interpret the EventFrame correctly."""

    spatial_crs: str | None = None
    temporal_resolution: str | None = None
    required_context: set[str] = Field(default_factory=set)
    notes: dict[str, Any] = Field(default_factory=dict)


class EventSchema(BaseModel):
    """
    Describes the structure of event data.

    Attributes:
        timestamp_col: Name of the timestamp column (required)
        lat_col: Name of latitude column (optional)
        lon_col: Name of longitude column (optional)
        geometry_col: Name of geometry column (optional, alternative to lat/lon)
        categorical_cols: List of categorical attribute columns
        numeric_cols: List of numeric attribute columns
        output_modalities: Supported downstream artefact modalities
        feature_provenance: Provenance metadata keyed by feature name
        context_requirements: Required context to interpret the data correctly
    """

    timestamp_col: str
    lat_col: str | None = None
    lon_col: str | None = None
    geometry_col: str | None = None
    categorical_cols: list[str] = Field(default_factory=list)
    numeric_cols: list[str] = Field(default_factory=list)
    output_modalities: set[OutputModality] = Field(default_factory=lambda: {OutputModality.TABLE})
    feature_provenance: dict[str, FeatureProvenance] = Field(default_factory=dict)
    context_requirements: ContextRequirementState = Field(default_factory=ContextRequirementState)

    @field_validator("lat_col", "lon_col")
    @classmethod
    def validate_coordinates(cls, v: str | None, info: Any) -> str | None:
        """Validate that lat/lon are provided together or geometry_col is set."""
        return v

    def model_post_init(self, __context: Any) -> None:
        """Validate that either lat/lon or geometry_col is provided."""
        has_latlon = self.lat_col is not None and self.lon_col is not None
        has_geometry = self.geometry_col is not None

        if not has_latlon and not has_geometry:
            raise ValueError("Either (lat_col and lon_col) or geometry_col must be provided")

    def compatibility_issues(self, other: EventSchema) -> list[str]:
        """Return human-readable compatibility issues when transitioning to *other*."""

        issues: list[str] = []

        if self.timestamp_col != other.timestamp_col:
            issues.append(
                "timestamp_col mismatch: " f"{self.timestamp_col!r} -> {other.timestamp_col!r}"
            )

        missing_modalities = {m for m in self.output_modalities if m not in other.output_modalities}
        if missing_modalities:
            issues.append(
                "missing output modalities: "
                + ", ".join(sorted(m.value for m in missing_modalities))
            )

        prev_features = set(self.feature_provenance)
        new_features = set(other.feature_provenance)
        missing_features = prev_features - new_features
        if missing_features:
            issues.append(
                "missing feature provenance entries: " + ", ".join(sorted(missing_features))
            )

        prev_ctx = self.context_requirements
        new_ctx = other.context_requirements

        if prev_ctx.spatial_crs and not new_ctx.spatial_crs:
            issues.append("spatial_crs removed")
        elif (
            prev_ctx.spatial_crs
            and new_ctx.spatial_crs
            and prev_ctx.spatial_crs != new_ctx.spatial_crs
        ):
            issues.append(
                "spatial_crs mismatch: " f"{prev_ctx.spatial_crs!r} -> {new_ctx.spatial_crs!r}"
            )

        if prev_ctx.temporal_resolution and not new_ctx.temporal_resolution:
            issues.append("temporal_resolution removed")
        elif (
            prev_ctx.temporal_resolution
            and new_ctx.temporal_resolution
            and prev_ctx.temporal_resolution != new_ctx.temporal_resolution
        ):
            issues.append(
                "temporal_resolution mismatch: "
                f"{prev_ctx.temporal_resolution!r} -> {new_ctx.temporal_resolution!r}"
            )

        missing_context = prev_ctx.required_context - new_ctx.required_context
        if missing_context:
            issues.append("missing required context tags: " + ", ".join(sorted(missing_context)))

        return issues


class ContextSchema(BaseModel):
    """
    Describes the structure of context data sources.

    Attributes:
        timestamp_col: Name of timestamp column (optional)
        interval_start_col: Start of time interval (optional)
        interval_end_col: End of time interval (optional)
        spatial_col: Name of spatial identifier column (e.g., grid_id, zone_id)
        geometry_col: Name of geometry column (optional)
        attribute_cols: List of attribute columns to join
    """

    timestamp_col: str | None = None
    interval_start_col: str | None = None
    interval_end_col: str | None = None
    spatial_col: str | None = None
    geometry_col: str | None = None
    attribute_cols: list[str] = Field(default_factory=list)

    def has_temporal(self) -> bool:
        """Check if schema has temporal dimension."""
        return self.timestamp_col is not None or (
            self.interval_start_col is not None and self.interval_end_col is not None
        )

    def has_spatial(self) -> bool:
        """Check if schema has spatial dimension."""
        return self.spatial_col is not None or self.geometry_col is not None


class EventMetadata(BaseModel):
    """
    Metadata about an event dataset.

    Attributes:
        dataset_name: Name of the dataset
        crs: Coordinate reference system (e.g., "EPSG:4326")
        time_zone: Time zone for timestamps (e.g., "America/Chicago")
        time_bin: Current time bin size if binned (e.g., "6h")
        grid_size_m: Grid cell size in meters if gridded
        bounds: Spatial bounds (minx, miny, maxx, maxy)
        date_range: Temporal range (start, end)
        custom: Additional custom metadata
    """

    dataset_name: str
    crs: str = "EPSG:4326"
    time_zone: str = "UTC"
    time_bin: str | None = None
    grid_size_m: float | None = None
    bounds: tuple[float, float, float, float] | None = None
    date_range: tuple[str, str] | None = None
    custom: dict[str, Any] = Field(default_factory=dict)
    output_modalities: set[str] = Field(default_factory=lambda: {"table"})
    feature_catalog: dict[str, Any] = Field(default_factory=dict)
    feature_provenance: dict[str, FeatureProvenance] = Field(default_factory=dict)
    context_requirements: ContextRequirementState = Field(default_factory=ContextRequirementState)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class DatasetConfig(BaseModel):
    """
    Configuration for loading a dataset.

    Attributes:
        dataset_name: Name of the dataset
        raw_root: Root path to raw data files
        layout: Directory layout ("flat" or "nested")
        crs: Target coordinate reference system
        time_zone: Time zone for timestamps
        quality: Optional data quality settings
    """

    dataset_name: str
    raw_root: str
    layout: str = "nested"
    crs: str = "EPSG:4326"
    time_zone: str = "UTC"
    quality: dict[str, Any] = Field(default_factory=dict)


# Step definitions accepted by ``FeatureConfig.steps`` — mirrors
# ``StepDefinition`` in ``shikong.core.registry`` (kept local to avoid a
# circular import): a step name, a ``(name, params)`` pair, or a mapping
# with ``name`` plus ``params``/``config``.
StepDefinition = str | tuple[str, Mapping[str, Any] | None] | Mapping[str, Any]

_FEATURE_KEYS = frozenset(
    {
        "steps",
        "count_features",
        "categorical_features",
        "categorical_top_n",
        "aggregation_windows",
        "complete_panel",
        "panel_min_periods",
        "ewma_halflife_days",
        "cyclical_encoding",
        "spatial_lag",
        "spatial_lag_rings",
        "kde_bandwidth_m",
        "stkde_bandwidth_m",
        "stkde_halflife_days",
        "near_repeat",
        "hawkes_halflife_days",
        "trend",
        "recency",
        "zero_features",
    }
)
_RECIPE_KEYS = frozenset({"dataset", "recipe", "grid", "temporal", "features", "context", "output"})
_GRID_KEYS = frozenset({"size_m", "origin"})
_TEMPORAL_KEYS = frozenset({"time_bin", "components"})

# Supported values for ``TemporalConfig.components`` — the single source of
# truth for the temporal-component vocabulary. ``ExtractTemporalComponentsStep``
# (``core/steps/temporal.py``) validates against the same set, so a typo is
# rejected identically at config parse time and at direct step construction.
# (The Literal in ``ExtractTemporalComponentsConfig`` lists the same names for
# registry-built steps.)
TEMPORAL_COMPONENTS = frozenset(
    {
        "hour_of_day",
        "day_of_week",
        "day_of_month",
        "day_of_year",
        "month",
        "year",
        "week_of_year",
        "quarter",
        "is_weekend",
    }
)


class GridConfig(BaseModel):
    """
    Spatial grid section of a recipe configuration.

    Phase 0: strict — unknown keys are rejected with the list of supported
    keys, so a typo such as ``grid.siz_m`` can no longer silently fall
    back to the default cell size.
    """

    model_config = ConfigDict(extra="forbid")

    size_m: float = 300.0
    origin: list[float] = Field(default_factory=lambda: [0.0, 0.0])

    @model_validator(mode="before")
    @classmethod
    def _reject_unknown_keys(cls, data: Any) -> Any:
        """Reject unknown keys with a message listing the supported keys."""
        if isinstance(data, dict):
            unknown = sorted(set(data) - _GRID_KEYS)
            if unknown:
                raise ValueError(
                    "Unknown grid key(s): " + ", ".join(unknown) + ". "
                    "Supported grid keys: " + ", ".join(sorted(_GRID_KEYS)) + "."
                )
        return data


class TemporalConfig(BaseModel):
    """
    Temporal section of a recipe configuration.

    Phase 0: strict, same rationale as ``GridConfig``.
    """

    model_config = ConfigDict(extra="forbid")

    time_bin: str = "6h"
    components: list[str] = Field(default_factory=lambda: ["hour_of_day", "day_of_week"])

    @field_validator("components")
    @classmethod
    def _reject_unknown_components(cls, components: list[str]) -> list[str]:
        """Reject unknown component values with the supported list."""
        unknown = sorted(set(components) - TEMPORAL_COMPONENTS)
        if unknown:
            raise ValueError(
                "Unknown temporal component(s): " + ", ".join(unknown) + ". "
                "Supported temporal components: " + ", ".join(sorted(TEMPORAL_COMPONENTS)) + "."
            )
        return components

    @model_validator(mode="before")
    @classmethod
    def _reject_unknown_keys(cls, data: Any) -> Any:
        """Reject unknown keys with a message listing the supported keys."""
        if isinstance(data, dict):
            unknown = sorted(set(data) - _TEMPORAL_KEYS)
            if unknown:
                raise ValueError(
                    "Unknown temporal key(s): " + ", ".join(unknown) + ". "
                    "Supported temporal keys: " + ", ".join(sorted(_TEMPORAL_KEYS)) + "."
                )
        return data


class FeatureConfig(BaseModel):
    """
    Feature-engineering section of a recipe configuration.

    Phase 0 (2026-11-15): only the keys below are accepted. Any other key
    is rejected with a message listing the supported keys — a typo or a
    leftover declaration must never be silently ignored again.

    Keys and their status in phase 1:

    - ``steps``: registry step definitions (``recipes/base.py``). When set,
      they replace the recipe's built-in pipeline.
    - ``count_features``: per-cell/bin counts. Implemented by the Chicago
      recipe (``ChicagoCountStep``); ``False`` is rejected there because
      the recipe always counts.
    - ``categorical_features``: per-category counts (e.g.
      ``count_primary_type_THEFT``), shares and Shannon entropy, via
      ``CategoryCountsStep``. Top-N guardrail: ``categorical_top_n``
      (default 10) plus an "other" bucket. Implemented in phase 1.
    - ``aggregation_windows``: rolling windows such as ``"1d"`` / ``"7d"`` /
      ``"30d"``. Each window drives a rolling mean (``count_ma_<days>``,
      trailing, closed-left) and a lag (``count_lag_<days>``, exact
      elapsed). Values must be whole days. Implemented in phase 1.
    - ``complete_panel``: complete the sparse unit x time panel with
      explicit zeros inside the observation support before any window or
      lag (``CompletePanelStep``). Opt-in; ``False`` by default.
    - ``panel_min_periods``: a unit enters the completed panel only if
      observed in at least this many distinct bins (default 1).
    - ``ewma_halflife_days``: exponentially weighted mean of ``count``
      (``count_ewma_<halflife>d``). ``None`` (default) disables it.
    - ``cyclical_encoding``: sin/cos calendar encodings
      (``CalendarEncodingStep``, cyclical) for hour, day-of-week, month.
    - ``spatial_lag``: multi-scale queen-ring spatial lag of the
      shortest-window lagged count
      (``count_lag_<shortest>_spatial_lag_r<r>`` for each ring in
      ``spatial_lag_rings``). Requires ``aggregation_windows``.
    - ``spatial_lag_rings``: queen rings for ``spatial_lag`` (default
      ``[1, 2, 3]``); ring 1 reproduces the phase-1 single-scale lag.
      Non-empty, unique, positive integers. Implemented in phase 2.
    - ``kde_bandwidth_m``: Gaussian KDE bandwidth in metres, evaluated on
      the shortest-window lagged count (``count_lag_<shortest>_kde``).
      ``None`` (default) disables it. Requires ``aggregation_windows``.
      Implemented in phase 2.
    - ``stkde_bandwidth_m`` / ``stkde_halflife_days``: separable
      space-time KDE (Gaussian in space, exponential decay in time) of
      ``count`` (``count_stkde``). Both must be set together. Requires
      ``complete_panel``. Implemented in phase 2.
    - ``near_repeat``: near-repeat ring counts over trailing 7 d and
      7-14 d windows (``count_nr_r<r>_7d`` / ``count_nr_r<r>_7_14d``) plus
      the ``count_nr_event_7d`` flag. Requires ``complete_panel``.
      Implemented in phase 2.
    - ``hawkes_halflife_days``: fixed-kernel Hawkes excitation of ``count``
      (``count_hawkes_<halflife>d``), strictly past bins only.
      ``None`` (default) disables it. Requires ``complete_panel``.
      Implemented in phase 2.
    - ``trend``: ratio of the shortest-window rolling mean to the
      longest-window one (``count_ma_<short>_vs_count_ma_<long>``).
      Requires at least two ``aggregation_windows``.
    - ``recency``: time since the last bin with ``count > 0`` and
      burstiness over a trailing 28 d window. Requires ``complete_panel``.
    - ``zero_features``: activity flag over trailing 7 d and share of
      non-zero bins over trailing 28 d. Requires ``complete_panel``.

    Cross-key rules are enforced by ``require_implemented`` (called by
    recipes at pipeline-build time) so an incoherent combination fails
    loudly instead of producing a silently wrong feature set.
    """

    model_config = ConfigDict(extra="forbid")

    steps: list[StepDefinition] | None = None
    count_features: bool = True
    categorical_features: list[str] = Field(default_factory=list)
    categorical_top_n: int = Field(default=10, ge=1)
    aggregation_windows: list[str] = Field(default_factory=list)
    complete_panel: bool = False
    panel_min_periods: int = Field(default=1, ge=1)
    ewma_halflife_days: float | None = Field(default=None, gt=0)
    cyclical_encoding: bool = False
    spatial_lag: bool = False
    spatial_lag_rings: list[int] = Field(default_factory=lambda: [1, 2, 3])
    kde_bandwidth_m: float | None = Field(default=None, gt=0)
    stkde_bandwidth_m: float | None = Field(default=None, gt=0)
    stkde_halflife_days: float | None = Field(default=None, gt=0)
    near_repeat: bool = False
    hawkes_halflife_days: float | None = Field(default=None, gt=0)
    trend: bool = False
    recency: bool = False
    zero_features: bool = False

    @model_validator(mode="before")
    @classmethod
    def _reject_unknown_keys(cls, data: Any) -> Any:
        """Reject unknown keys with a message listing the supported keys."""
        if isinstance(data, dict):
            unknown = sorted(set(data) - _FEATURE_KEYS)
            if unknown:
                raise ValueError(
                    "Unknown features key(s): " + ", ".join(unknown) + ". "
                    "Supported features keys: " + ", ".join(sorted(_FEATURE_KEYS)) + "."
                )
        return data

    @field_validator("aggregation_windows")
    @classmethod
    def _validate_windows_are_whole_days(cls, windows: list[str]) -> list[str]:
        """Aggregation windows must be whole days (lags/rolling use days)."""
        for window in windows:
            try:
                seconds = duration_seconds(window)
            except ValueError as exc:
                raise ValueError(
                    f"Unknown aggregation window {window!r}: durations such as "
                    "'1d' / '7d' / '30d' are supported"
                ) from exc
            if seconds % 86400 != 0:
                raise ValueError(f"Aggregation window {window!r} must be a whole number of days")
        return windows

    @field_validator("spatial_lag_rings", mode="before")
    @classmethod
    def _validate_rings(cls, rings: object) -> object:
        """Spatial lag rings must be non-empty, unique, positive integers.

        Runs before coercion so a boolean (``True``) is rejected instead
        of being silently coerced to ring 1.
        """
        if not isinstance(rings, list) or not rings:
            raise ValueError("spatial_lag_rings must not be empty")
        if any(not isinstance(r, int) or isinstance(r, bool) or r < 1 for r in rings):
            raise ValueError("spatial_lag_rings must contain positive integers only")
        if len(set(rings)) != len(rings):
            raise ValueError("spatial_lag_rings must not contain duplicates")
        return rings

    def window_days(self) -> list[int]:
        """Aggregation windows as whole days, ascending, de-duplicated."""
        return sorted({int(duration_seconds(w) // 86400) for w in self.aggregation_windows})

    def require_implemented(self, recipe_name: str) -> None:
        """
        Raise a blocking error for incoherent feature combinations.

        Recipes call this at pipeline-build time so a declared feature is
        never silently dropped or silently miscomputed. Phase 1 implements
        every key in this section; what is still rejected here is a
        combination whose inputs do not exist.
        """
        if self.spatial_lag and not self.aggregation_windows:
            raise ValueError(
                f"{recipe_name}: spatial_lag needs aggregation_windows "
                "(the spatial lag is computed on the lagged count)"
            )
        if self.kde_bandwidth_m is not None and not self.aggregation_windows:
            raise ValueError(
                f"{recipe_name}: kde_bandwidth_m needs aggregation_windows "
                "(the KDE is evaluated on the shortest-window lagged count)"
            )
        if (self.stkde_bandwidth_m is None) != (self.stkde_halflife_days is None):
            raise ValueError(
                f"{recipe_name}: stkde_bandwidth_m and stkde_halflife_days " "must be set together"
            )
        if self.trend and len(self.aggregation_windows) < 2:
            raise ValueError(
                f"{recipe_name}: trend needs at least two aggregation_windows "
                "(ratio of the shortest-window mean to the longest-window mean)"
            )
        needs_panel = [
            name
            for name, enabled in (
                ("recency", self.recency),
                ("zero_features", self.zero_features),
                ("ewma_halflife_days", self.ewma_halflife_days is not None),
            )
            if enabled
        ]
        if self.near_repeat:
            needs_panel.append("near_repeat")
        if self.hawkes_halflife_days is not None:
            needs_panel.append("hawkes_halflife_days")
        if self.stkde_bandwidth_m is not None:
            needs_panel.append("stkde")
        if needs_panel and not self.complete_panel:
            raise ValueError(
                f"{recipe_name}: {', '.join(needs_panel)} cannot be used "
                "without complete_panel (trailing statistics over a sparse "
                "panel would be misleading)"
            )


class RecipeConfig(BaseModel):
    """
    Configuration for a recipe.

    Attributes:
        dataset: Dataset name
        recipe: Recipe name
        grid: Spatial grid configuration
        temporal: Temporal configuration
        features: Feature engineering configuration
        context: Context enrichment configuration
        output: Output format configuration
    """

    model_config = ConfigDict(extra="forbid")

    dataset: str
    recipe: str
    grid: GridConfig = Field(default_factory=GridConfig)
    temporal: TemporalConfig = Field(default_factory=TemporalConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    context: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _reject_unknown_keys(cls, data: Any) -> Any:
        """Reject unknown top-level keys with the list of supported keys."""
        if isinstance(data, dict):
            unknown = sorted(set(data) - _RECIPE_KEYS)
            if unknown:
                raise ValueError(
                    "Unknown recipe key(s): " + ", ".join(unknown) + ". "
                    "Supported recipe keys: " + ", ".join(sorted(_RECIPE_KEYS)) + "."
                )
        return data
