"""Schema definitions for events and context sources."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


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

    dataset: str
    recipe: str
    grid: dict[str, Any] = Field(default_factory=dict)
    temporal: dict[str, Any] = Field(default_factory=dict)
    features: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
