"""Schema definitions for events and context sources."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


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
    """

    timestamp_col: str
    lat_col: str | None = None
    lon_col: str | None = None
    geometry_col: str | None = None
    categorical_cols: list[str] = Field(default_factory=list)
    numeric_cols: list[str] = Field(default_factory=list)

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
