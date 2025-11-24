# Eventflow API Reference

This document provides detailed API documentation for the Eventflow library.

## Core Module

### eventflow.core.schema

#### EventSchema

Describes the structure of event data.

```python
class EventSchema(BaseModel):
    timestamp_col: str
    lat_col: str | None = None
    lon_col: str | None = None
    geometry_col: str | None = None
    categorical_cols: list[str] = []
    numeric_cols: list[str] = []
```

#### ContextSchema

Describes the structure of context data sources.

```python
class ContextSchema(BaseModel):
    timestamp_col: str | None = None
    spatial_col: str | None = None
    attribute_cols: list[str]
```

### eventflow.core.event_frame

#### EventFrame

Central abstraction for event data.

```python
class EventFrame:
    def __init__(
        self,
        lazy_frame: pl.LazyFrame,
        schema: EventSchema,
        metadata: EventMetadata
    ):
        ...
    
    def with_spatial_grid(self, size_m: float) -> EventFrame:
        """Add spatial grid cells to events."""
        ...
    
    def with_time_bins(self, bin_size: str) -> EventFrame:
        """Add time bins to events."""
        ...
    
    def collect(self) -> pl.DataFrame:
        """Materialize the lazy frame."""
        ...
```

### eventflow.core.spatial

Functions for spatial operations:

- `create_grid(bounds, size_m, crs)`: Generate spatial grid
- `assign_to_grid(event_frame, grid)`: Assign events to grid cells
- `transform_crs(event_frame, target_crs)`: Transform coordinate system

### eventflow.core.temporal

Functions for temporal operations:

- `extract_temporal_components(event_frame, components)`: Extract hour, day, etc.
- `create_time_bins(event_frame, bin_size)`: Create time bins
- `align_temporal(events, context, strategy)`: Align events with context data

### eventflow.core.features

Functions for feature engineering:

- `aggregate_counts(event_frame, group_by)`: Count aggregations
- `moving_window(event_frame, window, agg_fn)`: Moving window features
- `encode_categorical(event_frame, col, method)`: Categorical encoding

## Dataset Module

### eventflow.datasets.chicago_crime

Chicago crime dataset adapter.

#### load_chicago_crime

```python
def load_chicago_crime(
    raw_path: str,
    config: DatasetConfig | None = None
) -> EventFrame:
    """Load Chicago crime data as EventFrame."""
    ...
```

## Recipe Module

### eventflow.recipes.base

#### BaseRecipe

```python
class BaseRecipe(ABC):
    @abstractmethod
    def build_pipeline(self) -> Pipeline:
        """Build the transformation pipeline."""
        ...
    
    def run(self, event_frame: EventFrame) -> EventFrame:
        """Run the recipe on event data."""
        ...
```

### eventflow.recipes.registry

Recipe registration and discovery:

- `register_recipe(dataset, name, recipe)`: Register a recipe
- `get_recipe(dataset, name)`: Get registered recipe
- `list_recipes(dataset)`: List available recipes

## Tracking Module

### eventflow.tracking.protocol

#### TrackerProtocol

```python
class TrackerProtocol(Protocol):
    def log_param(self, key: str, value: Any) -> None: ...
    def log_metric(self, key: str, value: float, step: int | None = None) -> None: ...
    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None: ...
    def set_tags(self, tags: dict[str, str]) -> None: ...
```

### eventflow.tracking.mlflow_tracker

MLflow implementation of TrackerProtocol.

## CLI Module

### eventflow.cli.main

Command-line interface:

```bash
eventflow run --dataset DATASET --recipe RECIPE --config CONFIG
eventflow list-datasets
eventflow list-recipes --dataset DATASET
eventflow validate --config CONFIG
```

## Examples

### Basic Usage

```python
from eventflow.datasets.chicago_crime import load_chicago_crime
from eventflow.core.pipeline import Pipeline
from eventflow.core.spatial import SpatialGridStep
from eventflow.core.temporal import TemporalBinStep

# Load data
ef = load_chicago_crime("data/raw/chicago_crime")

# Build pipeline
pipeline = Pipeline([
    SpatialGridStep(size_m=300),
    TemporalBinStep(bin_size="6h"),
])

# Run pipeline
result = pipeline.run(ef)
df = result.collect()
```

### With Context Enrichment

```python
from eventflow.datasets.chicago_crime.context import ChicagoWeatherSource
from eventflow.core.context import EnricherStep, TemporalJoin

# Create context source
weather = ChicagoWeatherSource("data/weather")

# Add to pipeline
pipeline = Pipeline([
    SpatialGridStep(size_m=300),
    TemporalBinStep(bin_size="6h"),
    EnricherStep(weather, TemporalJoin(strategy="nearest")),
])
```

## See Also

- [Architecture Documentation](architecture.md)
- [Dataset Documentation](datasets.md)
- [GitHub Repository](https://github.com/eventflow/eventflow)
