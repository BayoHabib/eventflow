# Dataset Documentation

This document describes the datasets supported by Eventflow and how to add new datasets.

## Supported Datasets

### Chicago Crime Data

**Source**: [City of Chicago Data Portal](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2)

**Description**: Crime incidents reported in Chicago from 2001 to present.

**Schema**:
- `date`: Timestamp of the crime
- `latitude`, `longitude`: Location coordinates
- `primary_type`: Type of crime
- `description`: Detailed description
- `location_description`: Where the crime occurred
- `arrest`: Whether an arrest was made
- `domestic`: Whether it was a domestic incident

**Configuration Example**:

```yaml
dataset_name: chicago_crime
raw_root: "data/raw/chicago_crime"
crs: "EPSG:26971"
time_zone: "America/Chicago"
```

**Available Recipes**:
- `chicago_crime_v1`: Baseline spatial-temporal features
- `chicago_crime_with_weather`: Includes weather context
- `chicago_crime_full`: All context sources

**Context Sources**:
- Weather: NOAA hourly weather data
- Special events: Chicago special events calendar
- Demographics: Census tract demographics

## Adding a New Dataset

To add a new dataset to Eventflow, follow these steps:

### 1. Create Dataset Package

Create a new package under `src/eventflow/datasets/`:

```
src/eventflow/datasets/your_dataset/
├── __init__.py
├── schema.py
├── mapping.py
├── context/
│   ├── __init__.py
│   └── your_context_source.py
└── recipes/
    ├── __init__.py
    └── your_recipe_v1.py
```

### 2. Define Schema

In `schema.py`, define the dataset schema:

```python
from eventflow.core.schema import EventSchema

YOUR_DATASET_SCHEMA = EventSchema(
    timestamp_col="timestamp",
    lat_col="latitude",
    lon_col="longitude",
    categorical_cols=["category1", "category2"],
    numeric_cols=["value1", "value2"],
)
```

### 3. Implement Data Loader

In `mapping.py`, implement the loader:

```python
import polars as pl
from eventflow.core.event_frame import EventFrame
from .schema import YOUR_DATASET_SCHEMA

def load_your_dataset(raw_path: str) -> EventFrame:
    """Load your dataset as EventFrame."""
    lf = pl.scan_parquet(f"{raw_path}/**/*.parquet")
    
    # Apply any necessary transformations
    lf = lf.with_columns([
        pl.col("timestamp").str.to_datetime(),
        # ... other transformations
    ])
    
    metadata = EventMetadata(
        dataset_name="your_dataset",
        crs="EPSG:4326",
        time_zone="UTC",
    )
    
    return EventFrame(lf, YOUR_DATASET_SCHEMA, metadata)
```

### 4. Create Context Sources (Optional)

In `context/`, implement context sources:

```python
from eventflow.core.context.sources import BaseContextSource
import polars as pl

class YourContextSource(BaseContextSource):
    def __init__(self, data_path: str):
        self.data_path = data_path
    
    def load(self) -> pl.LazyFrame:
        return pl.scan_parquet(self.data_path)
    
    @property
    def schema(self) -> ContextSchema:
        return ContextSchema(
            timestamp_col="timestamp",
            attribute_cols=["attr1", "attr2"],
        )
```

### 5. Create Recipes

In `recipes/`, create pre-built pipelines:

```python
from eventflow.recipes.base import BaseRecipe
from eventflow.core.pipeline import Pipeline

class YourDatasetV1Recipe(BaseRecipe):
    def __init__(self, config: RecipeConfig):
        self.config = config
    
    def build_pipeline(self) -> Pipeline:
        return Pipeline([
            SpatialGridStep(size_m=self.config.grid.size_m),
            TemporalBinStep(bin_size=self.config.temporal.time_bin),
            FeatureAggregationStep(),
        ])
```

### 6. Register Dataset and Recipes

In `__init__.py`:

```python
from eventflow.recipes.registry import register_recipe
from .recipes.your_recipe_v1 import YourDatasetV1Recipe

register_recipe("your_dataset", "v1", YourDatasetV1Recipe)
```

### 7. Add Configuration

Create `configs/datasets/your_dataset_example.yaml`:

```yaml
dataset_name: your_dataset
raw_root: "data/raw/your_dataset"
crs: "EPSG:4326"
time_zone: "UTC"
```

### 8. Add Tests

Create tests in `tests/datasets/your_dataset/`:

```python
def test_load_your_dataset():
    ef = load_your_dataset("path/to/test/data")
    assert ef.schema.timestamp_col == "timestamp"
    df = ef.collect()
    assert len(df) > 0
```

### 9. Document

Add dataset documentation to this file with:
- Data source and license
- Schema description
- Available recipes
- Example usage

## Dataset Guidelines

When adding a dataset:

1. **Keep raw data separate**: Never commit raw data to the repository
2. **Use lazy evaluation**: Always use `pl.scan_*` instead of `pl.read_*`
3. **Validate schema**: Use Pandera in tests to validate data structure
4. **Document context**: Clearly document all context sources and their schemas
5. **Provide examples**: Include example configurations and recipes
6. **Test thoroughly**: Write unit and integration tests
7. **Follow naming conventions**: Use snake_case for dataset names

## Common Patterns

### Loading Multiple Files

```python
lf = pl.scan_parquet(f"{raw_path}/**/*.parquet")
```

### Date Parsing

```python
lf = lf.with_columns([
    pl.col("date_str").str.to_datetime("%Y-%m-%d %H:%M:%S").alias("timestamp")
])
```

### Coordinate Validation

```python
lf = lf.filter(
    (pl.col("latitude").is_not_null()) &
    (pl.col("longitude").is_not_null()) &
    (pl.col("latitude").is_between(-90, 90)) &
    (pl.col("longitude").is_between(-180, 180))
)
```

## Need Help?

- Check existing dataset implementations in `src/eventflow/datasets/`
- See the [API Reference](api_reference.md) for core abstractions
- Open an issue on GitHub for questions
