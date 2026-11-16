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
- `chicago_crime_v1`: Baseline spatial-temporal features — 300 m cells,
  6-hour bins, completed panel with explicit zeros, per-category counts,
  lags, rolling means, EWMA, cyclical encodings, multi-scale spatial lag,
  KDE, STKDE, near-repeat, Hawkes, trend, recency and zero features
  (see `docs/api_reference.md` and `configs/recipes/chicago_crime_v1.yaml`).
  Replay on real Q1 2024 data: 1,908,088 rows × 90 columns,
  sum `count` = 60,235.

**Context Sources**:
- Weather: NOAA hourly weather data
- Special events: Chicago special events calendar
- Demographics: Census tract demographics

> These context sources exist but are unwired and untested
> (experimental); enabling any `context.<source>.enabled` in
> `chicago_crime_v1` raises a blocking error. For new datasets, use the
> generic `TabularFileSource` connector instead (see
> `docs/api_reference.md`).

### Beijing Multi-Site Air Quality Data

**Source**: Chen, S. (2017). Beijing Multi-Site Air Quality [Dataset].
UCI Machine Learning Repository. DOI 10.24432/C5RK5G.

**Description**: Hourly air-quality measurements from 12 monitoring
stations, 2013-03-01 → 2017-02-28 (35,064 hours each); 6 pollutants
(PM2.5, PM10, SO2, NO2, CO, O3) + 6 meteorological variables
(TEMP, PRES, DEWP, RAIN, wd, WSPM). Missing values are the literal
string `NA`. Total 420,768 rows.

**Loader**: `load_beijing_air_quality(raw_path, config=None,
apply_cleaning=True)` in `src/eventflow/datasets/beijing_air_quality/mapping.py`
— accepts a directory of per-station CSVs, one CSV, or one Parquet file.

**Available Recipes**:
- `beijing_air_quality_v1`: Station-hour panel — unit = station, bin =
  hour, only temporal steps (lags 1 h / 24 h / 168 h, rolling means
  24 h / 168 h, calendar encoding). `CompletePanelStep` runs with
  `measure_cols` (measurements are never zero-filled). Replay: 420,768
  rows × 64 columns, count sum 420,768, NA PM2.5 8,739 → 8,739.
  Honest limit: the shipped panel is already complete, so the proof
  covers gaps / missing-stays-missing and per-station separation, not
  irregular sampling — see `docs/design-notes/phase3-beijing.md`.

### GeoLife GPS Trajectories

**Source**: Microsoft Research GeoLife Trajectories 1.3.

**Description**: GPS fixes from 182 users (2007–2012). One fix is one
event; entity = user (`user_id`), a trajectory (`trajectory_id`, the
`.plt` file stem) groups the fixes of one continuous log. Coordinates
are WGS 84, projected to UTM 50N (EPSG:32650) for distance work.

**Loader**: `load_geolife(paths, *, user_ids=None, metadata=None)` in
`src/eventflow/datasets/geolife/loader.py` — loads `.plt` files, one
trajectory each.

**Available Recipes**:
- `geolife_v1`: CRS transform → trajectory kinematics (distance,
  elapsed time, speed; speed withheld across gaps > 300 s and on
  `dt <= 0`) → temporal components → cyclical calendar →
  inter-arrival → rows-mode lag / rolling mean of speed → 5-min bins →
  100 m grid → fix counts per (cell, bin). Replay on the documented
  subset (users `010` and `000`, 1,471 fixes): point-level output
  1,471 rows × 26 columns; panel 112 rows × 8 columns, count sum
  1,471. The panel-family steps were evaluated on the subset and do
  **not** apply as-is — see the verdict table in
  `docs/design-notes/phase3-geolife-boundary.md`. Timezone is an
  assumption (`.plt` times labelled `Asia/Shanghai`, no conversion).

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
from eventflow.core.steps.temporal import TimeBinsStep, TemporalLagStep

class YourDatasetV1Recipe(BaseRecipe):
    def build_pipeline(self) -> Pipeline:
        return Pipeline([
            TimeBinsStep(bin_size=self.config.temporal.time_bin),
            TemporalLagStep(
                value_cols=["count"], lag_periods=[1], time_unit="1d",
            ),
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
