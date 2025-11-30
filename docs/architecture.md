# Eventflow Architecture

## Overview

Eventflow is designed as a layered architecture that separates generic event transformation logic from dataset-specific implementations. This design enables high performance, reusability, and extensibility.

## Architecture Layers

### 1. Core Layer (`eventflow.core`)

The core layer contains all dataset-agnostic abstractions and primitives:

- **Schema definitions** (`schema.py`): Event and context schemas using Pydantic
- **Event frames** (`event_frame.py`): Central abstraction wrapping Polars LazyFrame
- **Spatial operations** (`spatial.py`): Grid construction, zone assignment, projections
- **Temporal operations** (`temporal.py`): Time binning, component extraction, alignment
- **Feature engineering** (`features.py`): Generic aggregations and encodings
- **Context enrichment** (`context/`): Abstract sources and join strategies
- **Pipeline orchestration** (`pipeline.py`): Composable step-based execution

### 2. Dataset Layer (`eventflow.datasets`)

Dataset-specific adapters that map real-world data to core abstractions:

- **Schema definitions**: Dataset-specific EventSchema instances
- **Data loaders**: Functions to load raw data into EventFrames
- **Context sources**: Dataset-specific implementations (weather, events, demographics)
- **Recipes**: Pre-built pipelines for common use cases

### 3. Recipe Layer (`eventflow.recipes`)

Generic recipe mechanism for composable feature engineering:

- **Base recipe interface**: Common API for all recipes
- **Recipe registry**: Discovery and instantiation of recipes
- **Configuration**: YAML-based recipe configuration

### 4. Tracking Layer (`eventflow.tracking`)

Optional experiment tracking integration:

- **Protocol definition**: Abstract interface for any tracking backend
- **MLflow implementation**: Concrete implementation for MLflow
- **Extensibility**: Easy to add other backends (Weights & Biases, etc.)

## Design Principles

### Lazy Evaluation

Eventflow uses Polars LazyFrame as the primary data structure, deferring computation until `.collect()` is called. This enables:

- Query optimization
- Reduced memory footprint
- Efficient pipelining

### Generic Core

The core layer never contains dataset-specific logic. All operations are parameterized through schemas and configurations.

### Type Safety

Extensive use of Pydantic models for configuration and schemas ensures:

- Runtime validation
- Clear documentation
- IDE support

### Composability

Operations are designed as pure functions or stateless steps that can be composed into pipelines.

## Key Components

### EventFrame

Central abstraction that wraps:
- A Polars LazyFrame (the data)
- An EventSchema (structure definition)
- EventMetadata (CRS, time zone, dataset info)

### Pipeline

A sequence of Steps that transform EventFrames:

```python
pipeline = Pipeline([
    SpatialGridStep(size_m=300),
    TemporalBinStep(bin_size="6h"),
    FeatureAggregationStep(),
    ContextEnricherStep(weather_source, temporal_join),
])
result = pipeline.run(event_frame)
```

For production workflows, steps can be registered centrally and resolved at runtime. The
`eventflow.core.registry.StepRegistry` keeps track of reusable step classes, supports
discoverability via tags, and can materialize a `Pipeline` from configuration-driven step
definitions (for example, `config.features["steps"]` in a recipe). This enables a clean
separation between step implementation and recipe assembly.

### Context Enrichment

Multi-stage process:
1. Define a context source (weather, demographics, etc.)
2. Choose a join strategy (temporal, spatial, or both)
3. Apply enrichment through EnricherStep

## Performance Considerations

- **Streaming**: LazyFrame evaluation enables streaming for large datasets
- **Projection pushdown**: Only load required columns
- **Predicate pushdown**: Filter data early in the pipeline
- **Parallel execution**: Polars handles parallelism automatically

## Extension Points

1. **New datasets**: Implement in `eventflow.datasets.<dataset_name>`
2. **New context sources**: Subclass `BaseContextSource`
3. **New features**: Add functions to `features.py`
4. **New recipes**: Register in recipe registry
5. **New tracking backends**: Implement `TrackerProtocol`

## Future Enhancements

- Distributed execution (Dask, Ray)
- GPU acceleration for spatial operations
- Streaming ingestion from Kafka/Pulsar
- Model training integration
- Web UI for recipe management
