# EventFlow

[![Test & Lint](https://github.com/BayoHabib/eventflow/actions/workflows/ci.yml/badge.svg)](https://github.com/BayoHabib/eventflow/actions)
[![Coverage](https://img.shields.io/badge/coverage-86%25-yellow.svg)](https://github.com/BayoHabib/eventflow)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-0.1.0-green.svg)](https://github.com/BayoHabib/eventflow/releases)

A generic library for preparing and transforming spatio-temporal datasets before analysis.

## Overview

Numerical definitions, migration notes and remaining foundations are described in
[preprocessing contracts](docs/preprocessing_contracts.md).
The [real-data audit fixes and migration guide](docs/real_data_audit_fixes.md)
cover shared grid identity, recipe support and window-normalized K/L/H.

[Observation panels](docs/observation_panel.md) count point-event records on
explicit space-time support while keeping observed zeros, missing totals and
outside-domain rows distinct, with optional exposure, rates and log offsets.
[Independent quality declarations and cadence checks](docs/observation_quality.md)
keep collection activity, batch completeness and scheduled gaps separate.

Pipelines support [asynchronous execution](docs/async_execution.md):

```python
prepared = await pipeline.run_async(events)
table = await prepared.collect_async()
```

Existing steps are offloaded from the asyncio event loop, and `AsyncStep` supports
native async operations. Ordered execution, error propagation and cancellation
cleanup are documented alongside the synchronous API.

The core supplies reusable spatial and temporal treatments. Dataset adapters and
recipes assemble them for a particular source; interpretation, statistical analysis,
and model training consume the prepared outputs downstream.

Configurable [spatial-domain preprocessing](docs/spatial_domains.md) supports
caller-supplied footprints and exclusions, optional event filtering, and geometric
cell coverage with provenance. It works with arbitrary EventFrame schemas and
introduces no default geographical or application-specific exclusions.

EventFlow transforms raw event data (crime records, sensor readings, transactions, etc.) into **ML-ready formats** for various model architectures:

| Adapter | Output Format | Target Models |
|---------|---------------|---------------|
| **TableAdapter** | `(N, F)` DataFrame | GLM, XGBoost, Poisson regression |
| **SequenceAdapter** | `(B, T, F)` tensors | LSTM, Transformer, temporal models |
| **RasterAdapter** | `(T, C, H, W)` arrays | CNN, ConvLSTM, U-Net |
| **GraphAdapter** | Nodes + edges | GNN, GAT, GCN |
| **StreamAdapter** | Continuous timestamps | Neural ODE, Hawkes processes |

## Key Features

- 🚀 **Lazy evaluation** with Polars for efficient processing of large datasets
- 🔄 **Multi-modal output** - same data, multiple ML-ready formats
- 🗺️ **Spatial operations** - grid binning, zone assignment, adjacency graphs
- ⏰ **Temporal features** - time bins, rolling windows, inter-event times
- 📊 **Point process features** - intensity estimation, kernel smoothing
- 🧪 **Composable pipelines** - registry-based step system
- ✅ **Validated** - 1,046 tests passing, 86% coverage (measured 2026-11-16; see `docs/audit/`)

## What EventFlow does not do

- **No model training and no prediction serving.** EventFlow prepares
  ML-ready features; models live downstream (e.g. `chicago_crime_api`).
- **Per-category and rolling-window features are opt-in, not default.**
  `CategoryCountsStep`, `TemporalLagStep`, `MovingAverageStep` and the
  phase-1/2 families are built by the recipe only when the corresponding
  `features.*` keys are set in the recipe YAML (see
  `docs/api_reference.md`); declaring them raises a blocking error if a
  required companion key is missing, instead of being silently ignored
  (see `docs/design-notes/chicago-declared-features.md`).
- **No partitioned output.** `output.partition_by` is not implemented; the
  CLI writes a single file.
- **No context enrichment in the Chicago recipe.** Weather, events and
  demographics sources exist but are unwired and untested (experimental);
  enabling any `context.<source>.enabled` raises a blocking error. For
  new datasets, use the generic `TabularFileSource` connector instead
  (see `docs/api_reference.md`).
- **No FastAPI service, PyPI release or Docker image yet.** These are the
  open high-priority TODO items; see `TODO.md`.

## Installation

```bash
# From source (PyPI coming soon)
git clone https://github.com/BayoHabib/eventflow.git
cd eventflow
pip install -e ".[dev]"
```

## Chicago downloader to features

Run the packaged Chicago recipe directly on downloaded CSV or Parquet chunks:

```bash
pip install "chicago-crime-downloader[parquet]"
chicago-crime-dl --mode daily --start-date 2020-01-10 --end-date 2020-01-10 \
  --out-root data/raw/chicago_crime --out-format parquet
eventflow run --dataset chicago_crime --recipe chicago_crime_v1 \
  --config configs/recipes/chicago_crime_v1.yaml \
  --input data/raw/chicago_crime --output data/processed/chicago_features.parquet
```

`--input` accepts a file or a directory of one format (CSV or Parquet). Alternatively,
pass `--dataset-config configs/datasets/chicago_crime_example.yaml`; `--input` overrides
its `raw_root` when both are supplied. The command writes one CSV or Parquet file.

The baseline normalizes ISO/legacy timestamps, numeric coordinate strings, and booleans,
and removes records without valid timestamps or geographic coordinates. It projects
Socrata longitude/latitude from EPSG:4326 to EPSG:26971 before assigning meter-based cells.
`DatasetConfig.crs` is a target CRS and now performs a coordinate transformation.

The bundled recipe keeps **300-meter cells and six-hour bins**. Its fixed grid origin
defaults to `(0, 0)` in projected meters, so cell identifiers remain stable across batches.
Set `grid.origin: [x, y]` to choose another fixed origin. Output contains `grid_id`,
`time_bin`, `count`, cell-center coordinates, bin-start `date`, and configured calendar
components. The shipped YAML completes the panel with explicit zeros
(`complete_panel: true`: 5,242 cells × 364 bins = 1,908,088 rows, sum `count` =
60,235) and builds the phase-1 and phase-2 feature families — per-category
counts, lags, rolling means, EWMA, cyclical encodings, multi-scale spatial
lag, KDE, STKDE, near-repeat, Hawkes, trend, recency and zero features
(see `docs/api_reference.md`).

For custom spatial pipelines, both CRS transformations update the active coordinate
columns in `EventSchema`. Meter-based grid operations reject geographic and foot-based
CRSs. Call `TransformCRSStep` first and pass a metric `crs` explicitly to `create_grid`.

## Quick Start

Every snippet below is executed by `tests/test_readme_examples.py`, so it
cannot drift from the code again. The examples share one frame: `df` must
contain the columns named in each snippet. In particular, the raster and
graph adapters work from a `grid_id` column of `"row_col"` strings (e.g.
`"0_0"`), not from `lat_bin`/`lon_bin` alone.

```python
import polars as pl
from eventflow.core.adapters import (
    TableAdapter, TableAdapterConfig,
    SequenceAdapter, SequenceAdapterConfig,
    RasterAdapter, RasterAdapterConfig,
    GraphAdapter, GraphAdapterConfig,
    StreamAdapter, StreamAdapterConfig,
)

# Load your event data
df = pl.read_csv("crime_data.csv")

# 1. TableAdapter - For regression models
# Requires df to contain event_count, lat_bin, lon_bin, day_of_week.
table_config = TableAdapterConfig(
    target_col="event_count",
    feature_cols=["lat_bin", "lon_bin", "day_of_week"],
)
table_adapter = TableAdapter(table_config)
table_output = table_adapter.convert(df)
X, y = table_output.get_X_y()  # Ready for sklearn!

# 2. SequenceAdapter - For LSTM/Transformer
# Requires a spatial id column (here cell_id) and a timestamp column.
seq_config = SequenceAdapterConfig(
    spatial_col="cell_id",
    timestamp_col="date",
    feature_cols=["event_count"],
    sequence_length=30,
)
seq_adapter = SequenceAdapter(seq_config)
seq_output = seq_adapter.convert(df)
# seq_output.sequences: (N_cells, 30, 1) tensor; seq_output.masks: (N_cells, 30)

# 3. RasterAdapter - For CNN models
# Requires a grid_id column ("row_col" strings) and a timestamp column.
# The first dimension is the number of distinct timestamps in the input,
# so (91, 2, 10, 10) needs 91 distinct dates; with 3 dates it is (3, 2, 10, 10).
raster_config = RasterAdapterConfig(
    grid_col="grid_id",
    timestamp_col="date",
    feature_cols=["event_count", "arrest_count"],
    grid_shape=(10, 10),
)
raster_adapter = RasterAdapter(raster_config)
raster_output = raster_adapter.convert(df)
# raster_output.raster: (T, 2, 10, 10) - T timesteps, 2 channels

# 4. GraphAdapter - For GNN models
# adjacency_type is "spatial", "temporal" or "both" (there is no "grid"
# mode). With "row_col" node ids and no latitude/longitude columns, edges
# fall back to 4-connectivity grid adjacency.
graph_config = GraphAdapterConfig(
    node_col="grid_id",
    feature_cols=["total_events", "centroid_lat", "centroid_lon"],
    adjacency_type="spatial",
)
graph_adapter = GraphAdapter(graph_config)
graph_output = graph_adapter.convert(df)
# graph_output.edge_index: (2, E) for PyTorch Geometric

# 5. StreamAdapter - For Neural ODEs
stream_config = StreamAdapterConfig(
    timestamp_col="timestamp",
    event_type_col="crime_type",
    state_cols=["latitude", "longitude"],
)
stream_adapter = StreamAdapter(stream_config)
stream_output = stream_adapter.convert(df)
# stream_output.timestamps, stream_output.inter_times
```

## Pipeline Steps

EventFlow provides composable pipeline steps for feature engineering
(every name below is importable from `eventflow.core.steps` — the list is
checked by `tests/test_readme_examples.py`):

```python
from eventflow.core.steps import (
    # Temporal
    TimeBinsStep, ExtractTemporalComponentsStep, CalendarEncodingStep,
    TemporalLagStep, MovingAverageStep, FillNullStep,
    RecencyWeightStep, InterArrivalStep,

    # Panel completion, categorical counts, dynamics
    CompletePanelStep, CategoryCountsStep,
    EWMAMeanStep, TrendStep, TimeSinceLastEventStep,
    BurstinessStep, ZeroFeaturesStep,

    # Spatial
    AssignToGridStep, AssignToZonesStep, GridAggregationStep,
    SpatialLagStep, TransformCRSStep, ComputeDistancesStep,
    KDEStep, LocalMoranStep, GetisOrdStep,

    # Point Process
    HawkesKernelStep, ConditionalIntensityStep,
    ContinuousInterEventStep,

    # Streaming
    StreamingWindowStep, StreamingInterEventStep, StreamingHawkesStep,

    # Observation
    ObservationPanelStep,
)
# Phase-2 steps are registered but not re-exported from
# eventflow.core.steps — import them from their own modules:
from eventflow.core.steps.density import KernelDensityStep, SpaceTimeKernelDensityStep
from eventflow.core.steps.hawkes import HawkesExcitationStep
from eventflow.core.steps.near_repeat import NearRepeatStep
from eventflow.core.steps.spatial import MultiScaleSpatialLagStep
from eventflow.core.steps.validation import (
    validate_temporal_ordering,
    validate_point_process_features,
)
```

## Project Structure

```
eventflow/
├── src/eventflow/
│   ├── core/
│   │   ├── adapters/      # Multi-modal output adapters
│   │   │   ├── table.py   # TableAdapter for DataFrames
│   │   │   ├── sequence.py # SequenceAdapter for RNNs
│   │   │   ├── raster.py  # RasterAdapter for CNNs
│   │   │   ├── graph.py   # GraphAdapter for GNNs
│   │   │   └── stream.py  # StreamAdapter for Neural ODEs
│   │   ├── steps/         # Pipeline steps
│   │   │   ├── temporal.py
│   │   │   ├── spatial.py
│   │   │   ├── panel.py       # CompletePanelStep
│   │   │   ├── dynamics.py    # EWMA, trend, recency, burstiness, zeros
│   │   │   ├── categorical.py # CategoryCountsStep (fit/freeze)
│   │   │   ├── density.py     # Gaussian KDE, space-time KDE
│   │   │   ├── near_repeat.py # Near-repeat ring counts
│   │   │   ├── hawkes.py      # Fixed-kernel Hawkes excitation
│   │   │   ├── point_process.py
│   │   │   ├── streaming.py
│   │   │   └── validation.py
│   │   ├── context/       # Context enrichment
│   │   │   └── tabular.py # TabularFileSource (generic connector)
│   │   ├── event_frame.py # Core EventFrame abstraction
│   │   ├── spatial.py     # Spatial utilities
│   │   └── temporal.py    # Temporal utilities
│   ├── datasets/          # Dataset-specific loaders
│   │   ├── chicago_crime/
│   │   ├── beijing_air_quality/
│   │   └── geolife/
│   ├── evaluation/        # Blocked splits, walk-forward, metrics
│   ├── recipes/           # Composable pipelines
│   └── tracking/          # MLflow integration
├── notebooks/             # Demo notebooks
├── tests/                 # 1,046 tests passing (measured 2026-11-16)
└── docs/                  # Documentation
```

## Example: Chicago Crime Analysis

See [`notebooks/demo_chicago_crime.ipynb`](notebooks/demo_chicago_crime.ipynb) for a complete example that:

1. Loads 60K+ real Chicago crime records
2. Converts to all 5 adapter formats
3. Visualizes spatial/temporal patterns
4. Demonstrates advanced architectures:
   - ST-GNN (Spatio-Temporal Graph Neural Network)
   - Neural Point Processes
   - Multi-Modal Fusion
   - Causal Inference
   - Anomaly Detection

> **Unverified (phase 0, confirmed 2026-11-16):** the demo notebook is not executed in CI
> and was not executed in the 2026-11-15 audit, so its figures (spatial
> correlation ρ = 0.515, 104 anomalies, etc.) are **not confirmed** — see
> `docs/audit/known-limitations.md`. A headless execution was attempted on
> 2026-11-16 (scratch venv, `jupyter nbconvert --execute`): it fails in the
> data-load cell with `ValueError: cannot concat empty list`, because the
> notebook reads monthly CSVs from a `chicago_crime_data_cli` directory
> that does not exist in this environment — see `notebooks/README.md`.
> The recipe itself, by contrast, is replayed on real Q1 2024 data by
> `make audit`.

## Development

```bash
# Clone the repository
git clone https://github.com/BayoHabib/eventflow.git
cd eventflow

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/eventflow --cov-report=html

# Linting
ruff check src tests
black src tests
mypy src
```

## Roadmap

See [TODO.md](TODO.md) for the full production readiness checklist.

**High Priority:**
- [ ] FastAPI prediction service
- [ ] PyPI publishing
- [ ] Docker support

**Coming Soon:**
- [ ] Pre-trained baseline models
- [ ] Real-time streaming support
- [ ] Dashboard integration

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

```bibtex
@software{eventflow2025,
  title = {EventFlow: A Generic Spatio-Temporal Event Transformation Engine},
  author = {Habib Bayo},
  year = {2025},
  url = {https://github.com/BayoHabib/eventflow}
}
```


