# EventFlow

[![Test & Lint](https://github.com/BayoHabib/eventflow/actions/workflows/ci.yml/badge.svg)](https://github.com/BayoHabib/eventflow/actions)
[![Coverage](https://img.shields.io/badge/coverage-68%25-yellow.svg)](https://github.com/BayoHabib/eventflow)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-0.1.0-green.svg)](https://github.com/BayoHabib/eventflow/releases)

A high-performance, generic spatio-temporal event transformation engine for feature engineering and MLOps pipelines.

## Overview

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
- ✅ **Validated** - 290 tests, 68% coverage

## Installation

```bash
# From source (PyPI coming soon)
git clone https://github.com/BayoHabib/eventflow.git
cd eventflow
pip install -e ".[dev]"
```

## Quick Start

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
table_config = TableAdapterConfig(
    target_col="event_count",
    feature_cols=["lat_bin", "lon_bin", "day_of_week"],
)
table_adapter = TableAdapter(table_config)
table_output = table_adapter.convert(df)
X, y = table_output.get_X_y()  # Ready for sklearn!

# 2. SequenceAdapter - For LSTM/Transformer
seq_config = SequenceAdapterConfig(
    sequence_col="cell_id",
    time_col="date",
    feature_cols=["event_count"],
    max_length=30,
)
seq_adapter = SequenceAdapter(seq_config)
seq_output = seq_adapter.convert(df)
# seq_output.sequences: (N_cells, 30, 1) tensor

# 3. RasterAdapter - For CNN models
raster_config = RasterAdapterConfig(
    time_col="date",
    lat_col="lat_bin",
    lon_col="lon_bin", 
    value_cols=["event_count", "arrest_count"],
    height=10, width=10,
)
raster_adapter = RasterAdapter(raster_config)
raster_output = raster_adapter.convert(df)
# raster_output.raster: (91, 2, 10, 10) - 91 days, 2 channels

# 4. GraphAdapter - For GNN models
graph_config = GraphAdapterConfig(
    node_id_col="cell_id",
    feature_cols=["total_events", "centroid_lat", "centroid_lon"],
    adjacency_type="grid",
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

EventFlow provides composable pipeline steps for feature engineering:

```python
from eventflow.core.steps import (
    # Temporal
    TimeOfDayStep, DayOfWeekStep, HourlyBinStep,
    RollingWindowStep, LagFeaturesStep,
    
    # Spatial
    GridBinningStep, SpatialLagStep, DistanceToStep,
    
    # Point Process
    IntensityEstimationStep, KernelDensityStep,
    InterEventTimeStep, EventCountStep,
    
    # Streaming
    SessionizeStep, TumblingWindowStep, SlidingWindowStep,
    
    # Validation
    ValidateTemporalOrderStep, ValidateSpatialBoundsStep,
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
│   │   │   ├── point_process.py
│   │   │   ├── streaming.py
│   │   │   └── validation.py
│   │   ├── event_frame.py # Core EventFrame abstraction
│   │   ├── spatial.py     # Spatial utilities
│   │   └── temporal.py    # Temporal utilities
│   ├── datasets/          # Dataset-specific loaders
│   ├── recipes/           # Composable pipelines
│   └── tracking/          # MLflow integration
├── notebooks/             # Demo notebooks
├── tests/                 # 290 tests
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

