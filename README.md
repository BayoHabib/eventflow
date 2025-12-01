# Eventflow

[![Test & Lint](https://github.com/BayoHabib/eventflow/actions/workflows/ci.yml/badge.svg)](https://github.com/BayoHabib/eventflow/actions)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-0.1.0-green.svg)](https://github.com/BayoHabib/eventflow/releases)

A high-performance, generic spatio-temporal event transformation engine for feature engineering and MLOps pipelines.

## Overview

Eventflow is designed to transform raw event data into machine learning-ready features while maintaining:
- **Generic abstractions** that work across different datasets
- **High performance** through lazy evaluation with Polars
- **Clean architecture** separating core engine from dataset-specific logic
- **Extensibility** for adding new datasets, context sources, and recipes

## Features

- 🚀 **Lazy evaluation** with Polars for efficient data processing
- 🗺️ **Spatial operations** with grid construction and zone assignment
- ⏰ **Temporal features** including time binning and windowing
- 🔗 **Context enrichment** from weather, events, demographics, etc.
- 📊 **Experiment tracking** with pluggable backends (MLflow, etc.)
- 🧪 **Recipe system** for composable feature engineering pipelines
- 📚 **Step registry** for reusable, config-driven pipeline assembly
- 🛠️ **CLI interface** for running experiments and recipes

## Installation

```bash
# Basic installation
pip install eventflow

# With development dependencies
pip install eventflow[dev]

# With MLflow tracking
pip install eventflow[tracking]

# All optional dependencies
pip install eventflow[all]
```

## Quick Start

```python
from eventflow.core import temporal
from eventflow.core.event_frame import EventFrame
from eventflow.core.schema import RecipeConfig
from eventflow.datasets.chicago_crime import load_chicago_crime
from eventflow.datasets.chicago_crime.recipes.chicago_crime_v1 import ChicagoCrimeV1Recipe
from eventflow.recipes.registry import get_recipe

# Load dataset
event_frame = load_chicago_crime("data/raw/chicago_crime")

# Get and run a recipe
recipe = get_recipe("chicago_crime", "chicago_crime_v1")
features = recipe.run(event_frame)

# Collect results
df = features.collect()
print(df.head())

# Build a pipeline from registered steps and configuration
from eventflow.core.pipeline import Step
from eventflow.core.registry import StepRegistry


class TemporalBinsStep(Step):
  def __init__(self, bin_size: str = "6h") -> None:
    self.bin_size = bin_size

  def run(self, event_frame: EventFrame) -> EventFrame:
    return temporal.create_time_bins(event_frame, self.bin_size)


registry = StepRegistry()
registry.register("temporal_bins", TemporalBinsStep, tags={"temporal"})

# Recipes consult config.features["steps"] when a registry is supplied
recipe_config = RecipeConfig(
  dataset="chicago_crime",
  recipe="custom",
  features={"steps": [{"name": "temporal_bins", "params": {"bin_size": "12h"}}]},
)
custom_recipe = ChicagoCrimeV1Recipe(recipe_config, step_registry=registry)
features = custom_recipe.run(event_frame)
```

## Command-Line Interface

```bash
# Run a recipe on a dataset
eventflow run --dataset chicago_crime --recipe chicago_crime_v1 --config configs/recipes/chicago_crime_v1.yaml

# List available datasets
eventflow list-datasets

# List recipes for a dataset
eventflow list-recipes --dataset chicago_crime
```

## Project Structure

```
eventflow/
├── src/eventflow/
│   ├── core/           # Generic engine (dataset-agnostic)
│   ├── datasets/       # Dataset-specific adapters
│   ├── recipes/        # Recipe mechanism and registry
│   ├── tracking/       # Experiment tracking (MLflow, etc.)
│   └── cli/            # Command-line interface
├── configs/            # YAML configurations
├── docs/               # Documentation
├── scripts/            # Example scripts
└── tests/              # Unit and integration tests
```

## Architecture

Eventflow follows a clean layered architecture:

1. **Core Layer** (`eventflow.core`): Generic primitives for events, spatial/temporal operations, features, and pipelines
2. **Dataset Layer** (`eventflow.datasets`): Adapters that map specific datasets to core abstractions
3. **Recipe Layer** (`eventflow.recipes`): Composable pipelines for feature engineering
4. **Tracking Layer** (`eventflow.tracking`): Optional experiment tracking integration

See [docs/architecture.md](docs/architecture.md) for details.

## Development

```bash
# Clone the repository
git clone https://github.com/eventflow/eventflow.git
cd eventflow

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run linting
ruff check src tests
black --check src tests

# Run type checking
mypy src
```

## Contributing

Contributions are welcome! Please see our contributing guidelines (coming soon).

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use Eventflow in your research, please cite:

```bibtex
@software{eventflow2024,
  title = {Eventflow: A Generic Spatio-Temporal Event Transformation Engine},
  author = {Habib Bayo},
  year = {2025},
  url = {https://github.com/eventflow/eventflow}
}
```


