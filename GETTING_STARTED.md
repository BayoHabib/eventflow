# Shikong Project - Getting Started

Welcome to **Shikong**, a high-performance, generic spatio-temporal event transformation engine!

## Project Structure Created

The complete Shikong project has been set up with the following structure:

```
shikong/
├── pyproject.toml           # Project configuration and dependencies
├── README.md                # Main documentation
├── LICENSE                  # MIT License
├── .gitignore              # Git ignore patterns
├── .pre-commit-config.yaml # Pre-commit hooks configuration
├── .github/
│   └── workflows/
│       └── ci.yml          # GitHub Actions CI workflow
├── configs/
│   ├── datasets/
│   │   └── chicago_crime_example.yaml
│   └── recipes/
│       └── chicago_crime_v1.yaml
├── docs/
│   ├── architecture.md      # Architecture documentation
│   ├── api_reference.md     # API reference (every example executed by tests)
│   ├── datasets.md          # Dataset documentation
│   ├── design-notes/        # Phase 1-4 design notes
│   └── audit/               # Audit evidence and known limitations
├── src/shikong/
│   ├── __init__.py
│   ├── core/               # Generic engine (dataset-agnostic)
│   │   ├── schema.py       # Schema definitions
│   │   ├── event_frame.py  # EventFrame abstraction
│   │   ├── spatial.py      # Spatial operations
│   │   ├── temporal.py     # Temporal operations
│   │   ├── features.py     # Feature engineering (partly experimental, phase 0)
│   │   ├── pipeline.py     # Pipeline orchestration
│   │   ├── utils.py        # Utilities
│   │   ├── steps/          # Pipeline steps (temporal, spatial, panel,
│   │   │                   # dynamics, categorical, density, near-repeat,
│   │   │                   # Hawkes, point process, streaming, validation)
│   │   └── context/        # Context enrichment
│   │       ├── sources.py
│   │       ├── joiners.py
│   │       ├── enricher.py
│   │       └── tabular.py  # TabularFileSource (generic connector)
│   ├── datasets/           # Dataset adapters
│   │   ├── chicago_crime/
│   │   │   ├── schema.py
│   │   │   ├── mapping.py
│   │   │   ├── context/
│   │   │   │   ├── weather.py
│   │   │   │   ├── events.py
│   │   │   │   └── demographics.py
│   │   │   └── recipes/
│   │   │       └── chicago_crime_v1.py
│   │   ├── beijing_air_quality/  # Second-domain proof (phase 3)
│   │   └── geolife/              # Trajectory proof (phase 3)
│   ├── evaluation/         # Blocked splits, walk-forward, metrics (phase 4)
│   ├── recipes/            # Recipe mechanism
│   │   ├── base.py
│   │   └── registry.py
│   ├── tracking/           # Experiment tracking
│   │   ├── protocol.py
│   │   └── mlflow_tracker.py
│   └── cli/                # Command-line interface
│       └── main.py
├── tests/                  # Unit and integration tests
│   ├── core/
│   │   ├── test_schema.py
│   │   └── test_event_frame.py
│   ├── datasets/
│   ├── evaluation/
│   ├── recipes/
│   │   └── test_registry.py
│   ├── integration/
│   ├── test_readme_examples.py   # Executes the README Quick Start
│   ├── test_docs_examples.py     # Executes the API reference examples
│   └── conftest.py
└── scripts/                # Example scripts
    ├── run_chicago_recipe.py
    ├── replay_beijing.py
    ├── replay_geolife.py
    ├── phase3_connectors_demo.py
    ├── phase4_chicago_evaluation.py
    └── run_experiment.py
```

## Next Steps

### 1. Install Dependencies

Navigate to the project directory and install in development mode:

```powershell
cd shikong
pip install -e ".[dev]"
```

This will install:
- Core dependencies: polars, pydantic, shapely, pyproj, typer, pyyaml
- Development tools: pytest, black, ruff, mypy, pre-commit
- Optional: Install MLflow with `pip install shikong[tracking]`

### 2. Set Up Pre-commit Hooks

```powershell
pre-commit install
```

This configures automatic code formatting and linting on commit.

### 3. Run Tests

```powershell
pytest
```

Or with coverage:

```powershell
pytest --cov=shikong --cov-report=html
```

### 4. Try the CLI

```powershell
# List available datasets
python -m shikong.cli.main list-datasets

# List recipes
python -m shikong.cli.main list-recipes

# Validate a config
python -m shikong.cli.main validate --config configs/recipes/chicago_crime_v1.yaml

# Check version
python -m shikong.cli.main version
```

### 5. Run Example Scripts

```powershell
# Run Chicago Crime recipe (requires data)
python scripts/run_chicago_recipe.py

# Run experiment with tracking
python scripts/run_experiment.py
```

## Key Features Implemented

### Core Module
- **EventFrame**: Central abstraction wrapping Polars LazyFrame with schema and metadata
- **Schema Definitions**: Pydantic models for events and context sources
- **Spatial Operations**: Grid construction, coordinate transformation, zone assignment
- **Temporal Operations**: Time binning, component extraction, temporal alignment
- **Feature Engineering**: Panel completion with explicit zeros, lags, rolling
  windows, EWMA, trend, recency, burstiness, zero features, per-category
  counts/shares/entropy, multi-scale spatial lag, Gaussian KDE / STKDE,
  near-repeat counts, fixed-kernel Hawkes excitation — all under the
  causality convention "a feature row at bin `t` predicts bin `t + 1`"
  (see `docs/api_reference.md` and `docs/design-notes/phase1-causality.md`)
- **Pipeline**: Composable step-based transformations
- **Context Enrichment**: Generic framework for joining external data, plus
  the `TabularFileSource` connector for local CSV/Parquet files

### Dataset Module
- **Chicago Crime Adapter**: Complete implementation with schema, mapping, and loaders
- **Beijing Air Quality Adapter** (phase 3): Sensor panel proof — same generic
  steps, no Chicago-specific assumption; see `docs/api_reference.md` and
  `docs/design-notes/phase3-beijing.md`
- **GeoLife Adapter** (phase 3): GPS trajectories with kinematics; the steps
  that do and do not apply are decided in
  `docs/design-notes/phase3-geolife-boundary.md`
- **Context Sources**: Weather, special events, demographics sources (Chicago
  ones unwired and untested); the generic `TabularFileSource` connector loads
  any local CSV/Parquet file as context
- **Recipes**: Pre-built feature engineering pipelines
  (`chicago_crime_v1`, `beijing_air_quality_v1`, `geolife_v1`)

### Evaluation Module (phase 4)
- **Blocked splits**: Temporal, spatial, entity and combined time × space
  splits with an embargo gap and a `validate_split` negative control
- **Walk-forward**: Expanding-window folds with per-fold refit of fittable
  steps (`CategoryCountsStep.fit/freeze`)
- **Metrics**: PAI (hotspot ranking) and PR-AUC as pure functions
- **Synthetic ground truth**: Planted-structure panels in
  `tests/evaluation/synthetic.py` (test-only helper)

### Recipe Module
- **Base Recipe Interface**: Abstract base class for all recipes
- **Recipe Registry**: Discovery and instantiation system

### Tracking Module
- **Protocol-based Design**: Tracker protocol for any backend
- **MLflow Implementation**: Full MLflow integration

### CLI Module
- **Typer-based CLI**: User-friendly command-line interface
- Commands for running recipes, listing datasets, validation

## Development Workflow

### Adding a New Dataset

1. Create package: `src/shikong/datasets/your_dataset/`
2. Define schema in `schema.py`
3. Implement loader in `mapping.py`
4. Add context sources in `context/`
5. Create recipes in `recipes/`
6. Add tests in `tests/datasets/your_dataset/`
7. Document in `docs/datasets.md`

### Adding a New Recipe

1. Create recipe class extending `BaseRecipe`
2. Implement `build_pipeline()` method
3. Register with `register_recipe()`
4. Add configuration YAML in `configs/recipes/`
5. Add tests
6. Document usage

### Code Quality

The project uses:
- **Black** for code formatting
- **Ruff** for linting
- **MyPy** for type checking
- **Pytest** for testing

Run all checks:

```powershell
black src tests
ruff check src tests
mypy src
pytest
```

## Architecture Principles

1. **Lazy Evaluation**: All operations use Polars LazyFrame until `.collect()`
2. **Generic Core**: No dataset-specific logic in `core/`
3. **Type Safety**: Extensive use of Pydantic for validation
4. **Composability**: Pure functions and stateless steps
5. **Extensibility**: Protocol-based design for tracking, context sources

## Important Notes

- **Import Errors**: You'll see import errors until dependencies are installed with `pip install -e ".[dev]"`
- **Data Not Included**: Raw data files are not part of the repository - configure paths in YAML files
- **Doc examples are executed**: Every code example in `docs/api_reference.md`
  is run by `tests/test_docs_examples.py`, and the README Quick Start by
  `tests/test_readme_examples.py` — if a doc example breaks, the suite fails
- **Demo notebook is unverified**: `notebooks/demo_chicago_crime.ipynb` is not
  executed in CI; see `notebooks/README.md` for the last execution attempt
  and why it fails (missing `chicago_crime_data_cli` data directory)

## Resources

- **Documentation**: See `docs/` directory for architecture, API reference, and dataset guides
- **Examples**: Check `scripts/` for usage examples
- **Tests**: Review `tests/` for usage patterns and fixtures
- **Configs**: Examine `configs/` for configuration examples

## Contributing

1. Create a feature branch
2. Make changes following the architecture principles
3. Add tests for new functionality
4. Run pre-commit hooks and tests
5. Submit pull request

## Getting Help

- Read the architecture documentation: `docs/architecture.md`
- Check API reference: `docs/api_reference.md`
- Review example scripts in `scripts/`
- Look at test files for usage examples

## License

MIT License - see LICENSE file for details.

---

**You're all set!** Start by installing dependencies and running the tests. Happy coding! 🚀
