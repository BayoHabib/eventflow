# Eventflow Project - Getting Started

Welcome to **Eventflow**, a high-performance, generic spatio-temporal event transformation engine!

## Project Structure Created

The complete Eventflow project has been set up with the following structure:

```
eventflow/
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
│   ├── api_reference.md     # API reference
│   └── datasets.md          # Dataset documentation
├── src/eventflow/
│   ├── __init__.py
│   ├── core/               # Generic engine (dataset-agnostic)
│   │   ├── schema.py       # Schema definitions
│   │   ├── event_frame.py  # EventFrame abstraction
│   │   ├── spatial.py      # Spatial operations
│   │   ├── temporal.py     # Temporal operations
│   │   ├── features.py     # Feature engineering (partly experimental, phase 0)
│   │   ├── pipeline.py     # Pipeline orchestration
│   │   ├── utils.py        # Utilities
│   │   └── context/        # Context enrichment
│   │       ├── sources.py
│   │       ├── joiners.py
│   │       └── enricher.py
│   ├── datasets/           # Dataset adapters
│   │   └── chicago_crime/
│   │       ├── schema.py
│   │       ├── mapping.py
│   │       ├── context/
│   │       │   ├── weather.py
│   │       │   ├── events.py
│   │       │   └── demographics.py
│   │       └── recipes/
│   │           └── chicago_crime_v1.py
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
│   ├── recipes/
│   │   └── test_registry.py
│   ├── integration/
│   └── conftest.py
└── scripts/                # Example scripts
    ├── run_chicago_recipe.py
    └── run_experiment.py
```

## Next Steps

### 1. Install Dependencies

Navigate to the project directory and install in development mode:

```powershell
cd eventflow
pip install -e ".[dev]"
```

This will install:
- Core dependencies: polars, pydantic, shapely, pyproj, typer, pyyaml
- Development tools: pytest, black, ruff, mypy, pre-commit
- Optional: Install MLflow with `pip install eventflow[tracking]`

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
pytest --cov=eventflow --cov-report=html
```

### 4. Try the CLI

```powershell
# List available datasets
python -m eventflow.cli.main list-datasets

# List recipes
python -m eventflow.cli.main list-recipes

# Validate a config
python -m eventflow.cli.main validate --config configs/recipes/chicago_crime_v1.yaml

# Check version
python -m eventflow.cli.main version
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
- **Feature Engineering**: Aggregations, moving windows, categorical encoding
- **Pipeline**: Composable step-based transformations
- **Context Enrichment**: Generic framework for joining external data

### Dataset Module
- **Chicago Crime Adapter**: Complete implementation with schema, mapping, and loaders
- **Context Sources**: Weather, special events, demographics sources
- **Recipes**: Pre-built feature engineering pipelines

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

1. Create package: `src/eventflow/datasets/your_dataset/`
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
- **Production Ready**: Core abstractions are production-ready; some implementations are placeholders for extension

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
