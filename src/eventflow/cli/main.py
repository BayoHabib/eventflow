"""Main CLI application using Typer."""

from pathlib import Path
from typing import Annotated

import polars as pl
import typer
import yaml  # type: ignore[import-untyped]

from eventflow.core.registry import (
    ContextSourceRegistry,
    FeatureStepRegistry,
    OutputAdapterRegistry,
)

app = typer.Typer(help="Eventflow: Spatio-temporal event transformation engine")

# --------------------------------------------------------------------------- #
# Global registries (lazily populated via entry points on first access)
# --------------------------------------------------------------------------- #
_step_registry: FeatureStepRegistry | None = None
_context_registry: ContextSourceRegistry | None = None
_output_registry: OutputAdapterRegistry | None = None


def get_step_registry() -> FeatureStepRegistry:
    """Load built-in treatments and plugin entry points on first access."""
    from eventflow.core.steps.registration import get_default_registry

    global _step_registry
    if _step_registry is None:
        _step_registry = get_default_registry()
        _step_registry.load_entry_points()
    return _step_registry


def get_context_registry() -> ContextSourceRegistry:
    """Return the global context source registry."""
    global _context_registry
    if _context_registry is None:
        _context_registry = ContextSourceRegistry()
        _context_registry.load_entry_points()
    return _context_registry


def get_output_registry() -> OutputAdapterRegistry:
    """Return the global output adapter registry."""
    global _output_registry
    if _output_registry is None:
        _output_registry = OutputAdapterRegistry()
        _output_registry.load_entry_points()
    return _output_registry


@app.command()
def run(
    dataset: str = typer.Option(..., help="Dataset name"),
    recipe: str = typer.Option(..., help="Recipe name"),
    config: str = typer.Option(..., help="Path to recipe config YAML"),
    output: str | None = typer.Option(None, help="Output path for results"),
    input_path: Path | None = typer.Option(
        None, "--input", help="Raw CSV/Parquet file or directory of chunks"
    ),
    dataset_config: Path | None = typer.Option(
        None, "--dataset-config", help="Dataset YAML specifying raw_root and target CRS"
    ),
) -> None:
    """
    Run a recipe on a dataset.

    Example:
        eventflow run --dataset chicago_crime --recipe chicago_crime_v1 --config configs/recipes/chicago_crime_v1.yaml
    """
    from eventflow.core.schema import DatasetConfig, RecipeConfig
    from eventflow.datasets.chicago_crime.mapping import load_chicago_crime
    from eventflow.recipes.registry import get_recipe

    typer.echo(f"Running recipe '{recipe}' on dataset '{dataset}'...")

    try:
        with open(config, encoding="utf-8") as f:
            recipe_config = RecipeConfig(**yaml.safe_load(f))
        if (recipe_config.dataset, recipe_config.recipe) != (dataset, recipe):
            raise ValueError("Dataset and recipe arguments must match the recipe configuration")
        if recipe_config.output.get("partition_by"):
            raise ValueError(
                "output.partition_by "
                f"{recipe_config.output.get('partition_by')} is declared but not "
                "implemented: the CLI writes a single output file. Remove "
                "partition_by from the recipe configuration."
            )
        if dataset != "chicago_crime":
            raise ValueError(f"No CLI data loader is available for dataset: {dataset}")
        data_config = None
        if dataset_config is not None:
            with dataset_config.open(encoding="utf-8") as f:
                data_config = DatasetConfig(**yaml.safe_load(f))
            if data_config.dataset_name != dataset:
                raise ValueError("Dataset configuration must match --dataset")
        if input_path is not None and data_config is not None:
            data_config = data_config.model_copy(update={"raw_root": str(input_path)})
        if input_path is not None:
            raw_path = str(input_path)
        elif data_config is not None:
            raw_path = data_config.raw_root
        else:
            raise ValueError("Provide --input or --dataset-config to locate raw data")

        recipe_instance = get_recipe(
            dataset,
            recipe,
            recipe_config,
            step_registry=get_step_registry(),
        )
        event_frame = load_chicago_crime(raw_path, config=data_config)
        result = recipe_instance.run(event_frame).collect()
        output_format = recipe_config.output.get("format", "parquet")
        destination = (
            Path(output) if output else Path("data/processed") / f"{recipe}.{output_format}"
        )
        if destination.suffix not in {".csv", ".parquet"}:
            raise ValueError("Output path must end with .csv or .parquet")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.suffix == ".parquet":
            result.write_parquet(
                destination, compression=recipe_config.output.get("compression", "snappy")
            )
        else:
            result.write_csv(destination)
    except (OSError, ValueError, TypeError, yaml.YAMLError, pl.exceptions.PolarsError) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from None

    typer.echo(f"Saved {result.height} cell/time-bin rows to {destination}")


@app.command()
def list_datasets() -> None:
    """List all available datasets."""
    from eventflow.recipes.registry import list_datasets as get_datasets

    datasets = get_datasets()

    if not datasets:
        typer.echo("No datasets registered")
        return

    typer.echo("Available datasets:")
    for dataset in datasets:
        typer.echo(f"  - {dataset}")


@app.command()
def list_recipes(dataset: str | None = typer.Option(None, help="Filter by dataset")) -> None:
    """List all available recipes."""
    from eventflow.recipes.registry import list_recipes as get_recipes

    recipes = get_recipes(dataset)

    if not recipes:
        typer.echo(
            f"No recipes found for dataset: {dataset}" if dataset else "No recipes registered"
        )
        return

    typer.echo("Available recipes:")
    for ds, recipe_list in recipes.items():
        typer.echo(f"\n{ds}:")
        for recipe_name in recipe_list:
            typer.echo(f"  - {recipe_name}")


@app.command()
def validate(
    config: str = typer.Option(..., help="Path to config YAML to validate"),
) -> None:
    """Validate a configuration file."""
    from eventflow.core.schema import DatasetConfig, RecipeConfig

    config_path = Path(config)
    if not config_path.exists():
        typer.echo(f"Error: Config file not found: {config}", err=True)
        raise typer.Exit(code=1) from None

    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    # Try to parse as recipe config
    try:
        if "recipe" in config_dict:
            recipe_config = RecipeConfig(**config_dict)
            # Schema validity is not enough: also build the pipeline, so a
            # declared-but-unimplemented feature fails here instead of at run
            # time (see FeatureConfig.require_implemented).
            from eventflow.recipes.registry import get_recipe

            recipe = get_recipe(
                recipe_config.dataset,
                recipe_config.recipe,
                recipe_config,
                step_registry=get_step_registry(),
            )
            pipeline = recipe.build_pipeline()
            typer.echo(
                f"✓ Valid recipe configuration: {config} "
                f"(pipeline builds: {len(pipeline.steps)} steps)"
            )
        elif "dataset_name" in config_dict:
            DatasetConfig(**config_dict)
            typer.echo(f"✓ Valid dataset configuration: {config}")
        else:
            typer.echo("Error: Unknown configuration type", err=True)
            raise typer.Exit(code=1) from None
    except Exception as e:
        typer.echo(f"Error: Invalid configuration: {e}", err=True)
        raise typer.Exit(code=1) from None


@app.command()
def version() -> None:
    """Show eventflow version."""
    from eventflow import __version__

    typer.echo(f"eventflow version {__version__}")


# --------------------------------------------------------------------------- #
# Registry inspection commands
# --------------------------------------------------------------------------- #


@app.command()
def list_steps(
    tag: Annotated[str | None, typer.Option(help="Filter steps by tag")] = None,
) -> None:
    """List registered pipeline steps."""
    registry = get_step_registry()
    specs = registry.list(tag=tag)

    if not specs:
        typer.echo("No steps registered" + (f" with tag '{tag}'" if tag else ""))
        return

    typer.echo("Registered steps:")
    for spec in specs:
        tags_str = ", ".join(sorted(spec.tags)) if spec.tags else "none"
        desc = spec.description or ""
        typer.echo(f"  {spec.name}  [tags: {tags_str}]")
        if desc:
            typer.echo(f"      {desc}")


@app.command()
def list_context_sources() -> None:
    """List registered context data sources."""
    registry = get_context_registry()
    specs = registry.list()

    if not specs:
        typer.echo("No context sources registered")
        return

    typer.echo("Registered context sources:")
    for spec in specs:
        tags_str = ", ".join(sorted(spec.tags)) if spec.tags else "none"
        desc = spec.description or ""
        typer.echo(f"  {spec.name}  [tags: {tags_str}]")
        if desc:
            typer.echo(f"      {desc}")


@app.command()
def list_output_adapters() -> None:
    """List registered output adapters."""
    registry = get_output_registry()
    specs = registry.list()

    if not specs:
        typer.echo("No output adapters registered")
        return

    typer.echo("Registered output adapters:")
    for spec in specs:
        tags_str = ", ".join(sorted(spec.tags)) if spec.tags else "none"
        desc = spec.description or ""
        typer.echo(f"  {spec.name}  [tags: {tags_str}]")
        if desc:
            typer.echo(f"      {desc}")


if __name__ == "__main__":
    app()
