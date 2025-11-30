"""Main CLI application using Typer."""

from pathlib import Path
from typing import Annotated

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
    """Return the global step registry, loading entry points on first call."""
    global _step_registry
    if _step_registry is None:
        _step_registry = FeatureStepRegistry()
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
) -> None:
    """
    Run a recipe on a dataset.

    Example:
        eventflow run --dataset chicago_crime --recipe chicago_crime_v1 --config configs/recipes/chicago_crime_v1.yaml
    """
    from eventflow.core.schema import RecipeConfig
    from eventflow.recipes.registry import get_recipe

    typer.echo(f"Running recipe '{recipe}' on dataset '{dataset}'...")

    # Load config
    config_path = Path(config)
    if not config_path.exists():
        typer.echo(f"Error: Config file not found: {config}", err=True)
        raise typer.Exit(code=1) from None

    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    recipe_config = RecipeConfig(**config_dict)

    # Get recipe, injecting step registry so it can resolve step names
    step_registry = get_step_registry()

    try:
        recipe_instance = get_recipe(
            dataset,
            recipe,
            recipe_config,
            step_registry=step_registry,
        )
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from None

    typer.echo(f"Recipe loaded: {recipe_instance}")
    typer.echo("Note: Full execution requires loading dataset - implement in your script")


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
            RecipeConfig(**config_dict)
            typer.echo(f"✓ Valid recipe configuration: {config}")
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
