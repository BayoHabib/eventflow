"""CLI tests using Typer's test runner."""

import textwrap

import pytest
from typer.testing import CliRunner

from eventflow.cli.main import app
from eventflow.recipes import registry
from eventflow.recipes.base import BaseRecipe
from eventflow.core.pipeline import Pipeline


class DummyRecipe(BaseRecipe):
    """Minimal recipe for CLI testing."""

    def build_pipeline(self) -> Pipeline:
        return Pipeline([])


@pytest.fixture(autouse=True)
def clear_registry() -> None:
    """Ensure recipe registry is reset between tests."""
    registry._RECIPE_REGISTRY.clear()


def test_list_commands_show_registered_entries() -> None:
    registry.register_recipe("dummy_ds", "dummy_recipe", DummyRecipe)

    runner = CliRunner()

    datasets_result = runner.invoke(app, ["list-datasets"])
    assert datasets_result.exit_code == 0
    assert "dummy_ds" in datasets_result.stdout

    recipes_result = runner.invoke(app, ["list-recipes", "--dataset", "dummy_ds"])
    assert recipes_result.exit_code == 0
    assert "dummy_recipe" in recipes_result.stdout


def test_validate_recipe_config(tmp_path) -> None:
    cfg = tmp_path / "recipe.yaml"
    cfg.write_text(
        textwrap.dedent(
            """
            dataset: dummy_ds
            recipe: dummy_recipe
            grid: {}
            temporal: {}
            """
        ).strip()
    )

    runner = CliRunner()
    result = runner.invoke(app, ["validate", "--config", str(cfg)])
    assert result.exit_code == 0
    assert "Valid recipe configuration" in result.stdout


def test_validate_dataset_config(tmp_path) -> None:
    cfg = tmp_path / "dataset.yaml"
    cfg.write_text(
        textwrap.dedent(
            """
            dataset_name: dummy_ds
            raw_root: data/raw/dummy
            """
        ).strip()
    )

    runner = CliRunner()
    result = runner.invoke(app, ["validate", "--config", str(cfg)])
    assert result.exit_code == 0
    assert "Valid dataset configuration" in result.stdout


def test_run_uses_registered_recipe(tmp_path) -> None:
    registry.register_recipe("dummy_ds", "dummy_recipe", DummyRecipe)
    cfg = tmp_path / "run.yaml"
    cfg.write_text(
        textwrap.dedent(
            """
            dataset: dummy_ds
            recipe: dummy_recipe
            """
        ).strip()
    )

    runner = CliRunner()
    result = runner.invoke(
        app, ["run", "--dataset", "dummy_ds", "--recipe", "dummy_recipe", "--config", str(cfg)]
    )
    assert result.exit_code == 0
    assert "Recipe loaded" in result.stdout


def test_version_command() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "eventflow version" in result.stdout
