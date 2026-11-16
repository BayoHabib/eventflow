"""Phase 3: the Beijing and GeoLife recipes are registered and buildable."""

from pathlib import Path

import pytest
import yaml

from eventflow.core.schema import RecipeConfig
from eventflow.recipes.registry import get_recipe, list_datasets

CONFIGS = Path(__file__).resolve().parents[2] / "configs" / "recipes"


def _load_recipe_config(name: str) -> RecipeConfig:
    with (CONFIGS / name).open(encoding="utf-8") as f:
        return RecipeConfig(**yaml.safe_load(f))


def test_phase3_recipes_registered() -> None:
    datasets = list_datasets()
    assert "beijing_air_quality" in datasets
    assert "geolife" in datasets


@pytest.mark.parametrize(
    ("dataset", "recipe", "config_file", "expected_steps"),
    [
        ("beijing_air_quality", "beijing_air_quality_v1", "beijing_air_quality_v1.yaml", 6),
        ("geolife", "geolife_v1", "geolife_v1.yaml", 10),
    ],
)
def test_phase3_recipes_build_from_yaml(
    dataset: str, recipe: str, config_file: str, expected_steps: int
) -> None:
    config = _load_recipe_config(config_file)
    assert config.dataset == dataset
    assert config.recipe == recipe
    built = get_recipe(dataset, recipe, config=config)
    pipeline = built.get_pipeline()
    assert len(pipeline.steps) == expected_steps
