"""Tests for recipe registry."""

import pytest
from eventflow.recipes.registry import register_recipe, get_recipe, list_recipes, list_datasets
from eventflow.recipes.base import BaseRecipe
from eventflow.core.pipeline import Pipeline
from eventflow.core.schema import RecipeConfig


class DummyRecipe(BaseRecipe):
    """Dummy recipe for testing."""
    
    def build_pipeline(self) -> Pipeline:
        return Pipeline([])


def test_register_and_get_recipe():
    """Test registering and retrieving a recipe."""
    register_recipe("test_dataset", "test_recipe", DummyRecipe)
    
    config = RecipeConfig(dataset="test_dataset", recipe="test_recipe")
    recipe = get_recipe("test_dataset", "test_recipe", config)
    
    assert isinstance(recipe, DummyRecipe)


def test_get_nonexistent_recipe():
    """Test getting a recipe that doesn't exist."""
    with pytest.raises(ValueError):
        get_recipe("nonexistent", "recipe")


def test_list_recipes():
    """Test listing recipes."""
    register_recipe("test_dataset", "recipe1", DummyRecipe)
    register_recipe("test_dataset", "recipe2", DummyRecipe)
    
    recipes = list_recipes("test_dataset")
    assert "test_dataset" in recipes
    assert "recipe1" in recipes["test_dataset"]
    assert "recipe2" in recipes["test_dataset"]


def test_list_datasets():
    """Test listing datasets."""
    register_recipe("dataset1", "recipe", DummyRecipe)
    register_recipe("dataset2", "recipe", DummyRecipe)
    
    datasets = list_datasets()
    assert "dataset1" in datasets
    assert "dataset2" in datasets
