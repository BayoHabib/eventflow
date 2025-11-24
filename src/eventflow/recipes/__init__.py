"""Recipe mechanism for composable feature engineering."""

from eventflow.recipes.base import BaseRecipe
from eventflow.recipes.registry import register_recipe, get_recipe, list_recipes

__all__ = [
    "BaseRecipe",
    "register_recipe",
    "get_recipe",
    "list_recipes",
]
