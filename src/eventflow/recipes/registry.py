"""Recipe registry for discovery and instantiation."""

from typing import Dict, Type
from eventflow.recipes.base import BaseRecipe
from eventflow.core.schema import RecipeConfig
from eventflow.core.utils import get_logger

logger = get_logger(__name__)

# Global recipe registry
_RECIPE_REGISTRY: Dict[str, Dict[str, Type[BaseRecipe]]] = {}


def register_recipe(
    dataset_name: str,
    recipe_name: str,
    recipe_class: Type[BaseRecipe],
) -> None:
    """
    Register a recipe for a dataset.
    
    Args:
        dataset_name: Name of the dataset
        recipe_name: Name of the recipe
        recipe_class: Recipe class to register
    """
    if dataset_name not in _RECIPE_REGISTRY:
        _RECIPE_REGISTRY[dataset_name] = {}
    
    _RECIPE_REGISTRY[dataset_name][recipe_name] = recipe_class
    logger.info(f"Registered recipe '{recipe_name}' for dataset '{dataset_name}'")


def get_recipe(
    dataset_name: str,
    recipe_name: str,
    config: RecipeConfig | None = None,
) -> BaseRecipe:
    """
    Get a recipe instance.
    
    Args:
        dataset_name: Name of the dataset
        recipe_name: Name of the recipe
        config: Optional recipe configuration
        
    Returns:
        Recipe instance
        
    Raises:
        ValueError: If dataset or recipe not found
    """
    if dataset_name not in _RECIPE_REGISTRY:
        raise ValueError(
            f"Dataset '{dataset_name}' not found. "
            f"Available datasets: {list(_RECIPE_REGISTRY.keys())}"
        )
    
    if recipe_name not in _RECIPE_REGISTRY[dataset_name]:
        raise ValueError(
            f"Recipe '{recipe_name}' not found for dataset '{dataset_name}'. "
            f"Available recipes: {list(_RECIPE_REGISTRY[dataset_name].keys())}"
        )
    
    recipe_class = _RECIPE_REGISTRY[dataset_name][recipe_name]
    
    # Create default config if not provided
    if config is None:
        config = RecipeConfig(dataset=dataset_name, recipe=recipe_name)
    
    logger.info(f"Creating recipe instance: {recipe_name} for dataset {dataset_name}")
    return recipe_class(config)


def list_recipes(dataset_name: str | None = None) -> Dict[str, list[str]]:
    """
    List available recipes.
    
    Args:
        dataset_name: Optional dataset name to filter by
        
    Returns:
        Dictionary mapping dataset names to lists of recipe names
    """
    if dataset_name is not None:
        if dataset_name not in _RECIPE_REGISTRY:
            return {}
        return {dataset_name: list(_RECIPE_REGISTRY[dataset_name].keys())}
    
    return {
        dataset: list(recipes.keys())
        for dataset, recipes in _RECIPE_REGISTRY.items()
    }


def list_datasets() -> list[str]:
    """
    List datasets with registered recipes.
    
    Returns:
        List of dataset names
    """
    return list(_RECIPE_REGISTRY.keys())
