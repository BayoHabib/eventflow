#!/usr/bin/env python
"""
Example script for running a Chicago Crime recipe.

This script demonstrates how to:
1. Load the Chicago Crime dataset
2. Apply a recipe for feature engineering
3. Save the results
"""

import sys
from pathlib import Path

import yaml  # type: ignore[import-untyped]

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eventflow.core.schema import DatasetConfig, RecipeConfig


def main() -> None:
    """Main execution function."""
    # Configuration paths
    dataset_config_path = Path("configs/datasets/chicago_crime_example.yaml")
    recipe_config_path = Path("configs/recipes/chicago_crime_v1.yaml")

    print("=" * 60)
    print("Chicago Crime Recipe Runner")
    print("=" * 60)

    # Load dataset configuration
    print(f"\n1. Loading dataset configuration from {dataset_config_path}")
    with open(dataset_config_path) as f:
        dataset_config_dict = yaml.safe_load(f)
    dataset_config = DatasetConfig(**dataset_config_dict)
    print(f"   Dataset: {dataset_config.dataset_name}")
    print(f"   Raw root: {dataset_config.raw_root}")
    print(f"   CRS: {dataset_config.crs}")

    # Load recipe configuration
    print(f"\n2. Loading recipe configuration from {recipe_config_path}")
    with open(recipe_config_path) as f:
        recipe_config_dict = yaml.safe_load(f)
    recipe_config = RecipeConfig(**recipe_config_dict)
    print(f"   Recipe: {recipe_config.recipe}")
    print(f"   Grid size: {recipe_config.grid.get('size_m', 'N/A')}m")
    print(f"   Time bin: {recipe_config.temporal.get('time_bin', 'N/A')}")

    # Load dataset
    print("\n3. Loading Chicago Crime dataset...")
    print("   Note: This requires actual data files in the configured path")
    print(f"   Expected path: {dataset_config.raw_root}")

    # Uncomment when data is available:
    # event_frame = load_chicago_crime(
    #     dataset_config.raw_root,
    #     config=dataset_config,
    # )
    # print(f"   Loaded {event_frame.count()} events")
    # print(f"   Schema: {event_frame.schema}")

    # Get and run recipe
    print("\n4. Getting recipe...")
    # Uncomment when ready:
    # recipe = get_recipe(
    #     dataset_config.dataset_name,
    #     recipe_config.recipe,
    #     recipe_config,
    # )
    # print(f"   Recipe: {recipe}")

    # Run pipeline
    print("\n5. Running feature engineering pipeline...")
    # Uncomment when ready:
    # result_frame = recipe.run(event_frame)
    # print(f"   Result shape: {result_frame.count()} rows")

    # Save results
    output_path = Path("data/processed/chicago_crime_features.parquet")
    print(f"\n6. Saving results to {output_path}...")
    # Uncomment when ready:
    # output_path.parent.mkdir(parents=True, exist_ok=True)
    # result_frame.collect().write_parquet(output_path)
    # print(f"   Saved successfully!")

    print("\n" + "=" * 60)
    print("Script execution complete")
    print("=" * 60)
    print("\nTo use this script with actual data:")
    print("1. Ensure Chicago Crime data is in:", dataset_config.raw_root)
    print("2. Uncomment the data loading and processing sections")
    print("3. Run: python scripts/run_chicago_recipe.py")


if __name__ == "__main__":
    main()
