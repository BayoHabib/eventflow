#!/usr/bin/env python
"""
Example script for running an experiment with tracking.

This script demonstrates how to:
1. Set up experiment tracking (MLflow)
2. Load and process data with eventflow
3. Train a model (placeholder)
4. Log parameters, metrics, and artifacts
"""

import sys
from datetime import datetime
from pathlib import Path

import yaml  # type: ignore[import-untyped]

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eventflow.core.schema import DatasetConfig, RecipeConfig
from eventflow.tracking.mlflow_tracker import MLflowTracker


def main() -> None:
    """Main execution function."""
    print("=" * 60)
    print("Eventflow Experiment Runner with MLflow Tracking")
    print("=" * 60)

    # Initialize MLflow tracker
    print("\n1. Initializing experiment tracker...")
    try:
        tracker: MLflowTracker | None = MLflowTracker()
        print("   MLflow tracker initialized")
    except ImportError:
        print("   MLflow not available - install with: pip install mlflow")
        print("   Continuing without tracking...")
        tracker = None

    # Configuration
    dataset_config_path = Path("configs/datasets/chicago_crime_example.yaml")
    recipe_config_path = Path("configs/recipes/chicago_crime_v1.yaml")

    # Load configurations
    print("\n2. Loading configurations...")
    with open(dataset_config_path) as f:
        dataset_config = DatasetConfig(**yaml.safe_load(f))
    with open(recipe_config_path) as f:
        recipe_config = RecipeConfig(**yaml.safe_load(f))

    # Start tracking run
    if tracker:
        run_name = f"chicago_crime_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"\n3. Starting MLflow run: {run_name}")
        tracker.start_run(run_name=run_name)

        # Log parameters
        tracker.set_tags(
            {
                "dataset": dataset_config.dataset_name,
                "recipe": recipe_config.recipe,
            }
        )
        tracker.log_param("grid_size_m", recipe_config.grid.get("size_m", 300))
        tracker.log_param("time_bin", recipe_config.temporal.get("time_bin", "6h"))
        tracker.log_param("crs", dataset_config.crs)

    # Feature engineering
    print("\n4. Feature engineering...")
    print("   Note: This requires actual data and model implementation")

    # Uncomment when ready:
    # event_frame = load_chicago_crime(dataset_config.raw_root, config=dataset_config)
    # recipe = get_recipe(dataset_config.dataset_name, recipe_config.recipe, recipe_config)
    # features = recipe.run(event_frame)
    #
    # if tracker:
    #     tracker.log_metric("num_events", event_frame.count())
    #     tracker.log_metric("num_features", len(features.lazy_frame.columns))

    # Model training (placeholder)
    print("\n5. Model training...")
    print("   Placeholder for model training")

    # Example metrics
    if tracker:
        # These would come from actual model evaluation
        tracker.log_metric("train_mae", 12.5)
        tracker.log_metric("val_mae", 15.3)
        tracker.log_metric("test_mae", 14.8)
        tracker.log_metric("train_r2", 0.82)
        tracker.log_metric("val_r2", 0.78)
        tracker.log_metric("test_r2", 0.79)

    # Save artifacts
    print("\n6. Saving artifacts...")
    if tracker:
        # Save configurations as artifacts
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            with open(config_path, "w") as f:
                json.dump(
                    {
                        "dataset": dataset_config.dict(),
                        "recipe": recipe_config.dict(),
                    },
                    f,
                    indent=2,
                )
            tracker.log_artifact(str(config_path))

    # End tracking
    if tracker:
        print("\n7. Ending MLflow run...")
        tracker.end_run()
        print("   Run completed and logged to MLflow")

    print("\n" + "=" * 60)
    print("Experiment complete")
    print("=" * 60)
    print("\nTo view results:")
    print("1. Run: mlflow ui")
    print("2. Open: http://localhost:5000")


if __name__ == "__main__":
    main()
