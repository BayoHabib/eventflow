"""MLflow implementation of TrackerProtocol."""

import json
import tempfile
from pathlib import Path
from typing import Any


class MLflowTracker:
    """
    MLflow implementation of experiment tracking.

    Note: This requires mlflow to be installed (optional dependency).
    """

    def __init__(self) -> None:
        """Initialize MLflow tracker."""
        try:
            import mlflow
            self.mlflow = mlflow
        except ImportError as err:
            raise ImportError(
                "MLflow is not installed. Install with: pip install mlflow"
            ) from err

    def log_param(self, key: str, value: Any) -> None:
        """Log a parameter to MLflow."""
        self.mlflow.log_param(key, value)

    def log_metric(
        self,
        key: str,
        value: float,
        step: int | None = None,
    ) -> None:
        """Log a metric to MLflow."""
        self.mlflow.log_metric(key, value, step=step)

    def log_artifact(
        self,
        local_path: str,
        artifact_path: str | None = None,
    ) -> None:
        """Log an artifact to MLflow."""
        self.mlflow.log_artifact(local_path, artifact_path=artifact_path)

    def set_tags(self, tags: dict[str, str]) -> None:
        """Set tags in MLflow."""
        self.mlflow.set_tags(tags)

    def log_dict(self, dictionary: dict[str, Any], artifact_file: str) -> None:
        """Log a dictionary as a JSON artifact."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / artifact_file
            with open(filepath, "w") as f:
                json.dump(dictionary, f, indent=2)
            self.mlflow.log_artifact(str(filepath))

    def start_run(self, run_name: str | None = None) -> Any:
        """Start an MLflow run."""
        return self.mlflow.start_run(run_name=run_name)

    def end_run(self) -> None:
        """End the current MLflow run."""
        self.mlflow.end_run()

    def __repr__(self) -> str:
        """String representation."""
        return "MLflowTracker()"
