"""Protocol definition for experiment tracking."""

from typing import Any, Protocol


class TrackerProtocol(Protocol):
    """
    Protocol for experiment tracking backends.

    This allows eventflow to remain independent of specific tracking
    implementations (MLflow, Weights & Biases, etc.).
    """

    def log_param(self, key: str, value: Any) -> None:
        """
        Log a parameter.

        Args:
            key: Parameter name
            value: Parameter value
        """
        ...

    def log_metric(
        self,
        key: str,
        value: float,
        step: int | None = None,
    ) -> None:
        """
        Log a metric.

        Args:
            key: Metric name
            value: Metric value
            step: Optional step number
        """
        ...

    def log_artifact(
        self,
        local_path: str,
        artifact_path: str | None = None,
    ) -> None:
        """
        Log an artifact (file).

        Args:
            local_path: Path to local file
            artifact_path: Optional remote artifact path
        """
        ...

    def set_tags(self, tags: dict[str, str]) -> None:
        """
        Set tags for the run.

        Args:
            tags: Dictionary of tag names and values
        """
        ...

    def log_dict(self, dictionary: dict[str, Any], artifact_file: str) -> None:
        """
        Log a dictionary as an artifact.

        Args:
            dictionary: Dictionary to log
            artifact_file: Artifact file name
        """
        ...
