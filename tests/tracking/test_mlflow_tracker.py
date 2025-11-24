"""Tests for MLflow tracker wrapper."""

import pathlib

import pytest

from eventflow.tracking.mlflow_tracker import MLflowTracker


class DummyRun:
    """Minimal MLflow run stub."""

    def __init__(self) -> None:
        self.params = {}
        self.metrics = []
        self.tags = {}
        self.artifacts = []
        self.ended = False

    def log_param(self, key: str, value) -> None:
        self.params[key] = value

    def log_metric(self, key: str, value, step=None) -> None:
        self.metrics.append((key, value, step))

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        self.artifacts.append((local_path, artifact_path))

    def set_tags(self, tags: dict[str, str]) -> None:
        self.tags.update(tags)

    def end_run(self) -> None:
        self.ended = True


class DummyMlflowModule:
    """Module-like stub that creates DummyRun objects."""

    def __init__(self) -> None:
        self.runs: list[DummyRun] = []

    def log_param(self, key, value):
        self.runs[-1].log_param(key, value)

    def log_metric(self, key, value, step=None):
        self.runs[-1].log_metric(key, value, step)

    def log_artifact(self, local_path, artifact_path=None):
        self.runs[-1].log_artifact(local_path, artifact_path)

    def set_tags(self, tags):
        self.runs[-1].set_tags(tags)

    def start_run(self, run_name=None):
        run = DummyRun()
        self.runs.append(run)
        return run

    def end_run(self):
        self.runs[-1].end_run()


@pytest.fixture
def tracker(monkeypatch) -> MLflowTracker:
    dummy = DummyMlflowModule()
    monkeypatch.setitem(__import__("sys").modules, "mlflow", dummy)
    t = MLflowTracker()
    # Start a run so subsequent log_* calls have a target
    t.start_run()
    return t


def test_logs_param_metric_tag_and_artifact(tracker: MLflowTracker, tmp_path: pathlib.Path) -> None:
    tracker.log_param("lr", 0.01)
    tracker.log_metric("acc", 0.9, step=1)
    tracker.set_tags({"stage": "dev"})

    # create temp file to log as artifact
    art = tmp_path / "file.txt"
    art.write_text("data")
    tracker.log_artifact(str(art), artifact_path="subdir")

    run = tracker.run
    assert run.params["lr"] == 0.01
    assert ("acc", 0.9, 1) in run.metrics
    assert run.tags["stage"] == "dev"
    assert run.artifacts == [(str(art), "subdir")]


def test_log_dict_and_end_run(tracker: MLflowTracker, tmp_path: pathlib.Path) -> None:
    tracker.log_dict({"a": 1}, "meta.json")
    tracker.end_run()

    run = tracker.run
    # log_dict writes a temp file as artifact; ensure it was logged
    assert any(path.endswith("meta.json") for path, _ in run.artifacts)
    assert run.ended is True
