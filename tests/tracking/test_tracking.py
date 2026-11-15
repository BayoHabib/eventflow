"""Tests for experiment tracking (phase 0).

``tracking`` is an optional seam: importing it must never require MLflow.
``MLflowTracker`` itself is EXPERIMENTAL (phase 0) — it has no test with a
real MLflow backend, and instantiating it without MLflow installed must
fail with a clear message.
"""

from __future__ import annotations

import pytest


def test_import_tracking_does_not_require_mlflow() -> None:
    """Importing eventflow.tracking works without mlflow installed."""
    pytest.importorskip("polars")  # the package itself needs polars
    import eventflow.tracking  # noqa: F401
    from eventflow.tracking import TrackerProtocol  # noqa: F401


def test_mlflow_tracker_requires_mlflow_installed() -> None:
    """Instantiating MLflowTracker without mlflow fails with a clear error."""
    pytest.importorskip("polars")
    try:
        import mlflow  # noqa: F401
    except ImportError:
        from eventflow.tracking.mlflow_tracker import MLflowTracker

        with pytest.raises(ImportError, match="MLflow is not installed"):
            MLflowTracker()
    else:
        pytest.skip("mlflow is installed; the no-mlflow error path is not exercised")
