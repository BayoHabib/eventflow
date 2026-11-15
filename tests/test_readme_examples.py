"""Tests that execute the README Quick Start examples (phase 0).

The README Quick Start was broken for 3 of 5 adapters (audit objective 3):
stale parameter names silently dropped by Pydantic, a wrong raster shape
with no error, and an invalid ``adjacency_type="grid"``. These tests run
the corrected snippets from the README against a synthetic frame with the
same columns the README assumes, and assert the shapes the README claims.
If the adapters change, the README must change with them — or this fails.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from eventflow.core.adapters import (
    GraphAdapter,
    GraphAdapterConfig,
    RasterAdapter,
    RasterAdapterConfig,
    SequenceAdapter,
    SequenceAdapterConfig,
    StreamAdapter,
    StreamAdapterConfig,
    TableAdapter,
    TableAdapterConfig,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _readme_frame() -> pl.DataFrame:
    """Synthetic frame with every column the README Quick Start uses."""
    base = datetime(2024, 1, 1)
    rows = []
    for d in range(3):
        for r in range(2):
            for c in range(2):
                rows.append(
                    {
                        "timestamp": base + timedelta(days=d),
                        "date": base + timedelta(days=d),
                        "grid_id": f"{r}_{c}",
                        "cell_id": f"{r}_{c}",
                        "event_count": 1 + d + r,
                        "arrest_count": d,
                        "lat_bin": r,
                        "lon_bin": c,
                        "day_of_week": (base + timedelta(days=d)).weekday(),
                        "total_events": 3 + d,
                        "centroid_lat": 41.88 + r * 0.01,
                        "centroid_lon": -87.62 + c * 0.01,
                        "latitude": 41.88 + r * 0.01,
                        "longitude": -87.62 + c * 0.01,
                        "crime_type": "THEFT",
                    }
                )
    return pl.DataFrame(rows)


def test_readme_table_adapter() -> None:
    """README §1: TableAdapter — X (12, 3), y (12,)."""
    df = _readme_frame()
    table_config = TableAdapterConfig(
        target_col="event_count",
        feature_cols=["lat_bin", "lon_bin", "day_of_week"],
    )
    table_output = TableAdapter(table_config).convert(df)
    x, y = table_output.get_X_y()
    assert x.shape == (12, 3)
    assert y.shape == (12,)


def test_readme_sequence_adapter() -> None:
    """README §2: SequenceAdapter — (4 cells, 30, 1) with masks (4, 30)."""
    df = _readme_frame()
    seq_config = SequenceAdapterConfig(
        spatial_col="cell_id",
        timestamp_col="date",
        feature_cols=["event_count"],
        sequence_length=30,
    )
    seq_output = SequenceAdapter(seq_config).convert(df)
    assert seq_output.sequences.shape == (4, 30, 1)
    assert seq_output.masks.shape == (4, 30)


def test_readme_raster_adapter() -> None:
    """README §3: RasterAdapter — (3 dates, 2 channels, 10, 10)."""
    df = _readme_frame()
    raster_config = RasterAdapterConfig(
        grid_col="grid_id",
        timestamp_col="date",
        feature_cols=["event_count", "arrest_count"],
        grid_shape=(10, 10),
    )
    raster_output = RasterAdapter(raster_config).convert(df)
    assert raster_output.raster.shape == (3, 2, 10, 10)


def test_readme_graph_adapter() -> None:
    """README §4: GraphAdapter — 4 nodes, 3 features, edge_index (2, E)."""
    df = _readme_frame()
    graph_config = GraphAdapterConfig(
        node_col="grid_id",
        feature_cols=["total_events", "centroid_lat", "centroid_lon"],
        adjacency_type="spatial",
    )
    graph_output = GraphAdapter(graph_config).convert(df)
    assert graph_output.node_features.shape == (4, 3)
    assert graph_output.edge_index.shape[0] == 2
    assert graph_output.edge_index.shape[1] > 0  # non-empty graph


def test_readme_stream_adapter() -> None:
    """README §5: StreamAdapter — 12 timestamps, inter-times, states (12, 2)."""
    df = _readme_frame()
    stream_config = StreamAdapterConfig(
        timestamp_col="timestamp",
        event_type_col="crime_type",
        state_cols=["latitude", "longitude"],
    )
    stream_output = StreamAdapter(stream_config).convert(df)
    assert stream_output.timestamps.shape == (12,)
    assert stream_output.inter_times.shape == (12,)
    assert stream_output.states.shape == (12, 2)


def test_readme_pipeline_steps_imports() -> None:
    """Every Step class named in the README Pipeline section must exist."""
    from eventflow.core.steps import (  # noqa: F401
        AssignToGridStep,
        CalendarEncodingStep,
        ConditionalIntensityStep,
        ContinuousInterEventStep,
        ExtractTemporalComponentsStep,
        GridAggregationStep,
        HawkesKernelStep,
        KDEStep,
        MovingAverageStep,
        SpatialLagStep,
        StreamingHawkesStep,
        StreamingInterEventStep,
        StreamingWindowStep,
        TemporalLagStep,
        TimeBinsStep,
        TransformCRSStep,
    )
    from eventflow.core.steps.validation import (  # noqa: F401
        validate_point_process_features,
        validate_temporal_ordering,
    )


def test_readme_has_no_stale_quick_start_names() -> None:
    """The Quick Start must not contain the stale parameter names."""
    text = (REPO_ROOT / "README.md").read_text()
    quick_start = text.split("## Quick Start")[1].split("## Pipeline Steps")[0]
    for stale in (
        "sequence_col",
        "max_length",
        "time_col",
        "lat_col",
        "lon_col",
        "value_cols",
        "node_id_col",
        'adjacency_type="grid"',
        "height=10",
        "width=10",
    ):
        assert stale not in quick_start, f"stale name still in Quick Start: {stale}"
    for current in (
        "spatial_col",
        "timestamp_col",
        "sequence_length",
        "grid_col",
        "grid_shape",
        "node_col",
        'adjacency_type="spatial"',
    ):
        assert current in quick_start, f"corrected name missing: {current}"


def test_readme_has_no_stale_test_counts() -> None:
    """The README must not repeat the stale 290 tests / 68% figures."""
    text = (REPO_ROOT / "README.md").read_text()
    assert "290 tests" not in text
    assert "68%" not in text
    assert "68% coverage" not in text
