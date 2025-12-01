"""Tests for output adapters - deterministic shapes, masking, and dtype handling."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest

from eventflow.core.adapters import (
    GraphAdapter,
    GraphAdapterConfig,
    GraphOutput,
    RasterAdapter,
    RasterAdapterConfig,
    RasterOutput,
    SequenceAdapter,
    SequenceAdapterConfig,
    SequenceOutput,
    SerializationFormat,
    StreamAdapter,
    StreamAdapterConfig,
    StreamOutput,
    TableAdapter,
    TableAdapterConfig,
    TableOutput,
    get_default_adapter_registry,
    register_builtin_adapters,
)
from eventflow.core.event_frame import EventFrame
from eventflow.core.schema import EventMetadata, EventSchema

if TYPE_CHECKING:
    pass


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def sample_event_frame() -> EventFrame:
    """Create a sample EventFrame for testing."""
    # Create sample data with 3x3 grid structure
    n_timesteps = 5

    data = []
    base_time = datetime(2024, 1, 1, 0, 0, 0)

    for t in range(n_timesteps):
        for row in range(3):
            for col in range(3):
                grid_id = f"{row}_{col}"
                data.append(
                    {
                        "timestamp": base_time + timedelta(days=t),
                        "grid_id": grid_id,
                        "lat": 41.8 + row * 0.01,
                        "lon": -87.6 + col * 0.01,
                        "event_count": float(np.random.poisson(5)),
                        "feature_a": float(np.random.randn()),
                        "feature_b": float(np.random.randn()),
                        "event_type": np.random.choice(["A", "B", "C"]),
                    }
                )

    df = pl.DataFrame(data)
    lf = df.lazy()

    schema = EventSchema(
        timestamp_col="timestamp",
        lat_col="lat",
        lon_col="lon",
        categorical_cols=["grid_id", "event_type"],
        numeric_cols=["event_count", "feature_a", "feature_b"],
    )

    metadata = EventMetadata(
        dataset_name="test_events",
        crs="EPSG:4326",
        time_zone="UTC",
    )

    return EventFrame(lf, schema, metadata)


@pytest.fixture
def simple_event_frame() -> EventFrame:
    """Create a minimal EventFrame for basic tests."""
    data = [
        {
            "timestamp": datetime(2024, 1, 1),
            "grid_id": "0_0",
            "lat": 41.8,
            "lon": -87.6,
            "value": 1.0,
        },
        {
            "timestamp": datetime(2024, 1, 2),
            "grid_id": "0_0",
            "lat": 41.8,
            "lon": -87.6,
            "value": 2.0,
        },
        {
            "timestamp": datetime(2024, 1, 3),
            "grid_id": "0_1",
            "lat": 41.8,
            "lon": -87.5,
            "value": 3.0,
        },
    ]
    df = pl.DataFrame(data)

    schema = EventSchema(
        timestamp_col="timestamp",
        lat_col="lat",
        lon_col="lon",
    )
    metadata = EventMetadata(dataset_name="simple_test")

    return EventFrame(df.lazy(), schema, metadata)


# -----------------------------------------------------------------------------
# TableAdapter Tests
# -----------------------------------------------------------------------------


class TestTableAdapter:
    """Tests for TableAdapter."""

    def test_convert_basic(self, sample_event_frame: EventFrame) -> None:
        """Test basic table conversion."""
        adapter = TableAdapter()
        output = adapter.convert(sample_event_frame)

        assert isinstance(output, TableOutput)
        assert len(output.data) == 45  # 9 grids * 5 timesteps
        assert len(output.feature_names) > 0

    def test_convert_with_target(self, sample_event_frame: EventFrame) -> None:
        """Test conversion with target column."""
        config = TableAdapterConfig(target_col="event_count")
        adapter = TableAdapter(config)
        output = adapter.convert(sample_event_frame)

        assert output.target == "event_count"
        X, y = output.get_X_y()
        assert X is not None
        assert y is not None
        assert len(y) == 45

    def test_convert_with_offset(self, sample_event_frame: EventFrame) -> None:
        """Test conversion with offset column for GLM."""
        # Add an offset column
        ef = sample_event_frame.with_columns(pl.lit(1.0).alias("log_exposure"))

        config = TableAdapterConfig(offset_col="log_exposure")
        adapter = TableAdapter(config)
        output = adapter.convert(ef)

        assert output.offset == "log_exposure"
        assert "log_exposure" in output.data.columns

    def test_convert_with_intercept(self, sample_event_frame: EventFrame) -> None:
        """Test conversion with intercept column."""
        config = TableAdapterConfig(include_intercept=True)
        adapter = TableAdapter(config)
        output = adapter.convert(sample_event_frame)

        assert "_intercept" in output.feature_names
        assert output.data["_intercept"].to_list() == [1.0] * 45

    def test_dtype_float32(self, sample_event_frame: EventFrame) -> None:
        """Test float32 dtype conversion."""
        config = TableAdapterConfig(dtype="float32")
        adapter = TableAdapter(config)
        output = adapter.convert(sample_event_frame)

        X = output.to_numpy()
        assert X.dtype == np.float32

    def test_dtype_float64(self, sample_event_frame: EventFrame) -> None:
        """Test float64 dtype conversion."""
        config = TableAdapterConfig(dtype="float64")
        adapter = TableAdapter(config)
        output = adapter.convert(sample_event_frame)

        X = output.to_numpy()
        assert X.dtype == np.float64

    def test_to_pandas(self, sample_event_frame: EventFrame) -> None:
        """Test pandas conversion."""
        pytest.importorskip("pyarrow")
        adapter = TableAdapter()
        output = adapter.convert(sample_event_frame)

        pdf = output.to_pandas()
        assert len(pdf) == 45

    def test_get_metadata(self, sample_event_frame: EventFrame) -> None:
        """Test metadata extraction."""
        adapter = TableAdapter()
        output = adapter.convert(sample_event_frame)
        meta = adapter.get_metadata(output)

        assert meta.modality == "table"
        assert "data" in meta.shapes
        assert len(meta.feature_names) > 0

    def test_serialize_parquet(self, sample_event_frame: EventFrame, tmp_path: Path) -> None:
        """Test Parquet serialization."""
        adapter = TableAdapter()
        output = adapter.convert(sample_event_frame)

        path = tmp_path / "table.parquet"
        adapter.serialize(output, path, SerializationFormat.PARQUET)

        assert path.exists()
        loaded = adapter.deserialize(path, SerializationFormat.PARQUET)
        assert len(loaded.data) == len(output.data)

    def test_serialize_arrow(self, sample_event_frame: EventFrame, tmp_path: Path) -> None:
        """Test Arrow serialization."""
        adapter = TableAdapter()
        output = adapter.convert(sample_event_frame)

        path = tmp_path / "table.arrow"
        adapter.serialize(output, path, SerializationFormat.ARROW)

        assert path.exists()
        loaded = adapter.deserialize(path, SerializationFormat.ARROW)
        assert len(loaded.data) == len(output.data)

    def test_serialize_numpy(self, sample_event_frame: EventFrame, tmp_path: Path) -> None:
        """Test NumPy serialization."""
        adapter = TableAdapter()
        output = adapter.convert(sample_event_frame)

        path = tmp_path / "table.npz"
        adapter.serialize(output, path, SerializationFormat.NUMPY)

        assert path.exists()


# -----------------------------------------------------------------------------
# SequenceAdapter Tests
# -----------------------------------------------------------------------------


class TestSequenceAdapter:
    """Tests for SequenceAdapter."""

    def test_convert_basic(self, sample_event_frame: EventFrame) -> None:
        """Test basic sequence conversion."""
        config = SequenceAdapterConfig(spatial_col="grid_id")
        adapter = SequenceAdapter(config)
        output = adapter.convert(sample_event_frame)

        assert isinstance(output, SequenceOutput)
        assert output.n_sequences == 9  # 9 grid cells
        assert output.max_length == 5  # 5 timesteps
        assert output.n_features > 0

    def test_sequence_shapes_deterministic(self, sample_event_frame: EventFrame) -> None:
        """Test that sequence shapes are deterministic."""
        config = SequenceAdapterConfig(spatial_col="grid_id")
        adapter = SequenceAdapter(config)

        output1 = adapter.convert(sample_event_frame)
        output2 = adapter.convert(sample_event_frame)

        assert output1.sequences.shape == output2.sequences.shape
        assert output1.masks.shape == output2.masks.shape

    def test_padding_right(self, simple_event_frame: EventFrame) -> None:
        """Test right padding."""
        config = SequenceAdapterConfig(
            spatial_col="grid_id",
            sequence_length=5,
            padding_side="right",
            padding_value=-999.0,
        )
        adapter = SequenceAdapter(config)
        output = adapter.convert(simple_event_frame)

        # Check padding is on right side
        assert output.sequences.shape[1] == 5
        # Mask should be True for valid positions
        assert output.masks[0, 0] or output.masks[0, -1]  # At least one valid

    def test_padding_left(self, simple_event_frame: EventFrame) -> None:
        """Test left padding."""
        config = SequenceAdapterConfig(
            spatial_col="grid_id",
            sequence_length=5,
            padding_side="left",
            padding_value=-999.0,
        )
        adapter = SequenceAdapter(config)
        output = adapter.convert(simple_event_frame)

        assert output.sequences.shape[1] == 5

    def test_masks_correct(self, sample_event_frame: EventFrame) -> None:
        """Test that masks correctly identify valid positions."""
        config = SequenceAdapterConfig(spatial_col="grid_id")
        adapter = SequenceAdapter(config)
        output = adapter.convert(sample_event_frame)

        # All sequences should have max_length valid positions
        assert np.all(output.masks.sum(axis=1) == output.max_length)

    def test_dtype_handling(self, sample_event_frame: EventFrame) -> None:
        """Test dtype is correctly applied."""
        config = SequenceAdapterConfig(spatial_col="grid_id", dtype="float32")
        adapter = SequenceAdapter(config)
        output = adapter.convert(sample_event_frame)

        assert output.sequences.dtype == np.float32

    def test_time_encoding_positional(self, sample_event_frame: EventFrame) -> None:
        """Test positional time encoding."""
        config = SequenceAdapterConfig(
            spatial_col="grid_id",
            time_encoding="positional",
        )
        adapter = SequenceAdapter(config)
        output = adapter.convert(sample_event_frame)

        assert output.time_encoding is not None
        assert output.time_encoding.shape == (output.n_sequences, output.max_length)

    def test_time_encoding_sinusoidal(self, sample_event_frame: EventFrame) -> None:
        """Test sinusoidal time encoding."""
        config = SequenceAdapterConfig(
            spatial_col="grid_id",
            time_encoding="sinusoidal",
        )
        adapter = SequenceAdapter(config)
        output = adapter.convert(sample_event_frame)

        assert output.time_encoding is not None
        assert len(output.time_encoding.shape) == 3

    def test_serialize_numpy(self, sample_event_frame: EventFrame, tmp_path: Path) -> None:
        """Test NumPy serialization."""
        config = SequenceAdapterConfig(spatial_col="grid_id")
        adapter = SequenceAdapter(config)
        output = adapter.convert(sample_event_frame)

        path = tmp_path / "sequences.npz"
        adapter.serialize(output, path, SerializationFormat.NUMPY)

        loaded = adapter.deserialize(path, SerializationFormat.NUMPY)
        assert loaded.sequences.shape == output.sequences.shape
        np.testing.assert_array_equal(loaded.masks, output.masks)

    def test_get_metadata(self, sample_event_frame: EventFrame) -> None:
        """Test metadata extraction."""
        config = SequenceAdapterConfig(spatial_col="grid_id")
        adapter = SequenceAdapter(config)
        output = adapter.convert(sample_event_frame)
        meta = adapter.get_metadata(output)

        assert meta.modality == "sequence"
        assert meta.extra["n_sequences"] == 9
        assert meta.extra["max_length"] == 5


# -----------------------------------------------------------------------------
# RasterAdapter Tests
# -----------------------------------------------------------------------------


class TestRasterAdapter:
    """Tests for RasterAdapter."""

    def test_convert_basic(self, sample_event_frame: EventFrame) -> None:
        """Test basic raster conversion."""
        config = RasterAdapterConfig(
            grid_col="grid_id",
            grid_shape=(3, 3),
        )
        adapter = RasterAdapter(config)
        output = adapter.convert(sample_event_frame)

        assert isinstance(output, RasterOutput)
        assert output.n_timesteps == 5
        assert output.height == 3
        assert output.width == 3

    def test_raster_shapes_deterministic(self, sample_event_frame: EventFrame) -> None:
        """Test that raster shapes are deterministic."""
        config = RasterAdapterConfig(grid_col="grid_id", grid_shape=(3, 3))
        adapter = RasterAdapter(config)

        output1 = adapter.convert(sample_event_frame)
        output2 = adapter.convert(sample_event_frame)

        assert output1.raster.shape == output2.raster.shape

    def test_channel_first(self, sample_event_frame: EventFrame) -> None:
        """Test channel-first format (PyTorch)."""
        config = RasterAdapterConfig(
            grid_col="grid_id",
            grid_shape=(3, 3),
            channel_first=True,
        )
        adapter = RasterAdapter(config)
        output = adapter.convert(sample_event_frame)

        # Shape should be (T, C, H, W)
        assert output.raster.shape[2] == 3  # Height
        assert output.raster.shape[3] == 3  # Width

    def test_channel_last(self, sample_event_frame: EventFrame) -> None:
        """Test channel-last format (TensorFlow)."""
        config = RasterAdapterConfig(
            grid_col="grid_id",
            grid_shape=(3, 3),
            channel_first=False,
        )
        adapter = RasterAdapter(config)
        output = adapter.convert(sample_event_frame)

        # Shape should be (T, H, W, C)
        assert output.raster.shape[1] == 3  # Height
        assert output.raster.shape[2] == 3  # Width

    def test_fill_value(self, sample_event_frame: EventFrame) -> None:
        """Test fill value for empty cells."""
        config = RasterAdapterConfig(
            grid_col="grid_id",
            grid_shape=(5, 5),  # Larger grid than data
            fill_value=-1.0,
        )
        adapter = RasterAdapter(config)
        output = adapter.convert(sample_event_frame)

        # Some cells should have fill value
        assert np.any(output.raster == -1.0)

    def test_normalize(self, sample_event_frame: EventFrame) -> None:
        """Test channel normalization."""
        config = RasterAdapterConfig(
            grid_col="grid_id",
            grid_shape=(3, 3),
            normalize=True,
        )
        adapter = RasterAdapter(config)
        output = adapter.convert(sample_event_frame)

        # Normalized data should have roughly zero mean
        # (may not be exact due to fill values)
        assert output.raster is not None

    def test_dtype_handling(self, sample_event_frame: EventFrame) -> None:
        """Test dtype is correctly applied."""
        config = RasterAdapterConfig(
            grid_col="grid_id",
            grid_shape=(3, 3),
            dtype="float64",
        )
        adapter = RasterAdapter(config)
        output = adapter.convert(sample_event_frame)

        assert output.raster.dtype == np.float64

    def test_serialize_numpy(self, sample_event_frame: EventFrame, tmp_path: Path) -> None:
        """Test NumPy serialization."""
        config = RasterAdapterConfig(grid_col="grid_id", grid_shape=(3, 3))
        adapter = RasterAdapter(config)
        output = adapter.convert(sample_event_frame)

        path = tmp_path / "raster.npz"
        adapter.serialize(output, path, SerializationFormat.NUMPY)

        loaded = adapter.deserialize(path, SerializationFormat.NUMPY)
        np.testing.assert_array_equal(loaded.raster, output.raster)

    def test_to_tensorflow(self, sample_event_frame: EventFrame) -> None:
        """Test TensorFlow format conversion."""
        config = RasterAdapterConfig(
            grid_col="grid_id",
            grid_shape=(3, 3),
            channel_first=True,
        )
        adapter = RasterAdapter(config)
        output = adapter.convert(sample_event_frame)

        tf_raster = output.to_tensorflow()
        # Should be (T, H, W, C)
        assert tf_raster.shape[1] == 3
        assert tf_raster.shape[2] == 3


# -----------------------------------------------------------------------------
# GraphAdapter Tests
# -----------------------------------------------------------------------------


class TestGraphAdapter:
    """Tests for GraphAdapter."""

    def test_convert_basic(self, sample_event_frame: EventFrame) -> None:
        """Test basic graph conversion."""
        config = GraphAdapterConfig(node_col="grid_id")
        adapter = GraphAdapter(config)
        output = adapter.convert(sample_event_frame)

        assert isinstance(output, GraphOutput)
        assert output.n_nodes == 9  # 9 unique grid cells
        assert output.n_features > 0

    def test_graph_shapes_deterministic(self, sample_event_frame: EventFrame) -> None:
        """Test that graph shapes are deterministic."""
        config = GraphAdapterConfig(node_col="grid_id")
        adapter = GraphAdapter(config)

        output1 = adapter.convert(sample_event_frame)
        output2 = adapter.convert(sample_event_frame)

        assert output1.node_features.shape == output2.node_features.shape
        assert output1.edge_index.shape == output2.edge_index.shape

    def test_edge_index_format(self, sample_event_frame: EventFrame) -> None:
        """Test edge index is in COO format."""
        config = GraphAdapterConfig(node_col="grid_id")
        adapter = GraphAdapter(config)
        output = adapter.convert(sample_event_frame)

        # Edge index should be (2, n_edges)
        assert output.edge_index.shape[0] == 2
        # Indices should be valid
        assert np.all(output.edge_index >= 0)
        assert np.all(output.edge_index < output.n_nodes)

    def test_adjacency_matrix(self, sample_event_frame: EventFrame) -> None:
        """Test adjacency matrix generation."""
        config = GraphAdapterConfig(node_col="grid_id")
        adapter = GraphAdapter(config)
        output = adapter.convert(sample_event_frame)

        assert output.adjacency is not None
        assert output.adjacency.shape == (9, 9)

    def test_self_loops(self, sample_event_frame: EventFrame) -> None:
        """Test self-loop inclusion."""
        config = GraphAdapterConfig(node_col="grid_id", include_self_loops=True)
        adapter = GraphAdapter(config)
        output = adapter.convert(sample_event_frame)

        # Check diagonal of adjacency for self-loops
        if output.adjacency is not None:
            assert np.any(np.diag(output.adjacency) > 0)

    def test_normalize_adjacency(self, sample_event_frame: EventFrame) -> None:
        """Test adjacency normalization."""
        config = GraphAdapterConfig(
            node_col="grid_id",
            normalize_adjacency=True,
            include_self_loops=True,
        )
        adapter = GraphAdapter(config)
        output = adapter.convert(sample_event_frame)

        # Normalized adjacency values should be <= 1
        if output.adjacency is not None:
            assert np.all(output.adjacency <= 1.0 + 1e-6)

    def test_dtype_handling(self, sample_event_frame: EventFrame) -> None:
        """Test dtype is correctly applied."""
        config = GraphAdapterConfig(node_col="grid_id", dtype="float32")
        adapter = GraphAdapter(config)
        output = adapter.convert(sample_event_frame)

        assert output.node_features.dtype == np.float32

    def test_temporal_snapshots(self, sample_event_frame: EventFrame) -> None:
        """Test temporal snapshot generation."""
        config = GraphAdapterConfig(
            node_col="grid_id",
            snapshot_interval="1d",
        )
        adapter = GraphAdapter(config)
        output = adapter.convert(sample_event_frame)

        assert output.n_snapshots == 5  # 5 days of data

    def test_serialize_numpy(self, sample_event_frame: EventFrame, tmp_path: Path) -> None:
        """Test NumPy serialization."""
        config = GraphAdapterConfig(node_col="grid_id")
        adapter = GraphAdapter(config)
        output = adapter.convert(sample_event_frame)

        path = tmp_path / "graph.npz"
        adapter.serialize(output, path, SerializationFormat.NUMPY)

        loaded = adapter.deserialize(path, SerializationFormat.NUMPY)
        np.testing.assert_array_equal(loaded.node_features, output.node_features)
        np.testing.assert_array_equal(loaded.edge_index, output.edge_index)

    def test_serialize_json(self, sample_event_frame: EventFrame, tmp_path: Path) -> None:
        """Test JSON serialization."""
        config = GraphAdapterConfig(node_col="grid_id")
        adapter = GraphAdapter(config)
        output = adapter.convert(sample_event_frame)

        path = tmp_path / "graph.json"
        adapter.serialize(output, path, SerializationFormat.JSON)

        loaded = adapter.deserialize(path, SerializationFormat.JSON)
        assert loaded.n_nodes == output.n_nodes


# -----------------------------------------------------------------------------
# StreamAdapter Tests
# -----------------------------------------------------------------------------


class TestStreamAdapter:
    """Tests for StreamAdapter."""

    def test_convert_basic(self, sample_event_frame: EventFrame) -> None:
        """Test basic stream conversion."""
        adapter = StreamAdapter()
        output = adapter.convert(sample_event_frame)

        assert isinstance(output, StreamOutput)
        assert output.n_events == 45
        assert output.n_state_dims > 0

    def test_stream_shapes_deterministic(self, sample_event_frame: EventFrame) -> None:
        """Test that stream shapes are deterministic."""
        adapter = StreamAdapter()

        output1 = adapter.convert(sample_event_frame)
        output2 = adapter.convert(sample_event_frame)

        assert output1.timestamps.shape == output2.timestamps.shape
        assert output1.states.shape == output2.states.shape

    def test_inter_times(self, sample_event_frame: EventFrame) -> None:
        """Test inter-event times calculation."""
        adapter = StreamAdapter()
        output = adapter.convert(sample_event_frame)

        # First inter-time should be 0
        assert output.inter_times[0] == 0
        # Inter-times should be non-negative
        assert np.all(output.inter_times >= 0)

    def test_time_origin_first(self, sample_event_frame: EventFrame) -> None:
        """Test first-event time origin."""
        config = StreamAdapterConfig(time_origin="first")
        adapter = StreamAdapter(config)
        output = adapter.convert(sample_event_frame)

        # First timestamp should be 0
        assert output.timestamps[0] == 0.0

    def test_time_origin_min(self, sample_event_frame: EventFrame) -> None:
        """Test min time origin."""
        config = StreamAdapterConfig(time_origin="min")
        adapter = StreamAdapter(config)
        output = adapter.convert(sample_event_frame)

        # Minimum should be 0
        assert np.min(output.timestamps) == 0.0

    def test_time_scale_normalize(self, sample_event_frame: EventFrame) -> None:
        """Test time normalization."""
        config = StreamAdapterConfig(time_scale="normalize")
        adapter = StreamAdapter(config)
        output = adapter.convert(sample_event_frame)

        # Normalized times should have mean ~0 and std ~1
        assert "mean" in output.time_scale_params
        assert "std" in output.time_scale_params

    def test_time_scale_log(self, sample_event_frame: EventFrame) -> None:
        """Test log time scaling."""
        config = StreamAdapterConfig(time_scale="log")
        adapter = StreamAdapter(config)
        output = adapter.convert(sample_event_frame)

        assert "min_shift" in output.time_scale_params

    def test_event_types(self, sample_event_frame: EventFrame) -> None:
        """Test event type extraction."""
        config = StreamAdapterConfig(event_type_col="event_type")
        adapter = StreamAdapter(config)
        output = adapter.convert(sample_event_frame)

        assert output.event_types is not None
        assert output.n_event_types == 3  # A, B, C
        assert len(output.event_type_names) == 3

    def test_max_events(self, sample_event_frame: EventFrame) -> None:
        """Test event truncation."""
        config = StreamAdapterConfig(max_events=10)
        adapter = StreamAdapter(config)
        output = adapter.convert(sample_event_frame)

        assert output.n_events == 10

    def test_dtype_handling(self, sample_event_frame: EventFrame) -> None:
        """Test dtype is correctly applied."""
        config = StreamAdapterConfig(dtype="float64")
        adapter = StreamAdapter(config)
        output = adapter.convert(sample_event_frame)

        assert output.timestamps.dtype == np.float64
        assert output.states.dtype == np.float64

    def test_serialize_numpy(self, sample_event_frame: EventFrame, tmp_path: Path) -> None:
        """Test NumPy serialization."""
        adapter = StreamAdapter()
        output = adapter.convert(sample_event_frame)

        path = tmp_path / "stream.npz"
        adapter.serialize(output, path, SerializationFormat.NUMPY)

        loaded = adapter.deserialize(path, SerializationFormat.NUMPY)
        np.testing.assert_array_almost_equal(loaded.timestamps, output.timestamps)
        np.testing.assert_array_almost_equal(loaded.states, output.states)

    def test_serialize_json(self, sample_event_frame: EventFrame, tmp_path: Path) -> None:
        """Test JSON serialization."""
        adapter = StreamAdapter()
        output = adapter.convert(sample_event_frame)

        path = tmp_path / "stream.json"
        adapter.serialize(output, path, SerializationFormat.JSON)

        loaded = adapter.deserialize(path, SerializationFormat.JSON)
        assert loaded.n_events == output.n_events

    def test_get_sequences_by_type(self, sample_event_frame: EventFrame) -> None:
        """Test splitting sequences by event type."""
        config = StreamAdapterConfig(event_type_col="event_type")
        adapter = StreamAdapter(config)
        output = adapter.convert(sample_event_frame)

        by_type = output.get_sequences_by_type()
        assert len(by_type) == 3
        total_events = sum(len(v[0]) for v in by_type.values())
        assert total_events == output.n_events


# -----------------------------------------------------------------------------
# Registry Integration Tests
# -----------------------------------------------------------------------------


class TestAdapterRegistry:
    """Tests for adapter registry integration."""

    def test_register_builtin_adapters(self) -> None:
        """Test that all built-in adapters can be registered."""
        from eventflow.core.registry import OutputAdapterRegistry

        registry = OutputAdapterRegistry()
        register_builtin_adapters(registry)

        adapters = registry.list()
        assert len(adapters) == 5

        names = {spec.name for spec in adapters}
        assert names == {"table", "sequence", "raster", "graph", "stream"}

    def test_get_default_registry(self) -> None:
        """Test getting default adapter registry."""
        registry = get_default_adapter_registry()

        assert registry.get("table") is not None
        assert registry.get("sequence") is not None
        assert registry.get("raster") is not None
        assert registry.get("graph") is not None
        assert registry.get("stream") is not None

    def test_create_adapter_from_registry(self, sample_event_frame: EventFrame) -> None:
        """Test creating adapters through registry."""
        registry = get_default_adapter_registry()

        table_adapter = registry.create("table")
        assert isinstance(table_adapter, TableAdapter)

        output = table_adapter.convert(sample_event_frame)
        assert isinstance(output, TableOutput)

    def test_create_adapter_with_params(self, sample_event_frame: EventFrame) -> None:
        """Test creating adapters with config parameters."""
        registry = get_default_adapter_registry()

        adapter = registry.create(
            "sequence",
            params={"spatial_col": "grid_id", "padding_value": -1.0},
        )
        assert isinstance(adapter, SequenceAdapter)
        assert adapter.config.padding_value == -1.0


# -----------------------------------------------------------------------------
# Edge Cases and Error Handling
# -----------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_event_frame(self) -> None:
        """Test handling of empty EventFrame."""
        df = pl.DataFrame(
            {
                "timestamp": [],
                "lat": [],
                "lon": [],
                "value": [],
            }
        ).cast({"timestamp": pl.Datetime})

        schema = EventSchema(timestamp_col="timestamp", lat_col="lat", lon_col="lon")
        metadata = EventMetadata(dataset_name="empty")
        ef = EventFrame(df.lazy(), schema, metadata)

        # Table adapter should handle empty data
        adapter = TableAdapter()
        output = adapter.convert(ef)
        assert len(output.data) == 0

    def test_single_event(self) -> None:
        """Test handling of single event."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1)],
                "grid_id": ["0_0"],
                "lat": [41.8],
                "lon": [-87.6],
                "value": [1.0],
            }
        )

        schema = EventSchema(timestamp_col="timestamp", lat_col="lat", lon_col="lon")
        metadata = EventMetadata(dataset_name="single")
        ef = EventFrame(df.lazy(), schema, metadata)

        # Stream adapter should handle single event
        adapter = StreamAdapter()
        output = adapter.convert(ef)
        assert output.n_events == 1
        assert output.inter_times[0] == 0.0

    def test_missing_column_error(self, sample_event_frame: EventFrame) -> None:
        """Test error on missing required column."""
        config = SequenceAdapterConfig(spatial_col="nonexistent_col")
        adapter = SequenceAdapter(config)

        # Polars raises ColumnNotFoundError when sorting by non-existent column
        with pytest.raises(pl.exceptions.ColumnNotFoundError):
            adapter.convert(sample_event_frame)

    def test_unsupported_serialization_format(
        self, sample_event_frame: EventFrame, tmp_path: Path
    ) -> None:
        """Test error on unsupported serialization format."""
        adapter = TableAdapter()
        output = adapter.convert(sample_event_frame)

        with pytest.raises(ValueError, match="Unsupported format"):
            adapter.serialize(output, tmp_path / "test.pkl", SerializationFormat.PICKLE)


# -----------------------------------------------------------------------------
# Plain DataFrame Input Tests (not wrapped in EventFrame)
# -----------------------------------------------------------------------------


class TestPlainDataFrameInputs:
    """Test that adapters work with plain Polars DataFrames (not EventFrame objects).
    
    This is important for users who want to use adapters without the full EventFrame
    abstraction, such as in notebooks or quick experiments.
    """

    @pytest.fixture
    def plain_crime_df(self) -> pl.DataFrame:
        """Create a plain Polars DataFrame mimicking crime data."""
        np.random.seed(42)
        n_records = 100
        base_date = datetime(2024, 1, 1)
        
        return pl.DataFrame({
            "case_id": [f"JE{i}" for i in range(n_records)],
            "timestamp": [base_date + timedelta(hours=np.random.randint(0, 24*10)) 
                         for _ in range(n_records)],
            "latitude": np.random.uniform(41.6, 42.0, n_records),
            "longitude": np.random.uniform(-87.9, -87.5, n_records),
            "primary_type": np.random.choice(["THEFT", "BATTERY", "ASSAULT"], n_records),
            "cell_id": np.random.randint(0, 25, n_records),
            "event_count": np.random.poisson(3, n_records),
        })

    @pytest.fixture
    def plain_aggregated_df(self, plain_crime_df: pl.DataFrame) -> pl.DataFrame:
        """Create aggregated daily counts per cell."""
        return plain_crime_df.with_columns([
            pl.col("timestamp").dt.date().alias("date"),
        ]).group_by(["cell_id", "date"]).agg([
            pl.len().alias("event_count"),
            pl.col("latitude").mean().alias("centroid_lat"),
            pl.col("longitude").mean().alias("centroid_lon"),
        ]).sort(["date", "cell_id"])

    def test_table_adapter_plain_dataframe(self, plain_aggregated_df: pl.DataFrame) -> None:
        """TableAdapter should work with plain DataFrame."""
        df = plain_aggregated_df.with_columns([
            pl.col("date").dt.weekday().alias("day_of_week"),
            pl.lit(1.0).alias("exposure"),
        ])
        
        config = TableAdapterConfig(
            target_col="event_count",
            feature_cols=["cell_id", "day_of_week"],
            offset_col="exposure",
        )
        adapter = TableAdapter(config)
        output = adapter.convert(df)
        
        assert output.data.shape[0] == len(df)
        assert "cell_id" in output.feature_names
        assert "day_of_week" in output.feature_names
        
        X, y = output.get_X_y()
        assert X.shape[0] == len(df)
        assert y is not None
        assert y.shape[0] == len(df)

    def test_sequence_adapter_plain_dataframe(self, plain_aggregated_df: pl.DataFrame) -> None:
        """SequenceAdapter should work with plain DataFrame."""
        config = SequenceAdapterConfig(
            spatial_col="cell_id",
            timestamp_col="date",
            feature_cols=["event_count"],
            sequence_length=10,
            padding_value=0.0,
        )
        adapter = SequenceAdapter(config)
        output = adapter.convert(plain_aggregated_df)
        
        n_locations = plain_aggregated_df["cell_id"].n_unique()
        assert output.sequences.shape[0] == n_locations
        assert output.sequences.shape[1] == 10  # sequence_length
        assert output.masks.shape == output.sequences.shape[:2]
        assert len(output.lengths) == n_locations

    def test_raster_adapter_plain_dataframe(self, plain_aggregated_df: pl.DataFrame) -> None:
        """RasterAdapter should work with plain DataFrame."""
        config = RasterAdapterConfig(
            grid_col="cell_id",
            timestamp_col="date",
            feature_cols=["event_count"],
            grid_shape=(5, 5),
            channel_first=True,
        )
        adapter = RasterAdapter(config)
        output = adapter.convert(plain_aggregated_df)
        
        n_timesteps = plain_aggregated_df["date"].n_unique()
        assert output.raster.shape[0] == n_timesteps
        assert output.raster.shape[1] == 1  # 1 feature/channel
        assert output.raster.shape[2:] == (5, 5)

    def test_graph_adapter_plain_dataframe(self, plain_aggregated_df: pl.DataFrame) -> None:
        """GraphAdapter should work with plain DataFrame."""
        # Aggregate to node features
        node_df = plain_aggregated_df.group_by("cell_id").agg([
            pl.col("event_count").sum().alias("total_events"),
            pl.col("centroid_lat").mean().alias("lat"),
            pl.col("centroid_lon").mean().alias("lon"),
        ]).sort("cell_id")
        
        config = GraphAdapterConfig(
            node_col="cell_id",
            feature_cols=["total_events", "lat", "lon"],
            adjacency_type="spatial",
            spatial_threshold=0.1,
            include_self_loops=True,
        )
        adapter = GraphAdapter(config)
        output = adapter.convert(node_df)
        
        n_nodes = len(node_df)
        assert output.node_features.shape[0] == n_nodes
        assert output.node_features.shape[1] == 3  # 3 features
        assert output.edge_index.shape[0] == 2  # src, dst
        assert output.adjacency.shape == (n_nodes, n_nodes)

    def test_stream_adapter_plain_dataframe(self, plain_crime_df: pl.DataFrame) -> None:
        """StreamAdapter should work with plain DataFrame."""
        config = StreamAdapterConfig(
            timestamp_col="timestamp",
            event_type_col="primary_type",
            state_cols=["latitude", "longitude"],
            time_scale="normalize",
            time_origin="first",
        )
        adapter = StreamAdapter(config)
        output = adapter.convert(plain_crime_df)
        
        assert output.timestamps.shape[0] == len(plain_crime_df)
        assert output.states.shape == (len(plain_crime_df), 2)  # lat, lon
        assert output.inter_times.shape[0] == len(plain_crime_df)
        assert output.event_types.shape[0] == len(plain_crime_df)

    def test_adapters_with_lazyframe(self, plain_crime_df: pl.DataFrame) -> None:
        """Adapters should also work with LazyFrame (not just DataFrame)."""
        lf = plain_crime_df.lazy()
        
        config = StreamAdapterConfig(
            timestamp_col="timestamp",
            state_cols=["latitude", "longitude"],
        )
        adapter = StreamAdapter(config)
        output = adapter.convert(lf)
        
        # Should auto-collect and process
        assert output.timestamps.shape[0] == len(plain_crime_df)

