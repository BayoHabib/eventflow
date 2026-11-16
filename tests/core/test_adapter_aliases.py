"""Tests for Phase 5 adapter naming aliases.

Proves that the new alias names return the identical arrays as the original
field names, that aliases survive serialization round-trips (including files
written before the aliases existed), and that the uniform ``.data`` accessor
behaves as documented in ``docs/design-notes/phase5-adapter-naming.md``.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from eventflow.core.adapters import (
    GraphAdapter,
    GraphOutput,
    RasterOutput,
    SequenceAdapter,
    SequenceOutput,
    SerializationFormat,
    StreamAdapter,
    StreamOutput,
    TableOutput,
)

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def sequence_output() -> SequenceOutput:
    """A small hand-built SequenceOutput."""
    return SequenceOutput(
        sequences=np.arange(24, dtype=np.float32).reshape(2, 3, 4),
        masks=np.array([[True, True, False], [True, True, True]]),
        sequence_ids=["a", "b"],
        feature_names=["f0", "f1", "f2", "f3"],
        lengths=np.array([2, 3]),
        dtypes={"sequences": "float32", "masks": "bool", "attention_mask": "bool"},
    )


@pytest.fixture
def stream_output() -> StreamOutput:
    """A small hand-built StreamOutput."""
    return StreamOutput(
        timestamps=np.array([0.0, 1.0, 2.5]),
        states=np.arange(6, dtype=np.float64).reshape(3, 2),
        inter_times=np.array([0.0, 1.0, 1.5]),
        state_names=["s0", "s1"],
        dtypes={"timestamps": "float64", "states": "float64", "state_features": "float64"},
    )


# -----------------------------------------------------------------------------
# SequenceOutput aliases
# -----------------------------------------------------------------------------


class TestSequenceAliases:
    """`SequenceOutput.masks` <-> `attention_mask`."""

    def test_attention_mask_is_masks(self, sequence_output: SequenceOutput) -> None:
        """Alias returns the identical array object, not a copy."""
        assert sequence_output.attention_mask is sequence_output.masks
        np.testing.assert_array_equal(sequence_output.attention_mask, sequence_output.masks)

    def test_attention_mask_read_only(self, sequence_output: SequenceOutput) -> None:
        """Aliases are read-only properties."""
        with pytest.raises(AttributeError):
            sequence_output.attention_mask = np.zeros((2, 3), dtype=bool)

    def test_metadata_carries_both_keys(self, sequence_output: SequenceOutput) -> None:
        """Metadata shapes/dtypes expose old and new keys side by side."""
        meta = SequenceAdapter().get_metadata(sequence_output)
        assert meta.shapes["masks"] == meta.shapes["attention_mask"] == (2, 3)
        assert meta.dtypes["masks"] == meta.dtypes["attention_mask"] == "bool"

    def test_to_torch_carries_both_keys(self, sequence_output: SequenceOutput) -> None:
        """to_torch() dict exposes old and new keys side by side."""
        pytest.importorskip("torch")
        import torch

        tensors = sequence_output.to_torch()
        assert isinstance(tensors["masks"], torch.Tensor)
        assert isinstance(tensors["attention_mask"], torch.Tensor)
        assert torch.equal(tensors["masks"], tensors["attention_mask"])

    def test_numpy_round_trip(self, sequence_output: SequenceOutput, tmp_path: Path) -> None:
        """Aliases survive a NumPy round-trip and stay identical to the originals."""
        adapter = SequenceAdapter()
        path = tmp_path / "seq.npz"
        adapter.serialize(sequence_output, path, SerializationFormat.NUMPY)

        stored = np.load(path, allow_pickle=True)
        assert "masks" in stored and "attention_mask" in stored

        loaded = adapter.deserialize(path, SerializationFormat.NUMPY)
        assert loaded.attention_mask is loaded.masks
        np.testing.assert_array_equal(loaded.attention_mask, sequence_output.masks)
        np.testing.assert_array_equal(loaded.sequences, sequence_output.sequences)

    def test_numpy_round_trip_pre_alias_file(
        self, sequence_output: SequenceOutput, tmp_path: Path
    ) -> None:
        """A file written with only the old key still loads and the alias works."""
        path = tmp_path / "seq_old.npz"
        np.savez(
            path,
            sequences=sequence_output.sequences,
            masks=sequence_output.masks,
            lengths=sequence_output.lengths,
            feature_names=np.array(sequence_output.feature_names),
            sequence_ids=np.array(sequence_output.sequence_ids),
        )
        loaded = SequenceAdapter().deserialize(path, SerializationFormat.NUMPY)
        assert loaded.attention_mask is loaded.masks
        np.testing.assert_array_equal(loaded.attention_mask, sequence_output.masks)


# -----------------------------------------------------------------------------
# StreamOutput aliases
# -----------------------------------------------------------------------------


class TestStreamAliases:
    """`StreamOutput.inter_times` <-> `inter_event_times`,
    `StreamOutput.states` <-> `state_features`."""

    def test_aliases_are_same_objects(self, stream_output: StreamOutput) -> None:
        """Aliases return the identical array objects, not copies."""
        assert stream_output.inter_event_times is stream_output.inter_times
        assert stream_output.state_features is stream_output.states
        np.testing.assert_array_equal(stream_output.inter_event_times, stream_output.inter_times)
        np.testing.assert_array_equal(stream_output.state_features, stream_output.states)

    def test_aliases_read_only(self, stream_output: StreamOutput) -> None:
        """Aliases are read-only properties."""
        with pytest.raises(AttributeError):
            stream_output.inter_event_times = np.zeros(3)
        with pytest.raises(AttributeError):
            stream_output.state_features = np.zeros((3, 2))

    def test_metadata_carries_both_keys(self, stream_output: StreamOutput) -> None:
        """Metadata shapes/dtypes expose old and new keys side by side."""
        meta = StreamAdapter().get_metadata(stream_output)
        assert meta.shapes["states"] == meta.shapes["state_features"] == (3, 2)
        assert meta.shapes["inter_times"] == meta.shapes["inter_event_times"] == (3,)
        assert meta.dtypes["states"] == meta.dtypes["state_features"] == "float64"

    def test_to_torch_carries_both_keys(self, stream_output: StreamOutput) -> None:
        """to_torch() dict exposes old and new keys side by side."""
        pytest.importorskip("torch")
        import torch

        tensors = stream_output.to_torch()
        assert torch.equal(tensors["states"], tensors["state_features"])
        assert torch.equal(tensors["inter_times"], tensors["inter_event_times"])

    def test_numpy_round_trip(self, stream_output: StreamOutput, tmp_path: Path) -> None:
        """Aliases survive a NumPy round-trip and stay identical to the originals."""
        adapter = StreamAdapter()
        path = tmp_path / "stream.npz"
        adapter.serialize(stream_output, path, SerializationFormat.NUMPY)

        stored = np.load(path, allow_pickle=True)
        assert "states" in stored and "state_features" in stored
        assert "inter_times" in stored and "inter_event_times" in stored

        loaded = adapter.deserialize(path, SerializationFormat.NUMPY)
        assert loaded.state_features is loaded.states
        assert loaded.inter_event_times is loaded.inter_times
        np.testing.assert_array_equal(loaded.state_features, stream_output.states)
        np.testing.assert_array_equal(loaded.inter_event_times, stream_output.inter_times)

    def test_numpy_round_trip_pre_alias_file(
        self, stream_output: StreamOutput, tmp_path: Path
    ) -> None:
        """A file written with only the old keys still loads and the aliases work."""
        path = tmp_path / "stream_old.npz"
        np.savez(
            path,
            timestamps=stream_output.timestamps,
            states=stream_output.states,
            inter_times=stream_output.inter_times,
            origin=np.array(stream_output.origin),
            state_names=np.array(stream_output.state_names),
        )
        loaded = StreamAdapter().deserialize(path, SerializationFormat.NUMPY)
        assert loaded.state_features is loaded.states
        assert loaded.inter_event_times is loaded.inter_times
        np.testing.assert_array_equal(loaded.states, stream_output.states)

    def test_json_round_trip(self, stream_output: StreamOutput, tmp_path: Path) -> None:
        """Aliases survive a JSON round-trip (new keys stored alongside old ones)."""
        adapter = StreamAdapter()
        path = tmp_path / "stream.json"
        adapter.serialize(stream_output, path, SerializationFormat.JSON)

        raw = json.loads(path.read_text(encoding="utf-8"))
        assert raw["states"] == raw["state_features"]
        assert raw["inter_times"] == raw["inter_event_times"]

        loaded = adapter.deserialize(path, SerializationFormat.JSON)
        np.testing.assert_array_equal(loaded.state_features, stream_output.states)
        np.testing.assert_array_equal(loaded.inter_event_times, stream_output.inter_times)

    def test_json_round_trip_pre_alias_file(
        self, stream_output: StreamOutput, tmp_path: Path
    ) -> None:
        """A JSON file with only the old keys still loads and the aliases work."""
        path = tmp_path / "stream_old.json"
        path.write_text(
            json.dumps(
                {
                    "timestamps": stream_output.timestamps.tolist(),
                    "states": stream_output.states.tolist(),
                    "inter_times": stream_output.inter_times.tolist(),
                    "origin": float(stream_output.origin),
                    "time_scale_params": {},
                    "state_names": stream_output.state_names,
                }
            ),
            encoding="utf-8",
        )
        loaded = StreamAdapter().deserialize(path, SerializationFormat.JSON)
        np.testing.assert_array_equal(loaded.state_features, stream_output.states)
        np.testing.assert_array_equal(loaded.inter_event_times, stream_output.inter_times)


# -----------------------------------------------------------------------------
# Uniform .data accessor
# -----------------------------------------------------------------------------


class TestDataAccessor:
    """Every adapter output exposes its main payload as `.data`."""

    def test_sequence_data(self, sequence_output: SequenceOutput) -> None:
        assert sequence_output.data is sequence_output.sequences

    def test_stream_data(self, stream_output: StreamOutput) -> None:
        assert stream_output.data is stream_output.states

    def test_raster_data(self) -> None:
        output = RasterOutput(raster=np.zeros((2, 2, 3, 3)))
        assert output.data is output.raster

    def test_graph_data(self) -> None:
        output = GraphOutput(node_features=np.zeros((4, 2)), edge_index=np.zeros((2, 3), dtype=int))
        assert output.data is output.node_features

    def test_table_data_is_field(self) -> None:
        """TableOutput already had `data` as a real field; it stays one."""
        df = pl.DataFrame({"a": [1, 2]})
        output = TableOutput(data=df)
        assert output.data is df

    def test_data_read_only_where_property(
        self, sequence_output: SequenceOutput, stream_output: StreamOutput
    ) -> None:
        """`.data` is a read-only property on the four outputs that lacked it."""
        with pytest.raises(AttributeError):
            sequence_output.data = np.zeros((2, 3, 4))
        with pytest.raises(AttributeError):
            stream_output.data = np.zeros((3, 2))
        with pytest.raises(AttributeError):
            RasterOutput(raster=np.zeros((1, 1, 1, 1))).data = np.zeros((1, 1, 1, 1))
        with pytest.raises(AttributeError):
            GraphOutput(
                node_features=np.zeros((1, 1)), edge_index=np.zeros((2, 0), dtype=int)
            ).data = np.zeros((1, 1))

    def test_graph_data_survives_numpy_round_trip(self, tmp_path: Path) -> None:
        """`.data` (via `node_features`) survives a NumPy round-trip."""
        adapter = GraphAdapter()
        output = GraphOutput(
            node_features=np.arange(8, dtype=np.float32).reshape(4, 2),
            edge_index=np.array([[0, 1], [1, 2]], dtype=np.int64),
            node_ids=[0, 1, 2, 3],
            feature_names=["x", "y"],
        )
        path = tmp_path / "graph.npz"
        adapter.serialize(output, path, SerializationFormat.NUMPY)
        loaded = adapter.deserialize(path, SerializationFormat.NUMPY)
        assert loaded.data is loaded.node_features
        np.testing.assert_array_equal(loaded.data, output.node_features)


# -----------------------------------------------------------------------------
# Pickling
# -----------------------------------------------------------------------------


class TestAliasPickling:
    """Aliases are plain properties: pickling is unaffected."""

    def test_sequence_pickle(self, sequence_output: SequenceOutput) -> None:
        restored = pickle.loads(pickle.dumps(sequence_output))
        assert restored.attention_mask is restored.masks
        np.testing.assert_array_equal(restored.attention_mask, sequence_output.masks)

    def test_stream_pickle(self, stream_output: StreamOutput) -> None:
        restored = pickle.loads(pickle.dumps(stream_output))
        assert restored.state_features is restored.states
        assert restored.inter_event_times is restored.inter_times
        np.testing.assert_array_equal(restored.states, stream_output.states)
