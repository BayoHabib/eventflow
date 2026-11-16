"""Pydantic configuration models for output adapters."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TableAdapterConfig(BaseModel):
    """Configuration for TableAdapter.

    Attributes:
        offset_col: Column name for exposure/offset in GLM (e.g., log population)
        weight_col: Column name for observation weights
        index_cols: Columns to use as index
        feature_cols: Explicit list of feature columns (auto-detected if None)
        target_col: Target column name for supervised learning
        categorical_encoding: How to encode categoricals (passthrough, ordinal, onehot)
        include_intercept: Whether to include intercept column for GLM
        dtype: Output dtype (float32, float64)
    """

    offset_col: str | None = None
    weight_col: str | None = None
    index_cols: list[str] = Field(default_factory=list)
    feature_cols: list[str] | None = None
    target_col: str | None = None
    categorical_encoding: Literal["passthrough", "ordinal", "onehot"] = "passthrough"
    include_intercept: bool = False
    dtype: Literal["float32", "float64"] = "float32"


class SequenceAdapterConfig(BaseModel):
    """Configuration for SequenceAdapter (RNN/Transformer).

    Attributes:
        spatial_col: Column identifying spatial units (e.g., grid_id, zone_id)
        timestamp_col: Column with timestamps (uses schema default if None)
        feature_cols: Columns to include as sequence features
        sequence_length: Fixed sequence length (None for variable with padding)
        padding_value: Value for padding shorter sequences
        padding_side: Side to pad on (left or right)
        time_encoding: How to encode time (none, positional, sinusoidal)
        return_masks: Whether to return attention masks
        dtype: Output dtype
    """

    spatial_col: str | None = None
    timestamp_col: str | None = None
    feature_cols: list[str] | None = None
    sequence_length: int | None = None
    padding_value: float = 0.0
    padding_side: Literal["left", "right"] = "right"
    time_encoding: Literal["none", "positional", "sinusoidal"] = "none"
    return_masks: bool = True
    dtype: Literal["float32", "float64"] = "float32"


class RasterAdapterConfig(BaseModel):
    """Configuration for RasterAdapter (CNN).

    Attributes:
        grid_col: Column identifying grid cell
        timestamp_col: Column with timestamps
        feature_cols: Columns to use as raster channels
        grid_shape: Shape of the spatial grid (rows, cols)
        time_steps: Number of time steps (None for all)
        fill_value: Value for empty cells
        normalize: Whether to normalize channels
        channel_first: Channel dimension first (PyTorch) or last (TensorFlow)
        dtype: Output dtype
    """

    grid_col: str | None = None
    timestamp_col: str | None = None
    feature_cols: list[str] | None = None
    grid_shape: tuple[int, int] | None = None
    time_steps: int | None = None
    fill_value: float = 0.0
    normalize: bool = False
    channel_first: bool = True
    dtype: Literal["float32", "float64"] = "float32"


class GraphAdapterConfig(BaseModel):
    """Configuration for GraphAdapter (GNN).

    Attributes:
        node_col: Column identifying nodes
        timestamp_col: Column with timestamps
        feature_cols: Node feature columns
        edge_feature_cols: Edge feature columns
        adjacency_type: How to compute adjacency (spatial, temporal, both)
        spatial_threshold: Distance threshold for spatial edges
        temporal_window: Time window for temporal edges
        include_self_loops: Whether to include self-loops
        normalize_adjacency: Whether to normalize adjacency matrix
        snapshot_interval: Interval for temporal snapshots (e.g., "1d")
        dtype: Output dtype
    """

    node_col: str | None = None
    timestamp_col: str | None = None
    feature_cols: list[str] | None = None
    edge_feature_cols: list[str] | None = None
    adjacency_type: Literal["spatial", "temporal", "both"] = "spatial"
    spatial_threshold: float | None = None
    temporal_window: str | None = None
    include_self_loops: bool = False
    normalize_adjacency: bool = True
    snapshot_interval: str | None = None
    dtype: Literal["float32", "float64"] = "float32"


class StreamAdapterConfig(BaseModel):
    """Configuration for StreamAdapter (Neural ODE/Continuous-time).

    Attributes:
        timestamp_col: Column with timestamps
        state_cols: Columns representing the state vector
        event_type_col: Column with event type (for marked point processes)
        intensity_cols: Columns for intensity features
        time_scale: How to scale time (none, normalize, log)
        time_origin: Origin for relative time (first, min, custom)
        custom_origin: Custom origin timestamp (ISO format)
        return_inter_times: Whether to return inter-event times
        max_events: Maximum events per sequence (for truncation)
        dtype: Output dtype
    """

    timestamp_col: str | None = None
    state_cols: list[str] | None = None
    event_type_col: str | None = None
    intensity_cols: list[str] | None = None
    time_scale: Literal["none", "normalize", "log"] = "none"
    time_origin: Literal["first", "min", "custom"] = "first"
    custom_origin: str | None = None
    return_inter_times: bool = True
    max_events: int | None = None
    dtype: Literal["float32", "float64"] = "float32"
