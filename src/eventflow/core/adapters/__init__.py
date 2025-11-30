"""Output adapters for converting EventFrames to model-ready data formats.

This module provides adapters for various ML modalities:
- TableAdapter: pandas/Polars DataFrames for GLM/Poisson regression
- SequenceAdapter: Padded tensors with masks for RNNs/Transformers
- RasterAdapter: 2D/3D arrays for CNNs (multi-channel grids per timestep)
- GraphAdapter: Node/edge arrays for GNNs with temporal snapshots
- StreamAdapter: (timestamp, state) sequences for neural ODEs
"""

from __future__ import annotations

from eventflow.core.adapters.base import (
    AdapterMetadata,
    BaseModalityAdapter,
    SerializationFormat,
)
from eventflow.core.adapters.configs import (
    GraphAdapterConfig,
    RasterAdapterConfig,
    SequenceAdapterConfig,
    StreamAdapterConfig,
    TableAdapterConfig,
)
from eventflow.core.adapters.graph import GraphAdapter, GraphOutput
from eventflow.core.adapters.raster import RasterAdapter, RasterOutput
from eventflow.core.adapters.registration import (
    get_default_adapter_registry,
    register_builtin_adapters,
)
from eventflow.core.adapters.sequence import SequenceAdapter, SequenceOutput
from eventflow.core.adapters.stream import StreamAdapter, StreamOutput
from eventflow.core.adapters.table import TableAdapter, TableOutput

__all__ = [
    # Base classes
    "BaseModalityAdapter",
    "AdapterMetadata",
    "SerializationFormat",
    # Configs
    "TableAdapterConfig",
    "SequenceAdapterConfig",
    "RasterAdapterConfig",
    "GraphAdapterConfig",
    "StreamAdapterConfig",
    # Adapters
    "TableAdapter",
    "SequenceAdapter",
    "RasterAdapter",
    "GraphAdapter",
    "StreamAdapter",
    # Outputs
    "TableOutput",
    "SequenceOutput",
    "RasterOutput",
    "GraphOutput",
    "StreamOutput",
    # Registration
    "register_builtin_adapters",
    "get_default_adapter_registry",
]
