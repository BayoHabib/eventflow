"""Output adapters for converting EventFrames to model-ready data formats.

This module provides adapters for various ML modalities:
- TableAdapter: pandas/Polars DataFrames for GLM/Poisson regression
- SequenceAdapter: Padded tensors with masks for RNNs/Transformers
- RasterAdapter: 2D/3D arrays for CNNs (multi-channel grids per timestep)
- GraphAdapter: Node/edge arrays for GNNs with temporal snapshots
- StreamAdapter: (timestamp, state) sequences for neural ODEs
"""

from __future__ import annotations

from shikong.core.adapters.base import (
    AdapterMetadata,
    BaseModalityAdapter,
    SerializationFormat,
)
from shikong.core.adapters.configs import (
    GraphAdapterConfig,
    RasterAdapterConfig,
    SequenceAdapterConfig,
    StreamAdapterConfig,
    TableAdapterConfig,
)
from shikong.core.adapters.graph import GraphAdapter, GraphOutput
from shikong.core.adapters.raster import RasterAdapter, RasterOutput
from shikong.core.adapters.registration import (
    get_default_adapter_registry,
    register_builtin_adapters,
)
from shikong.core.adapters.sequence import SequenceAdapter, SequenceOutput
from shikong.core.adapters.stream import StreamAdapter, StreamOutput
from shikong.core.adapters.table import TableAdapter, TableOutput

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
