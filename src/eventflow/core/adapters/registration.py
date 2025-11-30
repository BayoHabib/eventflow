"""Registration of built-in output adapters with the registry."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from eventflow.core.adapters.configs import (
    GraphAdapterConfig,
    RasterAdapterConfig,
    SequenceAdapterConfig,
    StreamAdapterConfig,
    TableAdapterConfig,
)
from eventflow.core.adapters.graph import GraphAdapter
from eventflow.core.adapters.raster import RasterAdapter
from eventflow.core.adapters.sequence import SequenceAdapter
from eventflow.core.adapters.stream import StreamAdapter
from eventflow.core.adapters.table import TableAdapter

if TYPE_CHECKING:
    from eventflow.core.registry import OutputAdapterRegistry


def register_builtin_adapters(registry: OutputAdapterRegistry) -> None:
    """Register all built-in output adapters with the given registry.

    Args:
        registry: The OutputAdapterRegistry to populate

    Note:
        The adapters use BaseModalityAdapter which is a more specific interface
        than BaseOutputAdapter. We cast for registry compatibility.
    """
    # Table adapter for GLM/Poisson
    registry.register(
        "table",
        cast(Any, TableAdapter),
        tags={"tabular", "glm", "poisson"},
        description="Convert EventFrames to tabular DataFrames for GLM/Poisson regression",
        config_model=TableAdapterConfig,
    )

    # Sequence adapter for RNN/Transformer
    registry.register(
        "sequence",
        cast(Any, SequenceAdapter),
        tags={"sequence", "rnn", "transformer"},
        description="Convert EventFrames to padded sequences for RNN/Transformer models",
        config_model=SequenceAdapterConfig,
    )

    # Raster adapter for CNN
    registry.register(
        "raster",
        cast(Any, RasterAdapter),
        tags={"raster", "cnn", "image"},
        description="Convert EventFrames to 2D/3D raster arrays for CNN models",
        config_model=RasterAdapterConfig,
    )

    # Graph adapter for GNN
    registry.register(
        "graph",
        cast(Any, GraphAdapter),
        tags={"graph", "gnn"},
        description="Convert EventFrames to graph structures for GNN models",
        config_model=GraphAdapterConfig,
    )

    # Stream adapter for Neural ODE
    registry.register(
        "stream",
        cast(Any, StreamAdapter),
        tags={"stream", "ode", "continuous"},
        description="Convert EventFrames to continuous-time sequences for Neural ODEs",
        config_model=StreamAdapterConfig,
    )


def get_default_adapter_registry() -> OutputAdapterRegistry:
    """Create and return a registry with all built-in adapters registered.

    Returns:
        OutputAdapterRegistry with built-in adapters
    """
    from eventflow.core.registry import OutputAdapterRegistry

    registry = OutputAdapterRegistry()
    register_builtin_adapters(registry)
    return registry
