"""Context enrichment module for joining external data sources."""

from eventflow.core.context.enricher import EnricherStep
from eventflow.core.context.joiners import (
    SpatialJoin,
    SpatioTemporalJoin,
    TemporalJoin,
)
from eventflow.core.context.sources import (
    BaseContextSource,
    DynamicTemporalSource,
    SpatioTemporalSource,
    StaticSpatialSource,
    StaticTemporalSource,
)

__all__ = [
    "BaseContextSource",
    "StaticSpatialSource",
    "StaticTemporalSource",
    "DynamicTemporalSource",
    "SpatioTemporalSource",
    "TemporalJoin",
    "SpatialJoin",
    "SpatioTemporalJoin",
    "EnricherStep",
]
