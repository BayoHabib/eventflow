"""
Registered Step implementations for spatial and temporal feature engineering.

All steps in this package:
- Inherit from eventflow.core.pipeline.Step
- Declare inputs/outputs via registry
- Update EventSchema with provenance tracking
- Support builder patterns for multi-window configurations
"""

from eventflow.core.steps.context import (
    ContextAggregationStep,
    SpatialContextJoinStep,
    SpatioTemporalContextJoinStep,
    TemporalContextJoinStep,
)
from eventflow.core.steps.registration import get_default_registry, register_builtin_steps
from eventflow.core.steps.spatial import (
    AssignToGridStep,
    AssignToZonesStep,
    ComputeDistancesStep,
    GetisOrdStep,
    GridAggregationStep,
    KDEStep,
    LocalMoranStep,
    SpatialLagStep,
    TransformCRSStep,
)
from eventflow.core.steps.temporal import (
    CalendarEncodingStep,
    ExtractTemporalComponentsStep,
    InterArrivalStep,
    MovingAverageStep,
    RecencyWeightStep,
    TemporalLagStep,
    TimeBinsStep,
)

__all__ = [
    # Registration
    "get_default_registry",
    "register_builtin_steps",
    # Context join steps
    "ContextAggregationStep",
    "SpatialContextJoinStep",
    "SpatioTemporalContextJoinStep",
    "TemporalContextJoinStep",
    # Spatial steps
    "AssignToGridStep",
    "AssignToZonesStep",
    "ComputeDistancesStep",
    "GetisOrdStep",
    "GridAggregationStep",
    "KDEStep",
    "LocalMoranStep",
    "SpatialLagStep",
    "TransformCRSStep",
    # Temporal steps
    "CalendarEncodingStep",
    "ExtractTemporalComponentsStep",
    "InterArrivalStep",
    "MovingAverageStep",
    "RecencyWeightStep",
    "TemporalLagStep",
    "TimeBinsStep",
]
