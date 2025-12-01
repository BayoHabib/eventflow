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
from eventflow.core.steps.point_process import (
    ConditionalIntensityStep,
    ContinuousInterEventStep,
    DurationFeaturesStep,
    ExponentialDecayStep,
    HawkesKernelStep,
    HazardRateStep,
    KFunctionStep,
    PairCorrelationStep,
    SurvivalTableStep,
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
from eventflow.core.steps.streaming import (
    EventBufferStep,
    OnlineStatisticsStep,
    StreamEvent,
    StreamingDecayStep,
    StreamingHawkesStep,
    StreamingInterEventStep,
    StreamingStep,
    StreamingWindowStep,
    StreamState,
    StreamWindow,
    batch_stream_iterator,
    event_stream_iterator,
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
from eventflow.core.steps.validation import (
    ValidationReport,
    ValidationResult,
    validate_cumulative_hazard_start,
    validate_hawkes_features,
    validate_hawkes_intensity,
    validate_hawkes_stability,
    validate_hazard_positivity,
    validate_intensity_integrability,
    validate_intensity_positivity,
    validate_inter_event_positivity,
    validate_point_process_features,
    validate_probability_bounds,
    validate_survival_boundaries,
    validate_survival_monotonicity,
    validate_temporal_ordering,
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
    # Point-process steps
    "ConditionalIntensityStep",
    "ContinuousInterEventStep",
    "DurationFeaturesStep",
    "ExponentialDecayStep",
    "HawkesKernelStep",
    "HazardRateStep",
    "KFunctionStep",
    "PairCorrelationStep",
    "SurvivalTableStep",
    # Streaming steps
    "EventBufferStep",
    "OnlineStatisticsStep",
    "StreamEvent",
    "StreamingDecayStep",
    "StreamingHawkesStep",
    "StreamingInterEventStep",
    "StreamingStep",
    "StreamingWindowStep",
    "StreamState",
    "StreamWindow",
    "batch_stream_iterator",
    "event_stream_iterator",
    # Validation utilities
    "ValidationReport",
    "ValidationResult",
    "validate_cumulative_hazard_start",
    "validate_hawkes_features",
    "validate_hawkes_intensity",
    "validate_hawkes_stability",
    "validate_hazard_positivity",
    "validate_intensity_integrability",
    "validate_intensity_positivity",
    "validate_inter_event_positivity",
    "validate_point_process_features",
    "validate_probability_bounds",
    "validate_survival_boundaries",
    "validate_survival_monotonicity",
    "validate_temporal_ordering",
]
