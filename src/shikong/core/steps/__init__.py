"""
Registered Step implementations for spatial and temporal feature engineering.

All steps in this package:
- Inherit from shikong.core.pipeline.Step
- Declare inputs/outputs via registry
- Update EventSchema with provenance tracking
- Support builder patterns for multi-window configurations
"""

from shikong.core.steps.categorical import CategoryCountsStep
from shikong.core.steps.context import (
    ContextAggregationStep,
    SpatialContextJoinStep,
    SpatioTemporalContextJoinStep,
    TemporalContextJoinStep,
)
from shikong.core.steps.domain import GridDomainCoverageStep, SpatialDomainMaskStep
from shikong.core.steps.dynamics import (
    BurstinessStep,
    EWMAMeanStep,
    TimeSinceLastEventStep,
    TrendStep,
    ZeroFeaturesStep,
)
from shikong.core.steps.observation import ObservationPanelStep
from shikong.core.steps.panel import CompletePanelStep
from shikong.core.steps.point_process import (
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
from shikong.core.steps.registration import get_default_registry, register_builtin_steps
from shikong.core.steps.spatial import (
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
from shikong.core.steps.streaming import (
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
from shikong.core.steps.temporal import (
    CalendarEncodingStep,
    ExtractTemporalComponentsStep,
    FillNullStep,
    InterArrivalStep,
    MovingAverageStep,
    RecencyWeightStep,
    TemporalLagStep,
    TimeBinsStep,
)
from shikong.core.steps.validation import (
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
    "ObservationPanelStep",
    "register_builtin_steps",
    # Context join steps
    "ContextAggregationStep",
    "SpatialContextJoinStep",
    "SpatioTemporalContextJoinStep",
    "TemporalContextJoinStep",
    # Category steps
    "CategoryCountsStep",
    "CompletePanelStep",
    # Spatial steps
    "AssignToGridStep",
    "SpatialDomainMaskStep",
    "GridDomainCoverageStep",
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
    "BurstinessStep",
    "EWMAMeanStep",
    "ExtractTemporalComponentsStep",
    "FillNullStep",
    "InterArrivalStep",
    "MovingAverageStep",
    "RecencyWeightStep",
    "TemporalLagStep",
    "TimeBinsStep",
    "TimeSinceLastEventStep",
    "TrendStep",
    "ZeroFeaturesStep",
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
