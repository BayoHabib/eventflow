"""Step registration for built-in spatial, temporal, and context steps.

This module registers all built-in steps with the FeatureStepRegistry.
It can be called explicitly or discovered via entry points.
"""

from __future__ import annotations

from eventflow.core.registry import FeatureStepRegistry
from eventflow.core.steps.categorical import (
    CategoryCountsConfig,
    CategoryCountsStep,
)
from eventflow.core.steps.context import (
    ContextAggregationStep,
    SpatialContextJoinStep,
    SpatioTemporalContextJoinStep,
    TemporalContextJoinStep,
)
from eventflow.core.steps.density import (
    KernelDensityConfig,
    KernelDensityStep,
    SpaceTimeKernelDensityConfig,
    SpaceTimeKernelDensityStep,
)
from eventflow.core.steps.domain import (
    GridDomainCoverageStep,
    SpatialDomainMaskConfig,
    SpatialDomainMaskStep,
)
from eventflow.core.steps.dynamics import (
    BurstinessConfig,
    BurstinessStep,
    EWMAMeanConfig,
    EWMAMeanStep,
    TimeSinceLastEventConfig,
    TimeSinceLastEventStep,
    TrendConfig,
    TrendStep,
    ZeroFeaturesConfig,
    ZeroFeaturesStep,
)
from eventflow.core.steps.hawkes import HawkesExcitationConfig, HawkesExcitationStep
from eventflow.core.steps.near_repeat import NearRepeatConfig, NearRepeatStep
from eventflow.core.steps.observation import ObservationPanelStep
from eventflow.core.steps.panel import CompletePanelConfig, CompletePanelStep
from eventflow.core.steps.point_process import (
    ConditionalIntensityConfig,
    ConditionalIntensityStep,
    ContinuousInterEventConfig,
    ContinuousInterEventStep,
    DurationFeaturesConfig,
    DurationFeaturesStep,
    ExponentialDecayConfig,
    ExponentialDecayStep,
    HawkesKernelConfig,
    HawkesKernelStep,
    HazardRateConfig,
    HazardRateStep,
    KFunctionConfig,
    KFunctionStep,
    PairCorrelationConfig,
    PairCorrelationStep,
    SurvivalTableConfig,
    SurvivalTableStep,
)
from eventflow.core.steps.spatial import (
    AssignToGridConfig,
    AssignToGridStep,
    AssignToZonesConfig,
    AssignToZonesStep,
    ComputeDistancesConfig,
    ComputeDistancesStep,
    GetisOrdConfig,
    GetisOrdStep,
    GridAggregationConfig,
    GridAggregationStep,
    KDEConfig,
    KDEStep,
    LocalAutocorrelationConfig,
    LocalMoranStep,
    MultiScaleSpatialLagConfig,
    MultiScaleSpatialLagStep,
    SpatialLagConfig,
    SpatialLagStep,
    TransformCRSConfig,
    TransformCRSStep,
)
from eventflow.core.steps.streaming import (
    EventBufferConfig,
    EventBufferStep,
    OnlineStatisticsConfig,
    OnlineStatisticsStep,
    StreamingDecayStep,
    StreamingHawkesConfig,
    StreamingHawkesStep,
    StreamingInterEventStep,
    StreamingWindowConfig,
    StreamingWindowStep,
)
from eventflow.core.steps.temporal import (
    CalendarEncodingConfig,
    CalendarEncodingStep,
    ExtractTemporalComponentsConfig,
    ExtractTemporalComponentsStep,
    FillNullConfig,
    FillNullStep,
    InterArrivalConfig,
    InterArrivalStep,
    MovingAverageConfig,
    MovingAverageStep,
    RecencyWeightConfig,
    RecencyWeightStep,
    TemporalLagConfig,
    TemporalLagStep,
    TimeBinsConfig,
    TimeBinsStep,
)


def register_builtin_steps(registry: FeatureStepRegistry) -> None:
    """Register all built-in steps with the given registry.

    This function can be called directly or via entry point discovery.

    Args:
        registry: The FeatureStepRegistry to register steps with.
    """
    registry.register(
        "observation_panel",
        ObservationPanelStep,
        tags=["spatial", "temporal"],
        description="Count raw records on explicit observation support and preserve unknown totals",
    )
    registry.register(
        "complete_panel",
        CompletePanelStep,
        tags=["spatial", "temporal"],
        description="Complete a sparse unit x time panel with explicit zeros inside the observation support",
        config_model=CompletePanelConfig,
    )
    registry.register(
        "category_counts",
        CategoryCountsStep,
        tags=["spatial", "temporal"],
        description="Per-category counts, shares and entropy per (unit, time) group",
        config_model=CategoryCountsConfig,
    )
    # Spatial steps
    registry.register(
        "spatial_domain_mask",
        SpatialDomainMaskStep,
        tags=["spatial"],
        description="Annotate or filter caller-defined spatial-domain membership",
        config_model=SpatialDomainMaskConfig,
    )
    registry.register(
        "grid_domain_coverage",
        GridDomainCoverageStep,
        tags=["spatial"],
        description="Join geometric domain coverage from a supplied cell grid",
    )
    registry.register(
        "transform_crs",
        TransformCRSStep,
        tags=["spatial"],
        description="Transform event coordinates to a different CRS",
        config_model=TransformCRSConfig,
    )

    registry.register(
        "assign_to_grid",
        AssignToGridStep,
        tags=["spatial"],
        description="Assign events to spatial grid cells",
        config_model=AssignToGridConfig,
    )

    registry.register(
        "assign_to_zones",
        AssignToZonesStep,
        tags=["spatial"],
        description="Assign events to predefined zones (neighborhoods, census tracts)",
        config_model=AssignToZonesConfig,
    )

    registry.register(
        "compute_distances",
        ComputeDistancesStep,
        tags=["spatial"],
        description="Compute distances from events to points of interest",
        config_model=ComputeDistancesConfig,
    )

    registry.register(
        "grid_aggregation",
        GridAggregationStep,
        tags=["spatial"],
        description="Aggregate event data at grid cell level",
        config_model=GridAggregationConfig,
    )

    registry.register(
        "kde",
        KDEStep,
        tags=["spatial", "continuous"],
        description="Compute Kernel Density Estimation for event locations",
        config_model=KDEConfig,
    )

    registry.register(
        "spatial_lag",
        SpatialLagStep,
        tags=["spatial"],
        description="Compute spatial lag (weighted average of neighbors)",
        config_model=SpatialLagConfig,
    )

    registry.register(
        "multiscale_spatial_lag",
        MultiScaleSpatialLagStep,
        tags=["spatial"],
        description="Compute spatial lag over disjoint queen rings (multi-scale)",
        config_model=MultiScaleSpatialLagConfig,
    )

    registry.register(
        "kernel_density",
        KernelDensityStep,
        tags=["spatial", "continuous"],
        description="Compute Gaussian kernel density at cell centres",
        config_model=KernelDensityConfig,
    )

    registry.register(
        "space_time_kde",
        SpaceTimeKernelDensityStep,
        tags=["spatial", "temporal", "continuous"],
        description="Compute separable space-time kernel density (Gaussian in space, exponential in time)",
        config_model=SpaceTimeKernelDensityConfig,
    )

    registry.register(
        "near_repeat",
        NearRepeatStep,
        tags=["spatial", "temporal"],
        description="Compute near-repeat ring counts over trailing time windows",
        config_model=NearRepeatConfig,
    )

    registry.register(
        "hawkes_excitation",
        HawkesExcitationStep,
        tags=["temporal", "continuous"],
        description="Compute fixed-kernel Hawkes excitation from past bins only",
        config_model=HawkesExcitationConfig,
    )

    registry.register(
        "local_moran",
        LocalMoranStep,
        tags=["spatial"],
        description="Compute Local Moran's I for spatial autocorrelation",
        config_model=LocalAutocorrelationConfig,
    )

    registry.register(
        "getis_ord",
        GetisOrdStep,
        tags=["spatial"],
        description="Compute Getis-Ord Gi* hotspot statistics",
        config_model=GetisOrdConfig,
    )

    # Temporal steps
    registry.register(
        "extract_temporal_components",
        ExtractTemporalComponentsStep,
        tags=["temporal"],
        description="Extract temporal components (hour, day of week, month, etc.)",
        config_model=ExtractTemporalComponentsConfig,
    )

    registry.register(
        "time_bins",
        TimeBinsStep,
        tags=["temporal"],
        description="Create time bins for temporal aggregation",
        config_model=TimeBinsConfig,
    )

    registry.register(
        "temporal_lag",
        TemporalLagStep,
        tags=["temporal"],
        description="Compute temporal lags for specified columns",
        config_model=TemporalLagConfig,
    )

    registry.register(
        "moving_average",
        MovingAverageStep,
        tags=["temporal", "continuous"],
        description="Compute moving averages with configurable windows",
        config_model=MovingAverageConfig,
    )

    registry.register(
        "recency_weight",
        RecencyWeightStep,
        tags=["temporal"],
        description="Compute exponential recency weights based on time decay",
        config_model=RecencyWeightConfig,
    )

    registry.register(
        "calendar_encoding",
        CalendarEncodingStep,
        tags=["temporal"],
        description="Encode calendar features (cyclical, one-hot, ordinal)",
        config_model=CalendarEncodingConfig,
    )

    registry.register(
        "fill_null",
        FillNullStep,
        tags=["temporal"],
        description="Fill nulls in named columns with an explicit constant",
        config_model=FillNullConfig,
    )

    registry.register(
        "ewma_mean",
        EWMAMeanStep,
        tags=["temporal"],
        description="Exponentially weighted mean of value columns (includes current bin)",
        config_model=EWMAMeanConfig,
    )

    registry.register(
        "trend",
        TrendStep,
        tags=["temporal"],
        description="Ratio of two rolling-mean columns (null on zero denominator)",
        config_model=TrendConfig,
    )

    registry.register(
        "time_since_last_event",
        TimeSinceLastEventStep,
        tags=["temporal"],
        description="Days since the most recent bin with value > 0",
        config_model=TimeSinceLastEventConfig,
    )

    registry.register(
        "burstiness",
        BurstinessStep,
        tags=["temporal"],
        description="Rolling coefficient of variation of counts (closed-left window)",
        config_model=BurstinessConfig,
    )

    registry.register(
        "zero_features",
        ZeroFeaturesStep,
        tags=["temporal"],
        description="Activity flag and share of nonzero bins (windows include current bin)",
        config_model=ZeroFeaturesConfig,
    )

    registry.register(
        "inter_arrival",
        InterArrivalStep,
        tags=["temporal"],
        description="Compute inter-arrival times between consecutive events",
        config_model=InterArrivalConfig,
    )

    # Context join steps
    registry.register(
        "spatial_context_join",
        SpatialContextJoinStep,
        tags=["spatial", "context"],
        description="Join events with spatial context data",
    )

    registry.register(
        "temporal_context_join",
        TemporalContextJoinStep,
        tags=["temporal", "context"],
        description="Join events with temporal context data",
    )

    registry.register(
        "spatiotemporal_context_join",
        SpatioTemporalContextJoinStep,
        tags=["spatial", "temporal", "context"],
        description="Join events with spatio-temporal context data",
    )

    registry.register(
        "context_aggregation",
        ContextAggregationStep,
        tags=["context"],
        description="Aggregate context data before joining to events",
    )

    # Point-process steps
    registry.register(
        "exponential_decay",
        ExponentialDecayStep,
        tags=["continuous"],
        description="Apply exponential decay weighting based on time",
        config_model=ExponentialDecayConfig,
    )

    registry.register(
        "hawkes_kernel",
        HawkesKernelStep,
        tags=["continuous"],
        description="Compute Hawkes process triggering kernel contributions",
        config_model=HawkesKernelConfig,
    )

    registry.register(
        "conditional_intensity",
        ConditionalIntensityStep,
        tags=["continuous"],
        description="Estimate conditional intensity function at event times",
        config_model=ConditionalIntensityConfig,
    )

    registry.register(
        "pair_correlation",
        PairCorrelationStep,
        tags=["spatial", "continuous"],
        description="Compute pair-correlation function (g-function)",
        config_model=PairCorrelationConfig,
    )

    registry.register(
        "k_function",
        KFunctionStep,
        tags=["spatial", "continuous"],
        description="Compute Ripley's K-function for spatial analysis",
        config_model=KFunctionConfig,
    )

    registry.register(
        "hazard_rate",
        HazardRateStep,
        tags=["continuous"],
        description="Estimate hazard rate for event occurrence",
        config_model=HazardRateConfig,
    )

    registry.register(
        "survival_table",
        SurvivalTableStep,
        tags=["continuous"],
        description="Generate survival table with Kaplan-Meier estimates",
        config_model=SurvivalTableConfig,
    )

    registry.register(
        "duration_features",
        DurationFeaturesStep,
        tags=["continuous", "temporal"],
        description="Build duration-based features with decay functions",
        config_model=DurationFeaturesConfig,
    )

    registry.register(
        "continuous_inter_event",
        ContinuousInterEventStep,
        tags=["continuous", "temporal"],
        description="Compute continuous inter-event time features",
        config_model=ContinuousInterEventConfig,
    )

    # Streaming steps
    registry.register(
        "streaming_window",
        StreamingWindowStep,
        tags=["continuous"],
        description="Maintain sliding window over event stream",
        config_model=StreamingWindowConfig,
    )

    registry.register(
        "online_statistics",
        OnlineStatisticsStep,
        tags=["continuous"],
        description="Compute online/streaming statistics using Welford's algorithm",
        config_model=OnlineStatisticsConfig,
    )

    registry.register(
        "streaming_hawkes",
        StreamingHawkesStep,
        tags=["continuous"],
        description="Compute Hawkes intensity in streaming fashion",
        config_model=StreamingHawkesConfig,
    )

    registry.register(
        "event_buffer",
        EventBufferStep,
        tags=["continuous"],
        description="Buffer events for batch processing",
        config_model=EventBufferConfig,
    )

    registry.register(
        "streaming_decay",
        StreamingDecayStep,
        tags=["continuous"],
        description="Apply streaming exponential decay to accumulated values",
    )

    registry.register(
        "streaming_inter_event",
        StreamingInterEventStep,
        tags=["continuous"],
        description="Compute inter-event features in streaming fashion",
    )


def get_default_registry() -> FeatureStepRegistry:
    """Create and return a registry with all built-in steps registered.

    Returns:
        FeatureStepRegistry with all built-in steps.
    """
    registry = FeatureStepRegistry()
    register_builtin_steps(registry)
    return registry
