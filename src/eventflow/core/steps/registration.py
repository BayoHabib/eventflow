"""Step registration for built-in spatial, temporal, and context steps.

This module registers all built-in steps with the FeatureStepRegistry.
It can be called explicitly or discovered via entry points.
"""

from __future__ import annotations

from eventflow.core.registry import FeatureStepRegistry
from eventflow.core.steps.context import (
    ContextAggregationStep,
    SpatialContextJoinStep,
    SpatioTemporalContextJoinStep,
    TemporalContextJoinStep,
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
    SpatialLagConfig,
    SpatialLagStep,
    TransformCRSConfig,
    TransformCRSStep,
)
from eventflow.core.steps.temporal import (
    CalendarEncodingConfig,
    CalendarEncodingStep,
    ExtractTemporalComponentsConfig,
    ExtractTemporalComponentsStep,
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
    # Spatial steps
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


def get_default_registry() -> FeatureStepRegistry:
    """Create and return a registry with all built-in steps registered.

    Returns:
        FeatureStepRegistry with all built-in steps.
    """
    registry = FeatureStepRegistry()
    register_builtin_steps(registry)
    return registry
