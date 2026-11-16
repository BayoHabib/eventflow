"""Core module containing generic, dataset-agnostic primitives."""

from shikong.core.cadence import CadenceRule
from shikong.core.context.sources import BaseContextSource
from shikong.core.domain import SpatialDomain, annotate_grid_coverage
from shikong.core.event_frame import EventFrame
from shikong.core.grid import GridSpec
from shikong.core.observation import ObservationPanel
from shikong.core.output_adapters import BaseOutputAdapter
from shikong.core.registry import (
    ContextSourceRegistry,
    ContextSourceSpec,
    FeatureStepRegistry,
    OutputAdapterRegistry,
    OutputAdapterSpec,
    StepRegistry,
    StepSpec,
)
from shikong.core.schema import (
    ContextRequirementState,
    ContextSchema,
    EventMetadata,
    EventSchema,
    FeatureProvenance,
    OutputModality,
)

__all__ = [
    "EventFrame",
    "GridSpec",
    "ObservationPanel",
    "CadenceRule",
    "SpatialDomain",
    "annotate_grid_coverage",
    "EventSchema",
    "ContextSchema",
    "EventMetadata",
    "OutputModality",
    "FeatureProvenance",
    "ContextRequirementState",
    "FeatureStepRegistry",
    "StepRegistry",
    "StepSpec",
    "ContextSourceRegistry",
    "ContextSourceSpec",
    "OutputAdapterRegistry",
    "OutputAdapterSpec",
    "BaseContextSource",
    "BaseOutputAdapter",
]
