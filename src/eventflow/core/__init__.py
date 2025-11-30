"""Core module containing generic, dataset-agnostic primitives."""

from eventflow.core.context.sources import BaseContextSource
from eventflow.core.event_frame import EventFrame
from eventflow.core.output_adapters import BaseOutputAdapter
from eventflow.core.registry import (
    ContextSourceRegistry,
    ContextSourceSpec,
    FeatureStepRegistry,
    OutputAdapterRegistry,
    OutputAdapterSpec,
    StepRegistry,
    StepSpec,
)
from eventflow.core.schema import (
    ContextRequirementState,
    ContextSchema,
    EventMetadata,
    EventSchema,
    FeatureProvenance,
    OutputModality,
)

__all__ = [
    "EventFrame",
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
