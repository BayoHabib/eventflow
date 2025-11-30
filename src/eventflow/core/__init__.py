"""Core module containing generic, dataset-agnostic primitives."""

from eventflow.core.event_frame import EventFrame
from eventflow.core.schema import EventSchema, ContextSchema, EventMetadata
from eventflow.core.registry import StepRegistry, StepSpec

__all__ = [
    "EventFrame",
    "EventSchema",
    "ContextSchema",
    "EventMetadata",
    "StepRegistry",
    "StepSpec",
]
