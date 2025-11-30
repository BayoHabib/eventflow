"""Core module containing generic, dataset-agnostic primitives."""

from eventflow.core.event_frame import EventFrame
from eventflow.core.registry import StepRegistry, StepSpec
from eventflow.core.schema import ContextSchema, EventMetadata, EventSchema

__all__ = [
    "EventFrame",
    "EventSchema",
    "ContextSchema",
    "EventMetadata",
    "StepRegistry",
    "StepSpec",
]
