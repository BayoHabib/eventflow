"""Core module containing generic, dataset-agnostic primitives."""

from eventflow.core.event_frame import EventFrame
from eventflow.core.schema import EventSchema, ContextSchema, EventMetadata

__all__ = [
    "EventFrame",
    "EventSchema",
    "ContextSchema",
    "EventMetadata",
]
