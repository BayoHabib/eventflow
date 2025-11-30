"""Eventflow: A high-performance spatio-temporal event transformation engine."""

__version__ = "0.1.0"

from eventflow.core.event_frame import EventFrame
from eventflow.core.schema import ContextSchema, EventMetadata, EventSchema

__all__ = [
    "EventFrame",
    "EventSchema",
    "ContextSchema",
    "EventMetadata",
    "__version__",
]
