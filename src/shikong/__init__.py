"""Shikong: A high-performance spatio-temporal event transformation engine."""

__version__ = "0.1.0"

from shikong.core.event_frame import EventFrame
from shikong.core.schema import ContextSchema, EventMetadata, EventSchema

__all__ = [
    "EventFrame",
    "EventSchema",
    "ContextSchema",
    "EventMetadata",
    "__version__",
]
