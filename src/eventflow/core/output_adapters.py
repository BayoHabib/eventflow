"""Base classes for output materialisation adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from eventflow.core.event_frame import EventFrame


class BaseOutputAdapter(ABC):
    """Abstract base class for materialising EventFrames to external targets."""

    @abstractmethod
    def write(self, event_frame: EventFrame, **kwargs: Any) -> None:
        """Persist the provided EventFrame to the adapter's target."""
        raise NotImplementedError

    def describe(self) -> str | None:  # pragma: no cover - simple accessor
        """Optional human-readable description of the adapter."""
        return None
