"""Base classes for output modality adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from pydantic import BaseModel

if TYPE_CHECKING:
    from eventflow.core.event_frame import EventFrame


class SerializationFormat(str, Enum):
    """Supported serialization formats for adapter outputs."""

    PARQUET = "parquet"
    ARROW = "arrow"
    PYTORCH = "pytorch"
    NUMPY = "numpy"
    JSON = "json"
    PICKLE = "pickle"

    def __str__(self) -> str:
        return self.value


@dataclass
class AdapterMetadata:
    """Metadata describing the adapter output structure.

    Attributes:
        modality: The output modality (table, sequence, raster, graph, stream)
        feature_names: List of feature column/channel names
        shapes: Dictionary of shape information by component
        dtypes: Dictionary of data types by component
        spatial_info: Spatial metadata (grid size, CRS, bounds)
        temporal_info: Temporal metadata (resolution, range)
        extra: Additional adapter-specific metadata
    """

    modality: str
    feature_names: list[str] = field(default_factory=list)
    shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    dtypes: dict[str, str] = field(default_factory=dict)
    spatial_info: dict[str, Any] = field(default_factory=dict)
    temporal_info: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary for serialization."""
        return {
            "modality": self.modality,
            "feature_names": self.feature_names,
            "shapes": {k: list(v) for k, v in self.shapes.items()},
            "dtypes": self.dtypes,
            "spatial_info": self.spatial_info,
            "temporal_info": self.temporal_info,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AdapterMetadata:
        """Create metadata from dictionary."""
        shapes = {k: tuple(v) for k, v in data.get("shapes", {}).items()}
        return cls(
            modality=data["modality"],
            feature_names=data.get("feature_names", []),
            shapes=shapes,
            dtypes=data.get("dtypes", {}),
            spatial_info=data.get("spatial_info", {}),
            temporal_info=data.get("temporal_info", {}),
            extra=data.get("extra", {}),
        )


# Type variable for adapter output types
T = TypeVar("T")


class BaseModalityAdapter(ABC, Generic[T]):
    """Abstract base class for modality-specific output adapters.

    Each adapter converts an EventFrame into a model-ready format and provides
    serialization capabilities.
    """

    @property
    @abstractmethod
    def modality(self) -> str:
        """Return the modality name (table, sequence, raster, graph, stream)."""
        ...

    @abstractmethod
    def convert(self, event_frame: EventFrame) -> T:
        """Convert an EventFrame to the adapter's output format.

        Args:
            event_frame: The EventFrame to convert

        Returns:
            The converted output in the adapter's format
        """
        ...

    @abstractmethod
    def get_metadata(self, output: T) -> AdapterMetadata:
        """Extract metadata from the converted output.

        Args:
            output: The converted adapter output

        Returns:
            AdapterMetadata describing the output structure
        """
        ...

    @abstractmethod
    def serialize(
        self,
        output: T,
        path: Path | str,
        fmt: SerializationFormat | str,
    ) -> None:
        """Serialize the output to disk.

        Args:
            output: The adapter output to serialize
            path: Destination path (file or directory)
            fmt: Serialization format
        """
        ...

    @abstractmethod
    def deserialize(
        self,
        path: Path | str,
        fmt: SerializationFormat | str,
    ) -> T:
        """Deserialize output from disk.

        Args:
            path: Source path (file or directory)
            fmt: Serialization format

        Returns:
            The deserialized adapter output
        """
        ...

    def describe(self) -> str | None:
        """Return a human-readable description of the adapter."""
        return f"{self.__class__.__name__} for {self.modality} modality"


class BaseAdapterConfig(BaseModel):
    """Base configuration for modality adapters."""

    pass
