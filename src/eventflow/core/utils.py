"""Utility functions and helpers."""

import logging
from enum import Enum
from typing import Any


# Logging setup
def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with standardized configuration.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger


# Common CRS definitions
class CRS(str, Enum):
    """Common coordinate reference systems."""

    WGS84 = "EPSG:4326"  # Standard lat/lon
    WEB_MERCATOR = "EPSG:3857"  # Web mapping
    NAD83_ILLINOIS_EAST = "EPSG:26971"  # Chicago area (meters)
    NAD83_CALIFORNIA_VI = "EPSG:26946"  # California area
    UTM_ZONE_16N = "EPSG:32616"  # Chicago area UTM


# Time zone helpers
COMMON_TIMEZONES = {
    "chicago": "America/Chicago",
    "new_york": "America/New_York",
    "los_angeles": "America/Los_Angeles",
    "london": "Europe/London",
    "paris": "Europe/Paris",
    "tokyo": "Asia/Tokyo",
}


def get_timezone(location: str) -> str:
    """
    Get timezone for a common location.

    Args:
        location: Location name (e.g., "chicago")

    Returns:
        Timezone string (e.g., "America/Chicago")
    """
    location = location.lower()
    if location in COMMON_TIMEZONES:
        return COMMON_TIMEZONES[location]
    return location  # Assume it's already a valid timezone


# Type checking helpers
def is_numeric_col(dtype: Any) -> bool:
    """Check if a Polars dtype is numeric."""
    import polars as pl

    return dtype in [
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
    ]


def is_categorical_col(dtype: Any) -> bool:
    """Check if a Polars dtype is categorical."""
    import polars as pl

    return dtype in [pl.Utf8, pl.Categorical, pl.Boolean]


def is_temporal_col(dtype: Any) -> bool:
    """Check if a Polars dtype is temporal."""
    import polars as pl

    return dtype in [pl.Date, pl.Datetime, pl.Time, pl.Duration]


# Data validation helpers
def validate_bounds(
    bounds: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """
    Validate and normalize spatial bounds.

    Args:
        bounds: (minx, miny, maxx, maxy)

    Returns:
        Validated bounds

    Raises:
        ValueError: If bounds are invalid
    """
    minx, miny, maxx, maxy = bounds

    if minx >= maxx:
        raise ValueError(f"minx ({minx}) must be less than maxx ({maxx})")
    if miny >= maxy:
        raise ValueError(f"miny ({miny}) must be less than maxy ({maxy})")

    return (minx, miny, maxx, maxy)


def parse_time_window(window: str) -> dict[str, int]:
    """
    Parse time window string into components.

    Args:
        window: Time window string (e.g., "7d", "1h", "30m")

    Returns:
        Dictionary with time components

    Examples:
        >>> parse_time_window("7d")
        {'days': 7}
        >>> parse_time_window("1h")
        {'hours': 1}
    """
    import re

    pattern = r"(\d+)([a-z]+)"
    match = re.match(pattern, window.lower())

    if not match:
        raise ValueError(f"Invalid time window format: {window}")

    value = int(match.group(1))
    unit = match.group(2)

    unit_map = {
        "s": "seconds",
        "m": "minutes",
        "h": "hours",
        "d": "days",
        "w": "weeks",
    }

    if unit not in unit_map:
        raise ValueError(f"Unknown time unit: {unit}")

    return {unit_map[unit]: value}


# Progress tracking
class ProgressTracker:
    """Simple progress tracker for long-running operations."""

    def __init__(self, total: int, description: str = "") -> None:
        """
        Initialize progress tracker.

        Args:
            total: Total number of items
            description: Description of the operation
        """
        self.total = total
        self.description = description
        self.current = 0
        self.logger = get_logger(__name__)

    def update(self, n: int = 1) -> None:
        """Update progress by n items."""
        self.current += n
        if self.current % max(1, self.total // 10) == 0:
            pct = (self.current / self.total) * 100
            self.logger.info(f"{self.description}: {self.current}/{self.total} ({pct:.1f}%)")

    def finish(self) -> None:
        """Mark progress as complete."""
        self.logger.info(f"{self.description}: Complete ({self.total} items)")
