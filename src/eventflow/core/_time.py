"""Explicit elapsed-time units shared by temporal computations."""

import math
import re
from decimal import Decimal


def duration_seconds(value: str) -> float:
    """Parse a positive fixed duration; a day is 24 hours, not a calendar day.

    Months/years and silently substituted defaults are deliberately unsupported.
    """
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)?\s*(us|ms|s|m|min|h|hr|d|day|w|week)\s*", value)
    if match is None:
        raise ValueError(f"Unsupported elapsed duration: {value!r}")
    factors = {
        "us": 1e-6,
        "ms": 1e-3,
        "s": 1,
        "m": 60,
        "min": 60,
        "h": 3600,
        "hr": 3600,
        "d": 86400,
        "day": 86400,
        "w": 604800,
        "week": 604800,
    }
    seconds = float(match[1] or 1) * factors[match[2]]
    if not math.isfinite(seconds) or seconds < 1e-6:
        raise ValueError("Elapsed duration must be finite and at least one microsecond")
    return seconds


def duration_ticks(value: str, time_unit: str, *, allow_zero: bool = False) -> int:
    """Parse an exactly representable duration without floating timestamp arithmetic."""
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)?\s*(us|ms|s|m|min|h|hr|d|day|w|week)\s*", value)
    if match is None or time_unit not in {"ns", "us", "ms"}:
        raise ValueError(f"Unsupported duration or timestamp precision: {value!r}, {time_unit!r}")
    factors = {
        "us": "0.000001",
        "ms": "0.001",
        "s": "1",
        "m": "60",
        "min": "60",
        "h": "3600",
        "hr": "3600",
        "d": "86400",
        "day": "86400",
        "w": "604800",
        "week": "604800",
    }
    seconds = Decimal(match[1] or "1") * Decimal(factors[match[2]])
    if (not allow_zero and seconds < Decimal("0.000001")) or seconds < 0:
        raise ValueError("Cadence duration must be positive and at least one microsecond")
    ticks = seconds * {"ns": 1_000_000_000, "us": 1_000_000, "ms": 1000}[time_unit]
    if ticks != ticks.to_integral_value() or ticks > 2**63 - 1:
        raise ValueError(f"Duration {value!r} is not representable in Datetime({time_unit})")
    return int(ticks)
