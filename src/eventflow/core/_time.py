"""Explicit elapsed-time units shared by temporal computations."""

import math
import re


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
