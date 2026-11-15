"""Opt-in scheduled-slot checks; no inference of collection or batch completeness."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from eventflow.core._time import duration_ticks


class CadenceRule(BaseModel):
    """A fixed elapsed schedule anchored at each interval start or a supplied origin.

    Tolerance is strictly less than half the period, so each input record can
    match at most one slot. Zero tolerance means exact timestamps. max_slots
    bounds the total evaluated schedule across the supplied panel.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    every: str
    tolerance: str | None = None
    origin: datetime | None = None
    max_slots: int = Field(default=100_000, strict=True, gt=0)

    @field_validator("origin", mode="before")
    @classmethod
    def validate_origin(cls, value: Any) -> Any:
        if value is not None and not isinstance(value, (str, datetime)):
            raise ValueError(
                "Origin requires an explicit datetime or ISO string, not an inferred epoch unit"
            )
        fraction = re.search(r"[.,](\d+)", value) if isinstance(value, str) else None
        if (fraction and fraction[1][6:].strip("0")) or getattr(value, "nanosecond", 0):
            raise ValueError(
                "Global origin supports microsecond precision; use interval starts for nanosecond phases"
            )
        return value

    @model_validator(mode="after")
    def validate_schedule(self) -> CadenceRule:
        period = duration_ticks(self.every, "ns")
        tolerance = (
            duration_ticks(self.tolerance, "ns", allow_zero=True)
            if self.tolerance is not None
            else 0
        )
        if 2 * tolerance >= period:
            raise ValueError("Cadence tolerance must be less than half the period")
        return self


def cadence_summary(
    reference: pl.DataFrame,
    accepted: pl.DataFrame,
    *,
    row_id: str,
    timestamp_col: str,
    start_col: str,
    end_col: str,
    applicable: pl.Expr,
    time_dtype: pl.Datetime,
    rule: CadenceRule | None,
    prefix: str,
    include_missing_times: bool,
) -> pl.DataFrame:
    """Return one diagnostic row per reference interval, never synthetic samples."""
    names = cadence_columns(prefix, include_missing_times)
    if set(names) & set(reference.columns):
        raise ValueError(
            "Cadence output columns collide with the panel reference; choose another prefix"
        )
    n = reference.height
    period = duration_ticks(rule.every, time_dtype.time_unit) if rule else 0
    tolerance = (
        duration_ticks(rule.tolerance, time_dtype.time_unit, allow_zero=True)
        if rule and rule.tolerance is not None
        else 0
    )
    origin = _origin_ticks(rule.origin, time_dtype) if rule and rule.origin is not None else None
    eligibility = reference.select(row_id, applicable.alias("applicable"))["applicable"].to_list()
    starts = reference[start_col].cast(pl.Int64).to_list()
    ends = reference[end_col].cast(pl.Int64).to_list()
    first: list[int] = [0] * n
    expected: list[int | None] = [None] * n
    status = ["not_applicable" if value is False else "unknown" for value in eligibility]
    total_slots = 0
    for i, (start, end, enabled) in enumerate(zip(starts, ends, eligibility, strict=True)):
        if rule is None or enabled is not True:
            continue
        first[i] = start if origin is None else start + (origin - start) % period
        count = max(0, (end - first[i] + period - 1) // period)
        total_slots += count
        if total_slots > rule.max_slots:
            raise ValueError(
                "Expected cadence slots exceed max_slots; narrow support or set an explicit larger budget"
            )
        expected[i] = count
        status[i] = "satisfied"
    seen: list[set[int]] = [set() for _ in range(n)]
    duplicates = [0] * n
    off_schedule = [0] * n
    # Store only distinct slot indices, not all input timestamps or a Cartesian grid.
    for i, timestamp in accepted.select(row_id, pl.col(timestamp_col).cast(pl.Int64)).iter_rows():
        count = expected[i]
        if count is None:
            continue
        index = (timestamp - first[i] + period // 2) // period
        if 0 <= index < count and abs(timestamp - (first[i] + index * period)) <= tolerance:
            if index in seen[i]:
                duplicates[i] += 1
            seen[i].add(index)
        else:
            off_schedule[i] += 1
    present: list[int | None] = []
    missing: list[int | None] = []
    missing_times: list[list[int] | None] = []
    for i, count in enumerate(expected):
        present.append(len(seen[i]) if count is not None else None)
        gaps = count - len(seen[i]) if count is not None else None
        missing.append(gaps)
        if gaps:
            status[i] = "gaps"
        if include_missing_times:
            missing_times.append(
                [first[i] + k * period for k in range(count) if k not in seen[i]]
                if count is not None
                else None
            )
    output = reference.select(row_id).with_columns(
        pl.Series(f"{prefix}_expected", expected, dtype=pl.Int64),
        pl.Series(f"{prefix}_present", present, dtype=pl.Int64),
        pl.Series(f"{prefix}_missing", missing, dtype=pl.Int64),
        pl.Series(
            f"{prefix}_duplicates",
            [x if expected[i] is not None else None for i, x in enumerate(duplicates)],
            dtype=pl.Int64,
        ),
        pl.Series(
            f"{prefix}_off_schedule",
            [x if expected[i] is not None else None for i, x in enumerate(off_schedule)],
            dtype=pl.Int64,
        ),
        pl.Series(f"{prefix}_status", status, dtype=pl.String),
    )
    if include_missing_times:
        output = output.with_columns(
            pl.Series(f"{prefix}_missing_times", missing_times, dtype=pl.List(pl.Int64)).cast(
                pl.List(time_dtype)
            )
        )
    return output


def cadence_columns(prefix: str, include_missing_times: bool) -> list[str]:
    if not isinstance(prefix, str) or not prefix:
        raise ValueError("Cadence prefix must be a non-empty string")
    fields = ["expected", "present", "missing", "duplicates", "off_schedule", "status"]
    if include_missing_times:
        fields.append("missing_times")
    return [f"{prefix}_{field}" for field in fields]


def _origin_ticks(origin: datetime, dtype: pl.Datetime) -> int:
    aware = origin.utcoffset() is not None
    if aware != (dtype.time_zone is not None):
        raise ValueError("Cadence origin and panel must agree on naive/aware time semantics")
    normalized = origin.astimezone(timezone.utc).replace(tzinfo=None) if aware else origin
    delta = normalized - datetime(1970, 1, 1)
    micros = (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds
    if dtype.time_unit == "ms":
        if micros % 1000:
            raise ValueError("Cadence origin is not representable in millisecond precision")
        return micros // 1000
    return micros * 1000 if dtype.time_unit == "ns" else micros
