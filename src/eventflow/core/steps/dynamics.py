"""Dynamics pipeline steps with registry integration.

These steps operate on the completed observation panel (one row per
unit x time bin, explicit zeros, never null counts) produced by
``ObservationPanelStep``, but they behave sensibly on any regular panel.

Global causality convention
---------------------------
A feature row at bin ``t`` is for predicting the count in bin ``t + 1``,
so no feature may use data from bins AFTER ``t``. Trailing statistics
use closed-left windows (strictly before ``t``) where the underlying
Polars operation supports it, except where a step documents otherwise:

- ``EWMAMeanStep`` and ``TimeSinceLastEventStep`` include the current
  bin. This is allowed under the convention above because the feature
  row predicts bin ``t + 1``: the current bin's count is known at
  prediction time.
- ``ZeroFeaturesStep`` uses windows closed on the right (including the
  current bin) for the same reason; this is documented on the step.
- ``BurstinessStep`` uses closed-left windows (strictly before ``t``).

Each step:
- Inherits from Step base class
- Registers inputs/outputs via FeatureProvenance
- Validates its inputs in ``__init__`` (a bad parameter fails at
  construction, never silently at ``run()`` time)
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import polars as pl
from pydantic import BaseModel, Field

from eventflow.core._time import duration_seconds
from eventflow.core.pipeline import Step
from eventflow.core.schema import FeatureProvenance
from eventflow.core.utils import get_logger

if TYPE_CHECKING:
    from eventflow.core.event_frame import EventFrame

logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# Pydantic Config Models for Step Parameters
# -----------------------------------------------------------------------------


class EWMAMeanConfig(BaseModel):
    """Configuration for exponentially weighted mean step."""

    value_cols: list[str] = Field(..., description="Columns to compute EWMA for")
    halflife_days: float = Field(
        default=7.0, gt=0, description="EWMA half-life in days (must be > 0)"
    )
    group_cols: list[str] = Field(default_factory=list, description="Optional columns to group by")
    min_periods: int = Field(
        default=1, ge=1, description="Minimum number of bins before a value is produced"
    )


class TrendConfig(BaseModel):
    """Configuration for trend (ratio of two rolling means) step."""

    recent_col: str = Field(..., description="Short-window rolling-mean column (numerator)")
    long_col: str = Field(..., description="Long-window rolling-mean column (denominator)")
    output_col: str | None = Field(
        default=None,
        description="Output column name (default: '{recent_col}_vs_{long_col}')",
    )


class TimeSinceLastEventConfig(BaseModel):
    """Configuration for time-since-last-event step."""

    value_col: str = Field(default="count", description="Count column to watch for events")
    group_cols: list[str] = Field(default_factory=list, description="Optional columns to group by")
    output_col: str = Field(default="count_time_since_last_d", description="Output column name")
    output_unit: str = Field(
        default="d", description="Output unit; only 'd' (days) is supported in v1"
    )


class BurstinessConfig(BaseModel):
    """Configuration for burstiness (rolling coefficient of variation) step."""

    value_col: str = Field(default="count", description="Count column to measure")
    window: str = Field(default="28d", description="Trailing window (e.g., '28d')")
    group_cols: list[str] = Field(default_factory=list, description="Optional columns to group by")
    min_periods: int = Field(
        default=1, ge=1, description="Minimum number of bins before a value is produced"
    )


class ZeroFeaturesConfig(BaseModel):
    """Configuration for zero/absence features step."""

    value_col: str = Field(default="count", description="Count column to measure")
    active_window: str = Field(
        default="7d", description="Trailing window for the active flag (e.g., '7d')"
    )
    share_window: str = Field(
        default="28d", description="Trailing window for the share of nonzero bins"
    )
    group_cols: list[str] = Field(default_factory=list, description="Optional columns to group by")
    min_periods: int = Field(
        default=1, ge=1, description="Minimum number of bins before a value is produced"
    )


# -----------------------------------------------------------------------------
# Dynamics Steps
# -----------------------------------------------------------------------------


class EWMAMeanStep(Step):
    """Exponentially weighted mean of value columns.

    Causality: the EWMA at bin ``t`` includes the current bin (standard
    EWMA, ``adjust=True``). This is allowed under the module convention
    because the feature row at ``t`` predicts the count in bin ``t + 1``,
    so the current bin's count is known at prediction time. No bin after
    ``t`` is ever used.

    The half-life in days is converted to a half-life in bins using the
    frame's ``metadata.time_bin`` (so a ``7d`` half-life on a daily panel
    is 7 bins, on a 6-hour panel 28 bins).

    Inputs:
        - value_cols to compute EWMA for

    Outputs:
        - {value_col}_ewma_{halflife}d columns
    """

    def __init__(
        self,
        value_cols: Sequence[str],
        halflife_days: float = 7.0,
        group_cols: Sequence[str] | None = None,
        min_periods: int = 1,
    ) -> None:
        self.value_cols = list(value_cols)
        if not self.value_cols:
            raise ValueError("value_cols must not be empty")
        if not math.isfinite(halflife_days) or halflife_days <= 0:
            raise ValueError(f"halflife_days must be a finite number > 0, got {halflife_days!r}")
        self.halflife_days = float(halflife_days)
        self.group_cols = list(group_cols) if group_cols else []
        _validate_min_periods(min_periods)
        self.min_periods = min_periods

    def run(self, event_frame: EventFrame) -> EventFrame:
        """Execute EWMA computation."""
        timestamp_col = event_frame.schema.timestamp_col
        lf = event_frame.lazy_frame
        result = event_frame

        _require_columns(lf, self.value_cols + self.group_cols)
        time_bin = event_frame.metadata.time_bin
        if time_bin is None:
            raise ValueError(
                "EWMAMeanStep requires metadata.time_bin to convert halflife_days "
                "to bins (run TimeBinsStep or ObservationPanelStep first)"
            )
        try:
            bin_seconds = duration_seconds(time_bin)
        except ValueError as exc:
            raise ValueError(
                f"EWMAMeanStep cannot parse metadata.time_bin {time_bin!r}: {exc}"
            ) from exc
        halflife_bins = self.halflife_days * 86400.0 / bin_seconds

        # Sort by timestamp (and group cols if specified)
        sort_cols = self.group_cols + [timestamp_col] if self.group_cols else [timestamp_col]
        lf = lf.sort(sort_cols, maintain_order=True)

        halflife_label = _format_number(self.halflife_days)
        for value_col in self.value_cols:
            ewma_col = f"{value_col}_ewma_{halflife_label}d"
            expr = pl.col(value_col).ewm_mean(half_life=halflife_bins, min_samples=self.min_periods)
            if self.group_cols:
                expr = expr.over(self.group_cols)
            lf = lf.with_columns(expr.alias(ewma_col))

            provenance = FeatureProvenance(
                produced_by="EWMAMeanStep",
                inputs=[value_col, timestamp_col],
                tags={"temporal"},
                description=(
                    f"Exponentially weighted mean of {value_col} "
                    f"with half-life {halflife_label}d (includes current bin)"
                ),
                metadata={
                    "halflife_days": self.halflife_days,
                    "halflife_bins": halflife_bins,
                    "time_bin": time_bin,
                    "min_periods": self.min_periods,
                },
            )
            result = result.register_feature(
                ewma_col,
                {
                    "source_step": "EWMAMeanStep",
                    "value_col": value_col,
                    "halflife_days": self.halflife_days,
                },
                provenance=provenance,
            )

        return result.with_lazy_frame(lf)


class TrendStep(Step):
    """Ratio of two existing rolling-mean columns.

    Causality: the ratio at bin ``t`` only combines columns that already
    exist at ``t``; it introduces no new look-ahead of its own. (Whether
    the inputs themselves are causal depends on how they were built, e.g.
    ``MovingAverageStep`` with its default ``closed="left"``.)

    The output is null when the long (denominator) column is null or 0:
    a zero denominator never produces ``inf`` or a divide-by-zero
    explosion.

    Inputs:
        - recent_col: short-window rolling-mean column
        - long_col: long-window rolling-mean column

    Outputs:
        - {output_col} (default {recent_col}_vs_{long_col})
    """

    def __init__(
        self,
        recent_col: str,
        long_col: str,
        output_col: str | None = None,
    ) -> None:
        self.recent_col = recent_col
        self.long_col = long_col
        self.output_col = output_col or f"{recent_col}_vs_{long_col}"

    def run(self, event_frame: EventFrame) -> EventFrame:
        """Execute trend ratio computation."""
        timestamp_col = event_frame.schema.timestamp_col
        lf = event_frame.lazy_frame

        _require_columns(lf, [self.recent_col, self.long_col])

        # Null when the denominator is null or 0: never inf, never a
        # divide-by-zero explosion. A null denominator makes the `when`
        # condition null, which also falls through to `otherwise(None)`.
        expr = (
            pl.when(pl.col(self.long_col) != 0)
            .then(pl.col(self.recent_col) / pl.col(self.long_col))
            .otherwise(None)
        )
        lf = lf.with_columns(expr.alias(self.output_col))

        provenance = FeatureProvenance(
            produced_by="TrendStep",
            inputs=[self.recent_col, self.long_col, timestamp_col],
            tags={"temporal"},
            description=f"Trend ratio {self.recent_col} / {self.long_col}",
            metadata={"recent_col": self.recent_col, "long_col": self.long_col},
        )
        result = event_frame.register_feature(
            self.output_col,
            {
                "source_step": "TrendStep",
                "recent_col": self.recent_col,
                "long_col": self.long_col,
            },
            provenance=provenance,
        )
        return result.with_lazy_frame(lf)


class TimeSinceLastEventStep(Step):
    """Time since the most recent bin with ``value_col > 0``.

    Causality: the look-back includes the current bin, so the output is
    ``0.0`` when the current bin itself has ``value_col > 0``. This is
    allowed under the module convention because the feature row at ``t``
    predicts the count in bin ``t + 1``. No bin after ``t`` is ever used.
    The output is null when no prior bin (including the current one) had
    ``value_col > 0``.

    Inputs:
        - value_col to watch for events (default "count")

    Outputs:
        - {output_col} in days (Float64), default "count_time_since_last_d"
    """

    def __init__(
        self,
        value_col: str = "count",
        group_cols: Sequence[str] | None = None,
        output_col: str = "count_time_since_last_d",
        output_unit: str = "d",
    ) -> None:
        if output_unit != "d":
            raise ValueError(
                f"Unsupported output_unit {output_unit!r}: only 'd' (days) is supported in v1"
            )
        self.value_col = value_col
        self.group_cols = list(group_cols) if group_cols else []
        self.output_col = output_col
        self.output_unit = output_unit

    def run(self, event_frame: EventFrame) -> EventFrame:
        """Execute time-since-last-event computation."""
        timestamp_col = event_frame.schema.timestamp_col
        lf = event_frame.lazy_frame

        _require_columns(lf, [self.value_col] + self.group_cols)

        # Sort by timestamp (and group cols if specified)
        sort_cols = self.group_cols + [timestamp_col] if self.group_cols else [timestamp_col]
        lf = lf.sort(sort_cols, maintain_order=True)

        # Timestamp of the last bin with value > 0, forward-filled per group.
        last_event = pl.when(pl.col(self.value_col) > 0).then(pl.col(timestamp_col)).forward_fill()
        if self.group_cols:
            last_event = last_event.over(self.group_cols)
        expr = ((pl.col(timestamp_col) - last_event).dt.total_seconds() / 86400.0).cast(pl.Float64)
        lf = lf.with_columns(expr.alias(self.output_col))

        provenance = FeatureProvenance(
            produced_by="TimeSinceLastEventStep",
            inputs=[self.value_col, timestamp_col],
            tags={"temporal"},
            description=(
                f"Days since the most recent bin with {self.value_col} > 0 "
                "(0.0 when the current bin itself has an event)"
            ),
            metadata={"value_col": self.value_col, "output_unit": self.output_unit},
        )
        result = event_frame.register_feature(
            self.output_col,
            {"source_step": "TimeSinceLastEventStep", "value_col": self.value_col},
            provenance=provenance,
        )
        return result.with_lazy_frame(lf)


class BurstinessStep(Step):
    """Coefficient of variation of counts over a trailing window.

    This is a burstiness *proxy*: the coefficient of variation (rolling
    std / rolling mean) of the counts. 0 means perfectly regular counts,
    higher means burstier. It is NOT the Goh-Barabasi inter-event-time
    ``B``, which needs continuous event times and is out of scope for v1.

    Causality: both rolling statistics use closed-left windows (strictly
    before ``t``), so the value at bin ``t`` does not depend on the count
    at ``t`` itself. The output is null when the trailing mean is null
    or 0 (a zero mean never produces ``inf``).

    Inputs:
        - value_col to measure (default "count")

    Outputs:
        - {value_col}_burstiness_{window}
    """

    def __init__(
        self,
        value_col: str = "count",
        window: str = "28d",
        group_cols: Sequence[str] | None = None,
        min_periods: int = 1,
    ) -> None:
        self.value_col = value_col
        self.window = window
        self.group_cols = list(group_cols) if group_cols else []
        _validate_min_periods(min_periods)
        self.min_periods = min_periods
        # Fail at construction on an unparseable window, never silently
        # at run() time.
        duration_seconds(window)

    def run(self, event_frame: EventFrame) -> EventFrame:
        """Execute burstiness computation."""
        timestamp_col = event_frame.schema.timestamp_col
        lf = event_frame.lazy_frame

        _require_columns(lf, [self.value_col] + self.group_cols)

        # Sort by timestamp (and group cols if specified)
        sort_cols = self.group_cols + [timestamp_col] if self.group_cols else [timestamp_col]
        lf = lf.sort(sort_cols, maintain_order=True)

        window_us = _window_micros(self.window)
        mean_expr = pl.col(self.value_col).rolling_mean_by(
            timestamp_col,
            window_size=window_us,
            closed="left",
            min_samples=self.min_periods,
        )
        std_expr = pl.col(self.value_col).rolling_std_by(
            timestamp_col,
            window_size=window_us,
            closed="left",
            min_samples=self.min_periods,
        )
        if self.group_cols:
            mean_expr = mean_expr.over(self.group_cols)
            std_expr = std_expr.over(self.group_cols)
        # Null when the mean is null or 0: never inf. A null mean makes
        # the `when` condition null, which also falls through to null.
        expr = (
            pl.when((mean_expr.is_null()) | (mean_expr == 0))
            .then(None)
            .otherwise(std_expr / mean_expr)
        )
        burstiness_col = f"{self.value_col}_burstiness_{self.window}"
        lf = lf.with_columns(expr.alias(burstiness_col))

        provenance = FeatureProvenance(
            produced_by="BurstinessStep",
            inputs=[self.value_col, timestamp_col],
            tags={"temporal"},
            description=(
                f"Rolling coefficient of variation of {self.value_col} over "
                f"a trailing {self.window} window (closed-left)"
            ),
            metadata={
                "value_col": self.value_col,
                "window": self.window,
                "min_periods": self.min_periods,
                "closed": "left",
            },
        )
        result = event_frame.register_feature(
            burstiness_col,
            {
                "source_step": "BurstinessStep",
                "value_col": self.value_col,
                "window": self.window,
            },
            provenance=provenance,
        )
        return result.with_lazy_frame(lf)


class ZeroFeaturesStep(Step):
    """Features of absence: activity flag and share of nonzero bins.

    Both outputs use windows closed on the RIGHT, i.e. they INCLUDE the
    current bin. This is deliberate and documented: the feature row at
    ``t`` predicts the count in bin ``t + 1``, so whether the current
    bin had any event is known at prediction time. No bin after ``t``
    is ever used.

    Inputs:
        - value_col to measure (default "count")

    Outputs:
        - {value_col}_active_{active_window} (Boolean): any value > 0 in
          the trailing window including the current bin.
        - {value_col}_share_nonzero_{share_window} (Float64): fraction of
          bins with value > 0 in the trailing window including the
          current bin.
    """

    def __init__(
        self,
        value_col: str = "count",
        active_window: str = "7d",
        share_window: str = "28d",
        group_cols: Sequence[str] | None = None,
        min_periods: int = 1,
    ) -> None:
        self.value_col = value_col
        self.active_window = active_window
        self.share_window = share_window
        self.group_cols = list(group_cols) if group_cols else []
        _validate_min_periods(min_periods)
        self.min_periods = min_periods
        # Fail at construction on an unparseable window, never silently
        # at run() time.
        duration_seconds(active_window)
        duration_seconds(share_window)

    def run(self, event_frame: EventFrame) -> EventFrame:
        """Execute zero-features computation."""
        timestamp_col = event_frame.schema.timestamp_col
        lf = event_frame.lazy_frame
        result = event_frame

        _require_columns(lf, [self.value_col] + self.group_cols)

        # Sort by timestamp (and group cols if specified)
        sort_cols = self.group_cols + [timestamp_col] if self.group_cols else [timestamp_col]
        lf = lf.sort(sort_cols, maintain_order=True)

        indicator = (pl.col(self.value_col) > 0).cast(pl.Int64)
        active_col = f"{self.value_col}_active_{self.active_window}"
        share_col = f"{self.value_col}_share_nonzero_{self.share_window}"

        active_expr = indicator.rolling_sum_by(
            timestamp_col,
            window_size=_window_micros(self.active_window),
            closed="right",
            min_samples=self.min_periods,
        )
        share_expr = indicator.rolling_mean_by(
            timestamp_col,
            window_size=_window_micros(self.share_window),
            closed="right",
            min_samples=self.min_periods,
        )
        if self.group_cols:
            active_expr = active_expr.over(self.group_cols)
            share_expr = share_expr.over(self.group_cols)
        # Null rolling sums stay null under the comparison (null > 0 is
        # null), so a window with too few bins yields a null flag.
        lf = lf.with_columns(
            [
                (active_expr > 0).alias(active_col),
                share_expr.alias(share_col),
            ]
        )

        for out_col, description, window in [
            (
                active_col,
                f"Any {self.value_col} > 0 in the trailing {self.active_window} "
                "window including the current bin",
                self.active_window,
            ),
            (
                share_col,
                f"Share of bins with {self.value_col} > 0 in the trailing "
                f"{self.share_window} window including the current bin",
                self.share_window,
            ),
        ]:
            provenance = FeatureProvenance(
                produced_by="ZeroFeaturesStep",
                inputs=[self.value_col, timestamp_col],
                tags={"temporal"},
                description=description,
                metadata={
                    "value_col": self.value_col,
                    "window": window,
                    "min_periods": self.min_periods,
                    "closed": "right",
                },
            )
            result = result.register_feature(
                out_col,
                {
                    "source_step": "ZeroFeaturesStep",
                    "value_col": self.value_col,
                    "window": window,
                },
                provenance=provenance,
            )

        return result.with_lazy_frame(lf)


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------


def _validate_min_periods(min_periods: int) -> None:
    if isinstance(min_periods, bool) or not isinstance(min_periods, int) or min_periods < 1:
        raise ValueError(f"min_periods must be an integer >= 1, got {min_periods!r}")


def _require_columns(lf: pl.LazyFrame, columns: Sequence[str]) -> None:
    missing = sorted(set(columns) - set(lf.collect_schema().names()))
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")


def _window_micros(window: str) -> str:
    """Convert a duration string to a microsecond window string for rolling_*_by."""
    return f"{round(duration_seconds(window) * 1e6)}us"


def _format_number(value: float) -> str:
    """Format a number without a trailing '.0' (7.0 -> '7', 2.5 -> '2.5')."""
    return f"{value:g}"
