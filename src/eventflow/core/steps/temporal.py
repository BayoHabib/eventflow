"""Temporal pipeline steps with registry integration.

Each step:
- Inherits from Step base class
- Registers inputs/outputs via FeatureProvenance
- Updates EventSchema properly
- Supports builder patterns for multi-window configurations
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import polars as pl
from pydantic import BaseModel, Field

from eventflow.core._time import duration_seconds
from eventflow.core.pipeline import Step
from eventflow.core.schema import TEMPORAL_COMPONENTS, FeatureProvenance
from eventflow.core.utils import get_logger

if TYPE_CHECKING:
    from eventflow.core.event_frame import EventFrame

logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# Pydantic Config Models for Step Parameters
# -----------------------------------------------------------------------------


class ExtractTemporalComponentsConfig(BaseModel):
    """Configuration for temporal component extraction step."""

    components: list[
        Literal[
            "hour_of_day",
            "day_of_week",
            "day_of_month",
            "day_of_year",
            "month",
            "year",
            "week_of_year",
            "quarter",
            "is_weekend",
        ]
    ] = Field(default=["hour_of_day", "day_of_week"], description="Temporal components to extract")


class TimeBinsConfig(BaseModel):
    """Configuration for time binning step."""

    bin_size: str = Field(default="1h", description="Time bin size (e.g., '1h', '6h', '1d')")
    bin_col: str = Field(default="time_bin", description="Output column name")


class TemporalLagConfig(BaseModel):
    """Configuration for temporal lag step."""

    value_cols: list[str] = Field(..., description="Columns to compute temporal lags for")
    lag_periods: list[int] = Field(default=[1], description="Lag periods to compute")
    time_unit: str = Field(default="1d", description="Elapsed time unit for lag periods")
    mode: Literal["duration", "rows"] = "duration"
    group_cols: list[str] = Field(
        default_factory=list, description="Optional columns to group by before computing lags"
    )


class MovingAverageConfig(BaseModel):
    """Configuration for moving average step."""

    value_cols: list[str] = Field(..., description="Columns to compute moving averages for")
    windows: list[int] = Field(default=[7], description="Window sizes for moving averages")
    time_unit: str = Field(default="1d", description="Time unit for window periods")
    group_cols: list[str] = Field(default_factory=list, description="Optional columns to group by")
    center: bool = Field(default=False, description="Centering is supported in rows mode only")
    mode: Literal["duration", "rows"] = "duration"
    closed: Literal["left", "right", "both", "none"] = "left"


class RecencyWeightConfig(BaseModel):
    """Configuration for recency weighting step."""

    half_life: float = Field(default=7.0, gt=0, description="Half-life in time units")
    time_unit: str = Field(default="1d", description="Time unit for half-life")
    reference_time: str | None = Field(
        default=None, description="Reference time (defaults to max timestamp)"
    )
    output_col: str = Field(default="recency_weight", description="Output column name")


class CalendarEncodingConfig(BaseModel):
    """Configuration for calendar encoding step."""

    encoding_type: Literal["cyclical", "onehot", "ordinal"] = Field(
        default="cyclical", description="Type of calendar encoding"
    )
    # Keep in sync with CALENDAR_ENCODING_COMPONENTS below: the Literal
    # guards the registry path, the frozenset guards direct construction.
    components: list[Literal["hour", "day_of_week", "day_of_month", "month", "week_of_year"]] = (
        Field(default=["hour", "day_of_week", "month"], description="Components to encode")
    )


# Components CalendarEncodingStep actually encodes. This is the single
# source of truth shared by the Literal in CalendarEncodingConfig above
# and the step's __init__ validation: the registry path is already safe
# (pydantic rejects values outside the Literal before the step is
# built), but direct construction bypasses it, so the step must
# validate too (issue #33).
CALENDAR_ENCODING_COMPONENTS = frozenset(
    {"hour", "day_of_week", "day_of_month", "month", "week_of_year"}
)


class FillNullConfig(BaseModel):
    """Configuration for null filling step."""

    value_cols: list[str] = Field(..., description="Columns to fill nulls in")
    fill_value: float | int = Field(default=0, description="Constant value used to replace nulls")


class InterArrivalConfig(BaseModel):
    """Configuration for inter-arrival time step."""

    group_cols: list[str] = Field(
        default_factory=list, description="Columns to group by for inter-arrival computation"
    )
    output_col: str = Field(default="inter_arrival_seconds", description="Output column name")
    include_stats: bool = Field(
        default=True, description="Include rolling statistics of inter-arrival times"
    )
    stat_windows: list[int] = Field(
        default=[10, 50], description="Window sizes for rolling statistics"
    )


# -----------------------------------------------------------------------------
# Temporal Steps
# -----------------------------------------------------------------------------


class ExtractTemporalComponentsStep(Step):
    """Extract temporal components from timestamp.

    Inputs:
        - timestamp_col from EventSchema

    Outputs:
        - hour_of_day, day_of_week, month, etc. based on configuration
    """

    # Builder API for multi-component configuration
    _builder_components: list[str] | None = None

    def __init__(self, components: Sequence[str] | None = None) -> None:
        if components is None:
            components = self._builder_components or ["hour_of_day", "day_of_week"]
        self.components = list(components)
        # An unknown component must fail here, not be silently skipped by
        # run(): the if-chains below only produce the supported components,
        # so a typo such as "hour_of_dya" used to yield neither the
        # misspelled nor the intended column (issue #30). Validated against
        # the same set as TemporalConfig.components in core/schema.py.
        unknown = sorted(set(self.components) - TEMPORAL_COMPONENTS)
        if unknown:
            raise ValueError(
                "Unknown temporal component(s): " + ", ".join(unknown) + ". "
                "Supported temporal components: " + ", ".join(sorted(TEMPORAL_COMPONENTS)) + "."
            )

    @classmethod
    def with_components(cls, components: Sequence[str]) -> type[ExtractTemporalComponentsStep]:
        """Builder API to configure components."""

        class ConfiguredStep(ExtractTemporalComponentsStep):
            _builder_components = list(components)

        return ConfiguredStep

    def run(self, event_frame: EventFrame) -> EventFrame:
        """Execute temporal component extraction."""
        timestamp_col = event_frame.schema.timestamp_col
        lf = event_frame.lazy_frame
        result = event_frame

        logger.info(f"Extracting temporal components: {self.components}")
        exprs: list[pl.Expr] = []
        feature_names: list[str] = []

        if "hour_of_day" in self.components:
            exprs.append(pl.col(timestamp_col).dt.hour().alias("hour_of_day"))
            feature_names.append("hour_of_day")

        if "day_of_week" in self.components:
            exprs.append(pl.col(timestamp_col).dt.weekday().alias("day_of_week"))
            feature_names.append("day_of_week")

        if "day_of_month" in self.components:
            exprs.append(pl.col(timestamp_col).dt.day().alias("day_of_month"))
            feature_names.append("day_of_month")

        if "day_of_year" in self.components:
            exprs.append(pl.col(timestamp_col).dt.ordinal_day().alias("day_of_year"))
            feature_names.append("day_of_year")

        if "month" in self.components:
            exprs.append(pl.col(timestamp_col).dt.month().alias("month"))
            feature_names.append("month")

        if "year" in self.components:
            exprs.append(pl.col(timestamp_col).dt.year().alias("year"))
            feature_names.append("year")

        if "week_of_year" in self.components:
            exprs.append(pl.col(timestamp_col).dt.week().alias("week_of_year"))
            feature_names.append("week_of_year")

        if "quarter" in self.components:
            exprs.append(pl.col(timestamp_col).dt.quarter().alias("quarter"))
            feature_names.append("quarter")

        if "is_weekend" in self.components:
            # Polars weekday(): 1=Monday, ..., 6=Saturday, 7=Sunday
            exprs.append((pl.col(timestamp_col).dt.weekday().is_in([6, 7])).alias("is_weekend"))
            feature_names.append("is_weekend")

        if exprs:
            lf = lf.with_columns(exprs)

        result = result.with_lazy_frame(lf)

        for feat_name in feature_names:
            provenance = FeatureProvenance(
                produced_by="ExtractTemporalComponentsStep",
                inputs=[timestamp_col],
                tags={"temporal"},
                description=f"Temporal component: {feat_name}",
            )
            result = result.register_feature(
                feat_name,
                {"source_step": "ExtractTemporalComponentsStep", "component": feat_name},
                provenance=provenance,
            )

        return result


class TimeBinsStep(Step):
    """Create time bins for temporal aggregation.

    Inputs:
        - timestamp_col from EventSchema

    Outputs:
        - time_bin column (or custom bin_col)
    """

    def __init__(self, bin_size: str = "1h", bin_col: str = "time_bin") -> None:
        self.bin_size = bin_size
        self.bin_col = bin_col

    def run(self, event_frame: EventFrame) -> EventFrame:
        """Execute time binning."""
        timestamp_col = event_frame.schema.timestamp_col

        logger.info(f"Creating time bins with size: {self.bin_size}")
        lf = event_frame.lazy_frame.with_columns(
            [pl.col(timestamp_col).dt.truncate(self.bin_size).alias(self.bin_col)]
        )

        provenance = FeatureProvenance(
            produced_by="TimeBinsStep",
            inputs=[timestamp_col],
            tags={"temporal"},
            description=f"Time binning at {self.bin_size} resolution",
            metadata={"bin_size": self.bin_size},
        )

        result = event_frame.with_lazy_frame(lf).with_metadata(time_bin=self.bin_size)
        result = result.register_feature(
            self.bin_col,
            {"source_step": "TimeBinsStep", "bin_size": self.bin_size},
            provenance=provenance,
        )
        return result.require_context(temporal_resolution=self.bin_size)


class TemporalLagStep(Step):
    """Compute temporal lags for specified columns.

    Inputs:
        - value_cols to compute lags for

    Outputs:
        - {value_col}_lag_{period} columns
    """

    # Builder API for multi-window configuration
    _builder_lags: list[int] | None = None

    def __init__(
        self,
        value_cols: Sequence[str],
        lag_periods: Sequence[int] | None = None,
        time_unit: str = "1d",
        group_cols: Sequence[str] | None = None,
        mode: Literal["duration", "rows"] = "duration",
    ) -> None:
        self.value_cols = list(value_cols)
        if lag_periods is None:
            lag_periods = self._builder_lags or [1]
        self.lag_periods = list(lag_periods)
        self.time_unit = time_unit
        self.mode = mode
        _validate_periods(self.lag_periods, mode)
        if mode == "duration":
            duration_seconds(time_unit)
        self.group_cols = list(group_cols) if group_cols else []

    @classmethod
    def with_lags(cls, lags: Sequence[int]) -> type[TemporalLagStep]:
        """Builder API to configure lag periods."""

        class ConfiguredStep(TemporalLagStep):
            _builder_lags = list(lags)

        return ConfiguredStep

    def run(self, event_frame: EventFrame) -> EventFrame:
        """Execute temporal lag computation."""
        timestamp_col = event_frame.schema.timestamp_col
        lf = event_frame.lazy_frame
        result = event_frame

        # Sort by timestamp (and group cols if specified)
        sort_cols = self.group_cols + [timestamp_col] if self.group_cols else [timestamp_col]
        lf = lf.sort(sort_cols, maintain_order=True)
        if self.mode == "duration":
            _validate_time_keys(lf, self.group_cols + [timestamp_col])
        source = lf

        for value_col in self.value_cols:
            for lag in self.lag_periods:
                lag_col = f"{value_col}_lag_{lag}"

                if self.mode == "duration":
                    # Exact elapsed lag: absence at t - duration stays unknown.
                    shifted = source.select(
                        *self.group_cols,
                        (
                            pl.col(timestamp_col)
                            + pl.duration(
                                microseconds=round(duration_seconds(self.time_unit) * lag * 1e6)
                            )
                        ).alias(timestamp_col),
                        pl.col(value_col).alias(lag_col),
                    )
                    lf = lf.join(
                        shifted,
                        on=self.group_cols + [timestamp_col],
                        how="left",
                        validate="m:1",
                        maintain_order="left",
                    )
                else:
                    expr = pl.col(value_col).shift(lag)
                    if self.group_cols:
                        expr = expr.over(self.group_cols)
                    lf = lf.with_columns(expr.alias(lag_col))

                provenance = FeatureProvenance(
                    produced_by="TemporalLagStep",
                    inputs=[value_col, timestamp_col],
                    tags={"temporal"},
                    description=f"Temporal lag of {value_col} by {lag} periods",
                    metadata={"lag": lag, "time_unit": self.time_unit, "mode": self.mode},
                )
                result = result.register_feature(
                    lag_col,
                    {"source_step": "TemporalLagStep", "value_col": value_col, "lag": lag},
                    provenance=provenance,
                )

        return result.with_lazy_frame(lf)


class MovingAverageStep(Step):
    """Compute moving averages for specified columns.

    Inputs:
        - value_cols to compute moving averages for

    Outputs:
        - {value_col}_ma_{window} columns
    """

    # Builder API for multi-window configuration
    _builder_windows: list[int] | None = None

    def __init__(
        self,
        value_cols: Sequence[str],
        windows: Sequence[int] | None = None,
        time_unit: str = "1d",
        group_cols: Sequence[str] | None = None,
        center: bool = False,
        mode: Literal["duration", "rows"] = "duration",
        closed: Literal["left", "right", "both", "none"] = "left",
    ) -> None:
        self.value_cols = list(value_cols)
        if windows is None:
            windows = self._builder_windows or [7]
        self.windows = list(windows)
        self.time_unit = time_unit
        self.group_cols = list(group_cols) if group_cols else []
        self.center = center
        self.mode = mode
        self.closed = closed
        _validate_periods(self.windows, mode)
        if mode == "duration":
            duration_seconds(time_unit)
            if center:
                raise ValueError("center=True requires mode='rows'; duration windows are trailing")
        if closed not in {"left", "right", "both", "none"}:
            raise ValueError("Unknown window boundary policy")

    @classmethod
    def with_windows(cls, windows: Sequence[int]) -> type[MovingAverageStep]:
        """Builder API to configure window sizes."""

        class ConfiguredStep(MovingAverageStep):
            _builder_windows = list(windows)

        return ConfiguredStep

    def run(self, event_frame: EventFrame) -> EventFrame:
        """Execute moving average computation."""
        timestamp_col = event_frame.schema.timestamp_col
        lf = event_frame.lazy_frame
        result = event_frame

        # Sort by timestamp (and group cols if specified)
        sort_cols = self.group_cols + [timestamp_col] if self.group_cols else [timestamp_col]
        lf = lf.sort(sort_cols, maintain_order=True)
        if self.mode == "duration":
            _validate_time_keys(lf, self.group_cols + [timestamp_col])

        for value_col in self.value_cols:
            for window in self.windows:
                ma_col = f"{value_col}_ma_{window}"

                if self.mode == "duration":
                    micros = round(duration_seconds(self.time_unit) * window * 1e6)
                    expr = pl.col(value_col).rolling_mean_by(
                        timestamp_col,
                        window_size=f"{micros}us",
                        closed=self.closed,
                        min_samples=1,
                    )
                else:
                    expr = pl.col(value_col).rolling_mean(
                        window_size=window, center=self.center, min_samples=1
                    )
                if self.group_cols:
                    expr = expr.over(self.group_cols)
                lf = lf.with_columns(expr.alias(ma_col))

                provenance = FeatureProvenance(
                    produced_by="MovingAverageStep",
                    inputs=[value_col, timestamp_col],
                    tags={"temporal"},
                    description=f"Moving average of {value_col} with window {window}",
                    metadata={
                        "window": window,
                        "time_unit": self.time_unit,
                        "center": self.center,
                        "mode": self.mode,
                        "closed": self.closed,
                    },
                )
                result = result.register_feature(
                    ma_col,
                    {
                        "source_step": "MovingAverageStep",
                        "value_col": value_col,
                        "window": window,
                    },
                    provenance=provenance,
                )

        return result.with_lazy_frame(lf)


class RecencyWeightStep(Step):
    """Compute exponential recency weights based on time decay.

    Inputs:
        - timestamp_col from EventSchema

    Outputs:
        - recency_weight column (exponential decay from reference time)
    """

    def __init__(
        self,
        half_life: float = 7.0,
        time_unit: str = "1d",
        reference_time: str | None = None,
        output_col: str = "recency_weight",
    ) -> None:
        self.half_life = half_life
        self.time_unit = time_unit
        self.reference_time = reference_time
        self.output_col = output_col

    def run(self, event_frame: EventFrame) -> EventFrame:
        """Execute recency weight computation."""
        timestamp_col = event_frame.schema.timestamp_col
        lf = event_frame.lazy_frame

        # Calculate time unit in seconds for conversion
        unit_seconds = _parse_time_unit_seconds(self.time_unit)
        half_life_seconds = self.half_life * unit_seconds
        decay_rate = 0.693147 / half_life_seconds  # ln(2) / half_life

        # Determine reference time
        if self.reference_time:
            ref_time = pl.lit(self.reference_time).str.to_datetime()
        else:
            # Use max timestamp as reference
            max_ts = lf.select(pl.col(timestamp_col).max()).collect().item()
            ref_time = pl.lit(max_ts)

        # Compute time difference and apply exponential decay
        lf = lf.with_columns(
            [
                ((-decay_rate * (ref_time - pl.col(timestamp_col)).dt.total_seconds()).exp()).alias(
                    self.output_col
                )
            ]
        )

        provenance = FeatureProvenance(
            produced_by="RecencyWeightStep",
            inputs=[timestamp_col],
            tags={"temporal"},
            description=f"Exponential recency weight with half-life {self.half_life} {self.time_unit}",
            metadata={"half_life": self.half_life, "time_unit": self.time_unit},
        )

        result = event_frame.with_lazy_frame(lf)
        return result.register_feature(
            self.output_col,
            {"source_step": "RecencyWeightStep", "half_life": self.half_life},
            provenance=provenance,
        )


class CalendarEncodingStep(Step):
    """Encode calendar features with cyclical, one-hot, or ordinal encoding.

    Inputs:
        - timestamp_col from EventSchema

    Outputs:
        - Encoded calendar features based on encoding_type
    """

    def __init__(
        self,
        encoding_type: str = "cyclical",
        components: Sequence[str] | None = None,
    ) -> None:
        self.encoding_type = encoding_type
        self.components = list(components) if components else ["hour", "day_of_week", "month"]
        # An unknown component must fail here, not be silently skipped by
        # run(): the branches below only produce the supported components,
        # so a typo such as "hour_of_dya" used to yield neither the
        # misspelled nor the intended column (issue #33). Validated against
        # the same set as the Literal in CalendarEncodingConfig.
        unknown = sorted(set(self.components) - CALENDAR_ENCODING_COMPONENTS)
        if unknown:
            raise ValueError(
                "Unknown calendar component(s): " + ", ".join(unknown) + ". "
                "Supported calendar components: "
                + ", ".join(sorted(CALENDAR_ENCODING_COMPONENTS))
                + "."
            )

    def run(self, event_frame: EventFrame) -> EventFrame:
        """Execute calendar encoding."""
        import math

        timestamp_col = event_frame.schema.timestamp_col
        lf = event_frame.lazy_frame
        result = event_frame

        for component in self.components:
            if self.encoding_type == "cyclical":
                # Cyclical encoding using sin/cos
                if component == "hour":
                    max_val = 24
                    lf = lf.with_columns(
                        [
                            (2 * math.pi * pl.col(timestamp_col).dt.hour() / max_val)
                            .sin()
                            .alias("hour_sin"),
                            (2 * math.pi * pl.col(timestamp_col).dt.hour() / max_val)
                            .cos()
                            .alias("hour_cos"),
                        ]
                    )
                    result = self._register_cyclical_features(
                        result, "hour", ["hour_sin", "hour_cos"]
                    )

                elif component == "day_of_week":
                    max_val = 7
                    lf = lf.with_columns(
                        [
                            (2 * math.pi * pl.col(timestamp_col).dt.weekday() / max_val)
                            .sin()
                            .alias("dow_sin"),
                            (2 * math.pi * pl.col(timestamp_col).dt.weekday() / max_val)
                            .cos()
                            .alias("dow_cos"),
                        ]
                    )
                    result = self._register_cyclical_features(
                        result, "day_of_week", ["dow_sin", "dow_cos"]
                    )

                elif component == "day_of_month":
                    max_val = 31
                    lf = lf.with_columns(
                        [
                            (2 * math.pi * pl.col(timestamp_col).dt.day() / max_val)
                            .sin()
                            .alias("dom_sin"),
                            (2 * math.pi * pl.col(timestamp_col).dt.day() / max_val)
                            .cos()
                            .alias("dom_cos"),
                        ]
                    )
                    result = self._register_cyclical_features(
                        result, "day_of_month", ["dom_sin", "dom_cos"]
                    )

                elif component == "month":
                    max_val = 12
                    lf = lf.with_columns(
                        [
                            (2 * math.pi * pl.col(timestamp_col).dt.month() / max_val)
                            .sin()
                            .alias("month_sin"),
                            (2 * math.pi * pl.col(timestamp_col).dt.month() / max_val)
                            .cos()
                            .alias("month_cos"),
                        ]
                    )
                    result = self._register_cyclical_features(
                        result, "month", ["month_sin", "month_cos"]
                    )

                elif component == "week_of_year":
                    max_val = 53
                    lf = lf.with_columns(
                        [
                            (2 * math.pi * pl.col(timestamp_col).dt.week() / max_val)
                            .sin()
                            .alias("woy_sin"),
                            (2 * math.pi * pl.col(timestamp_col).dt.week() / max_val)
                            .cos()
                            .alias("woy_cos"),
                        ]
                    )
                    result = self._register_cyclical_features(
                        result, "week_of_year", ["woy_sin", "woy_cos"]
                    )

            elif self.encoding_type == "ordinal":
                # Simple ordinal encoding (just extract the component)
                if component == "hour":
                    lf = lf.with_columns([pl.col(timestamp_col).dt.hour().alias("hour_ordinal")])
                    result = self._register_ordinal_feature(result, "hour_ordinal")
                elif component == "day_of_week":
                    lf = lf.with_columns([pl.col(timestamp_col).dt.weekday().alias("dow_ordinal")])
                    result = self._register_ordinal_feature(result, "dow_ordinal")
                elif component == "day_of_month":
                    lf = lf.with_columns([pl.col(timestamp_col).dt.day().alias("dom_ordinal")])
                    result = self._register_ordinal_feature(result, "dom_ordinal")
                elif component == "month":
                    lf = lf.with_columns([pl.col(timestamp_col).dt.month().alias("month_ordinal")])
                    result = self._register_ordinal_feature(result, "month_ordinal")
                elif component == "week_of_year":
                    lf = lf.with_columns([pl.col(timestamp_col).dt.week().alias("woy_ordinal")])
                    result = self._register_ordinal_feature(result, "woy_ordinal")

            # Note: One-hot encoding would expand columns significantly
            # and is better done at model training time

        return result.with_lazy_frame(lf)

    def _register_cyclical_features(
        self, event_frame: EventFrame, component: str, feature_names: list[str]
    ) -> EventFrame:
        """Register cyclical encoding features."""
        result = event_frame
        for feat_name in feature_names:
            provenance = FeatureProvenance(
                produced_by="CalendarEncodingStep",
                inputs=[event_frame.schema.timestamp_col],
                tags={"temporal"},
                description=f"Cyclical encoding for {component}",
            )
            result = result.register_feature(
                feat_name,
                {"source_step": "CalendarEncodingStep", "component": component},
                provenance=provenance,
            )
        return result

    def _register_ordinal_feature(self, event_frame: EventFrame, feat_name: str) -> EventFrame:
        """Register ordinal encoding feature."""
        provenance = FeatureProvenance(
            produced_by="CalendarEncodingStep",
            inputs=[event_frame.schema.timestamp_col],
            tags={"temporal"},
            description=f"Ordinal encoding: {feat_name}",
        )
        return event_frame.register_feature(
            feat_name,
            {"source_step": "CalendarEncodingStep", "type": "ordinal"},
            provenance=provenance,
        )


class FillNullStep(Step):
    """Fill nulls in named columns with a constant.

    This is a boundary policy made explicit, not a silent imputation:
    the caller must state why a null means this value. The phase-1 use
    case is a lag before the panel start (for example ``count_lag_1`` on
    the first day): no data exists before the observation window, so the
    lagged count is treated as 0 — "no events recorded". The step never
    invents a value for a null that has another meaning; use it only
    where the null's meaning is exactly "absent, counts as the fill
    value".

    Inputs:
        - value_cols to fill nulls in

    Outputs:
        - the same columns, with nulls replaced by ``fill_value``
    """

    def __init__(
        self,
        value_cols: Sequence[str],
        fill_value: float | int = 0,
    ) -> None:
        self.value_cols = list(value_cols)
        self.fill_value = fill_value
        if not self.value_cols:
            raise ValueError("value_cols must not be empty")
        if len(set(self.value_cols)) != len(self.value_cols):
            raise ValueError("value_cols must not contain duplicates")
        if not isinstance(fill_value, (int, float)) or isinstance(fill_value, bool):
            raise ValueError("fill_value must be a number")

    def run(self, event_frame: EventFrame) -> EventFrame:
        """Execute null filling."""
        schema = event_frame.lazy_frame.collect_schema()
        for value_col in self.value_cols:
            if value_col not in schema:
                raise ValueError(f"Column {value_col!r} is missing from the frame")
            if not schema[value_col].is_numeric():
                raise ValueError(f"Column {value_col!r} must be numeric, got {schema[value_col]}")
        lf = event_frame.lazy_frame.with_columns(
            [
                pl.col(value_col)
                .fill_null(pl.lit(self.fill_value).cast(schema[value_col]))
                .alias(value_col)
                for value_col in self.value_cols
            ]
        )
        result = event_frame.with_lazy_frame(lf)
        for value_col in self.value_cols:
            provenance = FeatureProvenance(
                produced_by="FillNullStep",
                inputs=[value_col],
                tags={"temporal"},
                description=f"Nulls in {value_col} filled with {self.fill_value}",
                metadata={"fill_value": self.fill_value},
            )
            result = result.register_feature(
                value_col,
                {"source_step": "FillNullStep", "fill_value": self.fill_value},
                provenance=provenance,
            )
        return result


class InterArrivalStep(Step):
    """Compute inter-arrival times between consecutive events.

    Inputs:
        - timestamp_col from EventSchema
        - Optional group_cols for per-group computation

    Outputs:
        - inter_arrival_seconds column
        - Optional rolling statistics of inter-arrival times
    """

    def __init__(
        self,
        group_cols: Sequence[str] | None = None,
        output_col: str = "inter_arrival_seconds",
        include_stats: bool = True,
        stat_windows: Sequence[int] | None = None,
    ) -> None:
        self.group_cols = list(group_cols) if group_cols else []
        self.output_col = output_col
        self.include_stats = include_stats
        self.stat_windows = list(stat_windows) if stat_windows else [10, 50]

    def run(self, event_frame: EventFrame) -> EventFrame:
        """Execute inter-arrival time computation."""
        timestamp_col = event_frame.schema.timestamp_col
        lf = event_frame.lazy_frame
        result = event_frame

        # Sort by timestamp (and group cols if specified)
        sort_cols = self.group_cols + [timestamp_col] if self.group_cols else [timestamp_col]
        lf = lf.sort(sort_cols)

        # Compute inter-arrival time
        if self.group_cols:
            lf = lf.with_columns(
                [
                    (pl.col(timestamp_col) - pl.col(timestamp_col).shift(1))
                    .dt.total_seconds()
                    .over(self.group_cols)
                    .alias(self.output_col)
                ]
            )
        else:
            lf = lf.with_columns(
                [
                    (pl.col(timestamp_col) - pl.col(timestamp_col).shift(1))
                    .dt.total_seconds()
                    .alias(self.output_col)
                ]
            )

        provenance = FeatureProvenance(
            produced_by="InterArrivalStep",
            inputs=[timestamp_col],
            tags={"temporal"},
            description="Inter-arrival time in seconds",
        )
        result = result.register_feature(
            self.output_col,
            {"source_step": "InterArrivalStep"},
            provenance=provenance,
        )

        # Add rolling statistics if requested
        if self.include_stats:
            for window in self.stat_windows:
                mean_col = f"{self.output_col}_ma_{window}"
                std_col = f"{self.output_col}_std_{window}"

                if self.group_cols:
                    lf = lf.with_columns(
                        [
                            pl.col(self.output_col)
                            .rolling_mean(window_size=window, min_samples=1)
                            .over(self.group_cols)
                            .alias(mean_col),
                            pl.col(self.output_col)
                            .rolling_std(window_size=window, min_samples=1)
                            .over(self.group_cols)
                            .alias(std_col),
                        ]
                    )
                else:
                    lf = lf.with_columns(
                        [
                            pl.col(self.output_col)
                            .rolling_mean(window_size=window, min_samples=1)
                            .alias(mean_col),
                            pl.col(self.output_col)
                            .rolling_std(window_size=window, min_samples=1)
                            .alias(std_col),
                        ]
                    )

                for stat_col in [mean_col, std_col]:
                    stat_provenance = FeatureProvenance(
                        produced_by="InterArrivalStep",
                        inputs=[self.output_col],
                        tags={"temporal"},
                        description=f"Rolling statistic of inter-arrival time (window={window})",
                    )
                    result = result.register_feature(
                        stat_col,
                        {"source_step": "InterArrivalStep", "window": window},
                        provenance=stat_provenance,
                    )

        return result.with_lazy_frame(lf)


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------


def _parse_time_unit_seconds(time_unit: str) -> float:
    return duration_seconds(time_unit)


def _validate_periods(periods: Sequence[int], mode: str) -> None:
    if mode not in {"duration", "rows"}:
        raise ValueError("mode must be 'duration' or 'rows'")
    if not periods or any(isinstance(p, bool) or not isinstance(p, int) or p <= 0 for p in periods):
        raise ValueError("Periods must be positive integers")


def _validate_time_keys(lf: pl.LazyFrame, keys: list[str]) -> None:
    data = lf.select(keys).collect()
    if any(data[col].null_count() for col in keys):
        raise ValueError("Temporal keys cannot contain nulls")
    if data.is_duplicated().any():
        raise ValueError("Duplicate temporal keys: aggregate or disambiguate each group/time first")
