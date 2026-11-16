"""Complete a sparse unit x time panel with explicit zeros.

A missing (unit, time-bin) pair is completed with an explicit zero for
every value column, but only inside the observation support: the units
kept are those observed in at least ``min_periods`` distinct time bins,
and the time axis runs from the earliest to the latest observed bin.
Pairs outside that support are simply not rows in the output — they are
never silently invented.

The step adds no information: it only makes absence explicit, so that
downstream windows and lags see a zero where nothing was recorded
instead of a gap they would have to guess about. Every row of the
output — observed or generated — carries ``observation_status =
"observed"``, matching the ``ObservationPanel`` states
("observed" / "missing" / "outside"): a generated pair is in-support
by construction.

Measurement columns (``measure_cols``) are the exception to the
zero-fill: a measurement is not a count, and a missing reading is not
a zero reading. Measure columns are carried through the (unit, time)
join like value columns, but they are never filled: generated pairs
get null and an observed null stays null, so the NA count per
measurement column is identical before and after completion. Declaring
a measurement in ``value_cols`` (or matching it with a
``value_prefixes`` entry) is an error, not a silent zero-fill.

Ordering guarantee: completion must run on an aggregated frame whose
time column is already binned (for example the ``time_bin`` produced by
``TimeBinsStep``), and before any feature derived from the time bin
itself (for example ``ExtractTemporalComponentsStep``). Columns other
than the unit, time, value and measure columns are classified per unit:
a column that is constant within each unit is a static attribute and
is carried to every pair of that unit; a column that varies within a
unit (for example a per-(unit, time) share or entropy) is joined on
(unit, time) instead, so observed pairs keep their value and generated
pairs get null — never silently collapsed with ``first()``.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING

import polars as pl
from pydantic import BaseModel, Field

from shikong.core._time import duration_seconds
from shikong.core.pipeline import Step
from shikong.core.schema import FeatureProvenance
from shikong.core.utils import get_logger

if TYPE_CHECKING:
    from shikong.core.event_frame import EventFrame

logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# Pydantic Config Model for Step Parameters
# -----------------------------------------------------------------------------


class CompletePanelConfig(BaseModel):
    """Configuration for panel completion step."""

    unit_col: str = Field(default="grid_id", description="Spatial unit column")
    time_col: str | None = Field(
        default=None,
        description="Time-bin column (defaults to the frame's schema.timestamp_col)",
    )
    bin_size: str | None = Field(
        default=None,
        description="Time-bin size (defaults to the frame's metadata.time_bin)",
    )
    value_cols: list[str] = Field(
        default=["count"], description="Numeric columns to zero-fill for generated pairs"
    )
    measure_cols: list[str] = Field(
        default_factory=list,
        description=(
            "Numeric measurement columns to carry through the (unit, time) "
            "join without zero-filling: generated pairs get null and an "
            "observed null stays null, so a missing reading is never "
            "turned into a zero reading"
        ),
    )
    value_prefixes: list[str] = Field(
        default_factory=list,
        description=(
            "Additional columns to zero-fill, matched by name prefix "
            "(for example 'count_' matches per-category count columns "
            "whose exact names depend on the data)"
        ),
    )
    min_periods: int = Field(
        default=1,
        ge=1,
        description="Minimum distinct time bins observed for a unit to enter the panel",
    )
    status_col: str = Field(
        default="observation_status", description="Output observation-status column name"
    )


# -----------------------------------------------------------------------------
# Panel Completion Step
# -----------------------------------------------------------------------------


class CompletePanelStep(Step):
    """Complete a sparse unit x time panel with explicit zeros.

    Inputs:
        - unit_col: spatial unit column
        - time_col: already-binned time column

    Outputs:
        - status_col ("observed" for every row: generated pairs are
          in-support by construction)
        - value_cols zero-filled for generated (unit, time-bin) pairs
        - measure_cols carried through the (unit, time) join, never
          zero-filled: null on generated pairs, nulls preserved on
          observed pairs
    """

    def __init__(
        self,
        unit_col: str = "grid_id",
        time_col: str | None = None,
        bin_size: str | None = None,
        value_cols: Sequence[str] | None = None,
        measure_cols: Sequence[str] | None = None,
        value_prefixes: Sequence[str] | None = None,
        min_periods: int = 1,
        status_col: str = "observation_status",
    ) -> None:
        self.unit_col = unit_col
        self.time_col = time_col
        self.bin_size = bin_size
        self.value_cols = list(value_cols) if value_cols is not None else ["count"]
        self.measure_cols = list(measure_cols) if measure_cols is not None else []
        self.value_prefixes = list(value_prefixes) if value_prefixes else []
        self.min_periods = min_periods
        self.status_col = status_col
        if not isinstance(min_periods, int) or isinstance(min_periods, bool) or min_periods < 1:
            raise ValueError("min_periods must be an integer >= 1")
        if not self.value_cols:
            raise ValueError("value_cols must not be empty")
        if any(not isinstance(p, str) or not p for p in self.value_prefixes):
            raise ValueError("value_prefixes must be non-empty strings")
        if len(set(self.value_cols)) != len(self.value_cols):
            raise ValueError("value_cols must not contain duplicates")
        if any(not isinstance(c, str) or not c for c in self.measure_cols):
            raise ValueError("measure_cols must be non-empty strings")
        if len(set(self.measure_cols)) != len(self.measure_cols):
            raise ValueError("measure_cols must not contain duplicates")
        if set(self.measure_cols) & set(self.value_cols):
            raise ValueError(
                "measure_cols must not overlap value_cols: a column is either "
                "a count (zero-filled) or a measurement (never zero-filled), not both"
            )
        if self.unit_col in self.value_cols or (
            self.time_col is not None and self.time_col in self.value_cols
        ):
            raise ValueError("value_cols must not include the unit or time column")
        if self.unit_col in self.measure_cols or (
            self.time_col is not None and self.time_col in self.measure_cols
        ):
            raise ValueError("measure_cols must not include the unit or time column")
        if self.unit_col == self.time_col or self.status_col in {
            self.unit_col,
            self.time_col,
            *self.value_cols,
            *self.measure_cols,
        }:
            raise ValueError(
                "unit_col, time_col, value_cols, measure_cols and status_col must be distinct"
            )
        if bin_size is not None:
            # Raises ValueError on unparseable values such as "1month".
            duration_seconds(bin_size)

    def run(self, event_frame: EventFrame) -> EventFrame:
        """Execute panel completion."""
        time_col = self.time_col or event_frame.schema.timestamp_col
        bin_size = self.bin_size or event_frame.metadata.time_bin
        if bin_size is None:
            raise ValueError(
                "CompletePanelStep needs a bin_size: pass bin_size explicitly or set "
                "metadata.time_bin (for example via TimeBinsStep)"
            )
        bin_seconds = duration_seconds(bin_size)
        unit_col = self.unit_col
        status_col = self.status_col

        lf = event_frame.lazy_frame
        schema = lf.collect_schema()

        if unit_col not in schema:
            raise ValueError(f"Unit column {unit_col!r} is missing from the frame")
        if time_col not in schema:
            raise ValueError(f"Time column {time_col!r} is missing from the frame")
        if status_col in schema:
            raise ValueError(
                f"Status column {status_col!r} already exists in the frame; "
                "choose a different status_col instead of overwriting it"
            )
        for value_col in self.value_cols:
            if value_col not in schema:
                raise ValueError(f"Value column {value_col!r} is missing from the frame")
        for measure_col in self.measure_cols:
            if measure_col not in schema:
                raise ValueError(f"Measure column {measure_col!r} is missing from the frame")
        # Prefix-matched value columns (for example per-category count
        # columns whose exact names depend on the data): resolved here,
        # once the frame schema is known.
        value_cols = list(self.value_cols)
        for name in schema.names():
            if (
                name not in value_cols
                and name not in {unit_col, time_col, status_col}
                and any(name.startswith(prefix) for prefix in self.value_prefixes)
            ):
                if name in self.measure_cols:
                    raise ValueError(
                        f"Value prefix matched measure column {name!r}: a column is "
                        "either a count (zero-filled) or a measurement "
                        "(never zero-filled), not both"
                    )
                value_cols.append(name)

        # Validate on the (unit, time) keys only, with small-result
        # aggregations: no key column is ever materialized and the full
        # frame stays lazy until the joins below (issue #39). The checks
        # below fire in the same order, with the same messages, as when
        # they ran on a collected key frame. The off-grid expression is
        # only added when the time column is Datetime (known from the
        # schema, no data read): on any other dtype the fused query would
        # fail inside ``.dt`` before the dtype check below could raise
        # its ValueError.
        bin_micros = round(bin_seconds * 1_000_000)
        stat_exprs = [
            pl.len().alias("_n_rows"),
            pl.col(unit_col).null_count().alias("_unit_nulls"),
            pl.col(time_col).null_count().alias("_time_nulls"),
            pl.col(time_col).min().alias("_min_time"),
            pl.col(time_col).max().alias("_max_time"),
            pl.struct([unit_col, time_col]).is_duplicated().any().alias("_has_duplicates"),
        ]
        if isinstance(schema[time_col], pl.Datetime):
            # Every observed bin must lie on the regular grid anchored at
            # the earliest bin; otherwise the time axis below would
            # silently drop the off-grid rows in the join. The earliest
            # bin is the column minimum, broadcast into the same row.
            stat_exprs.append(
                (
                    (pl.col(time_col) - pl.col(time_col).min()).dt.total_microseconds() % bin_micros
                    != 0
                )
                .any()
                .alias("_has_off_grid")
            )
        key_stats = lf.select(stat_exprs).collect().row(0, named=True)
        if key_stats["_n_rows"] == 0:
            raise ValueError("Cannot complete an empty frame: no units or time bins observed")
        # An empty frame infers Null dtypes, so dtype checks only make
        # sense once rows exist.
        if not isinstance(schema[time_col], pl.Datetime):
            raise ValueError(f"Time column {time_col!r} must be Datetime, got {schema[time_col]}")
        for value_col in value_cols:
            if not schema[value_col].is_numeric():
                raise ValueError(
                    f"Value column {value_col!r} must be numeric, got {schema[value_col]}"
                )
        for measure_col in self.measure_cols:
            if not schema[measure_col].is_numeric():
                raise ValueError(
                    f"Measure column {measure_col!r} must be numeric, got {schema[measure_col]}"
                )
        if key_stats["_unit_nulls"] or key_stats["_time_nulls"]:
            raise ValueError("Unit and time columns cannot contain nulls")
        if key_stats["_has_duplicates"]:
            raise ValueError(
                "Duplicate (unit, time) pairs in the input: aggregate to one row per "
                "(unit, time) pair before completing the panel"
            )
        if key_stats["_has_off_grid"]:
            raise ValueError(
                f"Time column {time_col!r} has bins that are not aligned to {bin_size!r} "
                f"from the earliest bin {key_stats['_min_time']}; bin the timestamps first "
                "(for example with TimeBinsStep)"
            )

        time_dtype = schema[time_col]
        assert isinstance(time_dtype, pl.Datetime)
        min_time = key_stats["_min_time"]
        max_time = key_stats["_max_time"]
        assert isinstance(min_time, datetime) and isinstance(max_time, datetime)

        # Units enter the panel only if observed in enough distinct bins.
        # One row per unit is collected: the per-unit bin counts, not the
        # key columns (issue #39).
        unit_counts = lf.group_by(unit_col).agg(pl.col(time_col).n_unique().alias("_n_bins"))
        kept = unit_counts.filter(pl.col("_n_bins") >= self.min_periods).collect()
        if kept.height == 0:
            raise ValueError(
                f"No spatial unit was observed in at least {self.min_periods} distinct "
                "time bins; lower min_periods or check the input"
            )
        n_units = kept.height

        # Columns other than the unit, time, value and measure columns are
        # classified per unit: constant within each unit means a static
        # attribute, carried to every pair of that unit; varying within
        # a unit means a per-(unit, time) value (for example a share or
        # an entropy), which is joined on (unit, time) instead — observed
        # pairs keep their value, generated pairs get null. Nothing is
        # ever silently collapsed with first().
        other_cols = [
            c
            for c in schema.names()
            if c not in {unit_col, time_col, *value_cols, *self.measure_cols}
        ]
        static_cols: list[str] = []
        varying_cols: list[str] = []
        if other_cols:
            # n_unique counts null as one distinct value, so a column that
            # is null everywhere still passes as constant. The aggregation
            # returns one row per unit, so the other columns are never
            # materialized in full (issue #39).
            varying = (
                lf.select([unit_col, *other_cols])
                .group_by(unit_col)
                .agg([pl.col(c).n_unique().alias(c) for c in other_cols])
                .select([pl.col(c).max().alias(c) for c in other_cols])
                .collect()
                .row(0, named=True)
            )
            for c in other_cols:
                (static_cols if varying[c] <= 1 else varying_cols).append(c)

        # Regular time axis from the earliest to the latest observed bin,
        # end included when it aligns (it does: alignment was checked).
        axis = pl.datetime_range(
            min_time,
            max_time,
            bin_size,
            eager=True,
            time_unit=time_dtype.time_unit,
            time_zone=time_dtype.time_zone,
        )
        assert isinstance(axis, pl.Series)
        n_bins = axis.len()
        logger.info(
            f"Completing panel: {n_units} units x {n_bins} bins "
            f"(bin_size={bin_size}, min_periods={self.min_periods})"
        )

        units_lf = kept.select(unit_col).lazy()
        axis_lf = axis.to_frame(time_col).lazy()
        panel = units_lf.join(axis_lf, how="cross")
        if static_cols:
            static_lf = lf.group_by(unit_col).agg([pl.col(c).first().alias(c) for c in static_cols])
            panel = panel.join(static_lf, on=unit_col, how="left", validate="m:1")
        observed = lf.select([unit_col, time_col, *value_cols, *self.measure_cols, *varying_cols])
        result_lf = (
            panel.join(observed, on=[unit_col, time_col], how="left", validate="1:1")
            .with_columns(
                [
                    pl.col(value_col).fill_null(0).cast(schema[value_col]).alias(value_col)
                    for value_col in value_cols
                ]
                # Measure columns are deliberately NOT filled: a generated
                # pair gets null and an observed null stays null, so a
                # missing reading is never turned into a zero reading.
                + [pl.lit("observed").alias(status_col)]
            )
            .sort([unit_col, time_col])
        )

        provenance = FeatureProvenance(
            produced_by="CompletePanelStep",
            inputs=[unit_col, time_col, *value_cols, *self.measure_cols],
            tags={"temporal", "spatial"},
            description=(
                "Observation status: every (unit, time-bin) pair in the completed "
                "panel is in-support, so generated pairs are 'observed' with "
                "explicit zero values for count columns and null for "
                "measurement columns"
            ),
            metadata={
                "min_periods": self.min_periods,
                "bin_size": bin_size,
                "n_units": n_units,
                "n_bins": n_bins,
                "measure_cols": list(self.measure_cols),
            },
        )
        result = event_frame.with_lazy_frame(result_lf)
        return result.register_feature(
            status_col,
            {"source_step": "CompletePanelStep", "status": "observed"},
            provenance=provenance,
        )
