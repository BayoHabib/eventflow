"""Near-repeat pipeline step.

For each (cell, bin ``t``), counts of events in queen rings 1..3 over
trailing time windows, computed from the completed panel counts
(explicit zeros make this exact).

Causality: both windows are **closed-left** (strictly before ``t``),
so the value at bin ``t`` does not depend on the count at ``t``
itself, and no bin after ``t`` is ever used. The feature row at ``t``
predicts the count in bin ``t + 1``, so a closed-left window is the
conservative choice here: only strictly past bins contribute.

Windows (fixed by the Phase 2 design note, in days):

- trailing 7 d: bins ``t - 7d .. t - 1 bin``;
- trailing 7-14 d: bins ``t - 14d .. t - 7d - 1 bin``, computed as
  (trailing 14 d sum) - (trailing 7 d sum), both closed-left.

The day windows are converted to bin counts using the frame's
``metadata.time_bin`` (a 7 d window on a 6 h panel is 28 bins, on a
daily panel 7 bins). The panel must therefore be regular and complete
(one row per (cell, bin), no gaps): positional rolling sums over bins
are only equal to time windows on a regular grid, which is exactly
what ``CompletePanelStep`` produces.

Early bins: before the panel start there are no bins, so the trailing
sums are taken over the bins that exist (a sum over zero available
bins is 0). Explicit zeros before the panel start do not exist and are
not invented.

Islands: a ring with no neighbours for a cell yields 0 (not null) for
that ring's counts. This differs deliberately from the spatial-lag
island -> null rule: a ring count is a *sum* of events over existing
neighbours, and zero neighbours means zero nearby events, whereas a
neighbour *mean* is undefined when there is no neighbour to average
over.

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
from pydantic import BaseModel, Field, field_validator

from shikong.core._time import duration_seconds
from shikong.core.pipeline import Step
from shikong.core.schema import FeatureProvenance
from shikong.core.utils import get_logger

if TYPE_CHECKING:
    from shikong.core.event_frame import EventFrame

logger = get_logger(__name__)

# Fixed window lengths for this version, in days (Phase 2 design note,
# Decision 4). If a caller needs other windows, that is a later change.
_WINDOW_7D_DAYS = 7
_WINDOW_14D_DAYS = 14


# -----------------------------------------------------------------------------
# Pydantic Config Model for Step Parameters
# -----------------------------------------------------------------------------


class NearRepeatConfig(BaseModel):
    """Configuration for the near-repeat step."""

    value_col: str = Field(default="count", description="Count column to sum over neighbours")
    rings: list[int] = Field(
        default=[1, 2, 3],
        description="Queen rings to report (ring r = cells at Chebyshev distance exactly r)",
    )
    unit_col: str = Field(default="grid_id", description="Spatial unit column")
    time_col: str = Field(default="time_bin", description="Time-bin column")

    @field_validator("rings")
    @classmethod
    def _validate_rings(cls, rings: list[int]) -> list[int]:
        _validate_rings(rings)
        return rings


def _validate_rings(rings: Sequence[int]) -> list[int]:
    """Validate a rings sequence; return it as a list.

    Rings must be non-empty, contain no duplicates, and every ring must
    be a positive integer (bools are rejected: ``True`` is not a ring).
    """
    if isinstance(rings, (str, bytes)) or rings is None:
        raise ValueError(f"rings must be a non-empty sequence of positive ints, got {rings!r}")
    rings_list = list(rings)
    if not rings_list:
        raise ValueError("rings must not be empty")
    for ring in rings_list:
        if isinstance(ring, bool) or not isinstance(ring, int) or ring < 1:
            raise ValueError(f"rings must be positive ints, got {ring!r}")
    if len(set(rings_list)) != len(rings_list):
        raise ValueError(f"rings must not contain duplicates, got {rings_list!r}")
    return rings_list


# -----------------------------------------------------------------------------
# Near-Repeat Step
# -----------------------------------------------------------------------------


class NearRepeatStep(Step):
    """Near-repeat counts: events in queen rings over trailing windows.

    For each (cell, bin ``t``) and each configured queen ring ``r``,
    the number of events in the ring over the trailing 7 d and over
    the trailing 7-14 d, both closed-left (strictly before ``t``),
    plus a binary flag that is 1 when any event occurred in ring 1 in
    the trailing 7 d.

    Inputs:
        - value_col (default "count") on the completed panel
        - unit_col (default "grid_id"), time_col (default "time_bin")
        - grid_row / grid_col: the grid coordinates of each unit, used
          to build the (cell, ring) neighbour edge table once

    Outputs:
        - {value_col}_nr_r{r}_7d (count of events in ring r, trailing 7 d)
        - {value_col}_nr_r{r}_7_14d (count of events in ring r, trailing
          7-14 d)
        - {value_col}_nr_event_7d (Int64 0/1: any event in ring 1,
          trailing 7 d)

    Window policy: both windows are closed-left. Early bins sum over
    the bins that exist (0 when the panel starts). A ring with no
    neighbours yields 0, not null (see the module docstring).

    Execution: this step is eager — each ring's edge join + group_by
    runs on its own and the result is collected before returning, so
    the ring joins never run together with the KDE joins at the final
    collect of the pipeline (that combination OOM-kills a 6 GB VM on a
    Chicago-size panel).
    """

    def __init__(
        self,
        value_col: str = "count",
        rings: Sequence[int] = (1, 2, 3),
        unit_col: str = "grid_id",
        time_col: str = "time_bin",
    ) -> None:
        if not isinstance(value_col, str) or not value_col:
            raise ValueError(f"value_col must be a non-empty string, got {value_col!r}")
        if not isinstance(unit_col, str) or not unit_col:
            raise ValueError(f"unit_col must be a non-empty string, got {unit_col!r}")
        if not isinstance(time_col, str) or not time_col:
            raise ValueError(f"time_col must be a non-empty string, got {time_col!r}")
        if len({value_col, unit_col, time_col}) != 3:
            raise ValueError("value_col, unit_col and time_col must be distinct")
        self.value_col = value_col
        self.rings = _validate_rings(rings)
        self.unit_col = unit_col
        self.time_col = time_col

    def run(self, event_frame: EventFrame) -> EventFrame:
        """Execute near-repeat computation."""
        lf = event_frame.lazy_frame
        value_col = self.value_col
        unit_col = self.unit_col
        time_col = self.time_col

        _require_columns(lf, [value_col, unit_col, time_col, "grid_row", "grid_col"])
        schema = lf.collect_schema()
        if not isinstance(schema[time_col], pl.Datetime):
            raise ValueError(f"Time column {time_col!r} must be Datetime, got {schema[time_col]}")
        if not schema[value_col].is_numeric():
            raise ValueError(f"Value column {value_col!r} must be numeric, got {schema[value_col]}")

        time_bin = event_frame.metadata.time_bin
        if time_bin is None:
            raise ValueError(
                "NearRepeatStep requires metadata.time_bin to convert the 7 d / "
                "14 d windows to bins (run TimeBinsStep or ObservationPanelStep first)"
            )
        try:
            bin_seconds = duration_seconds(time_bin)
        except ValueError as exc:
            raise ValueError(
                f"NearRepeatStep cannot parse metadata.time_bin {time_bin!r}: {exc}"
            ) from exc
        window_7_bins = _days_to_bins(_WINDOW_7D_DAYS, bin_seconds, time_bin)
        window_14_bins = _days_to_bins(_WINDOW_14D_DAYS, bin_seconds, time_bin)

        # Blocking validation of the value column: null / non-finite
        # counts are an error, never silently treated as zero. This is
        # a one-row aggregate collect (two booleans); the panel itself
        # stays lazy for the computation below.
        _validate_values(lf, value_col, unit_col, time_col)

        # Ring 1 is always computed internally: the binary flag uses
        # ring 1 even when the caller only asked for other rings.
        compute_rings = sorted(set(self.rings) | {1})

        # Eager evaluation, one ring at a time (memory-bounded). The
        # values frame (unit, time, value) is collected once, then each
        # ring's edge join + group_by runs on its own and is freed
        # before the next ring. A single join over all rings at once
        # builds a ~91M-row intermediate on a Chicago-size panel, and
        # collecting it together with the KDE joins OOM-kills the 6 GB
        # VM (phase 2, 2026-11-15). The widest single ring is ~46M rows.
        values = lf.select([unit_col, time_col, value_col]).collect()
        keys = values.select([unit_col, time_col]).unique(maintain_order=True)
        edges = _ring_edges(lf, unit_col, compute_rings).collect()

        def _ring_sums(ring: int) -> pl.DataFrame:
            """Trailing 7 d / 7-14 d ring counts for one ring (eager).

            Ring counts per (cell, bin) are the neighbour values summed
            over the ring's edges; the full (cell, bin) grid is then
            filled with 0 where a ring has no neighbours for a cell.
            Trailing rolling sums are closed-left: shift by one bin
            first (so the current bin never contributes), then roll over
            the previous bins. The 7-14 d window is (trailing 14 d) -
            (trailing 7 d), both closed-left. The shift's leading null
            per cell is 0: no bin exists before the panel start.
            Returns columns [unit_col, time_col, _sum_7d, _sum_7_14d].
            """
            counts = (
                edges.filter(pl.col("_ring") == ring)
                .join(
                    values.rename({unit_col: "_neighbour"}),
                    on="_neighbour",
                    how="inner",
                )
                .group_by(["_cell", time_col])
                .agg(pl.col(value_col).sum().alias("_ring_count"))
            )
            full = (
                keys.rename({unit_col: "_cell"})
                .join(counts, on=["_cell", time_col], how="left")
                .with_columns(pl.col("_ring_count").fill_null(0))
                .sort(["_cell", time_col], maintain_order=True)
            )
            return (
                full.with_columns(
                    pl.col("_ring_count").shift(1).over("_cell").fill_null(0).alias("_prev")
                )
                .with_columns(
                    [
                        pl.col("_prev")
                        .rolling_sum(window_size=window_7_bins, min_samples=1)
                        .over("_cell")
                        .alias("_sum_7d"),
                        pl.col("_prev")
                        .rolling_sum(window_size=window_14_bins, min_samples=1)
                        .over("_cell")
                        .alias("_sum_14d"),
                    ]
                )
                .with_columns((pl.col("_sum_14d") - pl.col("_sum_7d")).alias("_sum_7_14d"))
                .select(
                    [
                        pl.col("_cell").alias(unit_col),
                        time_col,
                        "_sum_7d",
                        "_sum_7_14d",
                    ]
                )
            )

        result_lf = lf
        result = event_frame
        for ring in self.rings:
            col_7d = f"{value_col}_nr_r{ring}_7d"
            col_7_14d = f"{value_col}_nr_r{ring}_7_14d"
            ring_lf = (
                _ring_sums(ring)
                .select(
                    [
                        unit_col,
                        time_col,
                        pl.col("_sum_7d").alias(col_7d),
                        pl.col("_sum_7_14d").alias(col_7_14d),
                    ]
                )
                .lazy()
            )
            result_lf = result_lf.join(
                ring_lf,
                on=[unit_col, time_col],
                how="left",
                validate="1:1",
                maintain_order="left",
            ).with_columns(
                [
                    pl.col(col_7d).fill_null(0).alias(col_7d),
                    pl.col(col_7_14d).fill_null(0).alias(col_7_14d),
                ]
            )
            for out_col, window in ((col_7d, "7d"), (col_7_14d, "7-14d")):
                result = _register(
                    result,
                    out_col,
                    [value_col, unit_col, time_col, "grid_row", "grid_col"],
                    f"Count of {value_col} in queen ring {ring} over the trailing "
                    f"{window} window (closed-left, strictly before the bin)",
                    {
                        "value_col": value_col,
                        "rings": self.rings,
                        "ring": ring,
                        "window": window,
                        "closed": "left",
                        "windows_days": [_WINDOW_7D_DAYS, _WINDOW_14D_DAYS],
                        "time_bin": time_bin,
                        "window_bins_7d": window_7_bins,
                        "window_bins_14d": window_14_bins,
                        "islands": "zero",
                    },
                )

        # Binary flag: any event in ring 1 in the trailing 7 d.
        flag_col = f"{value_col}_nr_event_7d"
        ring1_7d = f"{value_col}_nr_r1_7d"
        if 1 in self.rings:
            # Ring 1 was already joined above; derive the flag from it.
            result_lf = result_lf.with_columns(
                (pl.col(ring1_7d) > 0).cast(pl.Int64).alias(flag_col)
            )
        else:
            ring1_lf = (
                _ring_sums(1)
                .select(
                    [
                        unit_col,
                        time_col,
                        pl.col("_sum_7d").alias("_ring1_7d"),
                    ]
                )
                .lazy()
            )
            result_lf = (
                result_lf.join(
                    ring1_lf,
                    on=[unit_col, time_col],
                    how="left",
                    validate="1:1",
                    maintain_order="left",
                )
                .with_columns((pl.col("_ring1_7d").fill_null(0) > 0).cast(pl.Int64).alias(flag_col))
                .drop("_ring1_7d")
            )
        result = _register(
            result,
            flag_col,
            [value_col, unit_col, time_col, "grid_row", "grid_col"],
            f"1 if any {value_col} > 0 in queen ring 1 over the trailing 7 d "
            "window (closed-left), else 0",
            {
                "value_col": value_col,
                "rings": self.rings,
                "ring": 1,
                "window": "7d",
                "closed": "left",
                "windows_days": [_WINDOW_7D_DAYS, _WINDOW_14D_DAYS],
                "time_bin": time_bin,
                "window_bins_7d": window_7_bins,
                "window_bins_14d": window_14_bins,
                "islands": "zero",
            },
        )

        # Eager: collect the result now so the ring joins are executed
        # here, one ring at a time, instead of being deferred to the
        # final collect of the whole pipeline — where they would run
        # together with the KDE joins and OOM the 6 GB VM (phase 2).
        # Downstream steps then read a materialized frame.
        return result.with_lazy_frame(result_lf.collect().lazy())


def _register(
    result: EventFrame,
    out_col: str,
    inputs: list[str],
    description: str,
    metadata: dict[str, object],
) -> EventFrame:
    provenance = FeatureProvenance(
        produced_by="NearRepeatStep",
        inputs=inputs,
        tags={"spatial", "temporal"},
        description=description,
        metadata=metadata,
    )
    return result.register_feature(
        out_col,
        {"source_step": "NearRepeatStep", **metadata},
        provenance=provenance,
    )


def _days_to_bins(days: int, bin_seconds: float, time_bin: str) -> int:
    """Convert a window in days to an exact number of bins."""
    bins = days * 86400.0 / bin_seconds
    if not math.isfinite(bins) or bins < 1 or bins != math.floor(bins):
        raise ValueError(
            f"NearRepeatStep cannot express a {days}d window in whole bins: "
            f"time_bin {time_bin!r} ({bin_seconds:g}s) does not divide {days}d evenly"
        )
    return int(bins)


def _validate_values(lf: pl.LazyFrame, value_col: str, unit_col: str, time_col: str) -> None:
    """Raise on null/non-finite values and on null grid coordinates."""
    check = (
        lf.select(
            [
                pl.col(value_col).is_null().any().alias("_value_null"),
                pl.col(value_col).cast(pl.Float64).is_finite().not_().any().alias("_value_bad"),
                pl.col("grid_row").is_null().any().alias("_row_null"),
                pl.col("grid_col").is_null().any().alias("_col_null"),
                # The time column drives the trailing windows: a null
                # time silently drops the row from its cell's series and
                # shifts every window (the pre-completion "time_bin"
                # column is null on generated panel rows — use the
                # completed panel's timestamp column instead).
                pl.col(time_col).is_null().any().alias("_time_null"),
            ]
        )
        .collect()
        .row(0, named=True)
    )
    if check["_value_null"] or check["_value_bad"]:
        raise ValueError(
            f"NearRepeatStep requires finite observed values in {value_col!r}; "
            "null or non-finite values are a blocking error (complete the panel first)"
        )
    if check["_row_null"] or check["_col_null"]:
        raise ValueError("NearRepeatStep requires non-null grid_row / grid_col for every unit")
    if check["_time_null"]:
        raise ValueError(
            f"NearRepeatStep requires non-null times in {time_col!r}; "
            "use the completed panel's timestamp column (the pre-completion "
            "'time_bin' column is null on generated rows)"
        )


def _ring_edges(lf: pl.LazyFrame, unit_col: str, rings: list[int]) -> pl.LazyFrame:
    """(cell, neighbour, ring) edge table, built once from grid_row/grid_col.

    For each ring ``r`` and each offset ``(dr, dc)`` with
    ``max(|dr|, |dc|) == r``, a cell's neighbour at that offset is found
    by joining the unit table to a copy of itself shifted by the offset.
    Pure lazy Polars: no per-row Python loop, no panel collect.
    """
    units = lf.select([unit_col, "grid_row", "grid_col"]).unique(
        subset=[unit_col], keep="first", maintain_order=True
    )
    edge_frames: list[pl.LazyFrame] = []
    for ring in rings:
        for dr in range(-ring, ring + 1):
            for dc in range(-ring, ring + 1):
                if max(abs(dr), abs(dc)) != ring:
                    continue
                neighbours = units.select(
                    [
                        pl.col(unit_col).alias("_neighbour"),
                        (pl.col("grid_row") - dr).alias("_row_key"),
                        (pl.col("grid_col") - dc).alias("_col_key"),
                    ]
                )
                edge_frames.append(
                    units.join(
                        neighbours,
                        left_on=["grid_row", "grid_col"],
                        right_on=["_row_key", "_col_key"],
                        how="inner",
                    ).select(
                        [
                            pl.col(unit_col).alias("_cell"),
                            "_neighbour",
                            pl.lit(ring, dtype=pl.Int64).alias("_ring"),
                        ]
                    )
                )
    return pl.concat(edge_frames, how="vertical")


def _require_columns(lf: pl.LazyFrame, columns: Sequence[str]) -> None:
    missing = sorted(set(columns) - set(lf.collect_schema().names()))
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")
