"""Gaussian kernel density (KDE) and space-time KDE (STKDE) steps.

Both steps operate on the completed observation panel (one row per
unit x time bin, explicit zeros, never null counts) and smooth the
values of a caller-chosen value column over space with a Gaussian
kernel. They supersede the neighbour-count approximation in
``KDEStep`` (``core/steps/spatial.py``), which ignores the kernel
shape; that step is kept for backwards compatibility.

Global causality convention
---------------------------
A feature row at bin ``t`` is for predicting the count in bin
``t + 1``, so no feature may use data from bins AFTER ``t``.

- ``KernelDensityStep`` is a **same-bin smoother**: the density at
  bin ``t`` only uses values from bin ``t`` (within each
  ``group_cols`` group). Fed lagged counts (for example
  ``count_lag_1``) it is a predictor feature; fed current-bin counts
  it is a same-bin descriptor, not a predictor feature. The caller
  chooses which one by choosing the value column.
- ``SpaceTimeKernelDensityStep`` looks backward in time only: the
  density at bin ``t`` sums the per-bin spatial densities of bins
  ``s <= t`` with exponential weights (weight 1 at lag 0, decaying
  backward). No bin after ``t`` is ever used.

Kernel and truncation
---------------------
Gaussian kernel, density in events per m^2 at each cell centre::

    density(i, t) = sum_j w_j * exp(-d_ij^2 / 2h^2) / (2 * pi * h^2)

where ``w_j`` are the values of the caller-chosen value column,
``d_ij`` is the Euclidean distance between cell centres (projected
metres; a metric CRS is required) and ``h`` is the bandwidth in
metres, declared explicitly by the caller. There is no automatic
bandwidth estimation in Phase 2: estimation would be a fitted
statistic on the full panel, i.e. a leakage surface.

The kernel is truncated at ``3h`` (documented approximation; the
weight beyond ``3h`` is under 1.2% of the peak). Every cell is its
own neighbour at distance 0, so the density is never null for an
observed cell, even an isolated one.

No edge correction is implemented in Phase 2: density near the
boundary of the observed support is underestimated because
neighbours outside the support do not exist.

Implementation
--------------
The neighbour edge table (cells within ``3h``, with the kernel
weight) is built once from the distinct cell centres with a lazy
cross join + distance filter, then one join of the values onto the
edges + one ``group_by`` in pure Polars. No per-row Python loop, no
``collect()`` of the panel. The only ``collect()`` calls are
one-row validation aggregates (null / non-finite value checks and
the one-centre-per-unit check); the panel itself stays lazy.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import polars as pl
from pydantic import BaseModel, Field

from eventflow.core._time import duration_seconds
from eventflow.core.grid import require_metric_crs
from eventflow.core.pipeline import Step
from eventflow.core.schema import FeatureProvenance
from eventflow.core.utils import get_logger

if TYPE_CHECKING:
    from eventflow.core.event_frame import EventFrame

logger = get_logger(__name__)

# Truncation radius in bandwidths. exp(-3^2 / 2) is ~1.1% of the peak.
TRUNCATION_BANDWIDTHS = 3.0

# Maximum edges per chunk in _spatial_kde_frame's edge join: joining
# the whole edge table against every group builds a ~68M-row
# intermediate at Chicago scale (186k edges x 364 bins) that OOM-kills
# a 6 GB VM together with the panel (phase 2, 2026-11-15). Chunking by
# 50k edges bounds each join to ~18M rows.
_KDE_JOIN_CHUNK_EDGES = 50_000


# -----------------------------------------------------------------------------
# Pydantic Config Models for Step Parameters
# -----------------------------------------------------------------------------


class KernelDensityConfig(BaseModel):
    """Configuration for Gaussian kernel density estimation step."""

    value_col: str = Field(default="count", description="Value column to smooth")
    bandwidth_m: float = Field(
        default=600.0, gt=0, description="Gaussian bandwidth h in metres (must be > 0)"
    )
    unit_col: str = Field(default="grid_id", description="Spatial unit identifier column")
    group_cols: list[str] = Field(
        default_factory=list,
        description="Columns defining separate snapshots (e.g. ['time_bin']); "
        "values are never pooled across groups",
    )


class SpaceTimeKernelDensityConfig(BaseModel):
    """Configuration for space-time kernel density estimation step."""

    value_col: str = Field(default="count", description="Value column to smooth")
    bandwidth_m: float = Field(
        default=600.0, gt=0, description="Spatial Gaussian bandwidth hs in metres (must be > 0)"
    )
    halflife_days: float = Field(
        default=7.0, gt=0, description="Temporal half-life ht in days (must be > 0)"
    )
    unit_col: str = Field(default="grid_id", description="Spatial unit identifier column")
    time_col: str = Field(default="time_bin", description="Time-bin column to smooth over")


# -----------------------------------------------------------------------------
# Density Steps
# -----------------------------------------------------------------------------


class KernelDensityStep(Step):
    """Gaussian kernel density of a value column at each cell centre.

    Causality: this is a **same-bin smoother**. The density at bin
    ``t`` only uses values from bin ``t`` (within each ``group_cols``
    group). Fed lagged counts (for example ``count_lag_1``) it is a
    predictor feature; fed current-bin counts it is a same-bin
    descriptor, not a predictor feature. No bin after ``t`` is ever
    used, and values are never pooled across ``group_cols`` groups.

    The bandwidth ``h`` is declared explicitly by the caller (no
    automatic estimation). The kernel is truncated at ``3h``; every
    cell is its own neighbour at distance 0, so the density is never
    null for an observed cell. No edge correction is applied: density
    near the boundary of the observed support is underestimated.

    Inputs:
        - value_col on the completed panel (one row per unit x group)
        - cell-centre coordinates in the frame's schema lon/lat
          columns (projected metres; a metric CRS is required)

    Outputs:
        - {value_col}_kde (Float64): density in events per m^2

    Execution: this step is eager — the result is collected before
    returning, so the edge join runs on its own instead of being
    deferred to the final collect of the pipeline together with the
    other giant joins (see the comment at the end of ``run()``).
    """

    def __init__(
        self,
        value_col: str = "count",
        bandwidth_m: float = 600.0,
        unit_col: str = "grid_id",
        group_cols: Sequence[str] | None = None,
    ) -> None:
        _validate_column_name("value_col", value_col)
        _validate_column_name("unit_col", unit_col)
        if value_col == unit_col:
            raise ValueError("value_col and unit_col must be distinct")
        _validate_bandwidth(bandwidth_m)
        self.value_col = value_col
        self.bandwidth_m = float(bandwidth_m)
        self.unit_col = unit_col
        self.group_cols = list(group_cols) if group_cols else []
        if self.unit_col in self.group_cols or self.value_col in self.group_cols:
            raise ValueError("group_cols must not contain the unit or value column")

    def run(self, event_frame: EventFrame) -> EventFrame:
        """Execute Gaussian kernel density computation."""
        lf = event_frame.lazy_frame
        _require_columns(lf, [self.unit_col, self.value_col, *self.group_cols])
        centres = _cell_centres(event_frame, lf, self.unit_col)
        _validate_values(lf, self.value_col)
        _validate_no_nulls(lf, self.group_cols, "KernelDensityStep")

        output_col = f"{self.value_col}_kde"
        _require_output_free(lf, output_col)
        result_lf = _spatial_kde_frame(
            lf,
            unit_col=self.unit_col,
            value_col=self.value_col,
            group_cols=self.group_cols,
            centres=centres,
            bandwidth_m=self.bandwidth_m,
            output_col=output_col,
        )

        provenance = FeatureProvenance(
            produced_by="KernelDensityStep",
            inputs=[self.value_col, self.unit_col],
            tags={"spatial"},
            description=(
                f"Gaussian kernel density of {self.value_col} "
                f"(bandwidth {self.bandwidth_m:g} m, truncated at 3h); "
                "same-bin smoother: fed lagged counts it is a predictor "
                "feature, fed current-bin counts a same-bin descriptor"
            ),
            metadata={
                "value_col": self.value_col,
                "bandwidth_m": self.bandwidth_m,
                "kernel": "gaussian",
                "truncation_bandwidths": TRUNCATION_BANDWIDTHS,
                "group_cols": self.group_cols,
                "causality": "same-bin",
                "edge_correction": None,
            },
        )
        result = event_frame.register_feature(
            output_col,
            {
                "source_step": "KernelDensityStep",
                "value_col": self.value_col,
                "bandwidth_m": self.bandwidth_m,
            },
            provenance=provenance,
        )
        # Eager: collect the result now so the edge join (edges x bins)
        # runs here, on its own, instead of being deferred to the final
        # collect of the pipeline — where it would run together with the
        # near-repeat and STKDE joins and OOM a 6 GB VM on a
        # Chicago-size panel (phase 2, 2026-11-15). Downstream steps
        # then read a materialized frame.
        return result.with_lazy_frame(result_lf.collect().lazy())


class SpaceTimeKernelDensityStep(Step):
    """Space-time kernel density: Gaussian in space, exponential in time.

    Separable product kernel: the per-bin spatial KDE (same Gaussian
    kernel as ``KernelDensityStep``) is smoothed over time with
    exponential decay of half-life ``halflife_days``::

        stkde(i, t) = sum_{s <= t} kde(i, s) * 0.5 ** ((t - s) / ht_bins)

    The temporal kernel only looks backward, including the current
    bin with weight 1 at lag 0 (same "includes current bin" precedent
    as ``EWMAMeanStep``). No bin after ``t`` is ever used.

    The half-life in days is converted to a half-life in bins using
    the frame's ``metadata.time_bin`` (so a ``7d`` half-life on a
    daily panel is 7 bins, on a 6-hour panel 28 bins). The smoothing
    assumes a regular completed panel (one row per unit per bin, no
    gaps), so the per-unit bin index is the elapsed time in bins.

    The time smoothing is exact, not an approximation: with
    ``beta = 0.5 ** (1 / halflife_bins)`` and ``n`` the 1-based bin
    index within the unit, the adjusted EWMA
    ``ewm_mean(half_life=halflife_bins, adjust=True)`` is the
    weight-normalised mean, so multiplying by the weight sum
    ``(1 - beta**n) / (1 - beta)`` recovers the unnormalised
    discounted sum above. (A plain ``adjust=False`` EWMA rescaled by
    ``1 / (1 - beta)`` would NOT be exact: its first value carries
    weight 1 instead of ``1 - beta``.)

    Inputs:
        - value_col on the completed panel (one row per unit x bin)
        - cell-centre coordinates in the frame's schema lon/lat
          columns (projected metres; a metric CRS is required)

    Outputs:
        - {value_col}_stkde (Float64): density in events per m^2

    Execution: this step is eager, as ``KernelDensityStep`` — the
    result is collected before returning.
    """

    def __init__(
        self,
        value_col: str = "count",
        bandwidth_m: float = 600.0,
        halflife_days: float = 7.0,
        unit_col: str = "grid_id",
        time_col: str = "time_bin",
    ) -> None:
        _validate_column_name("value_col", value_col)
        _validate_column_name("unit_col", unit_col)
        _validate_column_name("time_col", time_col)
        if value_col == unit_col or value_col == time_col or unit_col == time_col:
            raise ValueError("value_col, unit_col and time_col must be distinct")
        _validate_bandwidth(bandwidth_m)
        if not math.isfinite(halflife_days) or halflife_days <= 0:
            raise ValueError(f"halflife_days must be a finite number > 0, got {halflife_days!r}")
        self.value_col = value_col
        self.bandwidth_m = float(bandwidth_m)
        self.halflife_days = float(halflife_days)
        self.unit_col = unit_col
        self.time_col = time_col

    def run(self, event_frame: EventFrame) -> EventFrame:
        """Execute space-time kernel density computation."""
        lf = event_frame.lazy_frame
        _require_columns(lf, [self.unit_col, self.value_col, self.time_col])
        centres = _cell_centres(event_frame, lf, self.unit_col)
        _validate_values(lf, self.value_col)
        _validate_no_nulls(lf, [self.time_col], "SpaceTimeKernelDensityStep")
        halflife_bins = _halflife_bins(
            event_frame, self.halflife_days, "SpaceTimeKernelDensityStep"
        )

        output_col = f"{self.value_col}_stkde"
        _require_output_free(lf, output_col)
        # Private intermediate: dropped before returning, so running
        # this step never leaves a stray KDE column behind.
        kde_col = f"__{self.value_col}_kde_for_stkde"
        _require_output_free(lf, kde_col)

        lf = _spatial_kde_frame(
            lf,
            unit_col=self.unit_col,
            value_col=self.value_col,
            group_cols=[self.time_col],
            centres=centres,
            bandwidth_m=self.bandwidth_m,
            output_col=kde_col,
        )

        # Exact discounted sum via the adjusted EWMA (see class
        # docstring): ewm_mean(adjust=True) is the weight-normalised
        # mean, so multiply back by the sum of the weights over the
        # bins seen so far, (1 - beta**n) / (1 - beta).
        beta = 0.5 ** (1.0 / halflife_bins)
        alpha = 1.0 - beta
        lf = lf.sort([self.unit_col, self.time_col], maintain_order=True)
        n_bins_seen = pl.col(kde_col).cum_count().over(self.unit_col)
        ewma = (
            pl.col(kde_col)
            .ewm_mean(half_life=halflife_bins, adjust=True, min_samples=1)
            .over(self.unit_col)
        )
        lf = lf.with_columns(
            (ewma * (1.0 - pl.lit(beta).pow(n_bins_seen)) / alpha).alias(output_col)
        ).drop(kde_col)

        provenance = FeatureProvenance(
            produced_by="SpaceTimeKernelDensityStep",
            inputs=[self.value_col, self.unit_col, self.time_col],
            tags={"spatial", "temporal"},
            description=(
                f"Space-time kernel density of {self.value_col} "
                f"(spatial bandwidth {self.bandwidth_m:g} m, temporal "
                f"half-life {self.halflife_days:g}d); backward-looking only, "
                "current bin included with weight 1"
            ),
            metadata={
                "value_col": self.value_col,
                "bandwidth_m": self.bandwidth_m,
                "halflife_days": self.halflife_days,
                "halflife_bins": halflife_bins,
                "time_bin": event_frame.metadata.time_bin,
                "kernel": "gaussian x exponential",
                "truncation_bandwidths": TRUNCATION_BANDWIDTHS,
                "causality": "backward-looking, includes current bin",
                "edge_correction": None,
            },
        )
        result = event_frame.register_feature(
            output_col,
            {
                "source_step": "SpaceTimeKernelDensityStep",
                "value_col": self.value_col,
                "bandwidth_m": self.bandwidth_m,
                "halflife_days": self.halflife_days,
            },
            provenance=provenance,
        )
        # Eager, as in KernelDensityStep: collect here so the spatial
        # KDE join runs on its own instead of being deferred to the
        # final collect together with the other giant joins.
        return result.with_lazy_frame(lf.collect().lazy())


# -----------------------------------------------------------------------------
# Shared implementation
# -----------------------------------------------------------------------------


def _spatial_kde_frame(
    lf: pl.LazyFrame,
    *,
    unit_col: str,
    value_col: str,
    group_cols: list[str],
    centres: pl.LazyFrame,
    bandwidth_m: float,
    output_col: str,
) -> pl.LazyFrame:
    """Add a Gaussian KDE column to ``lf`` via one edge join + group_by.

    ``centres`` is the distinct-centres frame with canonical columns
    ``_unit`` / ``_x`` / ``_y``. The edge table (cells within 3h, with
    the kernel weight) is built once from it; the values are joined
    onto the edges on the neighbour unit and summed per (unit, group).
    The self-edge (distance 0) guarantees a non-null density for
    every (unit, group) present in ``lf``.

    The edge join is evaluated eagerly in chunks of at most
    ``_KDE_JOIN_CHUNK_EDGES`` edges (the values frame is collected
    once first, so the upstream lazy chain runs a single time): each
    chunk's partial sums are added up afterwards, which is exact up
    to floating-point summation order. Without chunking, the join
    builds a ~68M-row intermediate at Chicago scale.
    """
    edges = _kernel_edges(centres, bandwidth_m).collect()
    values = lf.select([pl.col(unit_col).alias("_unit_j"), *group_cols, value_col]).collect()
    partials: list[pl.DataFrame] = []
    # max(..., 1): an empty edge table still runs one (empty) chunk so
    # the partial keeps its schema instead of concat([]) failing.
    for offset in range(0, max(edges.height, 1), _KDE_JOIN_CHUNK_EDGES):
        chunk = edges.slice(offset, _KDE_JOIN_CHUNK_EDGES)
        weighted = chunk.join(values, on="_unit_j", how="left")
        partials.append(
            weighted.group_by(["_unit_i", *group_cols]).agg(
                (pl.col(value_col) * pl.col("_kernel_w")).sum().alias(output_col)
            )
        )
    density = (
        pl.concat(partials, how="vertical")
        .group_by(["_unit_i", *group_cols])
        .agg(pl.col(output_col).sum())
        .rename({"_unit_i": unit_col})
    )
    return lf.join(
        density.lazy(),
        on=[unit_col, *group_cols],
        how="left",
        validate="m:1",
        maintain_order="left",
    )


def _kernel_edges(centres: pl.LazyFrame, bandwidth_m: float) -> pl.LazyFrame:
    """Lazy edge table (unit_i, unit_j, kernel weight) for pairs within 3h.

    ``centres`` carries canonical columns ``_unit`` / ``_x`` / ``_y``.
    The Gaussian weight at squared distance ``d2`` is
    ``exp(-d2 / 2h^2) / (2 * pi * h^2)``; pairs beyond ``3h`` are
    dropped (truncation). The self-pair (distance 0) is always kept.
    """
    left = centres.rename({"_unit": "_unit_i", "_x": "_x_i", "_y": "_y_i"})
    right = centres.rename({"_unit": "_unit_j", "_x": "_x_j", "_y": "_y_j"})
    radius2 = (TRUNCATION_BANDWIDTHS * bandwidth_m) ** 2
    normaliser = 2.0 * math.pi * bandwidth_m**2
    return (
        left.join(right, how="cross")
        .with_columns(
            ((pl.col("_x_i") - pl.col("_x_j")) ** 2 + (pl.col("_y_i") - pl.col("_y_j")) ** 2).alias(
                "_d2"
            )
        )
        .filter(pl.col("_d2") <= radius2)
        .with_columns(
            ((-pl.col("_d2") / (2.0 * bandwidth_m**2)).exp() / normaliser).alias("_kernel_w")
        )
        .select(["_unit_i", "_unit_j", "_kernel_w"])
    )


def _cell_centres(event_frame: EventFrame, lf: pl.LazyFrame, unit_col: str) -> pl.LazyFrame:
    """Distinct cell centres as a lazy frame with canonical columns.

    The centres are the frame's schema lon/lat columns, which on the
    completed panel hold the cell-centre coordinates in projected
    metres (see ``ChicagoCountStep``). A metric CRS is required, and
    every unit must have exactly one (finite, non-null) centre:
    anything else fails loudly instead of silently smoothing over
    the wrong geometry.
    """
    require_metric_crs(event_frame.metadata.crs)
    schema = event_frame.schema
    if schema.lon_col is None or schema.lat_col is None:
        raise ValueError(
            "Kernel density steps require schema lon/lat columns holding the "
            "cell-centre coordinates (projected metres)"
        )
    x_col, y_col = schema.lon_col, schema.lat_col
    _require_columns(lf, [x_col, y_col])

    # One-row validation aggregates only; the panel itself stays lazy.
    checks = (
        lf.select(
            pl.col(x_col).is_null().any().alias("_null_x"),
            pl.col(y_col).is_null().any().alias("_null_y"),
            pl.col(x_col).cast(pl.Float64).is_nan().any().alias("_nan_x"),
            pl.col(y_col).cast(pl.Float64).is_nan().any().alias("_nan_y"),
        )
        .collect()
        .row(0, named=True)
    )
    if checks["_null_x"] or checks["_null_y"]:
        raise ValueError("Cell-centre coordinates cannot contain nulls")
    if checks["_nan_x"] or checks["_nan_y"]:
        raise ValueError("Cell-centre coordinates cannot contain NaN")

    centres = lf.select(
        [
            pl.col(unit_col).alias("_unit"),
            pl.col(x_col).alias("_x"),
            pl.col(y_col).alias("_y"),
        ]
    ).unique(maintain_order=True)
    max_centres = (
        centres.group_by("_unit")
        .agg(pl.len().alias("_n"))
        .select(pl.col("_n").max())
        .collect()
        .item(0, 0)
    )
    if max_centres is not None and max_centres != 1:
        raise ValueError(
            "Each spatial unit must have exactly one cell centre; "
            "run this step on the completed panel, where the lon/lat "
            "columns hold the (static) cell-centre coordinates"
        )
    return centres


def _validate_values(lf: pl.LazyFrame, value_col: str) -> None:
    """Blocking check: the value column must be finite and non-null.

    One-row validation aggregate only; the panel itself stays lazy.
    """
    checks = (
        lf.select(
            pl.col(value_col).is_null().any().alias("_null_value"),
            pl.col(value_col).cast(pl.Float64).is_finite().all().alias("_all_finite"),
        )
        .collect()
        .row(0, named=True)
    )
    if checks["_null_value"]:
        raise ValueError(
            f"Value column {value_col!r} contains nulls; complete the panel "
            "with explicit zeros (or choose a missing-data policy) first"
        )
    if not checks["_all_finite"]:
        raise ValueError(
            f"Value column {value_col!r} contains non-finite values; "
            "choose a missing-data policy first"
        )


def _validate_no_nulls(lf: pl.LazyFrame, cols: list[str], step_name: str) -> None:
    """Blocking check: grouping / time columns must not contain nulls.

    A null group silently merges every generated panel row into one
    group (and a null time silently drops the row from its cell's
    time series). The pre-completion "time_bin" column is null on
    generated rows — use the completed panel's timestamp column.
    One-row validation aggregate only; the panel itself stays lazy.
    """
    if not cols:
        return
    checks = (
        lf.select([pl.col(c).is_null().any().alias(f"_null_{i}") for i, c in enumerate(cols)])
        .collect()
        .row(0, named=True)
    )
    for i, col in enumerate(cols):
        if checks[f"_null_{i}"]:
            raise ValueError(
                f"{step_name} requires non-null values in {col!r}; "
                "use the completed panel's timestamp column (the "
                "pre-completion 'time_bin' column is null on generated rows)"
            )


def _halflife_bins(event_frame: EventFrame, halflife_days: float, step_name: str) -> float:
    """Convert a half-life in days to bins via metadata.time_bin."""
    time_bin = event_frame.metadata.time_bin
    if time_bin is None:
        raise ValueError(
            f"{step_name} requires metadata.time_bin to convert halflife_days "
            "to bins (run TimeBinsStep or ObservationPanelStep first)"
        )
    try:
        bin_seconds = duration_seconds(time_bin)
    except ValueError as exc:
        raise ValueError(f"{step_name} cannot parse metadata.time_bin {time_bin!r}: {exc}") from exc
    return halflife_days * 86400.0 / bin_seconds


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------


def _validate_bandwidth(bandwidth_m: float) -> None:
    if not math.isfinite(bandwidth_m) or bandwidth_m <= 0:
        raise ValueError(f"bandwidth_m must be a finite number > 0, got {bandwidth_m!r}")


def _validate_column_name(name: str, value: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string, got {value!r}")


def _require_columns(lf: pl.LazyFrame, columns: Sequence[str]) -> None:
    missing = sorted(set(columns) - set(lf.collect_schema().names()))
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")


def _require_output_free(lf: pl.LazyFrame, output_col: str) -> None:
    if output_col in lf.collect_schema().names():
        raise ValueError(
            f"Output column {output_col!r} already exists in the frame; "
            "drop or rename it before running this step"
        )
