"""Fixed-kernel Hawkes excitation feature (panel level, Phase 2).

This module provides a single step, :class:`HawkesExcitationStep`: a
**fixed-kernel excitation feature, not a fitted Hawkes process**. There
is no parameter estimation (no MLE for alpha/beta/mu), no
background-intensity separation, and no branching-ratio estimation.
The caller declares the decay half-life; the kernel is then fixed.

Per unit, the excitation at bin ``t`` is the exponentially decayed sum
of the strictly past bins::

    E(t) = sum_{s < t} count(s) * exp(-(t - s) / tau)

with ``tau = halflife_days / ln 2`` (the same half-life convention as
``EWMAMeanStep``). The output is in units of "decayed event count".
The first bin of each unit has ``E = 0`` (no past bins).

How this differs from ``HawkesKernelStep``
-------------------------------------------
The event-level ``HawkesKernelStep`` (``core/steps/point_process.py``)
works on continuous event times with fitted-style parameters
(alpha/beta/mu) and splits the conditional intensity into background
plus triggering parts. This step works on the completed panel (one
row per unit x time bin, explicit zeros), uses one caller-declared
half-life, and produces a single strictly-past excitation feature.
Spatial excitation is composition, not a second implementation: feed
the excitation column to ``SpatialLagStep`` /
``MultiScaleSpatialLagStep`` to spread it over neighbours.

Implementation (and a correction to the design note)
----------------------------------------------------
The recurrence is ``E(t) = d * (E(t-1) + count(t-1))`` with
``d = exp(-bin_days / tau)``. The design note
(``docs/design-notes/phase2-spacetime-v2.md``, Decision 5) proposes
computing it as ``d / (1 - d) * shift(ewm_mean(count, adjust=False), 1)``.
That identity is **wrong in general**: Polars' ``adjust=False`` EWMA
is initialised at ``y(0) = count(0)`` instead of ``(1 - d) * count(0)``,
so the proposed formula over-counts the first bin's contribution by
exactly ``d**t * count(0) * d / (1 - d)``. It only looks right when
``count(0) == 0``. The error grows with the half-life (``d -> 1``) and
is proven, with the analytic error term, in
``tests/core/test_hawkes_steps.py::TestHawkesEwmIdentity``.

The implementation therefore uses the validated closed form based on
``adjust=True``. With ``y(t)`` the ``adjust=True`` EWMA and ``t`` the
0-based index of the bin within its unit::

    E(t) = d * y(t-1) * (1 - d**t) / (1 - d)          (t >= 1)
    E(0) = 0

because ``y(t) = sum_{s<=t} d**(t-s) * count(s) / sum_{k=0..t} d**k``
and ``sum_{k=0..t-1} d**k = (1 - d**t) / (1 - d)``. This form is
verified against a brute-force Python loop in the reference tests
(maximum absolute error ~1e-14 even for a half-life of 100 bins).

Numerical behaviour
-------------------
All arithmetic is float64 and nothing is clipped. When the half-life
is much longer than one bin, ``d`` is close to 1 and ``d / (1 - d)``
is large (about ``halflife_bins / ln 2``): the excitation then sums
many past bins with weights near 1, so large outputs are correct,
not an overflow. The factor ``(1 - d**t) / (1 - d)`` is the exact
geometric sum; for ``d`` close to 1 and small ``t`` it is computed
from values near 1, which float64 handles with relative error around
1e-12 for panel lengths in scope. No silent clipping or rescaling is
applied.

Causality
---------
Strictly past only (``s < t``): the value at bin ``t`` does not depend
on ``count(t)`` itself, and changing data at bins after ``t`` never
changes ``E`` at bins ``<= t``. This is stricter than ``EWMAMeanStep``,
whose EWMA includes the current bin.
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

if TYPE_CHECKING:
    from eventflow.core.event_frame import EventFrame


# -----------------------------------------------------------------------------
# Pydantic Config Model for Step Parameters
# -----------------------------------------------------------------------------


class HawkesExcitationConfig(BaseModel):
    """Configuration for the fixed-kernel Hawkes excitation step."""

    value_col: str = Field(default="count", description="Count column to excite")
    halflife_days: float = Field(
        default=7.0,
        gt=0,
        allow_inf_nan=False,
        description="Excitation half-life in days (must be finite and > 0)",
    )
    group_cols: list[str] = Field(default_factory=list, description="Optional columns to group by")


# -----------------------------------------------------------------------------
# Hawkes Excitation Step
# -----------------------------------------------------------------------------


class HawkesExcitationStep(Step):
    """Fixed-kernel Hawkes excitation of a panel value column.

    Fixed kernel, strictly past, no fitting: per unit,
    ``E(t) = sum_{s<t} count(s) * exp(-(t-s)/tau)`` with
    ``tau = halflife_days / ln 2``. This is a feature, not a fitted
    Hawkes process — no parameter estimation, no background/excitation
    separation, no branching ratio. It differs from the event-level
    ``HawkesKernelStep`` (alpha/beta/mu on continuous event times,
    background plus triggering intensity) by operating on the
    completed panel with a single caller-declared half-life.

    Causality: strictly past bins only (``s < t``). The value at bin
    ``t`` does not depend on ``count(t)`` itself, and no bin after
    ``t`` is ever used. The first bin of each unit is 0.

    The half-life in days is converted to a half-life in bins using
    the frame's ``metadata.time_bin`` (so a ``7d`` half-life on a
    daily panel is 7 bins, on a 6-hour panel 28 bins).

    Inputs:
        - value_col to excite (default "count")

    Outputs:
        - {value_col}_hawkes_{halflife}d in units of decayed event count
    """

    def __init__(
        self,
        value_col: str = "count",
        halflife_days: float = 7.0,
        group_cols: Sequence[str] | None = None,
        value_cols: Sequence[str] | None = None,
    ) -> None:
        if value_cols is not None:
            if value_col != "count":
                raise ValueError("Pass either value_col or value_cols, not both")
            self.value_cols = list(value_cols)
        else:
            self.value_cols = [value_col]
        if not self.value_cols or any(not col for col in self.value_cols):
            raise ValueError("value_cols must not be empty and must not contain empty names")
        self.value_col = self.value_cols[0]
        if not math.isfinite(halflife_days) or halflife_days <= 0:
            raise ValueError(f"halflife_days must be a finite number > 0, got {halflife_days!r}")
        self.halflife_days = float(halflife_days)
        self.group_cols = list(group_cols) if group_cols else []

    def run(self, event_frame: EventFrame) -> EventFrame:
        """Execute fixed-kernel Hawkes excitation."""
        timestamp_col = event_frame.schema.timestamp_col
        lf = event_frame.lazy_frame
        result = event_frame

        _require_columns(lf, self.value_cols + self.group_cols)
        time_bin = event_frame.metadata.time_bin
        if time_bin is None:
            raise ValueError(
                "HawkesExcitationStep requires metadata.time_bin to convert halflife_days "
                "to bins (run TimeBinsStep or ObservationPanelStep first)"
            )
        try:
            bin_seconds = duration_seconds(time_bin)
        except ValueError as exc:
            raise ValueError(
                f"HawkesExcitationStep cannot parse metadata.time_bin {time_bin!r}: {exc}"
            ) from exc
        bin_days = bin_seconds / 86400.0
        halflife_bins = self.halflife_days / bin_days
        tau_days = self.halflife_days / math.log(2)
        # d = exp(-bin_days / tau) = 0.5 ** (1 / halflife_bins).
        decay_per_bin = math.exp(-math.log(2) / halflife_bins)

        for value_col in self.value_cols:
            _require_finite_values(lf, value_col)

        # Sort by timestamp (and group cols if specified)
        sort_cols = self.group_cols + [timestamp_col] if self.group_cols else [timestamp_col]
        lf = lf.sort(sort_cols, maintain_order=True)

        halflife_label = _format_number(self.halflife_days)
        for value_col in self.value_cols:
            out_col = f"{value_col}_hawkes_{halflife_label}d"
            lf = lf.with_columns(
                _excitation_expr(
                    value_col,
                    halflife_bins,
                    decay_per_bin,
                    timestamp_col,
                    self.group_cols,
                ).alias(out_col)
            )

            provenance = FeatureProvenance(
                produced_by="HawkesExcitationStep",
                inputs=[value_col, timestamp_col],
                tags={"temporal"},
                description=(
                    f"Fixed-kernel Hawkes excitation of {value_col} with half-life "
                    f"{halflife_label}d (strictly past bins only, no fitting)"
                ),
                metadata={
                    "halflife_days": self.halflife_days,
                    "tau_days": tau_days,
                    "halflife_bins": halflife_bins,
                    "bin_days": bin_days,
                    "decay_per_bin": decay_per_bin,
                    "time_bin": time_bin,
                    "strictly_past": True,
                    "window": "strictly past (s < t)",
                    "formula": (
                        "d * shift(ewm_mean(adjust=True), 1) * (1 - d**t) / (1 - d), "
                        "t = 0-based bin index within the unit; E(0) = 0"
                    ),
                },
            )
            result = result.register_feature(
                out_col,
                {
                    "source_step": "HawkesExcitationStep",
                    "value_col": value_col,
                    "halflife_days": self.halflife_days,
                },
                provenance=provenance,
            )

        return result.with_lazy_frame(lf)


def _excitation_expr(
    value_col: str,
    halflife_bins: float,
    decay_per_bin: float,
    timestamp_col: str,
    group_cols: list[str],
) -> pl.Expr:
    """Closed-form excitation from the ``adjust=True`` EWMA.

    ``E(t) = d * y(t-1) * (1 - d**t) / (1 - d)`` where ``y`` is the
    ``adjust=True`` EWMA and ``t`` is the 0-based bin index within the
    unit. The shift is applied inside the ``over`` partition, so it
    never leaks across units; the first bin's shifted EWMA is null and
    ``fill_null(0.0)`` makes ``E(0) = 0``.
    """
    d = decay_per_bin
    ewm = pl.col(value_col).ewm_mean(half_life=halflife_bins, adjust=True)
    if group_cols:
        shifted = ewm.shift(1).over(group_cols)
        index = (pl.col(timestamp_col).cum_count().over(group_cols) - 1).cast(pl.Float64)
    else:
        shifted = ewm.shift(1)
        index = (pl.col(timestamp_col).cum_count() - 1).cast(pl.Float64)
    geometric_sum = (1.0 - pl.lit(d).pow(index)) / (1.0 - d)
    return (pl.lit(d) * shifted * geometric_sum).fill_null(0.0)


def _require_columns(lf: pl.LazyFrame, columns: Sequence[str]) -> None:
    missing = sorted(set(columns) - set(lf.collect_schema().names()))
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")


def _require_finite_values(lf: pl.LazyFrame, value_col: str) -> None:
    """Blocking check: null/non-finite values are an error, never zero.

    Same policy as ``SpatialLagStep``: on a sparse panel the caller
    must complete it first. The probe aggregates one column, so it
    costs a single pass over that column.
    """
    try:
        probe = lf.select(
            pl.col(value_col).is_null().sum().alias("n_null"),
            (~pl.col(value_col).is_finite()).fill_null(False).sum().alias("n_nonfinite"),
        ).collect()
    except Exception as exc:
        raise ValueError(
            f"HawkesExcitationStep requires a numeric value column, got non-numeric "
            f"{value_col!r}: {exc}"
        ) from exc
    n_null = probe["n_null"][0]
    n_nonfinite = probe["n_nonfinite"][0]
    if n_null:
        raise ValueError(
            f"HawkesExcitationStep requires finite non-null values in {value_col!r}: "
            f"found {n_null} null(s); complete the panel first"
        )
    if n_nonfinite:
        raise ValueError(
            f"HawkesExcitationStep requires finite non-null values in {value_col!r}: "
            f"found {n_nonfinite} non-finite value(s); complete the panel first"
        )


def _format_number(value: float) -> str:
    """Format a number without a trailing '.0' (7.0 -> '7', 2.5 -> '2.5')."""
    return f"{value:g}"
