# Phase 1 design — panel completion and causality

**Date:** 2026-11-15 (phase 1)
**Status:** design note — implemented in phase 1, see the code it cites.

## Decision already made (maintainer, 2026-11-15)

A missing cell/bin pair is completed with **explicit zeros** inside the
observation support, before any window or lag is computed. Zeros are
distinct from missing / out-of-domain via the observation panel states
(`ObservationPanel`, PR #24): a row present in the completed panel with
`observation_status == "observed"` and `count == 0` is an explicit zero —
the cell was observed and nothing happened there. Pairs outside the support
are not rows at all; they are not silently invented.

## Panel completion (`CompletePanelStep`, `core/steps/panel.py`)

- **Support, spatial:** the distinct spatial units observed in the input
  (e.g. the 5,242 Chicago grid cells with at least one crime in Q1 2024).
  Cells never observed are outside the support and produce no rows.
- **Support, temporal:** the regular bins of `bin_size` from the minimum to
  the maximum observed bin, inclusive (e.g. 364 bins of 6 h for Q1 2024).
- **Result:** the full cross product units x bins (Chicago: 5,242 x 364 =
  1,908,088 rows). Observed pairs keep their counts; generated pairs get
  `count = 0` and `observation_status = "observed"`.
- **`min_periods`:** a unit enters the completed panel only if it was
  observed in at least `min_periods` distinct bins (default 1 — every
  observed unit is kept). It is a noise filter on units, not a window
  parameter: rolling/lag steps have their own `min_periods` / `min_samples`
  for the number of observations required inside a window.
- **Non-key, non-value columns** are classified per unit, never silently
  collapsed: a column constant within a unit (e.g. `grid_row`, `grid_col`,
  cell-centre coordinates) is carried as that unit's static value; a column
  that varies within a unit (e.g. per-bin shares or entropy) is joined on
  (unit, time) — observed pairs keep their value, generated pairs get null.
  (`first()` over the whole panel would silently pick one bin's value.)
- The step adds no information: it only makes absence explicit.

## Boundary policy for lags before the panel start

A lagged count before the panel start has no earlier bin to read from, so
it is null by construction (`count_lag_1` is null for the first day of the
panel). The recipe fills **only the lag that feeds the spatial lag**
(`FillNullStep`, `count_lag_<shortest>`, 0) with the explicit policy:
before the panel start, lagged counts are treated as 0 ("no events
recorded") — `SpatialLagStep` requires finite non-null values. The other
lags keep their nulls; the fill is a boundary convention, not data.

## Causality convention

A feature row at bin `t` is for **predicting the count in bin `t+1`**.
Therefore no feature may use data from bins after `t`. Within that bound,
each feature documents its exact window policy:

| Feature | Window policy |
|---|---|
| Lags (`TemporalLagStep`, duration mode) | Exact elapsed lag; a missing value at `t - lag` stays null. |
| Rolling means (`MovingAverageStep`) | Trailing, `closed="left"` — strictly before `t`. |
| EWMA (`EWMAMeanStep`) | Standard EWMA, includes the current bin. |
| Category counts / shares / entropy | Describe bin `t` itself (events in that bin). |
| Cyclical encodings (`CalendarEncodingStep`) | Derived from the timestamp only, not from event data. |
| Spatial lag (`SpatialLagStep`) | Contemporaneous within a bin by construction — the recipe feeds it **lagged** counts (`count_lag_1`), so it uses neighbours' counts from the previous day, strictly before `t`. Never feed it the current bin's counts for a same-bin target. |
| Trend (`TrendStep`) | Ratio of two closed-left rolling means. |
| Time since last event | Most recent bin `<= t` with `count > 0` (0.0 when bin `t` itself has events); null if none. |
| Burstiness | Coefficient of variation over the trailing window, `closed="left"` (strictly before `t`). |
| Zero features (active / share of non-zero bins) | Trailing window **including** the current bin (`closed="right"`). |

Leakage tests pin the global guarantee: changing data at bins `> t` must
not change any feature at bins `<= t`; and a closed-left feature at bin
`t` must not change when the count at bin `t` itself is changed.

## What phase 1 deliberately does not do

- **Multi-scale spatial lag, KDE/STKDE, near-repeat, simplified Hawkes:**
  phase 2. Only the basic queen-adjacency spatial lag is in phase 1.
- **Partitioned output** (`output.partition_by`): still unimplemented and
  still a blocking error in the CLI; it is a CLI-level feature, not a
  feature-engineering one.
- **Goh-Barabasi burstiness** (inter-event-time B): needs continuous event
  times; phase 1 uses the coefficient of variation of counts as a
  documented proxy.
- **`is_holiday`:** still unimplemented (no holiday-calendar dependency);
  requesting it is a loud error since PR #35.
