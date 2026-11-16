# Phase 3 — Beijing Multi-Site Air Quality (UCI 501)

Second-domain proof for Shikong: the same generic Steps that built
the Chicago crime panel build a sensor panel, with no Chicago-specific
assumption (no grid, no crime columns, no spatial step).

## Data

- Source: Chen, S. (2017). Beijing Multi-Site Air Quality [Dataset].
  UCI Machine Learning Repository. DOI 10.24432/C5RK5G. Download:
  `PRSA2017_Data_20130301-20170228.zip` (7.6 MB) from the UCI dataset
  page. Note: `fetch_ucirepo(id=501)` refuses this dataset ("not
  available for import"), and the old
  `archive.ics.uci.edu/static/public/501/PRSA2017_Data_20130301-20170228.zip`
  URL 404s; the working download URL is
  `https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip`
  (a zip containing the inner `PRSA2017_Data_20130301-20170228.zip`).
  Raw data live under `phase3/data/beijing/` and are never committed.
- Contents: 12 stations x hourly, 2013-03-01 00:00 -> 2017-02-28
  23:00 (35,064 hours), 6 pollutants (PM2.5, PM10, SO2, NO2, CO, O3)
  + 6 meteorological variables (TEMP, PRES, DEWP, RAIN, wd, WSPM).
  Missing values are the literal string `NA`. Total 420,768 rows.
- The per-station CSVs carry no coordinates. Station locations in
  `schema.py` come from Ma et al. (2023), PLOS ONE, Table 1
  (DMS -> decimal degrees) for 11 stations; Shunyi is absent from
  that table, so its coordinate comes from Zhang et al. (2020),
  Table S1. Independent sources disagree by up to ~0.04 deg (~4 km)
  for some stations; no step in the v1 recipe consumes coordinates,
  so that imprecision cannot leak into any feature.

## EDA (measured, not assumed)

- The shipped data are already a **complete panel**: every station
  has exactly 35,064 rows over the full hourly axis, 0 duplicate
  (station, hour) pairs, 420,768 = 12 x 35,064. There are **no
  missing station-hours in the raw data** — all missingness is NA
  values inside present rows.
- NA rates (all 420,768 rows): CO 4.92% (20,701), NO2 2.88%
  (12,116), O3 3.16% (13,277), SO2 2.14% (9,021), PM2.5 2.08%
  (8,739), PM10 1.53% (6,449); meteorology ~0.1% each (TEMP 398,
  PRES 393, DEWP 403, RAIN 390, WSPM 318); wind direction 1,822
  nulls (0.43%). Per-station PM2.5 NA rates range 1.09% (Wanliu) to
  2.72% (Huairou). Full per-station coverage is in
  `audit/phase-3/beijing_replay.json`.
- One parsing trap: RAIN looks integer in the first rows but takes
  values like 0.4 later, so the loader pins explicit dtypes instead
  of relying on schema inference.

## How the sensor panel maps onto the observation-support/completion contract

- **Unit = station, bin = hour.** One raw row is one observed
  station-hour. The loader adds `count` = 1 per row: the count is the
  only column that completion ever zero-fills.
- **Support** = the stations observed in >= `min_periods` bins
  (1 here) x the hourly axis from the earliest to the latest observed
  bin. A station that goes down for 3 hours is *inside* support for
  those hours: completion generates those (station, hour) pairs with
  `count` = 0 and **null measurements** — "the station recorded
  nothing" is not "the station recorded 0 ug/m^3". A station-hour
  outside the axis (before the first / after the last observed hour)
  is never invented.
- **Measurements are `measure_cols`, not `value_cols`.** This is the
  generic fix this phase made to `CompletePanelStep` (see below):
  measurement columns are carried through the (unit, time) join and
  never zero-filled. Wind direction (`wd`) is a varying categorical
  column: joined on (station, hour), null on generated pairs.
- **"Missing stays missing", precisely.** The invariant is:
  `NA_after = NA_before + n_generated_pairs` per measurement column
  (each generated pair contributes one null), and no observed null
  ever becomes non-null. On the shipped data `n_generated_pairs`
  is 0, so the NA counts are *exactly identical* before and after
  completion (verified: PM2.5 8,739 -> 8,739, CO 20,701 -> 20,701,
  etc.). The synthetic gap test (3 generated pairs) verifies the
  general form: PM2.5 2 -> 5, TEMP 1 -> 4, RAIN 0 -> 3.

## Which Steps apply

`beijing_air_quality_v1` (6 steps, all generic):

1. `TimeBinsStep("1h")` — validates hourly alignment, records the bin.
2. `CompletePanelStep(unit_col="station", value_cols=["count"],
   measure_cols=[11 measurements])` — see above.
3. `ExtractTemporalComponentsStep` (hour of day, day of week, day of
   month, month, weekend) and `CalendarEncodingStep` (cyclical
   hour / day-of-week / month) — derived from the timestamp, which is
   the canonical bin datetime. (The pre-completion `time_bin` column
   is null on generated pairs, as in Chicago — nothing downstream
   reads it.)
4. `TemporalLagStep` (pollutants, lags 1h / 24h / 168h, duration mode,
   grouped by station) and `MovingAverageStep` (pollutants, windows
   24h / 168h, closed-left, grouped by station). Both are exact
   duration operations: a lag at `t` reads the same station at
   `t - lag`, and a missing value there stays null — absence stays
   unknown, never borrowed from another station (leakage test:
   rewriting Tiantan's PM2.5/PM10 to 999 leaves every Dongsi feature
   bit-identical) or filled with 0.

Steps that do **not** apply: everything spatial (no grid — stations
are named entities, not cells; no spatial lag / KDE / near-repeat /
Hawkes over station coordinates, which would need a neighbour
definition the dataset does not ship) and everything count-shaped
(`CategoryCountsStep`, `ZeroFeaturesStep`, `TimeSinceLastEventStep`
on a count that is 1-or-0 by construction). Rolling means skip
nulls (Polars `rolling_mean` ignores null values, `min_samples=1`),
so a 24h mean over a window with missing hours is the mean of the
available hours — documented here, not hidden.

## The `CompletePanelStep` fix (`measure_cols`)

Verified empirically before fixing: with a measurement declared in
`value_cols` (the natural declaration for a sensor panel), the
step's blanket `fill_null(0)` turned an *observed* null PM2.5 into
0.0 and every generated pair into 0.0 — a fabricated clean-air
reading. Leaving the column undeclared avoided it only by accident
(the varying-column join path), which is not a contract a recipe
author can rely on.

Fix (generic, in `core/steps/panel.py`, no Beijing special-casing):
a `measure_cols` parameter on `CompletePanelStep` /
`CompletePanelConfig`. Measure columns must be numeric, must not
overlap `value_cols`, and must not match any `value_prefixes` entry
(all three are hard errors). They are carried through the
(unit, time) join and never filled: null on generated pairs, nulls
preserved on observed rows. Chicago behaviour is unchanged
(`value_cols=["count"]`, no `measure_cols`; its tests pass
unmodified).

Known remaining sharp edge, flagged for the maintainer rather than
fixed here: an observed null *inside `value_cols`* is still
zero-filled by the blanket `fill_null(0)` — the step's contract is
"count columns get explicit zeros", and a null count on an observed
row is arguably corrupt input, but the fill does apply to observed
rows, not only to generated pairs. Restricting the fill to generated
pairs only would be a second, separable semantic tightening.

## Replay (measured)

Script: `scripts/replay_beijing.py`. Evidence:
`audit/phase-3/beijing_replay.json`; features:
`phase3/beijing_features.parquet` (34 MB, not committed).

- Stage 1 — Dongsi x 2016 (leap year): 8,784 rows in, 8,784 rows x
  64 cols out, count sum 8,784, 0 generated pairs, NA counts
  identical before/after completion (PM2.5 353, CO 318, TEMP 7, ...),
  2.2 s.
- Stage 2 — all 12 stations, full range: 420,768 rows in, 420,768
  rows x 64 cols out, count sum 420,768, 0 generated pairs, NA
  counts identical before/after completion *and* after the full
  pipeline, 9.8 s. The staged run was asserted equal to the packaged
  recipe run.
- Hand-checks (re-derived from the raw CSVs with an independent
  read, all PASS): Dongsi 2016-06-15 12:00 — PM2.5 20.0, TEMP 28.4,
  PM2.5_lag_24 41.0, PM2.5_ma_24 22.75; Dongsi 2016-01-12 13:00 —
  PM2.5 null (first 2016 null), and its 1h lag at 14:00 null;
  Shunyi 2015-03-10 08:00 — O3 9.0, PM10 22.0, PM10_lag_168 99.0.

## Honest limit

Timestamps are regular hourly and the shipped panel is already
complete, so this replay proves **gaps / missing-stays-missing and
per-entity separation, NOT irregular sampling**. Completion with
generated pairs is exercised only on synthetic gaps in the test
suite (a station down for 3 hours), never on real missing
station-hours, because the dataset contains none. A dataset with
genuinely irregular timestamps would still need a binning/validity
decision this recipe does not make.
