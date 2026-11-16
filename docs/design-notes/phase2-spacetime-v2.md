# Phase 2 design — space-time v2 (multi-scale lag, KDE/STKDE, near-repeat, simplified Hawkes)

**Date:** 2026-11-15 (phase 2)
**Status:** design note — decisions taken by the coordinator before implementation;
the maintainer decides by rejection. Code is in the modules cited below.

## Baseline

- Base: `main` @ `36aa7ec` (merge of PR #40, TODO Phase 1 DONE; code identical
  to `992a69f4a`, merge of PR #37). Baseline verified before any change:
  **705 tests passed / 0 failed, coverage 84%**, `make audit` PASS 5/5.
- The panel the new steps operate on is the Phase 1 completed panel
  (`CompletePanelStep`, opt-in, `min_periods`): Chicago Q1 2024,
  5,242 cells x 364 bins of 6 h = 1,908,088 rows, explicit zeros,
  `observation_status == "observed"` on generated rows.

## Exit criteria (defined before, binary, verified by execution)

1. Full suite green; coverage >= 84% (baseline 84%).
2. Every new step has reference tests vs hand-computed values (small synthetic
   panels, values computed by hand or by an independent brute-force loop in
   the test, not by calling the step itself) AND leakage (cutoff) tests:
   changing data at bins `> t` must not change any feature at bins `<= t`.
3. `SpatialLagStep` outputs are bit-identical (up to float tolerance 1e-9)
   before/after the vectorized rewrite, proven by a dedicated equivalence
   test run against the old implementation; the Chicago replay value
   `count_lag_1_spatial_lag` on the busiest cell (grid 1930/1195, bin
   2024-02-17 12:00) is still 0.375.
4. Chicago replay (Q1 2024) shows every new column with correct values;
   at least one new column per family is spot-checked by hand against the
   raw CSV (`data/raw/chicago_crime/crimes_q1_2024.csv`); row count stays
   1,908,088 and sum(`count`) stays 60,235.
5. Per-step timings on the Chicago replay are measured and reported;
   `SpatialLagStep` (single scale) must be at most ~2 s (was 51.4 s),
   total pipeline at most ~20 s (was ~61 s).
6. `make audit` PASS 5/5; evidence pack under `audit/phase-2/`;
   `docs/audit/traceability.md` gains an "After phase 2" section and
   `docs/audit/known-limitations.md` records what was knowingly left out.
7. Ruff, black, mypy clean on all touched files.

## Verification results (2026-11-15)

1. **PASS** — full suite 838 passed / 0 failed (705 baseline + 133 new),
   coverage 85% (baseline 84%). Ruff, black, mypy clean.
2. **PASS** — every new step has reference tests vs hand-computed /
   brute-force values and leakage (cutoff) tests
   (`tests/core/test_density_steps.py` 29 tests,
   `tests/core/test_near_repeat_steps.py` 27,
   `tests/core/test_hawkes_steps.py` 30,
   `tests/core/test_spatial_lag_vectorized.py` 38 + 2 smoke tests).
3. **PASS** — vectorized `SpatialLagStep` is equivalent to the verbatim
   phase-1 implementation to 1e-9 (dedicated equivalence test); the
   Chicago replay still pins `count_lag_1_spatial_lag` = 0.375 on the
   busiest cell (grid 1930/1195, bin 2024-02-17 12:00), re-verified on
   the replay output itself.
4. **PASS** — Chicago replay: 1,908,088 rows × 90 columns, sum(`count`)
   = 60,235, 1,851,597 explicit-zero rows, 0 null counts. All 14
   hand-checks pass against the raw CSV (independent pyproj + numpy
   repro, `~/workspace/eventflow-eda/phase2/check_phase2_values.py`):
   multi-scale lags r1/r2/r3 (0.375 / 0.2 / 0.142857), KDE
   (2.380353574e-06), STKDE (7.595752197e-05), near-repeat r1/r2/r3 ×
   7 d / 7–14 d (50/46, 52/67, 49/55) + flag (1), Hawkes 7 d
   (29.77515485). See "Bugs found by the hand-check" below.
5. **NOT MET as written** — per-step timings on the Chicago replay
   (6 GB VM, 1.5 shared CPU; timings vary run to run, ±2× observed):
   MultiScaleSpatialLagStep (3 rings) 21.2 s, KernelDensityStep 24.6 s,
   SpaceTimeKernelDensityStep 29.4 s, NearRepeatStep 16.9 s,
   Hawkes/TimeSinceLast/Burstiness/ZeroFeatures ~0 s (lazy, executed at
   the final collect, 2.1 s). Total pipeline ~98 s vs ~61 s in phase 1.
   The single-scale `SpatialLagStep` measured standalone on the full
   panel is 6.19 s (was 51.4 s — an 8.3× speedup), not ≤ ~2 s; the
   ~2 s / ~20 s targets were set from a synthetic-panel measurement
   and do not hold on this VM at Chicago scale. The vectorization
   itself is proven (criterion 3); the remaining cost is the
   group_by over the edge×bin intermediates, and the new families'
   joins dominate the new total. A perf regression test is still
   missing (known limitation).
6. **PASS** — `make audit` PASS 5/5 (ruff, black, mypy, pytest,
   replay_chicago), evidence pack under `audit/phase-2/`
   (manifest.json, replay_chicago.json, chicago_features.parquet,
   coverage.xml, logs, verdict.txt).
7. **PASS** — ruff, black, mypy clean.

### Bugs found by the hand-check (both fixed in this PR)

- **Wrong time column (near-repeat + STKDE).** The recipe wired
  `NearRepeatStep` and `SpaceTimeKernelDensityStep` to the default
  `time_col="time_bin"`. On the completed panel `time_bin` is null on
  every generated row (only the schema timestamp column, `date`, holds
  the bin datetime on all rows), so both steps silently computed their
  windows over the sparse observed-bins-only series: near-repeat
  ring-1 7 d = 60 instead of 50, STKDE = 9.43e-05 instead of
  7.60e-05. Fixed by passing `time_col="date"` in the recipe, and all
  three steps now reject a null time / group column with a blocking
  `ValueError` (regression tests added; the Hawkes step already used
  the schema timestamp column, which is why it was correct).
- **OOM on the 6 GB VM (twice).** The first replay was killed during
  the Hawkes validation probe: every giant-join step was lazy, so the
  near-repeat (~91M-row) and KDE (~68M-row) joins executed together
  at the final collect / probe. Fixed by (a) making
  `KernelDensityStep`, `SpaceTimeKernelDensityStep` and
  `NearRepeatStep` eager — each collects its result before returning;
  (b) computing near-repeat ring counts one ring at a time (widest
  ring ~46M rows instead of ~91M); (c) chunking the KDE edge join
  (50k edges per chunk, ~18M joined rows each) in
  `_spatial_kde_frame`. A second OOM during `KernelDensityStep`
  itself (44.5 s, killed) motivated (c). The replay now completes in
  ~98 s on this VM.

## Shared rules (inherited from Phase 0/1)

- Generic core: no Chicago-specific code in `core/`; durations, grouping
  keys, category policies configurable; every parameter declared explicitly
  (unknown/unimplemented value = blocking `ValueError`, never a silent
  no-op — same rule as PR #32/#35).
- Causality convention (`docs/design-notes/phase1-causality.md`): a feature
  row at bin `t` predicts the count in bin `t+1`. No feature may use data
  from bins after `t`. Each step documents its exact window policy in its
  docstring and provenance metadata.
- All new steps operate on the **completed panel**. Null/non-finite values
  in a value column are a blocking error (same policy as `SpatialLagStep`),
  never silently treated as zero: on a sparse panel the caller must complete
  it first. Islands (units with no neighbours in a scale) yield null.
- Neighbour tables are built **once per step run** from the completed
  panel's unit set (same observation support as Phase 1); no unit is ever
  invented. The "extension of the completion contract to additional spatial
  supports" in the roadmap is exactly this: multi-scale neighbourhoods and
  distance bands are defined over the completed panel's units, with the
  same observation-quality semantics. Zone/areal supports remain out of
  scope (see "Not covered").

## Decision 1 — vectorized spatial lag (addresses issue #38)

`SpatialLagStep` currently takes 51.4 s of a ~61 s pipeline: `_spatial_snapshots`
(`core/steps/spatial.py`) does `collect()` + `partition_by` into 364
per-bin DataFrames, rebuilds `SpatialWeights.from_units(...)` per snapshot,
and computes the lag with a pure-Python list comprehension over 5,242 cells
per snapshot. `LocalMoranStep` and `GetisOrdStep` share the helper and are
NOT in scope for the rewrite — they keep using `_spatial_snapshots`.

New implementation of `SpatialLagStep.run()` (same class, same outputs,
same column names):

1. Build the neighbour edge list **once** from the distinct units of the
   panel (grid_row/grid_col when available, else the existing
   `SpatialWeights.from_units` reference logic): a Polars frame
   `(unit, neighbour)` — 5,242 cells x ~8 neighbours ≈ 42k edges.
2. One join of the (per-bin) values onto the edges + one `group_by`
   (`time_bin`, `unit`) mean in pure Polars. No Python loop, no per-bin
   snapshots. The step does one initial `collect()` of the panel (to
   validate values and build the edge list); there are no per-snapshot
   collects or partitionings.
3. Semantics preserved: row-standardized binary weights, islands -> null,
   separate snapshots per `group_cols` (no pooling across time), finite
   non-null values required.

Equivalence is proven by a test that runs the old implementation (kept as a
reference function in the test file, copied verbatim from the pre-rewrite
code) and the new one on the same synthetic panels and on a Chicago slice,
asserting equal outputs to 1e-9.

## Decision 2 — multi-scale spatial lag (`MultiScaleSpatialLagStep`)

- Scales are **queen rings** 1, 2, 3 (ring r = cells at Chebyshev distance
  exactly r on the grid; ring 1 is the Phase 1 queen neighbourhood).
  Configurable: `rings: list[int]` (default `[1, 2, 3]`).
- Output: one column per (value, ring):
  `{value}_spatial_lag_r{r}` (e.g. `count_lag_1_spatial_lag_r2`).
  Ring 1 reproduces `SpatialLagStep` exactly.
- Each ring is row-standardized binary (mean of neighbours in that ring
  only — rings are disjoint, not cumulative). Islands in a ring -> null.
- Same vectorized edge-join implementation as Decision 1; edges carry the
  ring number. Rings are computed **one at a time** (one join + `group_by`
  per ring, not one over all rings at once): the all-rings intermediate
  (~88M rows on a Chicago-size panel) OOM-killed the 6 GB VM during
  development; the widest ring alone bounds peak memory. Same method,
  bounded intermediate.
- The Chicago recipe wires rings 1-3 on the shortest-window lagged count
  (`count_lag_1`), replacing the single-scale `SpatialLagStep` there.
  The single-scale step stays available and stays tested.

## Decision 3 — KDE and STKDE (`KernelDensityStep`, `SpaceTimeKernelDensityStep`)

The existing `KDEStep` is NOT a KDE: it counts neighbours inside the
bandwidth and divides by the buffer area, ignores kernel shape, and uses a
per-row Python `map_elements` loop. It is superseded by the steps below
(kept registered for backwards compatibility, marked as an approximation
in its docstring; its tests still pass).

- **Spatial KDE** (`KernelDensityStep`): Gaussian kernel, density in
  events per m^2 at each cell centre:
  `density(i, t) = sum_j w_j * exp(-d_ij^2 / 2h^2) / (2*pi*h^2)`,
  where `w_j` are the values of the caller-chosen value column.
  Bandwidth `h` in metres is **declared explicitly** by the caller
  (no automatic/Silverman estimation in Phase 2 — estimation would be a
  fitted statistic on the full panel, i.e. a leakage surface; a
  cross-validated bandwidth belongs to Phase 4 evaluation).
  Kernel truncated at 3h (documented approximation; weight beyond 3h is
  < 1.2% of the peak).
  Implementation: neighbour edge table (cells within 3h) built once from
  cell centres, one join + one `group_by` — same pattern as Decision 1.
  Causality: the step itself is a same-bin smoother; the Chicago recipe
  feeds it `count_lag_1` (same convention as the spatial lag), so the
  density at bin `t` describes neighbours' counts from the previous day.
  Feeding it current-bin counts is allowed by the API but must be
  documented as a same-bin descriptor, not a predictor feature.
- **STKDE** (`SpaceTimeKernelDensityStep`): separable product kernel —
  Gaussian in space (bandwidth `hs`, metres) x exponential decay in time
  (half-life `ht`, days). Implemented as spatial KDE per bin followed by
  per-cell exponential time-smoothing of the density series (the temporal
  kernel only looks backward, including the current bin with weight 1 —
  same "includes current bin" precedent as `EWMAMeanStep`). No future bins
  are used. Output: `{value}_stkde` (density, events per m^2).
- Edge correction: NOT implemented in Phase 2. Density near the boundary
  of the observed support is underestimated (neighbours outside the
  support do not exist). Recorded in `known-limitations.md`; the
  `KFunctionStep` translation correction is a point-pattern tool and does
  not transfer to panel KDE.
- Chicago recipe: KDE `h = 600 m` on `count_lag_1`; STKDE `hs = 600 m`,
  `ht = 7 d` on `count`.

## Decision 4 — near-repeat (`NearRepeatStep`)

- Definition (literature: Mohler et al. 2011; near-repeat ~200-400 m /
  7-30 days, risk x4-x12): for each (cell, bin `t`), counts of events in
  queen rings 1..3 over trailing time windows, computed from the completed
  panel counts (explicit zeros make this exact).
- Rings: queen rings 1, 2, 3 (configurable). Windows: trailing 7 d and
  trailing 7-14 d, both **closed-left** (strictly before `t`).
- Outputs: `{value}_nr_r{r}_7d`, `{value}_nr_r{r}_7_14d` (counts), plus a
  binary flag `{value}_nr_event_7d` = 1 if any event occurred in ring 1 in
  the trailing 7 d. Column names use the `nr` prefix.
- Implementation: ring-count join (same edge pattern as Decision 1),
  then per-(cell, ring) trailing rolling sums over bins in pure Polars.
- Chicago recipe: rings 1-3, windows 7 d / 7-14 d, on `count`.

## Decision 5 — simplified Hawkes (`HawkesExcitationStep`)

- This is a **fixed-kernel excitation feature, not a fitted Hawkes
  process**: no parameter estimation (no MLE for alpha/beta/mu), no
  background-intensity separation, no branching-ratio estimation. The
  existing event-level `HawkesKernelStep` (alpha/beta/mu) stays as is.
- Per unit, excitation at bin `t`:
  `E(t) = sum_{s<t} count(s) * exp(-(t-s)/tau)`,
  `tau` = decay constant in days, parameterised by the caller as
  `halflife_days` (same convention as `EWMAMeanStep`:
  `tau = halflife / ln 2`). Strictly past bins only (s < t).
- Implementation in pure Polars via the recurrence
  `E(t) = d * (E(t-1) + count(t-1))`, `d = exp(-bin_days / tau)`:
  `E(t) = d * shift(ewm_mean(count, half_life=halflife_bins, adjust=True), 1) * (1 - d**t) / (1 - d)`
  per unit, with `t` the 0-based bin index per unit. **Correction to the
  first draft of this note (caught by the reference tests, 2026-11-15):**
  the `adjust=False` form `E(t) = d / (1 - d) * shift(ewm_mean(adjust=False), 1)`
  does NOT hold — Polars' `adjust=False` EWMA initializes at `y(0) =
  count(0)` instead of `(1-d) * count(0)`, over-counting the first bin by
  `d**t * count(0) * d / (1-d)` at every `t >= 1` (it only looks right
  when `count(0) == 0`). The shipped implementation uses the `adjust=True`
  closed form above, proven against a brute-force loop to ~1e-14; the
  test suite pins both the failure of the `adjust=False` form (exact
  analytic error term) and the correctness of the shipped form. The same
  `adjust=True` form is used by `SpaceTimeKernelDensityStep` for its
  temporal smoothing.
- Output: `{value}_hawkes_{halflife}d` (e.g. `count_hawkes_7d`), in units
  of "decayed event count". Spatial excitation (neighbours' past events)
  is composition, not a second implementation: the recipe may feed the
  excitation column to `SpatialLagStep`/`MultiScaleSpatialLagStep`.
- Chicago recipe: `halflife_days = 7` on `count`.
- Null/non-finite values are a blocking error (same policy as
  `SpatialLagStep`).

## Wiring (coordinator, at integration — not the children's job)

- `FeatureConfig` gains: `spatial_lag_rings: list[int]` (default `[1, 2, 3]`,
  requires `aggregation_windows`), `kde_bandwidth_m: float | None`,
  `stkde_bandwidth_m: float | None` + `stkde_halflife_days: float | None`,
  `near_repeat: bool`, `hawkes_halflife_days: float | None` — all requiring
  `complete_panel` where they read panel counts (KDE/STKDE on the lagged
  count need `aggregation_windows`; near-repeat and Hawkes need the panel).
  Cross-key rules in `require_implemented`, same pattern as Phase 1.
- The shipped Chicago YAML enables the full Phase 2 set (the coordinator
  updates the YAML + `tests/integration/test_chicago_pipeline.py` step
  count, currently "17 steps").
- New steps are registered in `core/steps/registration.py` with config
  models. The coordinator does the registration and the recipe wiring;
  children must NOT touch `registration.py`, `schema.py`, the recipes,
  or each other's files.

## Not covered in Phase 2

- Fitting a Hawkes process (MLE), background/excitation separation,
  branching-ratio estimation.
- Automatic bandwidth selection (Silverman, likelihood cross-validation).
- KDE edge correction; zone/areal supports for the completion contract;
  `output.partition_by` (still a blocking CLI error, unrelated).
- `LocalMoranStep` / `GetisOrdStep` vectorization (they keep the
  per-snapshot helper; only `SpatialLagStep` is rewritten).
- `KDEStep` removal (kept, docstring corrected to call it an approximation).

## References

- Literature tour: `~/workspace/research_notes/spatio-temporal-features-literature-20261115-1127/report.md`
  §1c (joint space-time features), §4 rows 4-7 and 12 (near-repeat,
  Hawkes, spatial lag, KDE/STKDE, multi-scale), §3 (leakage pitfalls).
- Phase 1 causality note: `docs/design-notes/phase1-causality.md`.
- Issue #38 (vectorize `SpatialLagStep`) — the multi-scale lag is built
  on the vectorized approach it describes; the issue itself is NOT closed
  by this PR (the maintainer decides).
