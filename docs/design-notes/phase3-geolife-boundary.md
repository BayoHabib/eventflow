# Phase 3 — GeoLife boundary statement (TODO item 2, criterion 4)

Date: 2026-11-15. Base `main` @ `91ea351`. All numbers below were
produced by execution: `scripts/replay_geolife.py` (artefacts in
`audit/phase-3/geolife_replay.json`, point features in
`~/workspace/eventflow-eda/phase3/geolife_features.parquet`) and
`tests/datasets/test_geolife.py` (24 tests, hand-computed values on
tiny synthetic trajectories/panels).

## Representation

One GPS fix is one event. Entity = user (`user_id`); a trajectory
(`trajectory_id`, the `.plt` file stem) groups the fixes of one
continuous log. Coordinates are WGS 84, projected to UTM 50N
(EPSG:32650, metres) for any distance work. There is no fixed spatial
unit and no categorical column in the raw files.

Subset (justification): user `010`, trajectory `20070804033032.plt`
(1,116 fixes, 2007-08-04 03:30:32 → 04:14:46, 44.2 min, median
sampling 1 s, path 7.57 km, displacement 4.83 km — a real trip) and
user `000`, trajectory `20090508100431.plt` (355 fixes, 2009-05-08
10:04:31 → 10:53:08, 48.6 min, median sampling 5 s, path 3.03 km,
displacement 0.15 km — near-stationary wandering). Two users, two
sampling rates, two movement regimes; 1,471 rows total. The full
archive (~24.8M rows) was never replayed, per the exit criteria.

Data quality found in the subset (not cleaned away): 5 duplicate
timestamps in user 010 (`dt = 0`); 2 logging gaps > 300 s (474 s,
997 s); speed spikes up to 69.4 m/s (249 km/h) from GPS jumps —
`TrajectoryKinematicsStep` reports the distance but withholds the
speed across gaps and on `dt <= 0`. Altitude is 0 ft on some fixes
of user 010 (suspect missing-value coding, not investigated).

## Verdict table

"Runs" means the step executes. A step can run and still be
DOES NOT APPLY: for trajectories the question is whether the output
means anything, and the replay measures exactly that.

| Step family | Verdict | Evidence (replay / test) |
|---|---|---|
| Per-point kinematics (`TrajectoryKinematicsStep`, new, generic) | APPLIES | Hand-computed distances/speeds; null speed on first fix, `dt <= 0`, gaps > 300 s. Speed nulls 9/1,471 (0.6%). |
| Inter-arrival times | APPLIES | Hand-computed `[null, 5, 10, 1]` s + rolling means. The sampling interval itself is the signal (1 s vs 5 s users). |
| Temporal lags, rows mode | APPLIES | Hand-computed lag-1 / rolling means of speed per trajectory. |
| Temporal lags, duration mode | DOES NOT APPLY | Fails on the real subset: `ValueError: Duplicate temporal keys` (5 duplicate timestamps). On distinct timestamps it is null everywhere (no fix exactly 60 s before any fix). Exact-elapsed lags need regular sampling. |
| Rolling means, rows mode | APPLIES WITH ADAPTATION | Hand-computed. Caveat: in rows mode `closed` is ignored and the window **includes the current fix** — a trailing mean including the current point, not strictly-past. Fine for a feature row predicting the *next* fix; must not be read as strictly-past. |
| EWMA (point-level speed) | APPLIES WITH ADAPTATION | Hand-computed `[1, 5/3, 2.43]` with a 5-min half-life on 5-min bins. Needs `metadata.time_bin` and a sub-day half-life; day-scale half-lives are meaningless on a 44-min trajectory. Row-wise smoothing, not bin-wise. |
| Trend ratio (of row-MAs) | APPLIES | Hand-computed `[1, 1, 1.25, 1.4]`. Inherits the rows-mode caveat above. |
| Recency weights | APPLIES WITH ADAPTATION | Hand-computed `[0.25, 0.5, 1.0]`. The reference is the **global** max timestamp: on the subset the whole 2007 trajectory underflows to weight exactly 0.0 against the 2009 reference. Needs a per-trajectory reference. |
| Temporal components / calendar (cyclical) | APPLIES | Hand-computed hour 3 → sin/cos, Saturday → weekday 6. Time-of-day is a property of the fix, unaffected by movement. |
| Time bins (5 min) | APPLIES | Aggregation key only. Panel count sums to 1,471 = input rows; 3 largest (cell, bin) counts (113, 69, 61) re-derived from the raw `.plt` files by an independent recount (pyproj + manual floor). |
| CRS transform / grid assignment | APPLIES | As an aggregation key. The grid is a binning device, not a fixed unit the entity inhabits: 84 cells × 17 bins = 112 observed (cell, bin) pairs. |
| Panel completion | DOES NOT APPLY (as-is) | Two independent failures. (1) **Scale**: the time axis runs min→max over all units, so 2 trajectories 21 months apart project to 185,273 bins × 84 cells = **15,562,932** rows for 112 observed — not run. (2) **Semantics**: on user 010 alone, 74 observed → 603 rows, 529 generated zeros; each zero means "person elsewhere", yet is labelled `observation_status="observed"`. Needs a trajectory-scoped completion (new step, per-trajectory time axis), not the shared panel. |
| Spatial lag (multi-scale queen rings) | DOES NOT APPLY | Runs. Ring-1 lag: 75.0% exactly 0, 1.8% null, 26/112 non-zero, max 91.0. Non-zero only where the path was one bin earlier. Neighbours are cells the person happened to visit, not a fixed field — the lag re-describes the path. Adaptation: along-track neighbours, not queen rings. |
| KDE (Gaussian, lagged count) | DOES NOT APPLY | Runs. Mean density where the lagged count is 0 is **5.07e-06 > 0** (vs 9.64e-05 where > 0): the KDE smears the path into cells the person was not in. It estimates sampling density along one path, not an event field. Adaptation: KDE over stay points / trip endpoints, or over many users' trajectories (population level). |
| STKDE | DOES NOT APPLY on the sparse panel; APPLIES WITH ADAPTATION on a completed per-trajectory panel | Runs on the completed user-010 panel (603 rows, max 6.6e-04). On the sparse panel the per-unit bin index is not elapsed time, so the temporal decay would silently use wrong lags. Inherits the panel-completion caveat. |
| Near-repeat | DOES NOT APPLY | Runs. Flag = 1 for 25% of visited cells; max ring-1 count 182; the 7 d window covers the whole 15-min panel history. The "repeat" detected is the trajectory's own earlier positions. Adaptation: near-repeat *across different users* in the same cells, or across repeated visits on different days. |
| Hawkes excitation (panel) | DOES NOT APPLY | Runs. Excitation is decayed cumulative self-count of GPS fixes (max 113.0). Worse, on the sparse panel the decay counts **observed bins, not elapsed bins** — proven on cell 19701: step value 6.9976 matches index-based decay exactly, elapsed-time decay would be 6.9904 (gap of 4 bins, 1 step). A Hawkes process models self-exciting events; GPS fixes are logger-driven, not self-exciting. |
| Category counts | DOES NOT APPLY | Fails: `ValueError: CategoryCountsStep: missing column(s): transport_mode`. `.plt` files carry no categorical column. Adaptation: join `labels.txt` transport modes → per-segment mode shares; not built. |
| Time since last event | APPLIES WITH ADAPTATION | On the raw panel it is identically 0.0 (every row has count > 0 by construction — void). On the completed user-010 panel it becomes a genuine per-cell **revisit time** (max 40 min). Needs a sub-day output unit — only `'d'` (days) is supported, so values are small fractions. |
| Burstiness (CV of counts) | DOES NOT APPLY | Runs, but 92.9% null on the raw panel; the CV of fix counts per cell measures sampling regularity, not event clustering. |
| Zero features | DOES NOT APPLY on the raw panel; APPLIES WITH ADAPTATION on a completed panel | Raw panel: nonzero share identically 1.0 (void). Completed user-010 panel: mean share 0.131 — reinterpretable as a *presence* share ("share of bins spent in this cell"), not event absence. |
| EWMA (panel count) | APPLIES WITH ADAPTATION | Runs with a 0.01 d half-life (`count_ewma_0.01d`, max 113.0). Smooths fix counts — a sampling smoother, not an event intensity. |

Not evaluated in this phase (no claim either way): `ComputeDistancesStep`,
`AssignToZonesStep`, context joins, domain steps, the `point_process`
family (`KFunction`, `PairCorrelation`, `HazardRate`, `SurvivalTable`,
`DurationFeatures`, `ContinuousInterEvent`, event-level `HawkesKernel`),
and streaming steps. The event-level Hawkes kernel is *suspect* for
the same reason as the panel excitation (fix times are logger-driven),
but that was not tested and is not claimed here.

## What the recipe ships

`GeoLifeV1Recipe` (`src/eventflow/datasets/geolife/recipes/geolife_v1.py`)
contains only the APPLIES rows: CRS transform → kinematics →
temporal components → cyclical calendar → inter-arrival → rows-mode
lag / rolling mean of speed → 5-min bins → 100 m grid → fix counts
per (cell, bin). Point-level output: 1,471 rows × 26 cols
(`geolife_features.parquet`); panel: 112 rows × 8 cols, count sum
1,471. Panel-family steps are deliberately **not** wired into the
recipe; they are exercised as experiments in
`scripts/replay_geolife.py`.

## Design decisions for maintainer review

1. **Timezone is an assumption**: the `.plt` date/time columns carry
   no zone; the loader labels them `Asia/Shanghai` (Beijing local)
   and performs no conversion. If the logger actually recorded UTC,
   every timestamp is off by 8 h. Flagged, not resolved.
2. **`max_gap_s = 300` in kinematics** is a judgement call: speed is
   withheld across longer logging gaps. A different threshold changes
   2 speed values on this subset.
3. **Rows-mode rolling means include the current fix** (upstream
   `MovingAverageStep` ignores `closed` in rows mode). The recipe
   relies on the "feature row predicts the next fix" reading; if the
   project later requires strictly-past windows, rows mode needs a
   shift or a `closed="left"` implementation.
4. **Grid origin is data-dependent** (`AssignToGridStep` with
   `origin=None` takes the min projected coordinate): cell IDs are
   not stable if the subset changes. Chicago pins the origin in
   config; GeoLife should too once a canonical extent is chosen.
5. **The archive holds 18,670 `.plt` files vs 17,621 trajectories**
   in the dataset description. Difference not investigated — possibly
   empty/duplicate logs. The subset itself was verified file by file.
6. **Download path**: the details page
   (`microsoft.com/.../details.aspx?id=52367`) returns HTML, not the
   zip. The file was fetched from the official Microsoft CDN direct
   link
   (`download.microsoft.com/download/F/4/8/F4894AA5-FDBC-481E-9285-D5F8C4C4F039/Geolife%20Trajectories%201.3.zip`,
   313,164,406 bytes = 298.7 MB, SHA not published by Microsoft to
   check against). No substitute dataset was used.
7. **No registration wiring**: per the task, `datasets/__init__.py`,
   registries and `recipes/__init__.py` were not touched; the recipe
   is constructed directly in tests and in the replay script. The
   coordinator wires registration at integration.
