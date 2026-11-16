# Known limitations — Shikong (phase 0, 2026-11-15)

This file lists, frankly, what is **not** proven, not implemented, or still
silently ignored. It is the first thing an auditor should read. Items are
removed only when a test or a measurement proves them gone — never by
rewriting the README.

## Not verified

- **Demo notebook not executed.** `notebooks/demo_chicago_crime.ipynb`
  (1.29 MB on GitHub) has never been executed in CI or in any audit. Its
  figures — spatial correlation ρ = 0.515, graph-sequence correlation
  r = 0.953, 104 anomalies (84 spikes, 20 drops) in Q1 2024, reverse
  causality claim — are **neither confirmed nor refuted**. The README and
  `TODO.md` now say so explicitly. Executing the notebook in CI is a
  later step, not phase 0.
- **Remote CI state not consulted.** Only `.github/workflows/ci.yml` was
  read. Whether GitHub Actions is currently green on `main` is unknown.
  Local runs (this tree, Python 3.12) are green; the 3 OS × 3 Python
  matrix is not reproduced locally.
- **Performance not measured.** Throughput and memory on 1M+ events were
  never measured; no performance regression test exists (audit objective 26).
- **No property-based tests.** Hypothesis is declared in dev dependencies
  but used nowhere in `tests/`.

## Still silently ignored (known, not fixed in phase 0)

- **`DatasetConfig.quality.*`** (`min_year`, `max_missing_coords`,
  `outlier_detection` in `configs/datasets/chicago_crime_example.yaml`):
  never read by `mapping.py`; cleaning is fixed in `clean_chicago_data`.
- **`RecipeConfig.context` and `RecipeConfig.output`** are still
  `dict[str, Any]` (only `grid`, `temporal` and `features` were typed in
  phase 0). A typo inside `context`/`output` can still be silently
  ignored — except `output.partition_by`, which the CLI `run` command now
  rejects explicitly, and `context.*.enabled: true`, which the Chicago
  recipe rejects explicitly.
- **`TransformCRSStep(target_crs="EPSG:26971")` is hardcoded** in the
  Chicago recipe, not taken from `DatasetConfig.crs`.
- **`EventFrame.with_lazy_frame()` keeps the input schema**: functions in
  `core/features.py` return frames whose schema does not describe the new
  columns. Tested Steps register provenance instead.

## Experimental code (marked in the source, untested)

- `core/features.py`: `aggregate_by_category`, `encode_categorical`,
  `normalize_features` — see the module docstring for each limitation.
  `normalize_features` computes statistics over the whole frame (leakage
  risk, no fit/transform split). Note (phase 1): `aggregate_by_category`
  is superseded for recipes by the tested `CategoryCountsStep`
  (`core/steps/categorical.py`, top-N + `"other"` bucket); the utility
  itself stays EXPERIMENTAL.
- `tracking/mlflow_tracker.py`: no test against a real MLflow backend.
- `datasets/chicago_crime/context/` (weather, events, demographics):
  unwired, untested sources.
- `core/context/` joiners: spatial joins other than `grid`/`zone` raise
  `NotImplementedError` (`joiners.py:139`).

## Identity and documentation gaps (known, out of phase-0 scope)

- **Author identity**: `pyproject.toml` still says
  `Shikong Team <team@shikong.dev>`. Changing it is Habib's decision.
- **No generated API docs**: no Sphinx/MkDocs, no ReadTheDocs site (the
  dead ReadTheDocs URL was removed from `pyproject.toml` in phase 0).
- **`tests/datasets/` now holds the phase-3 dataset tests**
  (`test_beijing_air_quality.py`, `test_geolife.py`); it was empty
  before phase 3.
- **Adapter conversion limits**: `to_tensorflow()` exists only on Raster
  (and is a NumPy transpose); `to_torch()` is absent from Table.
- **Custom exceptions**: none — everything raises `ValueError`,
  `ImportError` or `NotImplementedError` (audit objective 19).

## Phase-3 scope reminders and new limitations (phase 3, 2026-11-16)

- **Beijing's shipped panel is already complete.** The 12 per-station
  CSVs hold all 35,064 hourly bins each (2013-03-01 → 2017-02-28);
  there are no missing station-hours. All missingness is NA values
  inside present rows. `CompletePanelStep` therefore generates 0 pairs
  on the real data; generated pairs are exercised only on synthetic
  gaps in tests. The proof covers gaps / missing-stays-missing and
  per-station separation, NOT irregular sampling (regular hourly
  timestamps).
- **Station coordinates are second-hand.** The UCI files carry no
  coordinates; the values come from Ma et al. (2023) Table 1 for 11
  stations and Zhang et al. (2020) Table S1 for Shunyi. Independent
  sources disagree by up to ~0.04° (~4 km) for some stations. No step
  in the v1 recipe consumes coordinates, so the imprecision cannot
  leak into any feature — but the values are not primary data.
- **`CompletePanelStep` still zero-fills observed nulls in
  `value_cols`.** The blanket `fill_null(0)` applies to the whole
  value column, not only to generated pairs. `measure_cols` fixes the
  measurement case (never zero-filled); a future `value_cols` column
  with observed nulls would still be silently zeroed. Chicago is
  unaffected (counts are never null on observed rows). Restricting the
  fill to generated pairs only is a separable follow-up.
- **`measure_cols` is not exposed in `FeatureConfig`.** The Beijing
  recipe constructs `CompletePanelStep` directly; the YAML-declarable
  config has no measurement-columns key. A recipe that wants
  YAML-declared `measure_cols` needs a `FeatureConfig` extension.
- **Rolling means skip nulls.** Polars `rolling_mean` with
  `min_samples=1` averages the available hours; a 24 h mean over a
  window with missing hours is the mean of the hours present.
  Documented in the Beijing design note, not hidden.
- **GeoLife timezone is an assumption.** `.plt` times are labelled
  `Asia/Shanghai` with no conversion. If the logger used UTC, every
  timestamp is off by 8 h.
- **GeoLife grid origin is data-dependent.** Cell IDs come from the
  minimum projected coordinate of the subset, so they are not stable
  across subsets. Chicago pins the origin in config; GeoLife should
  too once a canonical extent is chosen.
- **GeoLife × Beijing do not overlap in time.** The GeoLife subset
  spans 2007–2009, the Beijing data start 2013-03-01 (gap 1,392.5
  days). The connector demo proves the tolerance correctly rejects
  all 1,471 points (0 matched) instead of forcing a match. A
  production GeoLife enrichment still needs an overlapping 2007–2009
  weather source (reanalysis or an API written to a local file);
  none is on disk and none was fabricated.
- **`TabularFileSource` is local files only.** No live APIs, no
  network fetching, no POI/rates providers — by design. The two seams
  for a future provider are documented in
  `docs/design-notes/phase3-connectors.md`.
- **`TemporalContextJoinStep` with `strategy="exact"` does not sort
  its output.** Values are correct; row order is the Polars left-join
  (hash) order. The step makes no order guarantee; changing shared
  join behaviour was out of scope for phase 3.
- **Audit manifest label is stale.** `scripts/audit.py` still writes
  `"audit": "shikong phase-1 audit"` in `manifest.json`, including
  in the phase-3 pack. Cosmetic only; the measured numbers in the
  manifest are the phase-3 run's.

## Phase-2 scope reminders and new limitations (phase 2, 2026-11-15)

- **No Hawkes fitting.** `HawkesExcitationStep` is a fixed-kernel
  excitation feature (`E(t) = sum_{s<t} count(s) * exp(-(t-s)/tau)`); there
  is no parameter estimation (no MLE for alpha/beta/mu), no
  background/excitation separation, and no branching-ratio estimation.
  The event-level `HawkesKernelStep` (alpha/beta/mu) is a separate,
  model-flavoured step, not a fitted process either.
- **No automatic bandwidth selection.** KDE/STKDE bandwidths are declared
  explicitly by the caller. No Silverman rule, no likelihood
  cross-validation — estimating a bandwidth on the full panel would be a
  fitted statistic and a leakage surface; cross-validated selection
  belongs to phase-4 evaluation.
- **No KDE edge correction.** Density near the boundary of the observed
  support is underestimated: neighbours outside the support do not exist.
  The `KFunctionStep` translation correction is a point-pattern tool and
  does not transfer to panel KDE.
- **`KDEStep` is an approximation, not a KDE.** It counts neighbours
  inside the bandwidth and divides by the buffer area; no kernel function
  is applied. Its docstring now says so. Use `KernelDensityStep` for a
  proper Gaussian KDE.
- **Zone/areal supports still out of scope.** The phase-2 completion
  contract extension covers multi-scale neighbourhoods and distance bands
  over the completed panel's units only; areal data by zone is not a
  panel support yet.
- **Positional time windows.** `NearRepeatStep` and
  `SpaceTimeKernelDensityStep` index time by per-unit bin position and
  assume a regular, complete panel (one row per unit per bin, no gaps) —
  the `CompletePanelStep` output. A gapped panel would silently misalign
  the windows.
- **One initial collect in the vectorized spatial steps.**
  `SpatialLagStep` / `MultiScaleSpatialLagStep` collect the panel once to
  validate values and build the edge list; the per-snapshot collects and
  partitionings of phase 1 are gone. Measured on the Chicago replay
  (6 GB VM, shared CPU — timings vary run to run): the single-scale lag
  is 51.4 s → 6.2 s, the 3-ring multi-scale lag takes ~21 s inside the
  replay, and the full 21-step pipeline takes ~98 s (was ~61 s in
  phase 1 — the multi-scale saving roughly offsets the four new
  families, whose joins dominate the new cost: KDE ~25 s, STKDE ~29 s,
  near-repeat ~17 s). The original exit criterion (single-scale ≤ ~2 s,
  total ≤ ~20 s) was **not** met on this VM; see the phase-2 design
  note. The KDE/STKDE/near-repeat steps are eager, not lazy: each
  collects its result before returning, one ring / one edge chunk at a
  time, so the giant joins never run together at the final collect
  (that combination OOM-killed the 6 GB VM twice during phase 2).
  The KDE edge join is chunked (50k edges per chunk) for the same
  reason. There is still no performance regression test in the suite
  (audit objective 26).
- **Performance now measured, no regression test yet.** Per-step timings
  on the Chicago replay were measured in phase 2 (see
  `audit/phase-2/manifest.json` and `docs/design-notes/phase2-spacetime-v2.md`).
- **Time-column trap, fixed and guarded.** On the completed panel the
  pre-completion `time_bin` column is null on every generated
  (explicit-zero) row — only the schema timestamp column (`date` on
  Chicago) holds the bin datetime on all rows. The phase-2 recipe
  initially wired `NearRepeatStep` and `SpaceTimeKernelDensityStep` to
  `time_bin` and both silently produced wrong values (near-repeat
  ring-1 7 d = 60 instead of 50 on the busiest cell). The recipe now
  passes the timestamp column, and all three steps reject a null time /
  group column with a blocking `ValueError` (regression tests in
  `tests/core/test_density_steps.py`,
  `tests/core/test_near_repeat_steps.py`,
  `tests/integration/test_chicago_pipeline.py`).

## Phase-5 scope reminders and new limitations (phase 5, 2026-11-16)

- **Not published on PyPI.** The package builds and passes `twine check`
  (`audit/phase-5/build.log`, `twine_check.log`), CHANGELOG.md and the
  release procedure exist (`docs/release.md`), but no upload has been
  performed and no author contact email exists yet (the placeholder
  `team@shikong.dev` was removed from `pyproject.toml` rather than
  shipped as a fake address). Publishing is the maintainer's gesture.
- **Dockerfile not built.** `examples/Dockerfile` (examples only, no
  service) exists and is documented, but there is no Docker in this VM;
  the equivalent steps (wheel install in a fresh venv, `--help`,
  `validate`, quick start, CLI run) were verified instead
  (`examples/README.md`). Build the image on a Docker machine before
  relying on it.
- **Demo notebook still unverified.** Execution was attempted in phase 5
  and fails (`ValueError: cannot concat empty list` — it reads monthly
  CSVs from a path that does not exist here); it is marked UNVERIFIED in
  its first cell and in `notebooks/README.md`. Its figures (ρ = 0.515,
  104 anomalies) are presented nowhere as facts.
- **`return_masks` / `return_inter_times` flags are not consulted.**
  `SequenceAdapter.convert()` and `StreamAdapter.convert()` always
  return masks / inter-event times regardless of the flags (the output
  fields are required, so "no masks" has no clean representation).
  Found during the phase-5 naming work; fixing it would change
  behaviour, so it is documented, not fixed
  (`docs/design-notes/phase5-adapter-naming.md`).
- **`CompletePanelStep` emits explicit-zero rows with a null `time_bin`.**
  Only observed (cell, bin) pairs carry a timestamp; generated pairs
  carry the bin in the schema timestamp column (`date` for Chicago)
  but a null `time_bin`. Observed during the phase-5 clean-install
  verification (nulls appear right after step 6, `CompletePanelStep`);
  the audit replay only checks null `count`, not null `time_bin`. The
  Phase 2 fix (NearRepeat/STKDE wired to the schema timestamp column)
  already accounts for this; a library-level fix is a separate task.
- **Evaluation splits now refuse nullable split columns (demo findings,
  2026-11-16).** Splitting the completed panel on `time_bin` silently
  kept 2.9% of the rows and reported a perfect, meaningless PR-AUC of
  1.0. `temporal_split`, `blocked_split`, `spatial_split`,
  `entity_split`, `partition_units`, `validate_split` and
  `walk_forward_splits` now raise a blocking `ValueError` naming the
  null count when the split column contains nulls, and the evaluation
  API accepts collected DataFrames (converted with `.lazy()`). Panel
  semantics are unchanged: generated rows still carry a null `time_bin`.
- **Audit manifest label still stale.** `scripts/audit.py` labels its
  manifest "shikong phase-1 audit" although the numbers are the
  phase-5 run's (same staleness noted in phases 3 and 4).
- **No Sphinx/MkDocs site, no ReadTheDocs.** Docs are Markdown in
  `docs/`; every code example is executed by a test, but there is no
  built documentation site.

## Phase-4 scope reminders and new limitations (phase 4, 2026-11-16)

- **The harness rejects leaking splits, not leaking columns.** A column
  computed from future information scores 1.0 even on temporally blocked
  test rows (`tests/evaluation/test_planted_leak_integration.py::
  test_blocked_split_does_not_sanitize_a_future_peeking_column`).
  Blocking forces time order and exposes the leak (random-split PR-AUC
  1.00 vs blocked causal 0.41), but the column itself must not be built.
  The harness cannot inspect column provenance.
- **No spatial buffer zone around held-out units.** Blocked spatial
  splits are disjoint unit sets with no guard band; neighbouring units
  can still leak through spatial-lag / KDE features across the boundary.
  Documented in `docs/design-notes/phase4-evaluation.md`, not implemented.
- **Walk-forward is expanding-window only.** No sliding-window mode;
  every fold refits from scratch (no warm-start / incremental refit).
- **Downstream metrics are examples, not acceptance criteria.** PAI and
  PR-AUC on the Chicago replay (PAI cellule 7,96 à 1 %, PR-AUC 0,0684
  pour `count_ma_7`) are worked examples with lag features as scores
  and no model trained; no threshold on them accepts or rejects the
  preprocessing (`scripts/phase4_chicago_evaluation.py`).
- **Synthetic recovery tolerances are generator-specific.** The planted
  parameters (lag-1 φ=0,6, hotspot ×3, week-end ×1,6) are recovered
  within the tolerances stated in the tests on 6×6 units × 56 daily
  bins, seeds 42 and 7; they are not a claim about recovery on real
  data at other scales.
- **Audit manifest label still stale.** `scripts/audit.py` labels its
  manifest "shikong phase-1 audit" although the numbers are the
  phase-4 run's (same staleness noted in phase 3).
- **`encode_categorical` still learns its vocabulary over the whole
  frame.** It is EXPERIMENTAL, used by no recipe or Step, and was
  deliberately left as-is; it must not be used across train/test
  (see `docs/design-notes/phase4-fitted-statistics.md`).

## Phase-0 scope reminders

- Phase 0 implemented **no new features**: no lags, no rolling windows,
  no per-category counts. Phase 1 delivered them
  (`docs/design-notes/chicago-declared-features.md`,
  `docs/design-notes/phase1-causality.md`): lags, rolling means, EWMA,
  trend, queen spatial lag, per-category counts/shares/entropy, time
  since last event, burstiness, zero features, cyclical encodings — all
  on the explicit-zero completed panel.
- Phase 0 started **none** of the production TODO items: adapter aliases,
  FastAPI, PyPI, Docker (audit objectives 21–24).
