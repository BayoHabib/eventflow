# Known limitations — EventFlow (phase 0, 2026-11-15)

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
  `EventFlow Team <team@eventflow.dev>`. Changing it is Habib's decision.
- **No generated API docs**: no Sphinx/MkDocs, no ReadTheDocs site (the
  dead ReadTheDocs URL was removed from `pyproject.toml` in phase 0).
- **`tests/datasets/` is empty** (only `__init__.py`).
- **Adapter conversion limits**: `to_tensorflow()` exists only on Raster
  (and is a NumPy transpose); `to_torch()` is absent from Table.
- **Custom exceptions**: none — everything raises `ValueError`,
  `ImportError` or `NotImplementedError` (audit objective 19).

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
