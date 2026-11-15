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
  risk, no fit/transform split).
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

## Phase-0 scope reminders

- Phase 0 implemented **no new features**: no lags, no rolling windows,
  no per-category counts. Those are phase 1
  (`docs/design-notes/chicago-declared-features.md`).
- Phase 0 started **none** of the production TODO items: adapter aliases,
  FastAPI, PyPI, Docker (audit objectives 21–24).
