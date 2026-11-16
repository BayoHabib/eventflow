# Changelog

All notable changes to Shikong will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Shikong is pre-1.0: `0.1.0` is an **Alpha** release and the public API
may still change.

## [Unreleased]

### Added

- **STPP exact-event emitters** (`src/shikong/stpp/`, first STPP
  increment — data only, no fitted model). `extract_exact_events`
  turns an `EventFrame` into chronologically sorted exact events
  (projected kilometres, sorted mark codebook, null/non-finite values
  refused loudly); `chronological_event_splits` chains the library's
  own `temporal_split` with embargo gaps and reconciles splits +
  embargos against the cleaned total; `emit_gatech` (Gatech / THP /
  EasyTPP `{split}.pkl` + `.json`), `emit_catalog` (EarthquakeNPP-style
  `{split}_catalog.csv` — no `magnitude` column, crime has none — plus
  float64 `{split}_txy.npy`) and `emit_tick` (per-node timestamp
  lists) write the three formats plus a deterministic `manifest.json`
  stating "no fitted model". Scripts `scripts/emit_chicago_stpp.py`
  and `scripts/emit_geolife_stpp.py` emit Chicago Q1 2024 (60,235
  cleaned events; aux 8,922 / train 30,464 / dev 8,065 / test 9,915,
  embargos 1,435 + 1,434) and the GeoLife Phase 3 subset (1,471 fixes;
  auxiliary 14 days, remainder split in thirds, embargo 0 — a 2-day
  embargo would swallow the 49-minute remainder). Emitted data files
  live outside the repo (`~/workspace/eventflow-eda/stpp-emitters/`).
  40 tests in `tests/stpp/test_emitters.py` (synthetic frames only);
  design note `docs/design-notes/stpp-emitters.md`; independent
  verification `audit/stpp-emitters/verify_emissions.py` (395 checks,
  0 failures). Compatibility is claimed only for the formats read
  verbatim (Gatech/THP, EarthquakeNPP CSV, tick); DeepSTPP/AutoSTPP/
  SMASH loaders and ETAS catalog columns remain unverified (inventory
  §7) and are not guessed.

### Changed

- **Renamed: EventFlow → Shikong (时空, shíkōng, "space-time").** The PyPI
  distribution name `eventflow` has been taken since 2017 (unrelated
  pre-releases, no description), the import package `eventflow` is also
  installed by `flowevent` on PyPI, and the name collides with the academic
  EventFlow visualization tool (HCIL Maryland, Monroe et al. 2013) and the
  .NET EventFlow framework. Package `src/eventflow/` → `src/shikong/`,
  all imports, the `shikong` console script, and docs updated. This is a
  clean break with **no compatibility shim**: version 0.1.0 Alpha was never
  published on PyPI and has no users. The GitHub repository itself is
  still named `BayoHabib/eventflow` for now (renamed separately after this
  lands); GitHub redirects the old repository URLs, so existing links keep
  working. Includes the evaluation guardrail from PR #53 (reject nullable
  split columns; accept DataFrames).

## [0.1.0] - 2026-11-15

**Status: Alpha, unreleased — no upload to PyPI has been performed.**
See `docs/release.md` for the steps the maintainer will run to release.

All figures below are measured by `make audit` in this repository
(`docs/audit/traceability.md`, `audit/phase-N/manifest.json`); nothing is
claimed from reading code alone. Where a criterion was not met, that is
stated instead of smoothed over.

### Added — Phase 0: foundations (2026-11-15)

- Generic, dataset-agnostic core: `EventFrame`, schema validation,
  spatial/temporal treatments, composable registry-based pipelines.
- Five output adapters: Table `(N, F)`, Sequence `(B, T, F)`,
  Raster `(T, C, H, W)`, Graph (nodes + edges), Stream (continuous
  timestamps).
- Chicago crime dataset adapter and `chicago_crime_v1` recipe; CLI
  (`shikong run / validate / list-datasets / list-recipes / version`).
- Executable README Quick Start examples, covered by
  `tests/test_readme_examples.py`.
- `make audit` (ruff, black, mypy, pytest, Chicago replay) — PASS 5/5.
- **Measured:** 564 tests collected, 564 passed, 0 failed, 0 skipped;
  coverage 82% (5,176 statements, 928 missed).

### Added — Phase 1: panel completion and dynamics (2026-11-15)

- Explicit-zero panel completion (`CompletePanelStep`), categorical
  counts/shares/entropy with a top-10 + `"other"` guardrail
  (`CategoryCountsStep`), lags and rolling means (`TemporalLagStep`,
  `MovingAverageStep`), EWMA, trend, queen spatial lag of the shortest
  lag, time-since-last-event, burstiness, zero features, cyclical
  calendar encodings, and a strict `FeatureConfig` (an unknown or
  unimplemented declared feature raises a blocking error instead of
  being silently ignored).
- Causality convention documented in
  `docs/design-notes/phase1-causality.md`: a feature row at bin `t`
  predicts bin `t + 1`.
- **Measured:** 705 tests passed; coverage 84% (5,718 statements,
  912 missed); Chicago replay (T1 2024) 1,908,088 rows
  (5,242 cells x 364 bins), `sum(count)` = 60,235, 1,851,597
  explicit-zero rows, all `observation_status == "observed"`, no null
  counts. Lag, rolling mean, spatial lag, EWMA, trend and entropy
  verified by hand against the raw CSV.
- Not done: `output.partition_by` is still unimplemented (the CLI
  raises a blocking error if it is declared).

### Added — Phase 2: space-time features, vectorised spatial lag (2026-11-15)

- Multi-scale queen-ring spatial lag (`SpatialLagStep`, rings 1/2/3),
  Gaussian KDE, separable space-time KDE, near-repeat ring counts,
  fixed-kernel Hawkes excitation. Two bugs fixed along the way: a
  wrong time column (near-repeat/STKDE ran on a null `time_bin` on
  generated rows) and OOM on large joins (eager evaluation + chunking
  for KDE/STKDE/near-repeat).
- **Measured:** 838 tests passed; coverage 85% (6,261 statements,
  944 missed); Chicago replay 1,908,088 rows x 90 columns,
  `sum(count)` = 60,235; the 4 new feature families verified by hand
  against the raw CSV (14 checks, busiest cell 1930/1195, bin
  2024-02-17 12:00).
- **Honest miss:** the performance criterion from issue #38 was not
  met as written — single-scale lag 6.19 s (down from 51.4 s, x8.3)
  but not <= 2 s, and the full pipeline runs in ~98 s (up from ~61 s)
  with the 4 new families. PR #42 implements #38 without closing it.

### Added — Phase 3: genericity proof, second and third domains (2026-11-15)

- Second-domain proof: Beijing air-quality (fixed sensors, measures
  with NA) replayed with the same Steps; GeoLife trajectories
  (mobile, third representation) replayed on a subset; generic
  `TabularFileSource` connector; `measure_cols` fix in
  `CompletePanelStep` (measures are no longer zero-filled).
- **Measured:** 916 tests passed; coverage 85% (6,598 statements,
  959 missed); Chicago replay unchanged (1,908,088 rows x 90 columns,
  `sum(count)` = 60,235); Beijing 420,768 rows x 64 columns
  (12 stations x 35,064 hours), NA PM2.5 8,739 -> 8,739 (no leakage
  across stations); GeoLife subset 1,471 rows, count panel 112 rows,
  `sum(count)` = 1,471; connector demo 0/1,471 matched (non-overlapping
  ranges, honest negative) with 8/9 on the positive control.
- **Honest limit:** the delivered Beijing panel was already complete
  with regular hourly timestamps — the proof covers
  missing-stays-missing and per-entity separation, NOT irregular
  sampling (variant C was not attempted).

### Added — Phase 4: anti-leakage evaluation (2026-11-15)

- Blocked splits (by time / units / entities, optional embargo, no
  spatial buffer zone), walk-forward with an expanding window, freeze
  of fitted statistics (`CategoryCountsStep.fit/freeze`), synthetic
  ground truth, and Chicago PAI / PR-AUC examples labelled as
  non-acceptance criteria. Target encoding was examined and found
  not applicable (no target exists) — not invented.
- **Measured:** 991 tests collected (991 passed locally; 988 passed +
  3 skipped on CI); coverage 86% (6,953 statements, 983 missed);
  Chicago / Beijing / GeoLife replays unchanged. Synthetic ground
  truth: lag-1 correlation 0.529 vs 0.6 planted; top-4 hotspots exact.
  Planted leakage detected: PR-AUC 1.0000 leaked vs 0.4031 with the
  blocked causal split.

### Added — Phase 5: library release (unreleased)

- Packaging audited for PyPI: `license` now points at the `LICENSE`
  file; the placeholder author email `team@shikong.dev` (not a real
  contact address) was removed — the maintainer adds a real contact
  email before the first upload (see `docs/release.md`). Version
  stays `0.1.0`, Development Status Alpha. `python -m build` from a
  clean checkout produces sdist + wheel; `twine check dist/*` is
  clean. No upload performed.
- Clean-install verification: the built wheel installs in a fresh
  venv (runtime dependencies only, no `-e`, no dev extras) and
  `shikong --help`, `shikong version`, `shikong validate` on the
  shipped Chicago recipe config (21 steps) and the end-to-end smoke
  example all pass there (see `examples/README.md` for the exact
  commands and outputs).
- Runnable examples: `examples/chicago_quickstart.py` runs the full
  `chicago_crime_v1` pipeline (21 steps) on 120 synthetic events and
  asserts the panel invariants (516 rows, 62 columns,
  `sum(count)` = 120, no null counts, all `observation_status ==
  "observed"`); executed by `tests/integration/test_examples.py`.
  `examples/README.md` documents the clean-install procedure and what
  it proves and does not prove. `examples/Dockerfile` is an
  examples-only image (not a prediction API, not a deployment
  service); it was **not built in this VM** (no Docker available) —
  the equivalent steps were verified in a fresh venv instead.
- Output-adapter aliases (backwards compatible, old names keep working):
  `SequenceOutput.attention_mask`, `StreamOutput.inter_event_times`,
  `StreamOutput.state_features` — read-only properties returning the
  identical array object; serialization carries both keys and
  deserializers accept either, so pre-Phase 5 files still load. Uniform
  `.data` accessor on all five outputs. `feature_names` is the canonical
  name vocabulary. Decisions recorded in
  `docs/design-notes/phase5-adapter-naming.md`; proven by
  `tests/core/test_adapter_aliases.py` (21 tests).
- Documentation completed for Phases 1–4: `docs/api_reference.md`
  rewritten (panel, dynamics, categorical + fit/freeze, spatial v2,
  evaluation, connectors, Beijing/GeoLife recipes); every code example
  is executed by `tests/test_docs_examples.py` (32 tests). README,
  GETTING_STARTED, datasets and architecture corrected. The demo
  notebook was executed, fails (it reads monthly CSVs from a path that
  does not exist here), and is marked UNVERIFIED — its figures are
  presented nowhere as facts.
- **Measured:** 1,046 tests collected, 1,046 passed, 0 failed,
  2 skipped; coverage 86% (6,982 statements, 979 missed).
  `make audit` PASS 5/5; Chicago replay unchanged (1,908,088 rows ×
  90 columns, sum `count` = 60,235); Beijing and GeoLife replays
  unchanged. Traceability counter: 16 attained / 10 partial /
  1 not attained / 1 unverifiable.

### Fixed — Phase 5

- Nothing in library behaviour. Packaging, configuration, examples
  and documentation only.

[0.1.0]: https://github.com/BayoHabib/eventflow/releases/tag/v0.1.0
