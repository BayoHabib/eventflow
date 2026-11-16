# EventFlow TODO List

> Production Readiness Checklist & Improvements

## 🗺️ Roadmap (phases 0–5)

> Scope (decided 2026-11-15): EventFlow is a generic spatio-temporal
> preprocessing library; Chicago crime is one use case. The target taxonomy
> includes point events,
> trajectories, time series on fixed locations (sensors), evolving
> fields/rasters, areal data by zone, origin-destination flows, graph data of
> different kinds, and exogenous context to join. These are target use cases,
> not a claim that every representation is supported today. Contracts must
> preserve the relevant timestamps, entity identities, geometries or spatial
> supports, observation states, and optional categories; timestamp + point
> coordinates alone is not a universal representation.
>
> Phase evaluation method: exit criteria defined before, binary, verified by
> execution not declaration; three levels (correctness, non-regression vs a
> frozen measured baseline, proof on real data); test count/coverage are
> guardrails, not goals; the maintainer's go-ahead is the final validation.

### Phase 0 — Honest foundation ✅ DONE (2026-11-15)

- Strict typed config (`FeatureConfig`/`GridConfig`/`TemporalConfig`/
  `RecipeConfig`, `extra="forbid"`); unknown or unimplemented config key =
  blocking error; declared-but-ignored Chicago features removed from the YAML
  into a design note (`docs/design-notes/chicago-declared-features.md`).
- Built-in audit system: `make audit`, evidence pack, traceability matrix of
  the 28 audit objectives (`docs/audit/traceability.md`), known-limitations
  register (`docs/audit/known-limitations.md`).
- README Quick Start rewritten from real signatures, executed by tests.
- Measured at merge (PR #27): 564 tests passed / 0 failed, coverage 82%,
  `make audit` PASS 5/5, Chicago replay identical (56,491 rows, sum `count` =
  60,235).
- Follow-ups merged the same day: PR #28 (adapter UTF-8 reads), PR #32
  (validate false positives, issues #29/#30), PR #34 (audit skipped replay
  reported as SKIP/INCOMPLETE, issue #31), PR #35 (unknown temporal
  components in legacy function and `CalendarEncodingStep`, issue #33).

### Phase 1 — Generic v1 features ✅ DONE (2026-11-15)

Seven generic feature groups as core Steps in Polars, YAML-declarable,
with explicit data-availability cutoffs and no Chicago-specific code in
`core/`. Durations, grouping keys, and category policies are configurable;
the durations below are examples for the Chicago recipe. Reuse and verify
existing Steps before adding implementations.

Delivered (PR #37, merged 2026-11-15): `CompletePanelStep` (explicit-zero
panel completion, opt-in per recipe, `min_periods`) + the 7 groups as
17 steps in the Chicago pipeline; 705 tests passed / 0 failed, coverage
84%, `make audit` PASS 5/5; Chicago Q1 2024 replay 1,908,088 rows × 78
columns (5,242 cells × 364 bins), sum `count` = 60,235, 1,851,597
explicit-zero rows; traceability updated (15 attained / 8 partial /
4 not attained / 1 unverifiable). Performance follow-ups opened as
issues #38 (vectorize `SpatialLagStep`, 51.4 s of ~61 s) and #39 (eager
collects at plan-build time in `_validate_time_keys`).

1. Lags t−1/t−7/t−28 + rolling means 7/14/28 d + EWMA
2. Sin/cos cyclical (hour, day-of-week, month)
3. Basic spatial lag via grid adjacency, using only values available at the
   cutoff (full multi-scale / KDE / Hawkes stay in Phase 2)
4. Counts by type + shares + Shannon entropy, with cardinality guardrail
   (top-N + "other")
5. Trend (recent vs longer history)
6. Time since last event + burstiness
7. Zero-event indicators and zero-run features where observation support
   establishes an interpretable zero count

This also implements generically the Chicago recipe's
`categorical_features` + `aggregation_windows` (see the Phase 0 design note).
Order: define the observation-support/completion contract first, implement
opt-in panel completion for features that need it, then features and wiring.
Each group needs reference tests vs hand-computed values, data-availability
cutoff (no-leakage) tests, and docs. Replay Chicago Q1 2024 and a second real
dataset from another domain; produce a `make audit` evidence pack and update
traceability. PRs are in English.

- [x] **Design decision (decided 2026-11-15, implemented in PR #37):**
  complete the panel with explicit zeros inside the observation support
  before any window/lag, opt-in per recipe with `min_periods`; zeros
  distinct from missing/out-of-domain via the observation panel states.
  The constraints below were kept:
  - A zero event count requires in-domain support, active collection, and a
    complete input batch under the observation-quality contract. An absent
    row alone is not evidence of zero events.
  - Missing sensor measurements remain missing; do not replace them with
    zeros. Unknown observation quality and out-of-domain pairs remain
    distinguishable from measured values and interpretable zero counts.
  - Completion is opt-in and limited to declared units and intervals, with
    a size guardrail. The Chicago expansion from 56k to roughly 1.9M rows
    is a planning estimate, not a generic requirement.
  - Declare elapsed-time vs row-based windows, timezone/bin boundaries,
    minimum valid history (`min_periods`), and null/zero handling per
    feature. Irregular observations must not silently become regular.
- [x] **Availability contract in Phase 1 (partially delivered):** every
  feature declares which source observations are available at the cutoff —
  trailing windows closed on the left, spatial lag computed from lagged
  counts only. Temporal history and spatial neighbors exclude unavailable
  values. The fitted-statistics half (learn top-N categories, vocabularies,
  and other fitted statistics from the permitted training/history slice
  only; freeze them for validation/test; define unseen-category behavior)
  was not implemented in PR #37 — carried to Phase 4.
- [x] Exit criteria (verified at merge, except the second-domain replay,
  moved to Phase 3): all Phase 0 criteria still pass, coverage 84%,
  every group has reference + leakage tests, Chicago replay shows the new
  columns with correct values (busiest cell hand-verified against the raw
  CSV). Skipped checks remain SKIP/INCOMPLETE.

### Phase 2 — Space-time v2 ✅ DONE (2026-11-15)

Delivered (PR #42): 5 new generic Steps in Polars, causal, on the
explicit-zero completed panel — `MultiScaleSpatialLagStep` (queen rings
1/2/3, built on the vectorized approach from issue #38),
`KernelDensityStep`, `SpaceTimeKernelDensityStep`, `NearRepeatStep`,
`HawkesExcitationStep` (fixed exponential kernel, no fitting). The
Chicago recipe now runs 21 steps (was 17) with 12 new columns; 835
tests passed + 3 skipped / 0 failed (838 collected), coverage 85%
(CI run 94, 10/10; local audit run 838 passed), `make audit` PASS 5/5;
Chicago Q1 2024 replay 1,908,088 rows × 90 columns, sum `count` =
60,235, 1,851,597 explicit-zero rows; 14/14 hand-checks vs the raw
CSV; traceability unchanged (15 attained / 8 partial / 4 not attained
/ 1 unverifiable). Two bugs caught by the hand-check and fixed in the
PR: NearRepeat/STKDE wired to the wrong time column (`time_bin`, null
on generated rows) and OOM on the giant joins (eager steps + chunked
KDE join).

- [x] Spatial lag multi-scale (queen rings 1/2/3, disjoint;
  `SpatialLagStep` vectorized the same way, equivalence to the Phase 1
  implementation proven to 1e-9, ring-1 value 0.375 preserved)
- [x] KDE / STKDE (Gaussian KDE in events/m², separable Gaussian ×
  exponential-decay product kernel; fixed explicit bandwidths, no
  auto-estimation, no edge correction)
- [x] Near-repeat (queen rings 1–3, trailing 7 d and 7–14 d windows,
  closed-left, plus `nr_event_7d` flag)
- [x] Simplified Hawkes (fixed exponential kernel with a half-life
  parameter; no fitting, no background separation, no branching ratio)
- [x] Extend the Phase 1 completion contract to additional spatial
  supports (partially delivered): neighbour tables built once from the
  completed panel's unit set; zone/areal supports remain out of scope
- [ ] **Exit criterion 5 (timings) NOT met as written:** single-scale
  `SpatialLagStep` 6.19 s (was 51.4 s, ×8.3) vs the ≤ ~2 s target;
  total pipeline ~98 s vs ~61 s in Phase 1 (≤ ~20 s target). The
  vectorization itself is proven; the remaining cost is the group_by
  over edge×bin intermediates, and the four new families add their own
  joins. Issue #38 is addressed by this PR but not closed — maintainer
  decides. Issues #39 and #41 untouched.

### Phase 3 — Broader genericity proof ✅ DONE (2026-11-15)

Delivered (PR #44, merged 2026-11-15): second-domain proof on Beijing
Multi-Site Air Quality (UCI 501) — 420,768 rows × 64 columns, 12 stations,
NA PM2.5 identical before/after completion and after the full pipeline
(8,739 → 8,739), no leakage across stations; third representation on a
GeoLife GPS subset (1,471 rows, count panel 112 rows) with the
supported/unsupported boundary statement; generic `TabularFileSource`
external-table connector (25 tests), demonstrated with an honest negative
(0/1,471 matched on non-overlapping ranges) and a positive control (8/9).
Generic fix found by the Beijing proof: `measure_cols` in
`CompletePanelStep` — measurement columns are never zero-filled. 916 tests
collected (913 passed + 3 skipped on CI, 0 failed), coverage 85% (CI run
96, 10/10), `make audit` PASS 5/5; Chicago replay unchanged (1,908,088
rows × 90 columns, sum `count` = 60,235); traceability unchanged
(15 attained / 8 partial / 4 not attained / 1 unverifiable).

- [x] **Second-domain proof:** Beijing Multi-Site Air Quality (UCI 501),
  12 stations, hourly 2013–2017; missing measurements stay missing
  (NA counts unchanged), histories separated by entity, applicable Steps
  work without Chicago-specific assumptions. Honest limit: the shipped
  panel is already complete and timestamps are regular hourly — the
  proof covers gaps / missing-stays-missing / per-entity separation,
  not irregular sampling (variant C, Intel Berkeley, not attempted).
- [x] Third real dataset with another representation (GeoLife GPS
  trajectories, subset of 1–2 users): supported/unsupported statement
  written in `docs/design-notes/phase3-geolife-boundary.md` — kinematics,
  rows-mode lags/rolling, temporal components apply; spatial lag, KDE,
  near-repeat and Hawkes do not apply to trajectories, with reasons.
- [x] External connectors: generic `TabularFileSource` (file-backed
  tabular context join), tested and demonstrated; a production GeoLife
  enrichment still needs an overlapping 2007–2009 weather source, none
  fabricated.

### Phase 4 — Extended anti-leakage evaluation ✅ DONE (2026-11-15)

Delivered (PR #46, merged 2026-11-15): blocked splits (disjoint time
intervals, unit sets and entity IDs; optional embargo gap) with
`validate_split` rejecting any overlap; expanding-window walk-forward
harness with refit per fold; fitted-statistics freeze
(`CategoryCountsStep.fit/freeze`, proven by test — a held-out slice
cannot change the fitted vocabulary); synthetic ground truth (lag-1
corr 0.529 vs 0.6 planted, hotspot top-4 exact) with a planted leak
detected (leaky random-split PR-AUC 1.0000 vs blocked causal 0.4031);
downstream PAI/PR-AUC examples on Chicago, labelled explicitly NOT
acceptance criteria. Target encoding: not applicable — none exists in
the library, none invented. 991 tests collected (988 passed + 3 skipped
on CI, 0 failed), coverage 86% (CI run 98, 10/10), `make audit` PASS
5/5; Chicago/Beijing/GeoLife replays unchanged; traceability unchanged
(15 attained / 8 partial / 4 not attained / 1 unverifiable).

- [x] Blocked splits, walk-forward
- [x] Out-of-fold target encoding, if introduced; fit only on training folds — not applicable (no target encoding in the library; none invented)
- [x] Fitted-statistics scoping (carried from Phase 1, 2026-11-15): `CategoryCountsStep.fit/freeze` — fitted statistics learned from the permitted training slice only, frozen for validation/test; unseen-category behavior defined
- [x] Optional downstream evaluation examples with task-appropriate metrics (PAI for hotspot ranking, PR-AUC for classification) — examples only, not universal preprocessing acceptance criteria
- [x] Synthetic ground truth

### Phase 5 — Library release ✅ DONE (2026-11-15)

Delivered (PR #48, merged 2026-11-15): adapter aliases as read-only
properties (`attention_mask`, `inter_event_times`, `state_features`),
old names kept working; `.data` on all 5 outputs, `feature_names`
canonical; package builds from a clean checkout, `twine check` PASS
(no upload performed — the maintainer decides the release moment);
clean install verified from the wheel alone (`validate` 21 steps,
quickstart 120 events → 516 rows); `examples/chicago_quickstart.py`
executed by test; CHANGELOG.md 0.1.0 Alpha + `docs/release.md`;
api_reference rewritten with 32 executing tests. 1,048 tests collected
on CI (1,045 passed + 3 skipped, 0 failed), coverage 86% (CI run 100,
10/10), `make audit` PASS 5/5; Chicago/Beijing/GeoLife replays
unchanged; traceability 16 attained / 10 partial / 1 not attained
(FastAPI, out of scope by the maintainer's decision) / 1 unverifiable.

Covers the library's High Priority items below. Prediction services and
trained models are optional downstream applications, with separate scope.

- [x] Adapter aliases (`attention_mask`, `inter_event_times`,
  `state_features`)
- [x] PyPI publication (+ CHANGELOG.md) — package ready, `twine check`
  PASS; upload itself not performed
- [x] Reproducible example environment (Docker if useful) — clean install
  + executable examples verified; Dockerfile added but not built (no
  Docker in the VM)
- [x] Docs

## 🔴 High Priority (Pre-Production)

### API Stability
- [x] **Standardize adapter output attribute names** (DONE in phase 5, PR #48)
  - [x] `SequenceOutput.masks` → add alias `attention_mask` (HuggingFace/PyTorch convention)
  - [x] `StreamOutput.inter_times` → add alias `inter_event_times` (full name)
  - [x] `StreamOutput.states` → add alias `state_features` (consistent suffix)
  - [x] Decide on naming convention: `feature_names` vs `state_names` vs `channel_names` — `feature_names` canonical

- [x] **Consistent main data attribute naming** (DONE in phase 5, PR #48 — `.data` on all 5 outputs)
  | Adapter | Current | Proposed |
  |---------|---------|----------|
  | TableOutput | `data` | `data` ✓ |
  | SequenceOutput | `sequences` | `data` or keep |
  | RasterOutput | `raster` | `data` or keep |
  | GraphOutput | `node_features` | `data` or keep |
  | StreamOutput | `states` | `data` or keep |

### Library Distribution
- [x] **Publish to PyPI** (package ready in phase 5, PR #48 — `twine check` PASS; upload itself not performed, maintainer decides)
  - [x] Finalize version number (currently 0.1.0)
  - [x] Create CHANGELOG.md
  - [ ] `python -m build && twine upload dist/*` — not performed

- [x] **Reproducible example environment** (DONE in phase 5, PR #48)
  - [x] Document a clean installation and executable examples
  - [x] Add a Dockerfile for examples if needed; a prediction API is not a
    prerequisite for releasing the library — Dockerfile added, not built (no Docker in the VM)

## 🟡 Medium Priority (Post-MVP)

### Error Handling
- [ ] Add user-friendly error messages for common mistakes
- [ ] Create custom exception classes (`EventFlowError`, `AdapterError`, etc.)
- [ ] Add input validation with clear error messages

### Documentation
- [ ] Add docstring examples to all public functions
- [ ] Create API reference docs (Sphinx or MkDocs)
- [ ] Add more tutorials/notebooks
- [ ] Document adapter output schemas clearly

### Testing
- [x] Increase coverage from 68% to 80%+ — DONE in phase 0 (measured 82%, 564 tests, 2026-11-15; see `docs/audit/`)
- [ ] Add integration tests for full pipelines
- [ ] Add property-based tests (Hypothesis) for adapters
- [ ] Add performance regression tests

### Logging
- [ ] Replace print statements with structured logging
- [ ] Add log levels (DEBUG, INFO, WARNING, ERROR)
- [ ] Support JSON logging for production

## 🟢 Low Priority (Nice to Have)

### Performance
- [ ] Add benchmarks for adapters (time & memory)
- [ ] Profile large datasets (1M+ events)
- [ ] Optimize RasterAdapter for large grids
- [ ] Add lazy loading for large datasets

### Features
- [ ] Add `StreamingAdapter` for real-time data
- [ ] Add `to_tensorflow()` method to all adapters
- [ ] Add `to_torch()` method to all adapters (some have it)

### DevOps
- [x] Keep documentation-only edits outside the full test/build workflow:
  Markdown-only changes skip CI; changes to code, configs, notebooks, or
  workflows still run it. Manual `workflow_dispatch` remains available.
- [ ] Add pre-commit hooks for all checks
- [ ] Add GitHub release automation
- [ ] Add code coverage badge to README
- [ ] Set up ReadTheDocs

## 📝 Notes from Demo

> **Unverified (phase 0, 2026-11-15):** the figures below come from the demo
> notebook, which is not executed in CI and was not executed in the
> 2026-11-15 audit. They are **not confirmed** — see
> `docs/audit/known-limitations.md`. Do not cite them as results.

### Key Insights from Chicago Crime Analysis
1. **Spatial correlation ρ = 0.515** → ST-GNN will benefit from neighbor info
2. **Graph-Sequence correlation r = 0.953** → High-crime areas are volatile
3. **104 anomalies detected** (84 spikes, 20 drops) in Q1 2024
4. **Causal analysis shows reverse causality** → High crime → more enforcement

## Optional downstream applications (separate from library release)

These are possible consumers of EventFlow outputs, not required capabilities
or validated production use cases of the preprocessing library.

- [ ] Model-training examples and baseline models, with their own evaluation
- [ ] Chicago forecast API: RasterAdapter → model → REST API, in an example
  or separate application; domain-specific prediction endpoints such as
  `/predict`, `/hotspots`, and `/anomalies` belong there
- [ ] Anomaly-monitoring example: StreamAdapter → detector → alerts, with
  explicit ingestion/latency requirements and separate validation
- [ ] Resource-allocation dashboard: GraphAdapter → downstream analysis,
  with its own decision and evaluation criteria

---

*Last updated: November 15, 2026 (all phases 0–5 DONE — PR #48 merged, 1,048 tests, 86%; library release-ready, PyPI upload not performed)*
