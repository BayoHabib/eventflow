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

### Phase 1 — Generic v1 features (next)

Seven generic feature groups as core Steps in Polars, YAML-declarable,
with explicit data-availability cutoffs and no Chicago-specific code in
`core/`. Durations, grouping keys, and category policies are configurable;
the durations below are examples for the Chicago recipe. Reuse and verify
existing Steps before adding implementations.

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

- [ ] **Open design decision (blocks panel-dependent features):** specify
  the completion API and each feature's policy for missing unit/time pairs.
  Keep these constraints explicit:
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
- [ ] **Second-domain proof in Phase 1:** use real sensor/time-series data
  with irregular timestamps or gaps; show that missing measurements stay
  missing, histories stay separated by entity, and applicable Steps work
  without Chicago-specific assumptions.
- [ ] **Availability contract in Phase 1:** every feature declares which
  source observations are available at the cutoff. Temporal history and
  spatial neighbors must exclude unavailable values, including late-arriving
  observations where applicable. Learn top-N categories, vocabularies, and
  other fitted statistics from the permitted training/history slice only;
  freeze them for validation/test and define unseen-category behavior.
- [ ] Exit criteria: all Phase 0 criteria still pass, coverage ≥ 82%, every
  group has reference + leakage tests, Chicago replay shows the new columns
  with correct values, and the second-domain replay verifies its applicable
  features and missing-data semantics. Skipped checks remain SKIP/INCOMPLETE.

### Phase 2 — Space-time v2

- [ ] Spatial lag multi-scale
- [ ] KDE / STKDE
- [ ] Near-repeat
- [ ] Simplified Hawkes
- [ ] Extend the Phase 1 completion contract to additional spatial supports
  where needed; preserve the same observation-quality semantics

### Phase 3 — Broader genericity proof

- [ ] Extend the second-domain proof established in Phase 1
- [ ] Add a third real dataset with another representation (e.g. trajectories,
  areal data, or flows); state what is supported and what remains unsupported
- [ ] External connectors (POI, weather, rates)

### Phase 4 — Extended anti-leakage evaluation

Build on the per-feature availability checks required from Phase 1.

- [ ] Blocked splits, walk-forward
- [ ] Out-of-fold target encoding, if introduced; fit only on training folds
- [ ] Optional downstream evaluation examples with task-appropriate metrics
  (e.g. PAI for hotspot ranking or PR-AUC for classification); these are not
  universal preprocessing acceptance criteria
- [ ] Synthetic ground truth

### Phase 5 — Library release

Covers the library's High Priority items below. Prediction services and
trained models are optional downstream applications, with separate scope.

- [ ] Adapter aliases (`attention_mask`, `inter_event_times`,
  `state_features`)
- [ ] PyPI publication (+ CHANGELOG.md)
- [ ] Reproducible example environment (Docker if useful)
- [ ] Docs

## 🔴 High Priority (Pre-Production)

### API Stability
- [ ] **Standardize adapter output attribute names**
  - [ ] `SequenceOutput.masks` → add alias `attention_mask` (HuggingFace/PyTorch convention)
  - [ ] `StreamOutput.inter_times` → add alias `inter_event_times` (full name)
  - [ ] `StreamOutput.states` → add alias `state_features` (consistent suffix)
  - [ ] Decide on naming convention: `feature_names` vs `state_names` vs `channel_names`
  
- [ ] **Consistent main data attribute naming**
  | Adapter | Current | Proposed |
  |---------|---------|----------|
  | TableOutput | `data` | `data` ✓ |
  | SequenceOutput | `sequences` | `data` or keep |
  | RasterOutput | `raster` | `data` or keep |
  | GraphOutput | `node_features` | `data` or keep |
  | StreamOutput | `states` | `data` or keep |

### Library Distribution
- [ ] **Publish to PyPI**
  - [ ] Finalize version number (currently 0.1.0)
  - [ ] Create CHANGELOG.md
  - [ ] `python -m build && twine upload dist/*`

- [ ] **Reproducible example environment**
  - [ ] Document a clean installation and executable examples
  - [ ] Add a Dockerfile for examples if needed; a prediction API is not a
    prerequisite for releasing the library

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

*Last updated: November 15, 2026 (roadmap phases 0–5 clarified: support-aware completion and second-domain proof in phase 1, availability contracts, optional downstream applications, Markdown-only CI policy)*
