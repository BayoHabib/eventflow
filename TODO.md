# EventFlow TODO List

> Production Readiness Checklist & Improvements

## 🗺️ Roadmap (phases 0–5)

> Scope (decided 2026-11-15): EventFlow must handle **any** spatio-temporal
> problem, not just Chicago crime — generic core (timestamp + coordinates +
> optional category), Chicago is only a use case. Taxonomy: point events,
> trajectories, time series on fixed locations (sensors), evolving
> fields/rasters, areal data by zone, origin-destination flows, graph data of
> any kind, exogenous context to join.
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
causal, no Chicago-specific code in `core/`:

1. Lags t−1/t−7/t−28 + rolling means 7/14/28 d + EWMA
2. Sin/cos cyclical (hour, day-of-week, month)
3. Basic spatial lag via grid adjacency, causal only (full multi-scale /
   KDE / Hawkes stay in Phase 2)
4. Counts by type + shares + Shannon entropy, with cardinality guardrail
   (top-N + "other")
5. Trend (recent vs longer history)
6. Time since last event + burstiness
7. Zero features

This also implements generically the Chicago recipe's
`categorical_features` + `aggregation_windows` (see the Phase 0 design note).
Order: panel completion first, then features, then wiring; each group with
reference tests vs hand-computed values, causal-cutoff (no-leakage) tests,
docs; Chicago replay on Q1 2024; `make audit`, evidence pack, traceability
update, PR in English.

- [ ] **Open decision (blocks Phase 1):** what is a missing cell/bin pair?
  Complete the panel with explicit zeros inside the observation support
  before any window/lag (zeros distinct from missing/out-of-domain via the
  observation panel states; Chicago panel grows 56k → ~1.9M rows;
  completion opt-in per recipe with `min_periods`), or compute windows/lags
  on observed pairs only, or decide per feature.
- [ ] Exit criteria: all Phase 0 criteria still pass, coverage ≥ 82%, every
  group has reference + leakage tests, Chicago replay shows the new columns
  with correct values.

### Phase 2 — Space-time v2

- [ ] Spatial lag multi-scale
- [ ] KDE / STKDE
- [ ] Near-repeat
- [ ] Simplified Hawkes
- [ ] Explicit-zero panel completion

### Phase 3 — Genericity proof

- [ ] 2nd dataset from another domain
- [ ] 3rd dataset from another domain
- [ ] External connectors (POI, weather, rates)

### Phase 4 — Anti-leakage evaluation

- [ ] Blocked splits, walk-forward
- [ ] Out-of-fold target encoding
- [ ] PAI / PR-AUC
- [ ] Synthetic ground truth

### Phase 5 — Product

Covers the High Priority items below.

- [ ] Adapter aliases (`attention_mask`, `inter_event_times`,
  `state_features`)
- [ ] PyPI publication (+ CHANGELOG.md)
- [ ] Docker support
- [ ] FastAPI service
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

### Production Deployment
- [ ] **Create FastAPI prediction service** (`src/eventflow/api/`)
  - [ ] `POST /predict` - Risk score for lat/lon/datetime
  - [ ] `GET /hotspots` - Daily hotspot map
  - [ ] `GET /anomalies` - Detected anomalies
  - [ ] `GET /health` - Health check endpoint
  
- [ ] **Publish to PyPI**
  - [ ] Finalize version number (currently 0.1.0)
  - [ ] Create CHANGELOG.md
  - [ ] `python -m build && twine upload dist/*`

- [ ] **Docker support**
  - [ ] Create `Dockerfile`
  - [ ] Create `docker-compose.yml` for API + dependencies
  - [ ] Add to CI/CD pipeline

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
- [ ] Add model training utilities
- [ ] Add pre-trained baseline models

### DevOps
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

### Recommended First Production Use Cases
1. **7-Day Crime Forecast API** - RasterAdapter → ConvLSTM → REST API
2. **Real-time Anomaly Alerts** - StreamAdapter → Z-score monitoring
3. **Resource Allocation Dashboard** - GraphAdapter → Patrol optimization

---

*Last updated: November 15, 2026 (phase 0: coverage item closed, demo figures marked unverified; roadmap phases 0–5 added)*
