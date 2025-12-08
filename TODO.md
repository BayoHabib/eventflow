# EventFlow TODO List

> Production Readiness Checklist & Improvements

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
- [ ] Increase coverage from 68% to 80%+
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

*Last updated: December 8, 2025*
