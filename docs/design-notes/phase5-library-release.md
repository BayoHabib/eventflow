# Phase 5 — Library release (2026-11-16)

Final phase of the roadmap. No new feature families, no behaviour change
to any computation: naming, packaging, reproducible examples, docs.

## What was delivered

### Adapter aliases (`docs/design-notes/phase5-adapter-naming.md`)
- `SequenceOutput.masks` → alias `attention_mask`;
  `StreamOutput.inter_times` → `inter_event_times`;
  `StreamOutput.states` → `state_features`.
- Aliases are read-only properties returning the identical array object
  (`alias is field`); old names keep working. Serialization carries both
  keys (`to_torch()` dicts, metadata shapes/dtypes, NumPy `.npz`, JSON);
  deserializers accept either key, preferring the new one, so pre-Phase 5
  files still load. Parquet/Arrow column names and PyTorch Geometric keys
  unchanged (storage-internal / another library's convention).
- Uniform `.data` accessor on all five outputs (plain field on
  `TableOutput`, read-only property elsewhere, pointing at the descriptive
  payload name, which is kept).
- Convention: `feature_names` is canonical; `state_names` /
  `channel_names` stay as domain-accurate spellings.
- Tests: `tests/core/test_adapter_aliases.py` (21 tests, incl. old-format
  round-trips); the api_reference adapter example is executed by
  `tests/test_docs_examples.py::test_docs_api_reference_adapter_aliases`.

### Packaging
- `pyproject.toml`: license now `{file = "LICENSE"}`; the fake author
  email `team@shikong.dev` removed (maintainer must add a real contact
  before the first upload — flagged in `docs/release.md`). Version stays
  `0.1.0`, Alpha. No missing runtime dependency found.
- `python -m build` from a clean checkout → `shikong-0.1.0.tar.gz` +
  `shikong-0.1.0-py3-none-any.whl`; `twine check` PASSED on both
  (logs: `audit/phase-5/build.log`, `audit/phase-5/twine_check.log`).
- Clean install verified in a fresh venv from the wheel only:
  `shikong --help`, `shikong version` → 0.1.0, `validate` on the
  Chicago recipe (21 steps), `examples/chicago_quickstart.py`
  (120 events → 516 rows, sum `count` = 120, 0 null counts), CLI run to
  Parquet. **No upload performed** — publishing is the maintainer's
  gesture.
- `CHANGELOG.md` (Keep a Changelog, 0.1.0, explicitly unreleased) with
  the measured numbers per phase, including the honest misses (Phase 2
  timings, Phase 3 irregular sampling). `docs/release.md` documents the
  exact release steps.

### Reproducible example environment
- `examples/chicago_quickstart.py` (full 21-step Chicago recipe on 120
  synthetic events, asserts panel invariants) executed by
  `tests/integration/test_examples.py`; `examples/README.md` documents
  the clean-install procedure and what it proves / does not prove.
- `examples/Dockerfile` (examples only — not an API, not a deployment
  service) **not built**: no Docker in this VM, stated honestly in the
  README and the Dockerfile header. Build it on a Docker machine before
  relying on it.

### Docs
- `docs/api_reference.md` rewritten: panel, dynamics, categorical +
  fit/freeze, spatial v2, evaluation, connectors, Beijing/GeoLife
  recipes, `FeatureConfig` table, CLI, tracking; fake methods, stale
  signatures, phantom steps and the wrong GitHub URL fixed. Every example
  is executed by `tests/test_docs_examples.py` (32 tests) plus a
  link-resolution guard.
- README / GETTING_STARTED / `docs/datasets.md` / `docs/architecture.md`
  corrected (stale counts, "does not do" bullets contradicted by Phase 1,
  phantom recipes, nonexistent steps in examples).
- Demo notebook: execution attempted, **fails** (`ValueError: cannot
  concat empty list` — it reads monthly CSVs from a path that does not
  exist here). Marked UNVERIFIED in its first cell and in
  `notebooks/README.md`; its figures are presented nowhere as facts.

## Measured results (exit criteria, see `phase5-exit-criteria.md`)

1. Suite green, coverage ≥ 86%: **1,046 passed, 2 skipped, 86%**
   (6,982 stmts, 979 missed) — baseline 991 / 86%.
2. Replays unchanged: Chicago 1,908,088 × 90, sum `count` = 60,235;
   Beijing 420,768 × 64, NA PM2.5 8,739 → 8,739; GeoLife 1,471 × 26,
   count panel 112 rows.
3. Aliases proven by tests (old + new names, round-trips): pass.
4. Main data attribute decided and documented: pass.
5. Package builds + `twine check` clean: pass (no upload).
6. Clean install verified: pass.
7. Reproducible example environment: pass, with the Dockerfile-build
   caveat above.
8. Docs complete for Phases 1–4, examples executed, notebook marked
   unverified: pass.
9. CHANGELOG.md + release procedure: pass.
10. `make audit` PASS 5/5, traceability updated: pass
    (`audit/phase-5/manifest.json`; counter now **16 attained /
    10 partial / 1 not attained / 1 unverifiable** — objectives 21
    attained, 23 and 24 partial; only FastAPI (22) remains not
    attained, out of scope by maintainer decision).

## Out of scope (maintainer decision)

No FastAPI service, no deployment Docker, no model-training utilities,
no prediction services. No PyPI upload.

## Left open (follow-ups, not Phase 5)

- `return_masks` / `return_inter_times` flags are never consulted by
  `convert()` (see known-limitations).
- `CompletePanelStep` emits explicit-zero rows with a null `time_bin`
  (see known-limitations).
- Dockerfile needs a real `docker build` + run on a Docker machine.
- Real author contact email needed before the first PyPI upload.
- Demo notebook needs its data path fixed (or the notebook retired).
- Issues #38, #39, #41 untouched (as scoped); `output.partition_by`
  still a blocking error; `scripts/check_real_data_audit.py` bare
  `read_text()` calls still outside CI.
