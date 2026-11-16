# Phase 5 — Library release: exit criteria

Defined before work starts. Binary, verified by execution, never by declaration.
Baseline (main @ `4afbde762`, PR #46 merge): 991 tests collected, 991 passed
locally, coverage 86% (6,953 stmts, 983 missed), `make audit` PASS 5/5.
Chicago replay: 1,908,088 rows × 90 cols, sum `count` = 60,235.
Beijing replay: 420,768 rows × 64 cols, NA PM2.5 8,739 → 8,739.
GeoLife subset: 1,471 rows, count panel 112 rows, sum 1,471.

## Criteria

1. **Suite green, coverage ≥ 86%.** Full suite passes locally and on CI;
   coverage does not drop below the 86% baseline. Test count/coverage are
   guardrails, not goals.
2. **Non-regression replays unchanged.** Chicago, Beijing and GeoLife replays
   produce exactly the baseline numbers above.
3. **Adapter aliases proven by tests.** `SequenceOutput.attention_mask`,
   `StreamOutput.inter_event_times`, `StreamOutput.state_features` exist and
   return exactly the same arrays as the old names; old names keep working
   (alias, not silent rename). Serialization round-trips (JSON/numpy/pytorch
   dict keys) cover the new names.
4. **Main data attribute decided and documented.** Each output's main payload
   attribute is settled (`data` everywhere vs keep per-adapter names) with a
   written justification in `docs/design-notes/phase5-adapter-naming.md`;
   the naming convention question (`feature_names` vs `state_names` vs
   `channel_names`) is decided and applied consistently.
5. **Package builds and passes `twine check`.** `python -m build` produces
   sdist + wheel from a clean checkout; `twine check dist/*` is clean.
   No upload is performed (maintainer decides the release moment).
6. **Clean install verified.** The built wheel installs in a fresh venv and
   `eventflow --help` plus a smoke example run there.
7. **Reproducible example environment.** A clean-install procedure and
   executable examples exist and were actually executed (Dockerfile for the
   examples if useful); the procedure states what it proves and what it
   does not.
8. **Docs complete for Phases 1–4.** `docs/api_reference.md` (and README /
   GETTING_STARTED where they overlap) cover panel completion, dynamics,
   spatial v2, evaluation utilities, connectors and recipes; every code
   example in the docs is executed by a test or a script, never claimed
   from reading. The demo notebook is either executed green or explicitly
   marked unverified — never presented as verified.
9. **CHANGELOG.md + release procedure.** CHANGELOG.md exists (Keep-a-Changelog
   style, version 0.1.0 Alpha) and `docs/` documents the release steps
   (build, check, upload, tag) the maintainer will run.
10. **`make audit` PASS 5/5** and traceability updated ("After phase 5"
    section in `docs/audit/traceability.md`; counter restated honestly).

## Out of scope (maintainer decision, TODO retouch)

No FastAPI service, no deployment Docker, no model-training utilities,
no prediction services. Those are optional downstream applications.
