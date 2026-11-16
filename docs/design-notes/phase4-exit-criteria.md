# Phase 4 — Exit criteria (defined before implementation)

Phase 4 is an evaluation phase: **no new feature families**. It proves the
features built in Phases 1–3 do not leak and carry signal. Criteria are
binary (PASS/FAIL) and verified by execution, not declaration.

Baseline (main @ `d888e134`, merge of PR #45): 916 tests passed, coverage
85% (6,598 stmts, 959 missed), `make audit` PASS 5/5. Chicago replay
1,908,088 rows × 90 cols, sum `count` = 60,235. Beijing replay 420,768
rows × 64 cols, NA PM2.5 8,739 → 8,739. GeoLife subset 1,471 rows, count
panel 112 rows, sum 1,471.

## Criteria

1. **Suite green, coverage ≥ 85%.** Full suite passes (ruff, black, mypy
   clean). Test count/coverage are guardrails, not goals.
2. **Non-regression replays unchanged.** Chicago 1,908,088 rows × 90 cols,
   sum `count` = 60,235; Beijing 420,768 rows × 64 cols, NA PM2.5
   8,739 → 8,739; GeoLife subset 1,471 rows, count panel 112 rows,
   sum 1,471.
3. **Blocked splits.** Temporal, spatial (unit-disjoint) and entity
   (entity-disjoint) blockers implemented as library utilities, each with
   reference tests. A `validate_split` rejects any train/test pair that
   shares time bins (temporal), units (spatial) or entities (entity) —
   the negative control: a deliberately leaking split MUST be rejected,
   not silently passed.
4. **Walk-forward harness.** Expanding-window walk-forward over time bins,
   one evaluation period per step, pipeline refit per fold. Fittable steps
   are fit on the train fold only. Worked evaluation on the Chicago replay
   with numbers reported (not just "it ran").
5. **Fitted-statistics freeze.** `CategoryCountsStep` gains fit/freeze
   semantics: `fit(train_frame)` learns the top-N vocabulary from the
   train slice only; `run()` on a later slice uses the frozen vocabulary,
   unseen categories land in `other`. A test proves a held-out slice
   cannot change the fitted vocabulary (produced columns identical
   before/after mutating the held-out slice). The Chicago recipe run
   unfitted is byte-identical in columns to the pre-Phase-4 output.
6. **Target encoding: not applicable, documented.** No target encoding
   exists in the library (removed in Phase 0: it must be out-of-fold).
   No new encoder is invented. The decision is recorded in a design note,
   not in code.
7. **Synthetic ground truth.** A generator produces panel data with a
   known signal; the pipeline recovers the planted parameters within a
   stated tolerance (numbers in tests). A planted leak (a feature using
   future information) scores ~perfect under a random split and collapses
   under a temporal blocked split (causal score); `validate_split`
   rejects a deliberately overlapping split. Boundary, pinned by test:
   blocking a split does NOT sanitize a column that was already computed
   from future information (it still scores 1.0 on blocked test rows) —
   the harness rejects leaking *splits*; it cannot inspect column
   provenance. A future-peeking column must not be built.
8. **Downstream evaluation examples.** PAI (hotspot ranking) and PR-AUC
   (classification) worked examples on the Chicago replay, kept separate
   from the library (scripts/examples + docs), explicitly labelled NOT
   universal preprocessing acceptance criteria.
9. **`make audit` PASS 5/5**, evidence pack under `audit/phase-4/`,
   traceability "After phase 4" section.

## Fixed design decisions (for the maintainer's review)

- **"Blocked" means:** temporal = disjoint time intervals, optional
  embargo gap between train end and test start; spatial = disjoint unit
  sets (no shared grid cells / stations); entity = disjoint entity IDs.
  Combined time×space blocking applies both conditions.
- **FittableStep protocol (contract for the harness and for
  `CategoryCountsStep`):**
  `fit(event_frame) -> Self` learns fitted statistics from that frame
  only; `is_fitted -> bool` reports whether fitted statistics are held.
  The harness calls `fit(train_frame)` on every step that implements
  `fit` before running a fold. Unfitted `CategoryCountsStep` keeps the
  current behaviour (learn top-N from the run frame) for backward
  compatibility with single-run recipes.
- **Metrics without new runtime dependencies.** Runtime deps are numpy,
  polars, pydantic, shapely, pyproj, typer, pyyaml, python-dateutil —
  no sklearn/scipy. PAI and PR-AUC are implemented on numpy.
- **Target encoding stays out.** See criterion 6.
- **What counts as passing synthetic recovery:** the planted parameter
  (e.g. lag-1 coefficient, hotspot location, seasonal amplitude) is
  recovered within a tolerance stated in the test — the tolerance itself
  is part of the deliverable, not adjusted after the fact.
