# Phase 4 — Blocked splits, walk-forward evaluation, metrics

Phase 4 criteria 3, 4 and 8 (metric functions). Implementation:
`src/shikong/evaluation/` (`splits.py`, `walkforward.py`, `metrics.py`).
Tests: `tests/evaluation/test_splits.py`, `test_walkforward.py`,
`test_metrics.py` (37 tests).

## Why random splits are invalid here

A random train/test split of spatio-temporal panel rows leaks through three
mechanisms, one per blocked dimension:

- **Time.** A feature at bin *t* is built to predict bin *t + 1* (Phase 1
  causality convention), and window/lag features read several bins back.
  If bin *t* appears in both train and test, a lag feature computed from
  train rows at *t* is evaluated against test rows at the same *t*: the
  model is scored on data its features were built from. Neighbouring bins
  leak the same way through rolling windows, which is why the temporal
  split supports an embargo gap: rows in `[cutoff, cutoff + gap)` belong
  to neither side.
- **Space.** Spatial lags, KDE and near-repeat features read neighbouring
  units. If a unit is in both sides, a test row's spatial-lag feature is
  computed from train events in that same unit. Even a unit-disjoint split
  still leaks through *adjacent* units when the feature bandwidth reaches
  across the boundary; unit-disjointness is the minimum, not a guarantee
  against all spatial leakage — the safe reading is "no shared unit", and
  the embargo analogue in space is a buffer zone, which this library does
  not provide (see "What the harness does not do").
- **Entity.** The same entity (station, vehicle, user) on both sides lets
  an entity-specific pattern learned in train apply verbatim in test.
  Entity-disjoint splits are the analogue of group K-fold with the entity
  as the group.

"Blocked" therefore means, concretely: temporal = disjoint time intervals
(`train < cutoff <= test`, plus optional embargo); spatial = disjoint unit
sets; entity = disjoint entity sets. Combined time x space blocking
applies both conditions at once, and the two off-diagonal quadrants (early
but held-out unit, late but train unit) are dropped — they cannot be
assigned without breaking one of the blocks.

`validate_split` is the negative control for all of this: it collects the
distinct values of the blocking column(s) on each side and raises
`ValueError` on any overlap. A deliberately leaking split MUST be
rejected there, never silently passed.

## Walk-forward protocol

`walk_forward_splits(frame, time_col, initial_train, step, gap=None,
max_folds=None)` yields expanding-window folds over the time column:

- fold 0 trains on `[t_min, t_min + initial_train)` and tests on the next
  *step* after the embargo *gap*;
- each later fold extends the train window by one *step* (expanding,
  never sliding) and tests on the following *step*;
- folds stop when the test window starts after the last observed time bin;
  a fold whose test window holds no rows (sparse time column) is skipped.

`initial_train` and `step` are `timedelta`s over the actual time values,
so the protocol works for any regular binning (6h Chicago bins, 1h
Beijing bins, 5m GeoLife bins) without a bin-count parameter. Every
yielded fold is checked with `validate_split` on the time column.

`evaluate_walk_forward(steps, frame, time_col, initial_train, step,
score_fn, ...)` runs a fresh `Pipeline` per fold. **Refit policy:** each
step that implements `fit` (detected with `hasattr(step, "fit")`, per the
`FittableStep` protocol in `shikong.core.pipeline`: `fit(event_frame) ->
Self`, `is_fitted -> bool`) is fit on the train fold only, before the
fold runs — fresh fit per fold, fitted statistics from a previous fold
are overwritten, never carried over. The pipeline then runs on the train
fold and on the test fold, and `score_fn(train_out, test_out)` produces
the fold's scores. One `FoldResult` per fold, in chronological order,
with fold index, window bounds, row counts and the score dict.

## Metric definitions

### PAI — Prediction Accuracy Index (hotspot ranking)

Rank all rows by score descending and take the top `x` fraction as the
predicted hotspot area. With `n` rows, the top set has
`n_top = ceil(area_fraction * n)` rows and realised area fraction
`a = n_top / n` (ranks are discrete, so `a` can differ from the requested
fraction by less than `1 / n`). Then:

    PAI = (events captured in the top set / total events) / a

PAI = 1 is chance level; PAI > 1 means the top area concentrates events;
the maximum is `1 / a` (all events in the top set). The denominator uses
the *realised* fraction `a`, not the requested one, so an all-positive
label vector scores exactly 1.0 instead of a ceil-rounding artefact.

Tie handling: stable sort by score descending, so tied scores keep input
order. A tie straddling the cutoff is resolved by input order,
deterministically but arbitrarily — PAI is a ranking metric, and tied
scores carry no ranking information.

Undefined (raises `ValueError`): no positive events (0/0), empty input,
non-binary labels, non-finite scores, `area_fraction` outside (0, 1].

### PR-AUC — area under the precision-recall curve

Rows are ranked by score descending; precision and recall are evaluated
after each *distinct* score value (the curve steps once per threshold,
starting at recall 0 / precision 1) and the area is integrated with the
trapezoid rule. The chance level is the positive rate, not 0.5.

Tie handling: rows sharing a score are consumed as one group, so the
curve — and the area — do not depend on the arbitrary order of tied rows.
This matches the `precision_recall_curve` convention.

Undefined (raises `ValueError`): no positive events (recall undefined),
empty input, non-binary labels, non-finite scores. An all-positive label
vector scores exactly 1.0.

Both metrics are pure functions on numpy arrays; no sklearn/scipy
(runtime deps are unchanged).

## What the harness deliberately does NOT do

- It does not choose a model, a threshold, or a feature set. Scoring is
  caller-supplied (`score_fn`), so the harness stays usable for any
  downstream task.
- It does not provide a spatial buffer zone around held-out units; only
  unit-disjointness is enforced (see the leakage discussion above).
- It does not run the full Chicago replay inside unit tests. The worked
  evaluation in `tests/evaluation/test_walkforward.py` uses a small
  in-test panel in the shape of the Chicago recipe output (6h `time_bin`,
  grid `unit`, `count`, `count_lag_1`): 4 folds, 160/240/320/400 train
  rows, 80 test rows each, PAI/PR-AUC in range. The full-replay worked
  examples (criterion 8) belong to the coordinator's scripts/examples,
  not to the unit tests.
- It does not freeze fitted statistics itself; that is the step's job
  (criterion 5, `CategoryCountsStep`). The harness only guarantees the
  step is fit on the train fold and nothing else.
- It does not invent target encoding (criterion 6): there is none in the
  library, and none is added here.

## Verification

37 tests, all passing; `ruff check`, `black --check` and `mypy` clean on
the touched files (see the worker report). The `FittableStep` protocol
already exists in `src/shikong/core/pipeline.py` (added by the worker
owning criterion 5 during this phase); the harness detects fittable steps
by `hasattr(step, "fit")` and is compatible with it.
