# Phase 4 — Fitted statistics: fit/freeze and inventory

Phase 4, criteria 5 (fitted-statistics freeze) and 6 (fitted-statistics
inventory). Baseline: main @ `d888e134` (merge of PR #45), 916 tests,
coverage 85%.

## 1. The `FittableStep` contract

Defined in `src/eventflow/core/pipeline.py`, next to `Step`:

```python
class FittableStep(Protocol):
    """A step that learns fitted statistics from one frame and freezes them.

    `fit` learns from the given frame only. `run` after `fit` must use the
    frozen statistics, never re-learn from the run frame. `is_fitted`
    reports whether fitted statistics are held.
    """

    def fit(self, event_frame: EventFrame) -> "FittableStep": ...

    @property
    def is_fitted(self) -> bool: ...
```

Two deliberate deviations from the sketch in the exit criteria, both
forced by the repo as it stands:

- The return annotation is `"FittableStep"` (a forward reference), not
  `typing.Self`. The package declares `requires-python = ">=3.10"` and
  mypy is configured with `python_version = "3.10"`; `typing.Self`
  exists only from 3.11, so `from typing import Self` fails mypy and
  would raise `ImportError` on a Python 3.10 install. `typing_extensions`
  was not used either: it is not a declared runtime dependency and the
  task forbids adding one. The runtime contract — method names,
  signatures, semantics — is unchanged; if the project ever drops 3.10,
  `Self` can replace the forward references mechanically.
  `CategoryCountsStep.fit` is annotated `-> CategoryCountsStep` for the
  same reason.
- The protocol is **not** `@runtime_checkable`, and steps do not inherit
  from it: conformance is structural. The walk-forward harness detects
  fittable steps with `hasattr(step, "fit")` (see
  `src/eventflow/evaluation/walkforward.py`), so no `isinstance` check
  is needed. `CategoryCountsStep` satisfies the protocol structurally.

`Step`, `Pipeline`, and every other behaviour in `pipeline.py` are
untouched.

## 2. `CategoryCountsStep` fit/freeze

API (all in `src/eventflow/core/steps/categorical.py`):

- `fit(event_frame) -> CategoryCountsStep` — computes the top-N
  vocabulary per category column from the given frame only, reusing the
  existing `_select_top_values` (including its collision/validation
  errors), and stores it as
  `self._fitted: dict[str, list[tuple[Any, str]]] | None`. Returns `self`
  for harness chaining. A frame missing a category column raises the
  same `ValueError` as `run()` (the missing-column check was factored
  into `_require_columns`, shared by both). `fit` does **not** reject
  null grouping keys — the vocabulary does not depend on them; `run()`
  still rejects them, as before.
- `is_fitted -> bool` — `False` before `fit`, `True` after.
- `fitted_values(cat_col) -> list[tuple[Any, str]]` — the frozen
  (value, sanitized) pairs for one column. Raises `ValueError` when
  unfitted and `KeyError` for a column that is not configured. The
  returned list is a copy: mutating it cannot change the frozen
  vocabulary.

`run()` uses the frozen vocabulary when fitted and **never** re-learns
from the run frame; unseen categories in the run frame land in the
`other` bucket (existing `_build_aggregation` semantics: anything not
in the kept values, nulls included, is `other`). When unfitted, `run()`
keeps the current behaviour exactly — it learns the top-N from the run
frame — for backward compatibility with single-run recipes. The
unfitted branch is literally the pre-Phase-4 code path; the only change
is that the vocabulary selection is now behind an `if fitted / else`
branch.

### Why refit replaces

The walk-forward harness refits the pipeline on each fold's train
window, reusing the same step instances. If `fit` accumulated or merged
vocabularies, a later fold would inherit categories from an earlier
fold's train window — a leak across folds, and a vocabulary that no
single fold ever produced. Refit therefore overwrites `self._fitted`
wholesale: after `fit(frame2)`, no trace of `frame1`'s vocabulary
remains. A failed `fit` (e.g. a sanitization collision) leaves the
previous frozen vocabulary untouched, because the new dict is assigned
only after all columns validate. This is covered by tests.

### What is frozen — and what is not

Only the **vocabulary** (which columns exist) is frozen. Counts, shares
and entropy are still computed from the run frame: they are per-
(unit, time) features of that frame, not statistics to be carried
across frames. A test slice therefore gets its own group totals — a
new category inflates the `other` count of its groups, which is correct
— but it cannot create new columns or change which categories are
tracked.

### Unseen categories and the `other` bucket

Three options were available for a run-frame value outside the frozen
vocabulary: error, drop the row, or fold it into `other`. Error would
make evaluation brittle — any new category in a test fold would abort
the run. Dropping rows would silently change the panel: row counts and
group totals would no longer match the frame the step was given.
`other` preserves both: every event row is kept, every group total is
the true group size, shares still sum to 1 and entropy stays defined.
It is also the conservative choice for signal: an unseen category
contributes nothing beyond "not one of the known top-N", so a model
cannot profit from test-only categories. Nulls already land in `other`,
so the semantics are consistent rather than a special case.

## 3. YAML / registry path

`CategoryCountsStep` built from `CategoryCountsConfig` via the registry
(`registry.create("category_counts", params={...})`) still works
unfitted — the existing config tests cover this, and
`tests/core/test_fitted_statistics.py` pins it again plus the shipped
Chicago recipe: `configs/recipes/chicago_crime_v1.yaml` declares
`categorical_features: [primary_type, location_description]` with
`categorical_top_n: 10`, and the step built from it has `is_fitted is
False`. `fit` is a Python-API / harness concern, not YAML-declarable:
recipes run unfitted, and only the evaluation harness (walk-forward)
calls `fit`, once per fold, on the train window.

## 4. Fitted-statistics inventory

Method: every `.collect()` call in `src/eventflow/core/steps/*.py` and
`src/eventflow/core/*.py` was inspected, plus `encode_categorical` in
`core/features.py`. Classification:

- **(a) per-frame computation** — the collected data is used to compute
  the output for the frame being run, or to validate it; nothing is
  kept on the step, so a later frame cannot be affected.
- **(b) fitted statistics** — statistics learned from data and reused
  across frames. Only one such case exists among Steps.

The claim "the only (b) in Steps is `CategoryCountsStep` top-N" is
enforced by a test (`tests/core/test_fitted_statistics.py`): among all
`Step` subclasses reachable from the steps package and the dataset
recipes, only `CategoryCountsStep` defines `fit` in its own `__dict__`.

| Location | What is collected / learned | Class | Reasoning |
|---|---|---|---|
| `steps/categorical.py` `_select_top_values` | Top-N vocabulary per category column (frequencies over the whole frame) | **(b)** | The only cross-frame statistic in Steps. Freezable via `fit()` since Phase 4; unfitted `run()` re-learns it per frame. |
| `steps/categorical.py` `_reject_null_keys` | Null counts of the grouping keys | (a) | One-row validation aggregate; raises, produces no state. |
| `steps/context.py` | Supplied context geometries; event coordinates for the spatial join | (a) | Context data is user-supplied reference data, not learned from the run frame; the join is computed per frame. |
| `steps/density.py` KDE / STKDE | Kernel edge list, per-unit values, cell centres | (a) | Edge lists and KDE sums are built from the run frame (collected eagerly in chunks only to avoid OOM on a Chicago-size panel); nothing persists. Validation aggregates likewise. |
| `steps/domain.py`, `steps/dynamics.py`, `steps/observation.py` | — | (a) | No `.collect()` on data at all; pure lazy / per-row computation. |
| `steps/hawkes.py` | Null / non-finite probe of the value column | (a) | Validation only. The kernel is a fixed exponential with constructor hyperparameters — no fitting. |
| `steps/near_repeat.py` | Ring edge lists, per-(unit, time) values | (a) | Edge lists + values of the run frame, joined eagerly per ring to avoid OOM; nothing persists. Validation aggregates likewise. |
| `steps/panel.py` | (unit, time) keys; per-unit values of non-key columns | (a) | Keys are validated (empty frame, time dtype); other columns are classified static-vs-varying **within** the run frame to decide the fill policy. No state. |
| `steps/point_process.py` | Max timestamp (decay reference); sorted timestamps; pairwise distances; global mean inter-event time | (a) | All recomputed from the run frame on every `run()`. Note the global-mean case (`ContinuousInterEventStep`): the mean is taken over the whole run frame, so running the step on a frame that spans train+test would mix folds — but the step holds no state, so separately-run frames cannot leak into each other. The same caveat applies to any per-frame aggregate; it is a property of the call, not of stored state. |
| `steps/spatial.py` | Grid origin (frame min, only when not supplied); coordinates for KDE; per-group means for lags; per-group snapshots for Moran / Getis-Ord | (a) | Recomputed from the run frame on every `run()`; constructor-supplied `origin` / weights are hyperparameters, not learned. |
| `steps/streaming.py` | Sorted events; running statistics in `OnlineStatisticsStep` | (a) | The running state is created fresh by `initialize_state()` inside each `run()` and scoped to that run; it is never reused across frames. |
| `steps/temporal.py` | Max timestamp (decay reference); time-key null / duplicate check | (a) | Reference time is recomputed from the run frame; the key check is validation only. |
| `steps/validation.py` (all checks) | Whole-frame collects for checks | (a) | Validation only: returns `ValidationResult`, produces no features and no state. |
| `core/features.py` `compute_ratios` | — (pure per-row expression) | (a) | Supported, tested; no data-derived quantity at all. |
| `core/features.py` `aggregate_by_category` | Full distinct-category set of the frame (per call) | **(b\*)** | See §5. |
| `core/features.py` `encode_categorical` | Full vocabulary (one-hot) / ordering (ordinal) over the whole frame (per call) | **(b\*)** | See §5. |
| `core/features.py` `normalize_features` | Mean / std / min / max over the whole frame (per call, lazy expressions) | **(b\*)** | See §5. |
| `core/spatial.py`, `core/observation.py`, `core/context/tabular.py` | Coordinates for projection; event/panel key validation; context-source scan | (a) | Materialization / validation boundaries; no statistics learned or kept. |
| `core/adapters/*`, `core/event_frame.py`, `core/pipeline.py` (`StreamPipeline`) | Frame materialization for output / batching | (a) | Orchestration, not statistics. |

**(b\*)** means: fitted-statistics *character within a single call* —
the function learns a vocabulary / ordering / moments from the frame it
is given and applies them to that same frame — but it holds **no state
between calls**, so it cannot leak across separately-called frames. The
leak mode is calling it on one frame that spans folds. These three
helpers are not Steps, are used by no recipe, and are out of scope for
the freeze work (§5).

Two further notes on the inventory:

- Constructor hyperparameters (bandwidths, half-lives, decay rates,
  ring lists, window sizes, `top_n`) are **not** fitted statistics:
  they are supplied by the caller, never derived from data.
- The `SpatialWeights` / grid objects some spatial steps accept are
  caller-supplied inputs (like context data), not learned by the step.

## 5. `encode_categorical` — left as-is, with a warning

`encode_categorical(event_frame, col, method)` in
`core/features.py` is EXPERIMENTAL (since Phase 0) and is used by **no
recipe and no step** — verified by grep over `src/`, `tests/`,
`scripts/` and `docs/`; its only references are its own docstring, the
API reference, the known-limitations note, and a test asserting it is
documented as EXPERIMENTAL. It was deliberately **not** redesigned in
Phase 4:

- Phase 4 is an evaluation phase: no new feature families, no redesign
  of experimental helpers.
- Giving it a fit/transform split would promote an untested helper
  (one-hot column order is sorted, but ordinal encoding still uses a
  Python `map_elements` UDF, nulls are unhandled, and the returned
  frame's schema does not describe the new columns) into something that
  looks production-ready. The honest move is to leave it experimental
  and say so.

**Warning — do not use `encode_categorical` (nor
`aggregate_by_category` or `normalize_features`) across a train/test
split.** Each learns its vocabulary / ordering / moments from the whole
frame it is called on: one-hot columns are created for every distinct
value present, ordinal codes follow the sorted order of the values
present, and `normalize_features` centres/scales by the whole frame's
mean/std (or min/max). Called on a combined train+test frame, test
values leak into the training features (new one-hot columns, shifted
ordinal codes, shifted moments); called separately per fold, the
columns no longer match across folds. If categorical encoding is ever
needed across folds, it must go through a fitted Step with a frozen
vocabulary — i.e. the `CategoryCountsStep` pattern, or a new Step built
the same way — not through these helpers. The same warning applies to
`aggregate_by_category` (full distinct-category set per call) and
`normalize_features` (whole-frame moments, no fit/transform split);
both are documented as EXPERIMENTAL in their docstrings for exactly
these reasons.

## 6. Verification

- `tests/core/test_categorical_steps.py`: 27 pre-existing tests pass
  **unmodified**; 13 new fit/freeze tests pass
  (`TestCategoryCountsFitFreeze`): train/test `other`-bucket behaviour,
  held-out-slice mutation leaving produced columns identical, refit
  replacing, `is_fitted` flag, `fitted_values` (incl. copy semantics
  and error cases), `fit` reusing `_select_top_values` errors,
  multi-column fit, unfitted-run-equals-fit-then-run on the same frame,
  and the pinned pre-Phase-4 unfitted column set.
- `tests/core/test_fitted_statistics.py` (new, 4 tests): the
  `FittableStep` protocol shape; only `CategoryCountsStep` defines
  `fit` among all Step classes; the registry builds an unfitted step
  from `CategoryCountsConfig`; the shipped Chicago recipe carries
  exactly one, unfitted `CategoryCountsStep`
  (`["primary_type", "location_description"]`, `top_n=10`).
- Total for the touched test files: **44 passed**. The full suite was
  not run (parallel Phase 4 workers share the work copy; the
  coordinator runs it at integration).
- `ruff check`, `black --check`, `mypy` clean on the touched files.
- Not re-run here: the full Chicago replay (1,908,088 rows) — the
  unfitted-behaviour proof at unit-test level is that the unfitted
  branch of `run()` is the pre-Phase-4 code path (same output as
  fit-then-run on the same frame, and the same pinned column set), plus
  the recipe-level check that the Chicago step is built unfitted. The
  replay non-regression itself is criterion 2, owned by the coordinator.
