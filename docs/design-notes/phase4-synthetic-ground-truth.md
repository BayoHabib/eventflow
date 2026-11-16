# Phase 4 — Synthetic ground truth and downstream evaluation examples

Criteria 7 (synthetic ground truth) and 8 (downstream evaluation
examples) of `phase4-exit-criteria.md`. Work by worker C; the
evaluation harness itself (splitters, walk-forward, PAI/PR-AUC as
library metrics) is worker A's and is not used here — the demo and
example scripts below compute their metrics locally with plain
Polars/NumPy and never import `src/shikong/evaluation/`.

## 1. The generator (`tests/evaluation/synthetic.py`)

`generate_synthetic_panel(seed=42, ...)` builds a 6x6 grid of units x
56 consecutive daily bins starting Monday 2024-01-01 (2,016 panel
rows), plus the event-level rows the counts summarise (~5,400 events),
plus the planted parameters. Every draw comes from one
`numpy.random.Generator`; the same seed reproduces the panel exactly.

Generative model — counts are Poisson draws around a rate

```
rate(u, t) = baseline * hotspot(u) * dow(t) * exp(latent_scale * x(u, t)) / E[exp(...)]
x(u, t)    = phi * x(u, t-1) + eps,   eps ~ N(0, 1) per unit
```

Planted parameters (defaults):

| Parameter | Planted value |
|---|---|
| Baseline rate | 2.0 events per unit-bin |
| Hotspot | centre 2x2 block, units 15, 16, 21, 22, rate x3.0 |
| Day-of-week | weekend (Sat/Sun) x1.6, weekday x1.0 |
| Lag-1 AR(1) | latent `phi = 0.6`, `latent_scale = 0.5` |
| Categories | THEFT 0.40, BATTERY 0.25, ASSAULT 0.20, BURGLARY 0.15 |

The log-normaliser `E[exp(latent_scale * x)]` (with
`Var(x) = 1 / (1 - phi^2)`) keeps the multipliers interpretable as
mean-rate multipliers. With `signal=False` every multiplier is 1 and
`phi = 0`: pure Poisson noise at a constant rate — the null control.

`make_sparse_panel` drops a seeded random fraction of (unit, bin)
pairs for the `CompletePanelStep` test; `panel_event_frame` /
`events_event_frame` wrap the frames for the Steps.

## 2. What counts as passing recovery, and why the tolerances are what they are

The tolerances were fixed from the generator's design (sample sizes
and the known Poisson attenuation) before the recovery runs; they were
not adjusted afterwards. The test file states them at the top.

| Quantity (step) | Planted | Tolerance | Justification |
|---|---|---|---|
| Pooled lag-1 corr of `count` vs `count_lag_1` (`TemporalLagStep`) | latent phi = 0.6 | [0.25, 0.75] | Poisson noise attenuates the latent correlation (it can only pull it below phi, never above); the pooled SE over ~2,000 pairs is ~0.02, so 0.25 is >10 SE above 0 and 0.75 leaves room above phi for estimation error. |
| Weekend/weekday mean-count ratio | 1.6 | +/- 0.25 | ~5x the ratio's sampling error (~0.05). |
| Top-4 units by mean count | hotspot units | exact set equality | 3x rate over 56 bins: the rate gap is ~12x the SE of a unit mean. |
| Ring-1 spatial-lag gap, hotspot-adjacent minus far units (`MultiScaleSpatialLagStep` on `count_lag_1`, nulls filled 0 per the Chicago recipe policy) | ~1.0 (by construction) | >= 0.5 | Half the expected gap; the gap is a mean over 12 adjacent vs 20 far units x 56 bins, so its SE is ~0.1. |
| Corr of `count` with trailing 7-day mean (`MovingAverageStep`, closed left) | persistent latent level | > 0.25 | The trailing mean tracks the latent level; 0.25 is ~10 pooled SE above 0. |
| `CategoryCountsStep` top_n=2 vocabulary | {THEFT, BATTERY} | exact columns; shares 0.40 +/- 0.05 (THEFT), 0.35 +/- 0.05 (`other`) | Share SE ~0.007 over ~5,400 events; +/- 0.05 is ~7 SE. |
| `CalendarEncodingStep` cyclical day-of-week sin/cos | 2*pi*weekday/7 | 1e-9 | Deterministic formula, not an estimate. |
| `CompletePanelStep` on a 10% sparse panel | 2,016 rows | exact: dropped pairs return as 0, observed counts unchanged | Exact by construction of the step. |
| Null control (`signal=False`): lag-1 corr, weekend ratio, MA-7 corr | 0, 1, 0 | \|corr\| < 0.10, \|ratio - 1\| < 0.15 | ~4x the pooled sampling error: a real signal cannot hide inside, pure noise essentially never escapes. |

Measured recovery (seed 42 / seed 7; noise control seed 42):

| Quantity | Seed 42 | Seed 7 | Noise |
|---|---|---|---|
| Lag-1 corr | 0.529 | 0.423 | -0.006 |
| Weekend ratio | 1.535 | 1.637 | 0.991 |
| Top-4 units | {15, 16, 21, 22} | {15, 16, 21, 22} | — |
| Spatial-lag gap | 0.975 | 0.816 | — |
| MA-7 corr | 0.443 | 0.380 | -0.014 |
| THEFT / other share | 0.403 / 0.344 | 0.409 / 0.340 | — |

16 tests, all passing, in ~2 s (`tests/evaluation/test_synthetic_ground_truth.py`).

## 3. The planted-leak demonstration (`scripts/phase4_synthetic_demo.py`)

One prediction row per (unit, bin `t`): target = next-bin count >= 6
(prevalence ~11%); leaky score = `count_next` (the actual count of bin
`t + 1` — future information, and the target's own underlying value);
causal score = count at `t`.

- RANDOM 70/30 row split, leaky score: **PR-AUC = 1.0000**.
- TEMPORAL blocked split (train < 2024-01-29, test >= 2024-01-29),
  causal score: **PR-AUC = 0.4031** (chance = prevalence 0.111).
- Gap: 0.5969. The script asserts leaky > 0.95, blocked < 0.60,
  gap > 0.35.

What this proves — and what it does not. It proves a future-peeking
column can look perfect under a random split and collapse once the
evaluation respects time order, because at prediction time `t` the
next bin has not happened yet. It does NOT prove that blocking alone
removes the leak: the script prints the check — the leaky column
evaluated naively on the blocked test rows still scores 1.0000.
A temporal block exposes the column as future information; removing
or rejecting it is the harness's job (`validate_split`, coordinator
integration). The blocked arm here stays at 0.40 rather than falling
to the 0.11 baseline because the planted AR(1) signal is real and the
causal lag-1 count genuinely carries it — the collapse is from 1.00,
not to exactly chance, and the script says so.

## 4. Downstream evaluation examples (`scripts/phase4_chicago_evaluation.py`)

On the Phase 3 Chicago replay (1,908,088 rows, 365 bins, 5,242 cells,
sum `count` = 60,235), scores are the lag features themselves — no
model is trained. PAI and PR-AUC are computed by small local
implementations in the script (step average precision; PAI =
capture share / area fraction).

| Metric | Value |
|---|---|
| PAI at 1% of (cell, bin) pairs by `count_lag_1` | 2.53 (capture share 0.025) |
| PAI at 10% of (cell, bin) pairs by `count_lag_1` | 1.10 (capture share 0.110) |
| PAI at 1% of cells by mean `count_lag_1` | 7.96 (capture share 0.080) |
| PAI at 10% of cells by mean `count_lag_1` | 3.62 (capture share 0.362) |
| PR-AUC, score `count_lag_1`, target `count > 0` (prevalence 0.030) | 0.0325 |
| PR-AUC, score `count_ma_7`, target `count > 0` | 0.0684 |

Reading notes. The cell-bin PAI at 10% is nearly chance (1.10)
because 97.1% of (cell, bin) pairs have `count_lag_1 = 0`: the top
10% is dominated by ties at score 0/1, so that number mostly measures
tie-breaking, not ranking. The cell-level PAI is the informative view:
the top 1% of cells by mean lag-1 count hold 8.0% of all crimes
(PAI 7.96) — crime concentrates in a few cells, and the lag feature
finds them. PR-AUC is barely above prevalence for `count_lag_1`
(0.0325 vs 0.030) and higher for the trailing mean (0.0684):
at 6 h granularity, whether a given cell-bin has any crime is nearly
unpredictable from the previous bin alone, while the persistent
hotspot level (trailing mean) carries more signal. Runtime ~2 s
reading the audit parquet; if no replay parquet is found the script
runs the Chicago recipe once as a fallback.

## 5. Disclaimer (criterion 8)

These are DOWNSTREAM EXAMPLES ONLY, explicitly NOT universal
preprocessing acceptance criteria — the scripts print that header and
this note repeats it. No threshold on PAI or PR-AUC accepts or rejects
the preprocessing. The numbers depend on the scoring choices above
(lag features as scores, no model, 6 h bins, Q1 2024 Chicago); a
different downstream task, model, or threshold would give different
numbers. Phase 4 acceptance rests on the blocked splits, the
walk-forward harness, the fitted-statistics freeze and the synthetic
ground truth — not on these examples.
