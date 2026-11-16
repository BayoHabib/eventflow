# STPP exact-event emitters — design note

**Status:** implemented 2026-11-16 (increment 1 of the STPP direction:
exact-event emitters only). **No fitted model, no likelihood, no
intensity (λ)** — this increment emits data for STPP models; it does
not fit any.

**Sources.** The formats below were read verbatim during the GitHub
inventory of 2026-11-16
(`~/workspace/research_notes/eventflow-stpp-github-inventory/STPP-GitHub-Inventory.md`):

- Gatech / THP layout — `SimiaoZuo/Transformer-Hawkes-Process`,
  `preprocess/Dataset.py` (three pickles `train/dev/test.pkl`, loaded
  with `pickle.load(f, encoding='latin-1')`, dict `{dim_process,
  <split>: [sequences]}`, per-event `{time_since_start,
  time_since_last_event, type_event}` (+1 for padding in the loader);
  `idx_event` and `time_since_last_same_event` are the EasyTPP
  additions) and the EasyTPP documentation
  (`ant-research/EasyTemporalPointProcess`, `user_guide/dataset.html`):
  one `.pkl` or `.json` per split, `data_format: json` or `pickle`,
  per-event keys `idx_event, time_since_start,
  time_since_last_event, time_since_last_same_event, type_event`.
- Catalog CSV — `ss15859/EarthquakeNPP` README: one CSV per catalog,
  columns `id, time, longitude, latitude, magnitude, x, y`; `time` UTC
  `YYYY-MM-DD HH:MM:SS.sss`; `x, y` azimuthal-equidistant projected km,
  explicitly the coordinates NPPs should use as spatial input.
- tick — `X-DataInitiative/tick`,
  `tick/hawkes/model/model_hawkes_expkern_loglik.py` (read verbatim):
  `fit(events, end_times=None)` where `events` is a list of
  realizations, each a list of `n_nodes` 1-D `numpy.ndarray`s of event
  timestamps.
- Splits — EarthquakeNPP again: chronological splits with an explicit
  **auxiliary period** (history before training, used to seed
  history-dependent models).

Everything else in the inventory (DeepSTPP, AutoSTPP, SMASH loaders,
ETAS catalog columns) was **not** read — see "Honest limits" below.
Compatibility is claimed only for the formats above.

## Why exact events, never the panel

No implementation in the survey consumes binned counts. Every serious
repo consumes exact events: binning destroys within-bin ordering and
within-cell location, which is exactly what identifies
self-excitation. The panel stays the library's feature-engineering
output; STPP preparation starts from the raw events and emits exact
`(time, x, y, mark)` sequences alongside it.

## Coordinates

Projected kilometres, chosen per dataset and recorded in the manifest:

| Dataset | Target CRS | Reason |
|---|---|---|
| Chicago crime | EPSG:26971 (NAD83 / Illinois East, State Plane) | The library's existing Chicago projected CRS (recipe + loader config) |
| GeoLife (Beijing) | EPSG:32650 (WGS 84 / UTM 50N) | Beijing sits in UTM zone 50N; already the library's `GEOLIFE_TARGET_CRS` |

The emitter projects lon/lat (EPSG:4326) itself, metres / 1000, via
`pyproj`. Loaders are called **without** a CRS transform so lon/lat
stay geographic. `target_crs` is an explicit parameter with no
default: the projection must be chosen and recorded, never silently
assumed. Neither review prescribes a CRS; this is our choice, and the
catalog keeps the original lon/lat columns next to `x`/`y` so any
consumer can re-project.

## Time

Seconds (float) for every `time_since_*` field and every tick
timestamp. Catalog `time` is `YYYY-MM-DD HH:MM:SS.sss` in the
dataset's local time; the time zone is recorded in the manifest, never
converted (Chicago `America/Chicago`, GeoLife `Asia/Shanghai` — the
latter is the loader's assumption, flagged in the GeoLife schema, not
a declared property of the `.plt` files).

## Marks

Ad hoc everywhere in the surveyed repos, so the codebook is ad hoc
here too and fully explicit: sorted unique mark labels
(deterministic), `mark` / `type_event` = 0-based index. Chicago:
`primary_type`, 30 labels (`ARSON` … `WEAPONS VIOLATION`). GeoLife:
no categorical column in the `.plt` files, so the single mark 0,
label `"fix"`. The catalog carries both the integer `mark` and the
raw `mark_label`; each model maps them as it likes.

## Sequences

One sequence per split (the full chronological run). No daily
windowing — any segmentation is arbitrary; models window internally
(attested in the deep-learning STPP review). Gatech files therefore
hold exactly one sequence each.

## Splits

Chronological, built by chaining `temporal_split` (the library's own
guardrailed splitter) with a gap = embargo at the train/dev and
dev/test cutoffs. Embargo events are counted by direct filters on the
embargo intervals and reconciled: `aux + train + dev + test +
embargo_1 + embargo_2 == cleaned total`.

**Chicago Q1 2024** (60,672 raw → 60,235 cleaned; 437 rows removed by
the loader's cleaning — null dates / non-finite or out-of-range
coordinates):

| Split | Window | Events |
|---|---|---|
| aux | [2024-01-01, 2024-01-15) | 8,922 |
| train | [2024-01-15, 2024-03-01) | 30,464 |
| embargo | [2024-03-01, 2024-03-03) | 1,435 |
| dev | [2024-03-03, 2024-03-15) | 8,065 |
| embargo | [2024-03-15, 2024-03-17) | 1,434 |
| test | [2024-03-17, 2024-04-01) | 9,915 |

Auxiliary = 14 days (covers the 7–14 d near-repeat window and the
library's 7 d Hawkes kernel); embargo = 2 days (the Phase 4 /
DS-project convention). These lengths are **our choice** — neither
review prescribes them.

**GeoLife Phase 3 subset** (1,471 fixes, two trajectories 643 days
apart: 2007-08-04 with 1,116 fixes over ~44 min; 2009-05-08 with 355
fixes over ~49 min). Boundaries are computed from the data: aux =
first 14 days `[t_min, t_min+14d)` (the whole first trajectory); the
remainder (second trajectory only) is split into train/dev/test by
equal thirds of its time span (train 180, dev 33, test 142 — uneven
because the trajectory has a ~16-minute gap, 10:23–10:39, falling in
the middle third; equal thirds of time, not of count). **Embargo is
0**: a 2-day embargo would swallow the entire ~49-minute remainder.
Dev/test are thin and exist for format parity with Chicago, not as a
serious held-out evaluation. This is a documented limitation of the
subset, not of the emitters.

## Formats emitted (per split)

- **Gatech** `{split}.pkl` (+ `{split}.json`, same content):
  `{dim_process, <split>: [sequence]}`; per-event keys exactly
  `idx_event, time_since_start, time_since_last_event,
  time_since_last_same_event, type_event`; first event of a type has
  `time_since_last_same_event == 0.0`. `dim_process` = codebook size
  (30 Chicago, 1 GeoLife). Note: THP's own loader stores types +1 (0
  is its padding value); EasyTPP's Gatech format uses 0-based
  `type_event` as emitted here.
- **Catalog** `{split}_catalog.csv`: columns exactly `id, time,
  longitude, latitude, mark, mark_label, x, y`. **Deliberate
  deviation from EarthquakeNPP: no `magnitude` column** — crime has
  no magnitude, and inventing one would be fabrication. Plus
  `{split}_txy.npy`: float64 `[n, 3]` = `(t_seconds_since_start,
  x_km, y_km)`, matching the CSV row for row. This is the minimal
  `(t, x, y)` shape DeepSTPP-style models expect (cf. the Boston
  crime repo in the inventory).
- **tick** `{split}_tick.pkl`: `{"events": [[array per node]],
  "end_times": [T], "node_labels": [...], "time_unit": "s"}`; one
  realization per split; nodes = marks in codebook order; `T` = the
  split window end in seconds since the sequence start (the
  observation end, not the last event).
- **`manifest.json`** per emission: dataset, `n_events`, cleaning
  counts, CRS, `time_zone`, codebook, split boundaries + embargo
  intervals + per-split counts, file list, library version, and the
  explicit statement "no fitted model". Sorted keys (deterministic).

## What this increment does NOT do

- No fitted model: no λ, no likelihood, no kernel, no learned
  parameter anywhere in `src/shikong/stpp/`.
- No DeepSTPP / AutoSTPP / NSTPP / DSTPP / SMASH adapter: their
  loaders were not read (see limits), so no claim is made that these
  files load into them — only that the shapes match the formats that
  *were* read (Gatech, EarthquakeNPP CSV, tick).
- No ETAS catalog: the ETAS repos' exact catalog columns were not
  verified (inventory §7); the catalog CSV here is the EarthquakeNPP
  shape minus `magnitude`, not an ETAS input claim.
- No windowing, no standardisation, no train-set statistics: the
  emitters are pure functions of the events (same input → byte-identical
  output, verified by hash comparison of two emissions).

## Honest limits

- **Inventory §7, still open.** DeepSTPP last-commit date and
  `src/data/` contents unverified (repeated GitHub 429s); SMASH loader
  paths unknown (root listing never succeeded); AutoSTPP
  `src/data/data.py` contents not read; `lmizrahi/etas`
  `example_catalog.csv` columns not verified (429). None of these is
  guessed here; the emitters claim compatibility only with the three
  formats read verbatim (Gatech/THP, EarthquakeNPP CSV, tick).
- **GeoLife splits are degenerate by construction** (see Splits):
  the subset is two short trajectories 643 days apart. A serious
  GeoLife evaluation needs the full 182-user release, not this subset.
- **Marks are the raw labels.** Chicago `primary_type` has 30 values
  of very uneven frequency (e.g. `HUMAN TRAFFICKING` is rare); no
  grouping, no rare-mark handling is applied — that is a modelling
  decision, left to the model.
- **Simultaneous events keep input order** (stable sort). Chicago
  timestamps have 1-minute resolution, so ties are common; no
  tie-breaking is invented.
- **The `aux` split is emitted but not a model input by itself**:
  it is history for history-dependent models (EarthquakeNPP's
  convention), not a fourth evaluation split.

## API

`src/shikong/stpp/emitters.py` (public surface re-exported from
`shikong.stpp`):

- `extract_exact_events(frame, *, mark_col, target_crs,
  source_crs="EPSG:4326") -> pl.DataFrame` — sorted exact events
  (`timestamp, longitude, latitude, mark, mark_label, x, y`); refuses
  null / non-finite timestamp, coordinates or mark with `ValueError`.
- `chronological_event_splits(events, *, time_col="timestamp",
  boundaries: list[SplitBoundary]) -> ChronologicalSplits` —
  `SplitBoundary(name, end, gap=timedelta(0))`; chains
  `temporal_split`; reports embargo counts/intervals; refuses empty
  splits and trailing events.
- `build_gatech_sequence(events)`, `emit_gatech(splits, *,
  output_dir, dim_process)`.
- `events_to_txy(events)`, `emit_catalog(splits, *, output_dir)`.
- `build_tick_payload(events, *, split_end, codebook)`,
  `emit_tick(splits, *, output_dir, split_ends, codebook)`.
- `build_manifest(*, ...)` / `write_manifest(output_dir, manifest)`.

Scripts (data files live outside the repo, under
`~/workspace/eventflow-eda/stpp-emitters/{chicago,geolife}/`):

- `scripts/emit_chicago_stpp.py [--output-dir DIR] [--raw-csv PATH]`
- `scripts/emit_geolife_stpp.py [--output-dir DIR] [--subset-dir DIR]`

Both print per-split counts, embargo counts and the reconciliation,
and exit non-zero if it fails.

## Verification

- `tests/stpp/test_emitters.py` — 40 tests, synthetic in-memory
  frames only, hand-computed Gatech fields, projection checked
  against an independent pyproj transformer, round-trips by
  independent re-readers (pickle `latin-1`, strict CSV parse,
  `np.load`), determinism by hash comparison. Module coverage
  100 % (measured with the full suite).
- `audit/stpp-emitters/verify_emissions.py` — independent
  verification of the real emitted files: 395 checks, 0 failures
  (Gatech/tick/catalog round-trips, x/y vs independent pyproj,
  Chicago mark counts per split vs the raw CSV, first/last
  timestamps vs the raw CSV, reconciliation, byte-identical
  re-emission). Log: `audit/stpp-emitters/verification.log`,
  report: `audit/stpp-emitters/verification.json`.
