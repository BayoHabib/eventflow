# STPP data preparation for deep-learning STPP — design note

**Status:** implemented 2026-11-16 (increment 2 of the STPP direction:
data preparation for deep-learning STPP). **No fitted model, no
likelihood, no intensity (λ), no training** — this increment emits
data for neural STPP models; it does not fit or train any. The
classical fitted layer is paused by the owner's decision
(2026-11-16); this increment is preparation only.

**Sources.** Every loader specification below was read verbatim from
the reference repositories on 2026-11-16, in three spaced,
rate-limit-clean sessions (no HTTP 429 in any of them). Notes:
`~/workspace/eventflow-eda/stpp-dl-data-prep/notes/` (one file per
model group, with quoted code).

| Model | Repo (HEAD read) | Files read in full | Status |
|---|---|---|---|
| DeepSTPP | `Rose-STL-Lab/DeepSTPP` @ `0e4c2542` | `src/data/dataset.py`, `adapter.py`, `synthetic.py` (except tail), README, root + `src/` listings | dormant (last commit 2023-09-21, README-only; last code 2022-04-04) |
| AutoSTPP | `Rose-STL-Lab/AutoSTPP` @ `ee48bd14` | `src/data/data.py`, `adapter.py`, `download_data.py`, `src/data/lightning/sliding_window.py`, `configs/autoint_stpp.yaml` | dormant (last commit 2024-10-14) |
| NSTPP | `facebookresearch/neural_stpp` @ `5029fa06` | `data/download_and_preprocess_earthquakes.py`, README, `train_stpp.py`, `viz_dataset.py`, `data/` listing | dead (last commit 2021-12-28) |
| DSTPP | `tsinghua-fib-lab/Spatio-temporal-Diffusion-Point-Processes` @ `6609e0e2` | `app.py`, `DSTPP/Dataset.py`, `DSTPP/Constants.py`, README; listings root, `DSTPP/`, `dataset/`, `dataset/Earthquake/` | dormant (last commit 2024-03-25) |
| SMASH | `zichongli5/SMASH` @ `f2f238c5` | `app.py`, `model/Dataset.py`, `scripts/train_earthquake.sh`; listings root, `model/`, `datasets/`, `datasets/Earthquake/` | dormant (last commit 2025-05-07) |
| Boston crime (DeepSTPP vendored) | `jaime-varela/boston-crime-stpp` @ `9985922a` | `DeepStpp/data/dataset.py`, README | dormant; vendored file byte-identical to canonical |

## Per-model input specification (what the loaders actually read)

### DeepSTPP + AutoSTPP — one shared sliding-window wrapper contract

The two repos ship (near-)identical `SlidingWindowWrapper` classes
(DeepSTPP `src/data/dataset.py`; AutoSTPP
`src/data/lightning/sliding_window.py`): input is a **list of
`[n, 3]` arrays, columns `[t, x, y]`**, time absolute from the
sequence start. The wrapper itself converts column 0 to gaps
(`np.diff(..., prepend=0)` — first gap = absolute `t0`), min–max
scales to [0, 1], and windows internally (`lookback=20,
lookahead=1`, hard-coded defaults; `roll=True` by default, so the
model sees `[x, y, Δt]`). Output float32. **No marks. No splits in
the wrapper class** (the caller supplies already-split lists).

Differences that matter: AutoSTPP's file layout is one `.npz` per
dataset at `./data/spatiotemporal/<name>.npz` with keys `train` /
`val` / `test` (loaded with `allow_pickle=True`); its Lightning
module passes `normalized=True` with min/max fitted on TRAIN only
and reused for val/test. DeepSTPP has **no canonical file format at
all** — the wrapper is in-memory, `src/data/` holds no real-data
loader and no data files; the repo's only file format is the
synthetic CSV `[x, y, t]` (`synthetic.py::save`, comma, `%f`).
DeepSTPP's two legacy paths are INCONSISTENT with the wrapper and
are NOT replicated: `DataAdapter` takes a single `(N, 3)` array
`[lat, lon, time]` (time LAST), lookback 10, and fits its
`MinMaxScaler` on the whole sequence BEFORE the 8:1:1 split
(leakage by construction); `synthetic.py::dataset` uses first gap 0
(vs `t0` in the wrapper) and the same pre-split scaler fit.

### NSTPP (earthquakes preprocessor)

`data/download_and_preprocess_earthquakes.py` (read in full): one
`.npz`, `earthquakes/earthquakes_jp.npz`, one key per 30-day window
named `YYYYMMDD` (window start), value float64 `[n, 3]` =
`[t_days_since_window_start, longitude, latitude]`. Time is DAYS
since 1990-01-01, re-based to days since the window start, range
[0, 30). Coordinates are RAW lon/lat in degrees (no projection; the
training code standardises via `dataset.S_mean`/`S_std`, which live
in the unread `datasets.py`). Windows are stepped WEEKLY (30-day
window, 7-day step, overlapping); all of 2011 and Nov–Dec 2010 are
skipped (no reason stated in the file). **No splits in the
preprocessor** — the `.npz` is a single dict of window keys; splits
happen at load in `datasets.py` (NOT read — split rule, `S_mean` /
`S_std` computation, and the collate function are UNVERIFIED).
No marks: magnitude is a download filter (≥ 2.5) and is dropped.
The README does NOT document the on-disk format at all.

### DSTPP

`app.py` `data_loader` + `DSTPP/Dataset.py` (read in full):
`dataset/<name>/data_train.pkl`, `data_val.pkl`, `data_test.pkl`
(pre-split). Each pickle is a list of sequences; each sequence a
list of events; each event a POSITIONAL list/tuple, NOT a dict (the
`Dataset.py` docstring claiming dicts with `time_since_start,
time_since_last_event, type_event` is stale — the code indexes
`elem[0..3]`). Raw event, dim=2: **`[t, lng, lat]`**; dim=1:
`[t, lng]`; dim=3: `[t, lng, lat, height]`. The loader inserts the
gap at index 1 → `[t_abs, dt, lng, lat]`, `dt = t − t_prev`,
first-event `dt = t`. Normalisation is min–max over
train+val+test COMBINED; absolute time is NOT normalised
(`Max=1, Min=0` → identity); gap and coords → [0, 1].
**`assert Min[1] > 0`**: the minimum gap over all splits must be
strictly positive (no two events share a timestamp). No windowing
in the loader; padding `PAD = 0` to batch max, float32. **Marks:
NONE.** **`event_type_num.json` does NOT exist in this repo and has
no role** — absent from all listings read, never referenced in
`app.py` or `Dataset.py`. (The 2026-11-16 inventory's table entry
claiming it is part of the DSTPP format was WRONG; corrected here.)
**Loader bug:** the validation loader is built from the TEST data
(`valloader = get_dataloader(test_data, …)`); `data_val.pkl` is
loaded but never used.

### SMASH

`app.py` `data_loader` + `model/Dataset.py` (read in full):
`dataset/<name>/data_train.pkl`, `data_val.pkl`, `data_test.pkl`.
**Path mismatch: the loader opens `dataset/<name>/` (singular),
but the repo ships `datasets/<name>/` (plural) and contains NO
`dataset/` directory** — a fresh clone cannot run `app.py` as-is.
Inner schema: same list-of-sequences of positional events. dim=2
(unmarked): raw `[t, lng, lat]` → `[t, gap, lng+1, lat]`; dim=3
(marked): raw `[t, mark, lng, lat]` → `[t, gap, mark+1, lng, lat]`
(the `+1` makes the mark 1-based). Time: absolute `t` + gap;
**default `--log_normalization 1`: gap stored as
`log(max(gap, 1e-4))`**, first event `log(max(t, 1e-4))`. Same
cross-split min–max normalisation; absolute time identity; mark
column forced identity (`Max[2]=1, Min[2]=0`). No windowing;
padding literal `0.`, float32. No dim=1 support. Same
valloader-from-test bug as DSTPP. **Marks exist ONLY at dim=3:**
mark = raw `i[1]`, integer event type, 0-based in the pickle,
categorical (`num_types = int(max mark+1)` over all splits).
**`scripts/train_earthquake.sh` runs Earthquake with `--dim 3`** —
SMASH's Earthquake pickle IS marked, unlike DSTPP's. **Mark
semantics per dataset (magnitude bin? crime type? football event
type?) are UNVERIFIED** — the code never says what `i[1]` means.
DSTPP and SMASH do NOT share an Earthquake schema despite the same
filenames (SMASH train pickle 3,527,855 B vs DSTPP 2,899,838 B).

## What Shikong emits (per model)

| Model | Files | Content |
|---|---|---|
| DeepSTPP + AutoSTPP | `<name>.npz` | Keys `train`/`val`/`test`; each an object array holding ONE float64 `[n, 3]` array `[t_seconds_since_start, x_km, y_km]` (the split's full run). Mapping train→train, dev→val, test→test; aux excluded. Loads with `np.load(..., allow_pickle=True)`. |
| DeepSTPP (file shape) | `{split}_deepstpp.csv` | Columns `[x, y, t]` (`x_km, y_km, t_seconds`), comma, `%f`. One per split (aux included). Secondary export matching `synthetic.py::save`. |
| NSTPP | `<name>_nstpp.npz` | Keys `YYYYMMDD` (window start, dataset local time); values float64 `[n, 3]` = `[t_days_since_window_start, lon_deg, lat_deg]`. 30-day windows / 7-day step over ALL cleaned events (no splits in the file); empty windows included. |
| DSTPP | `dataset/<name>_dstpp/data_{train,val,test}.pkl` | List with ONE sequence per split of positional `[t, lng, lat]` lists (`t` seconds since split start; lng/lat degrees). dev→val; aux excluded. |
| SMASH | `dataset/<name>_smash/data_{train,val,test}.pkl` | List with ONE sequence per split of positional `[t, mark, lng, lat]` lists (`t` seconds since split start; mark 0-based codebook int; lng/lat degrees). dev→val; aux excluded. |

DSTPP and SMASH pickles are written under separate subdirectories
(`<name>_dstpp` / `<name>_smash`) because both loaders expect
`dataset/<name>/data_*.pkl` with the same filenames but different
schemas — same-named files would collide. The loader path each model
expects is `dataset/<name>/`; rename the subdirectory to the dataset
name the loader is run with. (SMASH additionally needs the shipped
`datasets/` renamed to `dataset/` in its own repo — a repo-side
mismatch, not Shikong's.)

## Coordinates, time, marks, sequences, splits

- **Coordinates:** projected kilometres (`x`/`y`) for the wrapper
  and DeepSTPP CSV (the loaders treat them opaquely and min–max
  scale); raw longitude/latitude DEGREES for NSTPP, DSTPP and SMASH
  (their field names are `lng`/`lat`). Neither review prescribes a
  CRS for the DL loaders; the choice is documented per format and
  pinned in each manifest.
- **Time:** seconds (float) since the split start for the wrapper,
  DSTPP, SMASH and DeepSTPP CSV — the same origin as the PR #55
  emitters. NSTPP windows use DAYS since the window start (the
  preprocessor's unit). Time units are UNVERIFIED in the
  DeepSTPP/AutoSTPP/DSTPP/SMASH loaders (never stated in the code
  read); our seconds are pinned explicitly in the manifest and the
  design note rather than inherited silently.
- **Marks:** none in the wrapper / NSTPP / DSTPP formats (3 columns
  only). SMASH dim=3 carries our codebook index (Chicago
  `primary_type`, 0-based int; GeoLife single mark 0, `"fix"`).
  `num_types` = 30 for Chicago, 1 for GeoLife. Our mark's meaning is
  documented; SMASH's own datasets' mark semantics stay unverified.
- **Sequences:** one sequence per split for the split-based formats
  (the full chronological run) — same as the PR #55 emitters. NSTPP
  windows are many overlapping sequences over the whole cleaned
  period. No daily windowing beyond NSTPP's 30-day windows — any
  further segmentation is the model's job (the wrappers window
  internally, lookback 20).
- **Splits:** chronological, same boundaries as the PR #55 emission
  (Chicago: aux 14 d, train, dev, test with 2-day embargos; GeoLife
  subset: aux 14 d, remainder in equal thirds, embargo 0). Mapping
  dev→val. The `aux` split is excluded from the fixed-schema
  containers (the loaders have no aux key); it stays available in the
  Gatech/catalog/tick emissions.

## The DSTPP tie blocker (honest limit, measured)

The DSTPP reference loader asserts the minimum gap over all splits
is strictly positive (`assert Min[1] > 0` in `app.py`). Two
independent causes make it fail on our data:

1. **First gap = 0.0 always.** The loader sets the first event's gap
   to its absolute `t`; our `t` is seconds since the split start, so
   the first event has `t = 0` and gap `0`. (The shipped datasets'
   `t` origin is unverified — necessarily first `t` > 0 there.)
2. **Same-minute ties: 23,683 tied consecutive pairs in Chicago**
   (measured by the verification script across the three splits).
   Chicago timestamps have 1-minute resolution; ties are common.

GeoLife has 0 tied pairs (1-second resolution) but still fails on
cause 1 (first gap 0.0). **The reference DSTPP loader as written
cannot load these emissions.** No tie-breaking is invented (that
would fabricate sub-minute precision) and no `t` origin is fudged
to dodge the assert. The emitter still produces the format
faithfully; the verification script implements the loader transform
faithfully and REPORTS the assert outcome with the exact tied-pair
count. DSTPP compatibility is format-level only, for Chicago and
GeoLife alike. SMASH has no such problem: its log floor (`1e-4`)
keeps zero gaps finite.

## What Shikong does NOT do (refusals, by design)

- **No pre-split normalisation.** DeepSTPP's `DataAdapter` and
  `synthetic.py` fit their `MinMaxScaler` on the whole sequence
  BEFORE the split (leakage by construction). Shikong ships
  un-normalised absolute values; normalisation is the loader's job,
  fitted on train only (as AutoSTPP's Lightning module does).
- **No pre-windowing.** The wrappers window internally (lookback
  20); DSTPP/SMASH use sequences as stored. Shikong does not window
  beyond NSTPP's 30-day windows (which the preprocessor itself
  defines).
- **No tie-breaking.** Simultaneous events keep input order (stable
  sort), as in the PR #55 emitters.
- **No replication of loader bugs.** The valloader-from-test bug
  (DSTPP + SMASH) and the SMASH `dataset/` vs `datasets/` path
  mismatch are documented, not worked around in the emitted data
  (the val file is still emitted, correctly).
- **No replication of dataset-specific quirks.** NSTPP's 2011 /
  Nov–Dec-2010 window exclusions are Japan-catalog quirks, not
  format; not replicated.
- **No training, no fitting, no likelihood, no model code** anywhere
  in `src/shikong/stpp/`.

## API

`src/shikong/stpp/dl_formats.py` (public surface re-exported from
`shikong.stpp`):

- `emit_wrapper_npz(splits, *, output_dir, name)` — one `<name>.npz`,
  keys `train`/`val`/`test` (dev→val, aux excluded).
- `window_events_nstpp(events, *, window_days=30, step_days=7)` —
  window key → float64 `[n, 3]` array; refuses empty input,
  non-positive parameters, and `step_days > window_days`.
- `emit_nstpp_npz(events, *, output_dir, name, window_days=30,
  step_days=7)` — one `<name>_nstpp.npz`.
- `emit_dstpp_pickles(splits, *, output_dir, name)` — positional
  `[t, lng, lat]` pickles under `dataset/<name>/`.
- `emit_smash_pickles(splits, *, output_dir, name)` — positional
  `[t, mark, lng, lat]` pickles under `dataset/<name>/`.
- `emit_deepstpp_csv(splits, *, output_dir)` — `{split}_deepstpp.csv`,
  columns `[x, y, t]`, `%f`.

Scripts (data files live outside the repo, under
`~/workspace/eventflow-eda/stpp-dl-data-prep/{chicago,geolife}/`):

- `scripts/emit_chicago_dl_stpp.py [--output-dir DIR] [--raw-csv PATH]`
- `scripts/emit_geolife_dl_stpp.py [--output-dir DIR] [--subset-dir DIR]`

Both print per-split counts, embargo counts, the reconciliation, the
NSTPP window summary, and exit non-zero if it fails.

## Verification

- `tests/stpp/test_dl_formats.py` — 18 tests, synthetic in-memory
  frames only, hand-computed values (wrapper keys/shapes/dtypes,
  wrapper windowing round-trip via a faithful re-implementation of
  `SlidingWindowWrapper`, NSTPP window keys/values incl. empty
  windows, DSTPP/SMASH positional schemas + loader transforms incl.
  the DSTPP assert logic and the SMASH log-gap floor, DeepSTPP CSV
  shape/`%f` format, determinism by hash comparison). Module
  coverage 100 % (measured with the new tests).
- `audit/stpp-dl-data-prep/verify_dl_emissions.py` — independent
  verification of the real emitted files, no imports from
  `shikong.stpp.dl_formats`: **61 checks, 0 failures.** Wrapper
  values vs the PR #55 catalog CSVs + independent pyproj; NSTPP
  windows vs the cleaned event set re-derived from the raw CSV
  (Chicago: 60,235 events, 226,676 window occurrences, every event
  in ≥ 1 window) and vs the PR #55 catalogs (GeoLife: 92 windows,
  87 empty); DSTPP/SMASH values vs catalog CSVs; DSTPP tie count
  (Chicago 23,683; GeoLife 0) and assert outcome; SMASH log-gaps
  finite, `num_types` 30 / 1; DeepSTPP CSVs vs catalog + pyproj;
  byte-identical re-emission (13 files per dataset).
  Log: `audit/stpp-dl-data-prep/verification.log`, report:
  `audit/stpp-dl-data-prep/verification.json`.

## Honest limits

- **Inventory §7, now closed for the five DL models.** DeepSTPP
  last-commit date (2023-09-21, dormant) and `src/data/` contents;
  AutoSTPP `src/data/data.py`; NSTPP earthquakes preprocessor;
  DSTPP pickle schema (from loader code, not from the binary);
  SMASH loader paths — all verified by direct reads in this
  increment, zero HTTP 429s across the three sessions. Remaining
  unverified: NSTPP `datasets.py` (split rule, `S_mean`/`S_std`,
  collate); AutoSTPP ready-`.npz` build step (time unit, CRS, split
  dates) and `synthetic.py`; DeepSTPP `notebooks/` and the
  `synthetic.py` tail; Boston notebook (lookback/normalisation/splits
  actually used); SMASH `crime`/`football` contents and mark
  semantics; DSTPP/SMASH time unit, coordinate CRS, sequence
  provenance, whether splits are chronological, and the binary pickle
  contents themselves (never downloaded). None of these is guessed;
  the emitters claim only what the code read establishes.
- **DSTPP tie blocker** (see above): format-level compatibility
  only.
- **GeoLife splits are degenerate by construction** (same as PR #55):
  the subset is two short trajectories 643 days apart; NSTPP windows
  over it are 92 windows, 87 of them empty. A serious GeoLife
  evaluation needs the full 182-user release.
- **Time units are pinned by Shikong, not by the loaders** (seconds
  for wrapper/DSTPP/SMASH/CSV, days for NSTPP). A consumer that
  assumes a different unit must convert; the manifest states ours.
- **Marks are the raw labels** (same as PR #55): Chicago
  `primary_type`, 30 values of very uneven frequency; no grouping, no
  rare-mark handling — a modelling decision, left to the model.
- **Simultaneous events keep input order** (stable sort); no
  tie-breaking is invented.
- **`event_type_num.json` correction:** the 2026-11-16 inventory
  listed it as part of the DSTPP format; it does not exist in the
  DSTPP repo and is never read. Any `event_type_num.json` belongs to
  other (NHP/THP-style) loaders, not to DSTPP.
