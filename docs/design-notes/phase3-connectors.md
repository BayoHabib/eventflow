# Phase 3 — External connectors (criterion 5)

## What was built

`TabularFileSource` (`src/shikong/core/context/tabular.py`), a generic
external-table connector in `core/` — no dataset-specific code. It
loads one local CSV or Parquet file under a declared schema:

- `timestamp_col` — the observation time (Datetime/Date, or a string
  column parsed strictly, optionally with an explicit `strptime`
  format);
- `key_cols` (optional) — join keys such as a station id, used by
  spatio-temporal joins to keep groups isolated;
- `value_cols` — the columns to join onto events (at least one).

It subclasses the existing `DynamicTemporalSource`, so it plugs into
the existing `EnricherStep` / `TemporalJoin` machinery, and its
`load()` output feeds the existing `TemporalContextJoinStep`,
`SpatialContextJoinStep`, and `SpatioTemporalContextJoinStep`
unchanged. Exported from `shikong.core.context`.

## Validation discipline

The declaration is validated at construction — missing file, unknown
suffix, empty / duplicate / overlapping declarations, missing declared
columns, non-temporal timestamp dtype — so a bad declaration fails
before any pipeline runs. The data are validated at `load()`:
unparseable or null timestamps, and duplicate `(key_cols, timestamp)`
pairs (an asof join over duplicate context keys would match
ambiguously). `load()` returns exactly the declared columns, sorted
by `key_cols + [timestamp_col]`. Unmatched events stay null after a
join; the connector never invents a value.

One deliberate trade-off: `load()` materializes the file once (after
selecting only the declared columns) so the data checks can run. The
context tables in scope are small (the Beijing parquet is 420,768
rows); a future very large source should filter before loading, or
subclass `BaseContextSource` with its own lazy path.

## The demonstration, and the honest finding

`scripts/phase3_connectors_demo.py` → `audit/phase-3/connectors_demo.json`.

The two real ranges **do not overlap**: GeoLife subset fixes run
2007-08-04 → 2009-05-08; the Beijing meteorology starts 2013-03-01
(gap 1,392.5 days). Part A runs the real join anyway (asof-backward,
1 h tolerance, nearest station Dongsi, ~0.1 km from the subset
centroid): **0 matched / 1,471 unmatched**. The tolerance correctly
refuses to match across a ~4-year gap. Widening it to force a match
would join 2009 fixes to 2013 weather and was not done; no
overlapping data was fabricated.

Parts B/C are the positive control on the same real Beijing table:
synthetic event timestamps inside the Beijing range (2013-03-01,
realistic GPS-fix cadence) — weather values real, event times
synthetic, stated as such. Asof-backward 1 h: 8/9 matched (the 9th
predates the first context row and correctly stays null); "exact":
2/9 matched (the two on-the-hour events). All 10 hand-checks
re-derive the joined TEMP/PRES/DEWP/WSPM from an independent read of
the raw PRSA CSV for the same station and match exactly.

Two scope notes from the data itself:

- There is **no HUMI (humidity) column in UCI 501**; the joined
  meteorological subset is TEMP, PRES, DEWP, WSPM. RAIN and `wd`
  were not joined.
- Beijing weather is already inside the UCI data, so Beijing needs no
  external connector at all; the connector's demonstration case is
  GeoLife enrichment, and a production version needs an overlapping
  weather source for 2007–2009, which is not on disk.

## What this is not (out of scope)

- No live API clients, no network fetching, no credentials, no
  caching, no rate limiting. The connector reads local files only.
- No POI / rates / geocoding / reverse-geocoding providers, no
  place search, no routing.
- No new join semantics: all join behaviour lives in the existing
  steps; the connector only supplies a validated table.

## How a future provider plugs in

Two seams, in increasing order of work:

1. **File-based provider (recommended).** A fetcher script (outside
   the library) calls the provider API, writes a CSV/Parquet file,
   and the pipeline loads it with `TabularFileSource` exactly as the
   demo does. The file is the audit boundary: the joined values are
   reproducible from a file on disk, and no credentials ever enter
   the library. This is the right shape for a weather API covering
   2007–2009, a reanalysis extract, or any bulk download.
2. **In-library source.** Subclass `BaseContextSource` (or
   `DynamicTemporalSource` / `SpatioTemporalSource` for the matching
   shape), implement `load() -> pl.LazyFrame` and the `schema`
   property, and pass it to `EnricherStep` with a `TemporalJoin` /
   `SpatialJoin`. The subclass owns its own validation and must meet
   the same bar: fail loudly on unknown columns, unparseable
   timestamps, and duplicate join keys; never silently fill or drop.
   The existing Chicago hand-rolled loader
   (`datasets/chicago_crime/context/weather.py`, untested and
   unwired) is the pattern this replaces — a future dataset loader
   should be a thin `TabularFileSource` declaration, not a new class.

## Pre-existing quirk found, not fixed

`TemporalContextJoinStep` with `strategy="exact"` does not sort its
output (the asof strategies do), and Polars' left join does not
preserve left-row order there — the returned rows are correct but in
hash-join order. The connector tests sort by time before asserting
for that strategy. The step makes no order guarantee, and changing
shared join behaviour is out of scope for this criterion; flagging
for the maintainer.
