# Shikong examples

Runnable examples for the library. Every example here is executed —
by `tests/integration/test_examples.py` in the suite and by hand in a
fresh venv during release verification — never claimed from reading.

## `chicago_quickstart.py`

Runs the full packaged `chicago_crime_v1` recipe (21 steps: CRS
transform, 300 m grid, 6 h time bins, categorical counts, panel
completion, lags, rolling means, EWMA, trend, spatial lag rings,
KDE / STKDE, near-repeat, Hawkes, recency, burstiness, zero features)
on 120 synthetic Chicago-shaped events over 3 days, then asserts the
panel invariants the full Chicago replay checks on real data.

```bash
python examples/chicago_quickstart.py [--config PATH] [--out-dir DIR]
```

Expected output (measured 2026-11-15, `shikong` 0.1.0):

```text
Input events:            120
Output rows:             516
Output columns:          62
Grid cells:              43
Time bins:               12
sum(count):              120
Null counts:             0
Observed (cell, bin):   70
Rows with count > 0:     70
observation_status:      ['observed']
Output written to:       <out-dir>/chicago_quickstart.parquet
Quick start OK: panel complete, every event counted exactly once.
```

`--config` defaults to `configs/recipes/chicago_crime_v1.yaml` in the
checkout containing the script; the recipe config is user-supplied
data, not part of the installed package (configs are not shipped in
the wheel). `--out-dir` defaults to a temporary directory; the
synthetic CSV and the output Parquet are written there.

Known quirk, documented in the script: the explicit-zero rows
produced by `CompletePanelStep` carry a null `time_bin` — only the
observed (cell, bin) pairs carry a bin timestamp. The shape assertion
therefore counts bins from the non-null `time_bin` values. This is
library behaviour, not an example bug; the full Chicago replay
(`scripts/audit.py`) checks the same invariants and likewise does not
assert non-null `time_bin` on zero rows.

## Clean-install procedure

What a clean install of the released wheel must satisfy before any
upload (full procedure: `docs/release.md`). Verified 2026-11-15
against the wheel built from a clean copy of the tree
(`shikong-0.1.0-py3-none-any.whl`, `twine check` PASSED):

```bash
# 1. Build from a clean checkout
python -m build
twine check dist/*

# 2. Fresh venv, wheel only: no -e, no dev extras
python -m venv /tmp/ef-clean-venv
/tmp/ef-clean-venv/bin/python -m pip install dist/shikong-0.1.0-py3-none-any.whl

# 3. CLI works from the installed package
/tmp/ef-clean-venv/bin/shikong --help        # lists run/validate/version/...
/tmp/ef-clean-venv/bin/shikong version        # shikong version 0.1.0

# 4. The shipped recipe config validates (21 steps)
/tmp/ef-clean-venv/bin/shikong validate --config configs/recipes/chicago_crime_v1.yaml
# ✓ Valid recipe configuration: ... (pipeline builds: 21 steps)

# 5. End-to-end smoke: the example above, plus the real CLI run path
/tmp/ef-clean-venv/bin/python examples/chicago_quickstart.py
/tmp/ef-clean-venv/bin/shikong run --dataset chicago_crime --recipe chicago_crime_v1 \
    --config configs/recipes/chicago_crime_v1.yaml \
    --input <synthetic csv> --output /tmp/cli_run.parquet
# Saved 516 cell/time-bin rows to /tmp/cli_run.parquet (62 cols, sum(count) = 120)
```

The wheel install pulled all runtime dependencies from PyPI
(`numpy`, `polars`, `pydantic`, `shapely`, `pyproj`, `typer`,
`pyyaml`, `python-dateutil`) — no missing runtime dependency was
exposed, so no `pyproject.toml` fix was needed beyond the metadata
audit (license file reference, placeholder author email removed).

### What this proves

- The wheel is self-contained: `pip install` of the wheel alone
  yields a working `shikong` command and importable package.
- The shipped recipe config builds a 21-step pipeline from the
  installed package, and the full recipe runs end to end on
  synthetic input with the replay invariants holding
  (`sum(count)` == input events, no null counts, complete panel).

### What it does not prove

- Nothing about real Chicago data: the smoke input is 120 synthetic
  rows, not the 1,908,088-row Q1 2024 replay (`scripts/audit.py`
  covers that, and it needs the raw data under `data/raw/`).
- Nothing about performance: the smoke run takes ~2 s; the real
  replay takes ~98 s (see `docs/design-notes/phase2-spacetime-v2.md`).
- Nothing about the dev extras, the test suite, or any downstream
  application (no API, no deployment service — those are out of
  scope for the library release).

## Docker

`examples/Dockerfile` builds an **examples-only** image: it installs
the library from the built wheel plus dev extras and runs
`chicago_quickstart.py`. It is not a prediction API and not a
deployment service.

```bash
# From the repository root, after `python -m build`:
docker build -f examples/Dockerfile -t shikong-examples:0.1.0 .
docker run --rm shikong-examples:0.1.0
```

**Status: the Dockerfile was not built in this VM — no Docker
daemon is available here** (`which docker` finds nothing). The
equivalent steps (wheel install in a fresh venv, `shikong --help`,
`validate`, the quick start and the CLI run) were verified in a
fresh venv instead, as documented above. Build the image on a machine
with Docker before relying on it, and re-run the example inside it;
until then, treat the Dockerfile as unexecuted.
