#!/usr/bin/env python
"""Chicago quick start: run the packaged ``chicago_crime_v1`` recipe on a tiny
synthetic dataset and check the output.

This is the smoke example for a clean install (see ``examples/README.md``):
it exercises the real recipe pipeline — grid assignment, 6 h time bins,
categorical counts, panel completion, lags, rolling means, spatial lag,
KDE / STKDE, near-repeat and Hawkes features — on 120 synthetic events
spread over 3 days, then asserts the invariants the full Chicago replay
checks on real data:

* every input event is counted exactly once (``sum(count) == n_input``);
* the completed panel has no null counts and every row is
  ``observation_status == "observed"``;
* the panel shape is cells x bins (rows == distinct cells x distinct
  non-null time bins).

Observed quirk (library behaviour, not an example bug): the explicit-zero
rows produced by ``CompletePanelStep`` carry a null ``time_bin`` — only the
70 observed (cell, bin) pairs carry a bin timestamp. The shape assertion
therefore counts bins from the non-null ``time_bin`` values. The full
Chicago replay (``scripts/audit.py``) checks the same invariants on real
data (``sum(count)``, no null counts, ``observation_status``); it does not
assert non-null ``time_bin`` on zero rows either.

Usage (from a checkout, or anywhere once ``shikong`` is installed)::

    python examples/chicago_quickstart.py [--config PATH] [--out-dir DIR]

``--config`` defaults to ``configs/recipes/chicago_crime_v1.yaml`` in the
repository that contains this file. The synthetic input is written to a
temporary directory unless ``--out-dir`` is given.
"""

from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import yaml

from shikong.core.schema import RecipeConfig
from shikong.core.steps.registration import get_default_registry
from shikong.datasets.chicago_crime.mapping import load_chicago_crime
from shikong.recipes.registry import get_recipe

N_EVENTS = 120
N_DAYS = 3
# Downtown Chicago, jittered within ~1.5 km so the events fall into a
# handful of 300 m grid cells.
BASE_LAT = 41.8781
BASE_LON = -87.6298
PRIMARY_TYPES = ("THEFT", "BATTERY", "BURGLARY")
LOCATIONS = ("STREET", "APARTMENT", "STORE")


def find_repo_root(start: Path) -> Path:
    """Walk up from ``start`` until a directory holding pyproject.toml."""
    for parent in [start, *start.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError(
        "Could not locate the Shikong checkout (no pyproject.toml found above "
        f"{start}). Pass --config explicitly."
    )


def write_synthetic_csv(path: Path, n_events: int = N_EVENTS) -> None:
    """Write ``n_events`` synthetic Chicago-shaped crime rows."""
    base = datetime(2024, 1, 1, 0, 0, 0)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "latitude", "longitude", "primary_type", "location_description"])
        for i in range(n_events):
            # Deterministic spread: one event per ~36 min, cycling types/locations
            # and drifting across a small lat/lon grid so several cells are hit.
            ts = base + timedelta(minutes=36 * i)
            lat = BASE_LAT + 0.004 * ((i // 7) % 5) + 0.001 * (i % 3)
            lon = BASE_LON + 0.004 * ((i // 11) % 5) + 0.001 * (i % 2)
            writer.writerow(
                [
                    ts.strftime("%Y-%m-%d %H:%M:%S"),
                    f"{lat:.6f}",
                    f"{lon:.6f}",
                    PRIMARY_TYPES[i % len(PRIMARY_TYPES)],
                    LOCATIONS[i % len(LOCATIONS)],
                ]
            )


def run_quickstart(config_path: Path, out_dir: Path) -> pl.DataFrame:
    """Run the Chicago recipe on synthetic data; return the collected panel."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "synthetic_chicago.csv"
    write_synthetic_csv(csv_path)

    with config_path.open(encoding="utf-8") as f:
        recipe_config = RecipeConfig(**yaml.safe_load(f))

    recipe = get_recipe(
        recipe_config.dataset,
        recipe_config.recipe,
        recipe_config,
        step_registry=get_default_registry(),
    )
    event_frame = load_chicago_crime(str(csv_path))
    result = recipe.run(event_frame).collect()
    result.write_parquet(out_dir / "chicago_quickstart.parquet")
    return result


def main(argv: list[str] | None = None) -> int:
    """Entry point: run the quick start and assert the output invariants."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Recipe config YAML (default: configs/recipes/chicago_crime_v1.yaml "
        "in the checkout containing this file).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for the synthetic CSV and output Parquet "
        "(default: a temporary directory).",
    )
    args = parser.parse_args(argv)

    config_path = args.config
    if config_path is None:
        config_path = (
            find_repo_root(Path(__file__).resolve().parent)
            / "configs"
            / "recipes"
            / "chicago_crime_v1.yaml"
        )
    if not config_path.exists():
        print(f"Recipe config not found: {config_path}", file=sys.stderr)
        return 1

    tmp = tempfile.TemporaryDirectory() if args.out_dir is None else None
    out_dir = Path(tmp.name) if tmp is not None else args.out_dir
    result = run_quickstart(config_path, out_dir)

    n_cells = result["grid_id"].n_unique()
    # Zero rows carry a null time_bin (see module docstring); count bins
    # from the observed pairs only.
    observed = result.filter(pl.col("time_bin").is_not_null())
    n_bins = observed["time_bin"].n_unique()
    count_sum = int(result["count"].sum())
    null_counts = int(result["count"].is_null().sum())
    n_observed_pairs = observed.select(["grid_id", "time_bin"]).unique().height
    n_positive = int((result["count"] > 0).sum())
    statuses = (
        sorted(result["observation_status"].unique().to_list())
        if "observation_status" in result.columns
        else []
    )

    print(f"Input events:            {N_EVENTS}")
    print(f"Output rows:             {result.height}")
    print(f"Output columns:          {result.width}")
    print(f"Grid cells:              {n_cells}")
    print(f"Time bins:               {n_bins}")
    print(f"sum(count):              {count_sum}")
    print(f"Null counts:             {null_counts}")
    print(f"Observed (cell, bin):   {n_observed_pairs}")
    print(f"Rows with count > 0:     {n_positive}")
    print(f"observation_status:      {statuses}")
    print(f"Output written to:       {out_dir / 'chicago_quickstart.parquet'}")

    assert result.height == n_cells * n_bins, (
        f"panel is not complete: {result.height} rows != " f"{n_cells} cells x {n_bins} bins"
    )
    assert count_sum == N_EVENTS, f"sum(count) {count_sum} != input events {N_EVENTS}"
    assert null_counts == 0, f"{null_counts} null counts in the completed panel"
    assert n_observed_pairs == n_positive, (
        f"duplicate or missing observed pairs: {n_observed_pairs} distinct "
        f"(cell, bin) with a timestamp vs {n_positive} rows with count > 0"
    )
    assert statuses == ["observed"], f"unexpected observation_status: {statuses}"
    print("Quick start OK: panel complete, every event counted exactly once.")
    if tmp is not None:
        tmp.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
