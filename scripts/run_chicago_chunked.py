"""
Chunked runner for Chicago Crime recipe.

Processes monthly Parquet chunks produced by `chicago-crime-dl` to avoid
large single-run memory usage. For each month present under the raw root,
this script:
 1) Loads only that month's parquet files lazily.
 2) Applies dataset cleaning (timestamp/coord coercion).
 3) Runs the ChicagoCrimeV1 recipe.
 4) Writes the aggregated result to a monthly parquet under the output root.

Usage:
    python scripts/run_chicago_chunked.py \\
        --raw-root data/raw_2024 \\
        --output-root data/processed/chicago_crime_v1_2024 \\
        --year 2024
"""

from __future__ import annotations

import argparse
from pathlib import Path
import polars as pl

from eventflow.datasets.chicago_crime.mapping import clean_chicago_data
from eventflow.datasets.chicago_crime.schema import CHICAGO_CRIME_SCHEMA, create_chicago_metadata
from eventflow.datasets.chicago_crime.recipes.chicago_crime_v1 import ChicagoCrimeV1Recipe
from eventflow.core.event_frame import EventFrame
from eventflow.core.schema import DatasetConfig, RecipeConfig
from eventflow.core.temporal import create_time_bins
from eventflow.core.features import aggregate_counts


def process_month(raw_root: Path, output_root: Path, year: int, month: int, mode: str) -> None:
    """Process a single month and write aggregated output."""
    pattern = raw_root / f"monthly_{year}-{month:02d}_*.parquet"
    matches = list(pattern.parent.glob(pattern.name))
    if not matches:
        print(f"[skip] No data for {year}-{month:02d}")
        return

    print(f"[load] {year}-{month:02d} ({len(matches)} files)")

    lf = pl.scan_parquet(str(pattern))
    lf = clean_chicago_data(lf)

    ef = EventFrame(
        lf,
        CHICAGO_CRIME_SCHEMA,
        create_chicago_metadata(crs="EPSG:4326", time_zone="America/Chicago"),
    )

    if mode == "recipe":
        recipe = ChicagoCrimeV1Recipe(
            RecipeConfig(dataset="chicago_crime", recipe="chicago_crime_v1")
        )
        result = recipe.run(ef)
    else:
        # Simple, safer aggregation: time bins + counts by primary_type and time_bin
        binned = create_time_bins(ef, "6h", bin_col="time_bin")
        result = aggregate_counts(binned, group_by=["primary_type", "time_bin"])

    df = result.collect()
    output_root.mkdir(parents=True, exist_ok=True)
    suffix = "recipe" if mode == "recipe" else "simple"
    out_file = output_root / f"chicago_crime_{suffix}_{year}-{month:02d}.parquet"
    df.write_parquet(out_file)
    print(f"[done] {year}-{month:02d}: {len(df)} rows -> {out_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ChicagoCrimeV1 recipe month by month.")
    parser.add_argument("--raw-root", required=True, help="Path to raw parquet files (flat layout).")
    parser.add_argument(
        "--output-root",
        required=True,
        help="Where to write per-month aggregated parquet outputs.",
    )
    parser.add_argument("--year", type=int, default=2024, help="Year to process (default: 2024).")
    parser.add_argument(
        "--mode",
        choices=["simple", "recipe"],
        default="simple",
        help="Transformation to run: 'simple' (time-bin counts) or 'recipe' (full spatial grid).",
    )
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    output_root = Path(args.output_root)

    cfg = DatasetConfig(
        dataset_name="chicago_crime",
        raw_root=str(raw_root),
        layout="flat",
        crs="EPSG:4326",
        time_zone="America/Chicago",
    )
    # cfg is currently unused but kept to signal expected structure.
    print(f"[config] raw={cfg.raw_root} year={args.year}")

    for month in range(1, 13):
        process_month(raw_root, output_root, args.year, month, args.mode)


if __name__ == "__main__":
    main()
