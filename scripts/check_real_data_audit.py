"""Replay the real-data audit after fixes, using an extracted frozen audit bundle.

python scripts/check_real_data_audit.py --audit-root /path/to/extracted/bundle
No download, imputation or modification of the original audit artifacts occurs.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import gzip
import hashlib
import json
import math
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import shapely
from shapely.geometry import shape

from shikong.core import CadenceRule, EventFrame, EventMetadata, EventSchema, ObservationPanel
from shikong.core.domain import SpatialDomain
from shikong.core.schema import RecipeConfig
from shikong.core.spatial import assign_to_grid, create_grid, transform_crs
from shikong.core.steps.domain import GridDomainCoverageStep, SpatialDomainMaskStep
from shikong.core.steps.observation import ObservationPanelStep
from shikong.core.steps.point_process import KFunctionStep
from shikong.core.steps.spatial import AssignToGridStep, SpatialLagStep
from shikong.core.steps.temporal import TemporalLagStep
from shikong.datasets.chicago_crime.mapping import load_chicago_crime
from shikong.datasets.chicago_crime.recipes.chicago_crime_v1 import ChicagoCrimeV1Recipe


def replay(audit_root: Path) -> dict[str, Any]:
    """Return reproducible measurements; assertions fail on numerical regressions."""
    manifest = json.loads((audit_root / "results/data_manifest.json").read_text())
    for entry in manifest:
        assert (
            hashlib.sha256((audit_root / entry["file"]).read_bytes()).hexdigest() == entry["sha256"]
        )
    records = json.loads((audit_root / "data/chicago_crime_2024_01.json").read_text())
    assert len(records) == int(
        json.loads((audit_root / "data/chicago_crime_count.json").read_text())[0]["n"]
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "crime.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "id",
                    "date",
                    "primary_type",
                    "latitude",
                    "longitude",
                    "community_area",
                ],
            )
            writer.writeheader()
            writer.writerows(records)
        events = load_chicago_crime(path)
        raw = events.collect()
        assert raw.height == len(records) - sum(
            not r.get("latitude") or not r.get("longitude") for r in records
        )
        projected = transform_crs(events, "EPSG:26971")
        df = projected.collect()
        x, y = projected.schema.lon_col, projected.schema.lat_col
        bounds = (df[x].min(), df[y].min(), df[x].max(), df[y].max())
        reference = create_grid(bounds, 1000, "EPSG:26971")
        core = assign_to_grid(projected, reference).collect()
        step = AssignToGridStep(1000, origin=bounds[:2]).run(projected)
        stepped = step.collect()
        assert core["grid_id"].equals(stepped["grid_id"])
        assert not core["grid_id"].null_count()
        communities = json.loads((audit_root / "data/communities.json").read_text())
        domain = SpatialDomain(
            shapely.union_all([shape(r["the_geom"]) for r in communities]),
            "EPSG:4326",
            "Chicago communities",
        )
        covered = GridDomainCoverageStep(domain, reference, "EPSG:26971").run(step).collect()
        assert covered["domain_fraction"].null_count() == 0
        assert covered["domain_fraction"].is_between(0, 1).all()
        recipe = ChicagoCrimeV1Recipe(
            RecipeConfig(dataset="chicago_crime", recipe="chicago_crime_v1")
        )
        aggregate = recipe.run(events)
        table = aggregate.collect()
        assert table["count"].sum() == raw.height
        lagged = SpatialLagStep(["count"], group_cols=["time_bin"]).run(aggregate).collect()
        # Exhaustive eight-offset lookup, independent of SpatialWeights.
        lookup = {
            (r["time_bin"], r["grid_row"], r["grid_col"]): r["count"] for r in table.to_dicts()
        }
        expected = []
        for r in lagged.to_dicts():
            neighbors = [
                lookup[(r["time_bin"], r["grid_row"] + dr, r["grid_col"] + dc)]
                for dr in [-1, 0, 1]
                for dc in [-1, 0, 1]
                if (dr or dc) and (r["time_bin"], r["grid_row"] + dr, r["grid_col"] + dc) in lookup
            ]
            expected.append(float(np.mean(neighbors)) if neighbors else np.nan)
        np.testing.assert_allclose(
            lagged["count_spatial_lag"], expected, atol=1e-12, equal_nan=True
        )

        async def scenario():
            return await (await recipe.run_async(events)).collect_async()

        assert asyncio.run(scenario()).equals(table)
        # Explicit domain filtering is separate from coordinate cleaning.
        requested = events.with_lazy_frame(events.lazy_frame.head(100))
        pattern = SpatialDomainMaskStep(domain, mode="filter").run(requested)
        n = pattern.collect().height
        window = domain.to_crs("EPSG:26971")
        local_l = KFunctionStep(1000, transform="L", window=window).run(pattern).collect()
        h = KFunctionStep(1000, transform="H", window=window).run(pattern).collect()
        errors = h["h_function_local"].to_numpy() - (local_l["l_function_local"].to_numpy() - 1000)
        assert np.max(np.abs(errors)) < 1e-10
        # Direct NumPy reference for the explicitly uncorrected estimator.
        p = transform_crs(pattern, "EPSG:26971")
        p_df = p.collect()
        xy = np.column_stack([p_df[p.schema.lon_col], p_df[p.schema.lat_col]])
        distances = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
        neighbors = np.sum((distances <= 1000) & ~np.eye(n, dtype=bool), axis=1)
        reference_k = window.geometry.area * neighbors / (n - 1)
        k = (
            KFunctionStep(1000, transform="K", window=window, edge_correction="none")
            .run(pattern)
            .collect()
        )
        np.testing.assert_allclose(k["k_function_local"], reference_k, atol=1e-8)
        # Historical station archives: preserve missing slots and nullable temperature.
        weather = []
        for station, file in [("ohare", "ohare_2024.gz"), ("midway", "midway_2024.gz")]:
            for line in (
                gzip.decompress((audit_root / "data" / file).read_bytes()).decode().splitlines()
            ):
                values = list(map(int, line.split()))
                weather.append(
                    {
                        "station": station,
                        "t": datetime(*values[:4], tzinfo=timezone.utc),
                        "x": 0.0,
                        "y": 0.0,
                        "temperature": None if values[4] == -9999 else values[4] / 10,
                    }
                )
        wdf = pl.DataFrame(weather)
        wef = EventFrame(
            wdf.lazy(),
            EventSchema(timestamp_col="t", lon_col="x", lat_col="y"),
            EventMetadata(dataset_name="NOAA-archive"),
        )
        cells = pl.DataFrame(
            {
                "station": ["ohare", "midway"],
                "start": [datetime(2024, 1, 1, tzinfo=timezone.utc)] * 2,
                "end": [datetime(2025, 1, 1, tzinfo=timezone.utc)] * 2,
                "x": [0.0, 0.0],
                "y": [0.0, 0.0],
            }
        )
        support = ObservationPanel(cells, "EPSG:4326", key_cols=["station"])
        cadence = (
            ObservationPanelStep(support, cadence_rule=CadenceRule(every="1h")).run(wef).collect()
        )
        assert cadence["cadence_missing"].sum() == 17568 - wdf.height == 38
        assert cadence["event_count"].null_count() == 2
        assert wdf["temperature"].null_count() == 1
        lag = (
            TemporalLagStep(
                ["temperature"], lag_periods=[24], time_unit="1h", group_cols=["station"]
            )
            .run(wef)
            .collect()
        )
        temperature = {(r["station"], r["t"]): r["temperature"] for r in weather}
        values = [
            temperature.get((r["station"], r["t"] - timedelta(hours=24))) for r in lag.to_dicts()
        ]
        np.testing.assert_allclose(
            lag["temperature_lag_24"],
            [v if v is not None else math.nan for v in values],
            equal_nan=True,
        )
    return {
        "status": "PASS",
        "input_files_sha256_verified": len(manifest),
        "crime_source_rows": len(records),
        "crime_retained_rows": raw.height,
        "grid_id_mismatches": 0,
        "coverage_unmatched_rows": 0,
        "recipe_output_rows": table.height,
        "count_sum": int(table["count"].sum()),
        "queen_lag_rows_compared": lagged.height,
        "async_equals_sync": True,
        "k_pattern_requested_rows": 100,
        "k_pattern_in_window_rows": n,
        "h_l_identity_max_error": float(np.max(np.abs(errors))),
        "k_normalization_numpy_reference": "PASS",
        "k_edge_correction": "translation",
        "weather_rows": wdf.height,
        "weather_missing_hours": 38,
        "weather_null_temperature": 1,
        "weather_elapsed_lag_reference": "PASS",
        "collection_completeness_remain_unknown": True,
        "scope": "Arithmetic and integration on frozen data; not inference/forecast calibration or scale certification",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    result = replay(args.audit_root)
    result["elapsed_seconds"] = time.perf_counter() - started
    source = Path(__file__).resolve().parents[1] / "src/shikong"
    result["tested_source_sha256"] = {
        str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(source.rglob("*.py"))
    }
    content = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)
    print(content)


if __name__ == "__main__":
    main()
