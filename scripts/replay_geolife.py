#!/usr/bin/env python
"""GeoLife subset replay (phase 3, criterion 4).

Loads the 2-trajectory subset, runs the GeoLife v1 recipe, hand-checks
a few (cell, bin) counts against an independent recount from the raw
.plt files, and exercises every panel-family step as an experiment —
recording what runs, what fails, and where the numbers show the
semantics collapse to re-describing one person's path.

Outputs:
- audit/phase-3/geolife_replay.json (this work copy)
- <phase3>/geolife_features.parquet (point-level features, outside
  the work copy — data artefacts are never committed)

Usage:
    PYTHONPATH=src python scripts/replay_geolife.py \
        [--data-dir .../phase3/data/geolife/subset] \
        [--json audit/phase-3/geolife_replay.json] \
        [--parquet .../phase3/geolife_features.parquet]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import polars as pl
from pyproj import Transformer

from eventflow.core.pipeline import Pipeline
from eventflow.core.schema import RecipeConfig
from eventflow.core.steps.categorical import CategoryCountsStep
from eventflow.core.steps.density import KernelDensityStep, SpaceTimeKernelDensityStep
from eventflow.core.steps.dynamics import (
    BurstinessStep,
    EWMAMeanStep,
    TimeSinceLastEventStep,
    ZeroFeaturesStep,
)
from eventflow.core.steps.hawkes import HawkesExcitationStep
from eventflow.core.steps.near_repeat import NearRepeatStep
from eventflow.core.steps.panel import CompletePanelStep
from eventflow.core.steps.spatial import MultiScaleSpatialLagStep, TransformCRSStep
from eventflow.core.steps.temporal import (
    FillNullStep,
    RecencyWeightStep,
    TemporalLagStep,
)
from eventflow.datasets.geolife.loader import load_geolife
from eventflow.datasets.geolife.recipes.geolife_v1 import (
    TRAJECTORY_GROUP_COLS,
    GeoLifeV1Recipe,
)

GRID_SIZE_M = 100.0
TIME_BIN = "5m"


def _share(series: pl.Series, predicate) -> float:
    values = [predicate(v) for v in series.to_list()]
    return sum(values) / len(values) if values else float("nan")


def _hand_recount(
    plt_paths: list[Path], origin_x: float, origin_y: float
) -> dict[tuple[int, datetime], int]:
    """Independent (cell, bin) recount from the raw .plt files.

    Projects with pyproj directly and assigns cells with plain
    floor() arithmetic — a different code path from AssignToGridStep.
    Returns {(grid_row, grid_col, time_bin): count} keyed by row/col.
    """
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32650", always_xy=True)
    counts: dict[tuple[int, int, datetime], int] = {}
    for path in plt_paths:
        lines = path.read_text(encoding="utf-8").splitlines()[6:]
        for line in lines:
            if not line.strip():
                continue
            lat, lon, _r, _alt, _d, date_s, time_s = line.split(",")
            x, y = transformer.transform(float(lon), float(lat))
            col = math.floor((x - origin_x) / GRID_SIZE_M)
            row = math.floor((y - origin_y) / GRID_SIZE_M)
            moment = datetime.strptime(f"{date_s} {time_s}", "%Y-%m-%d %H:%M:%S")
            bin_start = moment.replace(minute=(moment.minute // 5) * 5, second=0, microsecond=0)
            key = (row, col, bin_start)
            counts[key] = counts.get(key, 0) + 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default=str(Path.home() / "workspace/eventflow-eda/phase3/data/geolife/subset"),
    )
    parser.add_argument(
        "--json",
        default=str(Path(__file__).resolve().parent.parent / "audit/phase-3/geolife_replay.json"),
    )
    parser.add_argument(
        "--parquet",
        default=str(Path.home() / "workspace/eventflow-eda/phase3/geolife_features.parquet"),
    )
    args = parser.parse_args()

    started = time.time()
    data_dir = Path(args.data_dir)
    plt_paths = sorted(data_dir.glob("*/Trajectory/*.plt"))
    assert len(plt_paths) == 2, f"expected 2 subset .plt files, found {len(plt_paths)}"
    subset = {
        "users": sorted({p.parent.parent.name for p in plt_paths}),
        "trajectories": [p.stem for p in plt_paths],
        "files": [str(p) for p in plt_paths],
    }

    events = load_geolife([str(p) for p in plt_paths])
    raw = events.collect()
    per_user = (
        raw.group_by("user_id")
        .agg(
            pl.len().alias("n"),
            pl.col("timestamp").min().alias("t0"),
            pl.col("timestamp").max().alias("t1"),
        )
        .sort("user_id")
    )
    subset["per_trajectory"] = [
        {
            "user_id": r["user_id"],
            "trajectory_id": subset["trajectories"][subset["users"].index(r["user_id"])],
            "n_points": r["n"],
            "t_start": r["t0"].isoformat(),
            "t_end": r["t1"].isoformat(),
            "span_minutes": (r["t1"] - r["t0"]).total_seconds() / 60.0,
        }
        for r in per_user.iter_rows(named=True)
    ]
    subset["n_rows"] = raw.height

    config = RecipeConfig(
        dataset="geolife",
        recipe="geolife_v1",
        grid={"size_m": GRID_SIZE_M},
        temporal={"time_bin": TIME_BIN},
    )
    recipe = GeoLifeV1Recipe(config)

    # Point-level frame (recipe steps 1-8) — the honest feature output.
    point_steps = recipe.build_pipeline().steps[:8]
    point_frame = Pipeline(point_steps).run(events).collect()
    point_frame.write_parquet(args.parquet)
    points = {
        "rows": point_frame.height,
        "columns": point_frame.columns,
        "speed_nulls": point_frame["speed_m_s"].null_count(),
        "speed_null_share": point_frame["speed_m_s"].null_count() / point_frame.height,
        "speed_median_m_s": point_frame["speed_m_s"].median(),
        "speed_max_m_s": point_frame["speed_m_s"].max(),
        "dt_nulls": point_frame["dt_s"].null_count(),
        "inter_arrival_nulls": point_frame["inter_arrival_seconds"].null_count(),
        "inter_arrival_median_s": point_frame["inter_arrival_seconds"].median(),
    }

    # Count panel (full recipe).
    panel_frame = recipe.run(events)
    panel = panel_frame.collect()
    panel_info = {
        "rows": panel.height,
        "columns": panel.columns,
        "count_sum": int(panel["count"].sum()),
        "n_cells": panel["grid_id"].n_unique(),
        "n_bins": panel["time_bin"].n_unique(),
        "max_count_per_cell_bin": int(panel["count"].max()),
    }
    assert panel_info["count_sum"] == subset["n_rows"], "panel must count every fix once"

    # -- Hand-checks: independent recount of 3 (cell, bin) counts. ------
    # Origin rule of AssignToGridStep with origin=None: min projected x/y.
    projected = Pipeline([TransformCRSStep(target_crs="EPSG:32650")]).run(events).collect()
    origin_x = float(projected["longitude_proj"].min())
    origin_y = float(projected["latitude_proj"].min())
    recount = _hand_recount(plt_paths, origin_x, origin_y)
    assert sum(recount.values()) == subset["n_rows"]
    panel_by_key = {
        (r["grid_row"], r["grid_col"], r["time_bin"]): r["count"]
        for r in panel.iter_rows(named=True)
    }
    assert set(recount) == set(panel_by_key), "recount keys must match panel keys"
    hand_checks = []
    for key in sorted(recount, key=lambda k: -recount[k])[:3]:
        hand_checks.append(
            {
                "grid_row": key[0],
                "grid_col": key[1],
                "time_bin": key[2].isoformat(),
                "recount": recount[key],
                "panel_count": panel_by_key[key],
                "match": recount[key] == panel_by_key[key],
            }
        )
    assert all(c["match"] for c in hand_checks)

    experiments: dict[str, dict] = {}

    def record(name: str, outcome: dict) -> None:
        experiments[name] = outcome

    # -- Duration-mode lag on point-level speed: fails outright. --------
    # Two reasons, both structural: (a) duplicate timestamps exist in
    # real GPS data (user 010 has 5), and duration mode rejects
    # duplicate (group, time) keys; (b) even with distinct timestamps,
    # no fix exists exactly 60 s before any fix, so the lag is null
    # everywhere (shown in tests/datasets/test_geolife.py).
    dup_ts = int(point_frame.filter(pl.col("dt_s") == 0).height)
    try:
        TemporalLagStep(
            ["speed_m_s"],
            lag_periods=[1],
            time_unit="60s",
            mode="duration",
            group_cols=TRAJECTORY_GROUP_COLS,
        ).run(panel_frame.with_lazy_frame(point_frame.lazy())).collect()
        dur_status: dict = {"status": "runs", "note": "unexpected"}
    except Exception as exc:  # noqa: BLE001 - the error message is the evidence
        dur_status = {
            "status": "fails",
            "error": f"{type(exc).__name__}: {exc}",
            "duplicate_timestamps_dt_eq_0": dup_ts,
            "note": "Duration-mode lags need unique exact timestamps; GPS "
            "fixes repeat timestamps and are irregularly spaced.",
        }
    record("temporal_lag_duration_mode_on_speed", dur_status)

    # -- Recency weights on point fixes: applies, with a sub-day halflife.
    recency = (
        RecencyWeightStep(half_life=600.0, time_unit="1s")
        .run(panel_frame.with_lazy_frame(point_frame.lazy()))
        .collect()
    )
    record(
        "recency_weight_halflife_600s",
        {
            "status": "runs",
            "min_weight": recency["recency_weight"].min(),
            "max_weight": recency["recency_weight"].max(),
            "note": "Per-fix exponential weight from the latest fix; the "
            "reference is the global max timestamp, so the two trajectories "
            "(21 months apart) share one reference — per-trajectory "
            "references would be the honest grouping.",
        },
    )

    # -- CompletePanel: infeasible on the combined panel, bounded on one
    # trajectory. The combined time axis spans 2007-08-04 -> 2009-05-08.
    t_min = panel["time_bin"].min()
    t_max = panel["time_bin"].max()
    n_bins_axis = int((t_max - t_min).total_seconds() // 300) + 1
    record(
        "complete_panel_combined",
        {
            "status": "not_run_infeasible",
            "reason": "Time axis runs min->max over all units: "
            f"{n_bins_axis} 5-min bins x {panel_info['n_cells']} cells = "
            f"{n_bins_axis * panel_info['n_cells']:,} pairs for 2 days of "
            "real coverage 21 months apart.",
            "projected_rows": n_bins_axis * panel_info["n_cells"],
            "observed_rows": panel_info["rows"],
        },
    )
    user010 = panel.filter(pl.col("time_bin") < datetime(2008, 1, 1))
    # Rebuild a one-trajectory panel EventFrame for the step.
    panel_ef_010 = panel_frame.with_lazy_frame(user010.lazy())
    completed_010 = (
        CompletePanelStep(unit_col="grid_id", value_cols=["count"], bin_size=TIME_BIN)
        .run(panel_ef_010)
        .collect()
    )
    record(
        "complete_panel_user010_only",
        {
            "status": "runs",
            "rows_before": user010.height,
            "rows_after": completed_010.height,
            "generated_zero_rows": int((completed_010["count"] == 0).sum()),
            "note": "Every generated row is a (cell, bin) where the person "
            "was not present; the zero means 'person elsewhere', and the "
            "step still labels it observation_status='observed'.",
        },
    )

    # -- Spatial lag on the lagged count (raw panel). --------------------
    lagged = TemporalLagStep(
        ["count"], lag_periods=[1], time_unit=TIME_BIN, mode="duration", group_cols=["grid_id"]
    ).run(panel_frame)
    filled = FillNullStep(value_cols=["count_lag_1"], fill_value=0).run(lagged)
    slag = (
        MultiScaleSpatialLagStep(
            value_cols=["count_lag_1"],
            neighbor_col="grid_id",
            rings=(1, 2, 3),
            group_cols=["timestamp"],
        )
        .run(filled)
        .collect()
    )
    record(
        "multiscale_spatial_lag_rings_1_2_3",
        {
            "status": "runs",
            "null_share_r1": _share(slag["count_lag_1_spatial_lag_r1"], lambda v: v is None),
            "zero_share_r1": _share(slag["count_lag_1_spatial_lag_r1"], lambda v: v == 0.0),
            "nonzero_r1": int((slag["count_lag_1_spatial_lag_r1"] > 0).sum()),
            "max_r1": slag["count_lag_1_spatial_lag_r1"].max(),
            "note": "Mostly null (no visited neighbour in the same bin) or "
            "0; non-zero only where the path was one bin earlier.",
        },
    )

    # -- KDE on the lagged count (raw panel). ----------------------------
    # Every raw-panel row has count > 0 by construction, so the smearing
    # test compares cells whose *lagged* count is 0 against the rest:
    # a positive density where the lagged count is 0 is smoothing, not data.
    kde = (
        KernelDensityStep(value_col="count_lag_1", bandwidth_m=300.0, group_cols=["timestamp"])
        .run(filled)
        .collect()
    )
    record(
        "kde_lagged_count_bandwidth_300m",
        {
            "status": "runs",
            "mean_kde_where_lagged_count_eq_0": kde.filter(pl.col("count_lag_1") == 0)[
                "count_lag_1_kde"
            ].mean(),
            "mean_kde_where_lagged_count_gt_0": kde.filter(pl.col("count_lag_1") > 0)[
                "count_lag_1_kde"
            ].mean(),
            "max_kde": kde["count_lag_1_kde"].max(),
            "note": "Density is positive in cells whose lagged count is 0: "
            "the KDE smears the path into cells the person was not in "
            "during the previous bin.",
        },
    )

    # -- STKDE: raw panel violates the regular-panel assumption, so run
    # it on the completed user-010 panel (where the assumption holds).
    stkde = (
        SpaceTimeKernelDensityStep(
            value_col="count",
            bandwidth_m=300.0,
            halflife_days=0.02,
            time_col="timestamp",
        )
        .run(panel_frame.with_lazy_frame(completed_010.lazy()))
        .collect()
    )
    record(
        "stkde_on_completed_user010_panel",
        {
            "status": "runs",
            "rows": stkde.height,
            "max_stkde": stkde["count_stkde"].max(),
            "null_share": stkde["count_stkde"].null_count() / stkde.height,
            "note": "Requires the completed panel: on the sparse raw panel "
            "the per-unit bin index is not elapsed time, so the temporal "
            "decay would silently use the wrong lags.",
        },
    )

    # -- Near-repeat (raw panel; 7 d window >> 15-min panel span). -------
    nr = (
        NearRepeatStep(value_col="count", time_col="timestamp", rings=(1,))
        .run(panel_frame)
        .collect()
    )
    record(
        "near_repeat_ring1",
        {
            "status": "runs",
            "share_event_flag_1": _share(nr["count_nr_event_7d"], lambda v: v == 1),
            "share_event_flag_1_among_visited": _share(
                nr.filter(pl.col("count") > 0)["count_nr_event_7d"], lambda v: v == 1
            ),
            "max_nr_r1_7d": int(nr["count_nr_r1_7d"].max()),
            "note": "The 7 d window covers the whole panel history, and the "
            "flag is 1 wherever the trajectory passed a neighbouring cell "
            "earlier — the 'repeat' detected is the trajectory itself.",
        },
    )

    # -- Hawkes on the raw panel: bin-index vs elapsed-time divergence. --
    hawkes = (
        HawkesExcitationStep(value_col="count", halflife_days=7.0, group_cols=["grid_id"])
        .run(panel_frame)
        .collect()
    )
    # Find a cell observed in >= 2 non-consecutive bins and compare the
    # step's excitation against elapsed-time decay.
    hawkes_check = None
    for grid_id in panel["grid_id"].unique().to_list():
        cell = panel.filter(pl.col("grid_id") == grid_id).sort("time_bin")
        if cell.height >= 2:
            gap_bins = int((cell["time_bin"][1] - cell["time_bin"][0]).total_seconds() // 300)
            if gap_bins > 1:
                decay = 0.5 ** (1.0 / 2016.0)  # 7 d halflife on 5-min bins
                c0 = int(cell["count"][0])
                got = hawkes.filter(pl.col("grid_id") == grid_id).sort("time_bin")[
                    "count_hawkes_7d"
                ][1]
                hawkes_check = {
                    "grid_id": int(grid_id),
                    "first_count": c0,
                    "elapsed_bins_between_observations": gap_bins,
                    "step_value_index_based": got,
                    "expected_index_based": c0 * decay,
                    "expected_elapsed_time_based": c0 * decay**gap_bins,
                    "matches_index_based": abs(got - c0 * decay) < 1e-9,
                }
                break
    record(
        "hawkes_excitation_halflife_7d",
        {
            "status": "runs",
            "max_excitation": hawkes["count_hawkes_7d"].max(),
            "bin_index_vs_elapsed_time": hawkes_check,
            "note": "Excitation is decayed cumulative self-count of GPS "
            "fixes; on the sparse panel the decay counts observed bins, "
            "not elapsed bins (see bin_index_vs_elapsed_time).",
        },
    )

    # -- Time-since-last / burstiness / zero-features / EWMA on count. ---
    # On the sparse raw panel these are void by construction: every row
    # has count > 0, so time-since-last is identically 0 and the
    # nonzero share identically 1. The completed user-010 panel is the
    # honest configuration for them.
    tsl = (
        TimeSinceLastEventStep(value_col="count", group_cols=["grid_id"]).run(panel_frame).collect()
    )
    bursty = (
        BurstinessStep(value_col="count", window="28d", group_cols=["grid_id"])
        .run(panel_frame)
        .collect()
    )
    zeroed = ZeroFeaturesStep(value_col="count", group_cols=["grid_id"]).run(panel_frame).collect()
    ewma = (
        EWMAMeanStep(value_cols=["count"], halflife_days=0.01, group_cols=["grid_id"])
        .run(panel_frame)
        .collect()
    )
    ewma_col = next(c for c in ewma.columns if c.startswith("count_ewma_"))
    tsl_completed = (
        TimeSinceLastEventStep(value_col="count", group_cols=["grid_id"])
        .run(panel_frame.with_lazy_frame(completed_010.lazy()))
        .collect()
    )
    zeroed_completed = (
        ZeroFeaturesStep(value_col="count", group_cols=["grid_id"])
        .run(panel_frame.with_lazy_frame(completed_010.lazy()))
        .collect()
    )
    record(
        "time_since_last_burstiness_zero_features_ewma",
        {
            "status": "runs",
            "raw_panel": {
                "time_since_last_null_share": tsl["count_time_since_last_d"].null_count()
                / tsl.height,
                "time_since_last_max_d": tsl["count_time_since_last_d"].max(),
                "burstiness_null_share": bursty["count_burstiness_28d"].null_count()
                / bursty.height,
                "share_nonzero_mean": zeroed["count_share_nonzero_28d"].mean(),
            },
            "completed_user010_panel": {
                "time_since_last_max_d": tsl_completed["count_time_since_last_d"].max(),
                "time_since_last_max_minutes": tsl_completed["count_time_since_last_d"].max()
                * 1440.0,
                "share_nonzero_mean": zeroed_completed["count_share_nonzero_28d"].mean(),
            },
            "ewma_column": ewma_col,
            "ewma_max": ewma[ewma_col].max(),
            "note": "On the raw panel time-since-last is 0 everywhere and "
            "the nonzero share is 1 everywhere (every row has count > 0 "
            "by construction). On the completed panel time-since-last "
            "becomes a genuine per-cell revisit time; burstiness / "
            "zero-features / EWMA of fix counts describe sampling, not "
            "an event process.",
        },
    )

    # -- CategoryCounts: no categorical column exists in .plt data. ------
    try:
        CategoryCountsStep(
            category_cols=["transport_mode"], unit_col="grid_id", time_col="time_bin"
        ).run(panel_frame)
        record("category_counts", {"status": "runs", "note": "unexpected"})
    except Exception as exc:  # noqa: BLE001 - the error message is the evidence
        record(
            "category_counts",
            {
                "status": "fails",
                "error": f"{type(exc).__name__}: {exc}",
                "note": ".plt files carry no categorical column.",
            },
        )

    report = {
        "dataset": "geolife",
        "source": {
            "name": "Geolife Trajectories 1.3.zip",
            "version": "1.2.2",
            "published": "2024-07-15",
            "url": "https://www.microsoft.com/en-us/download/details.aspx?id=52367",
            "direct_url": "https://download.microsoft.com/download/F/4/8/"
            "F4894AA5-FDBC-481E-9285-D5F8C4C4F039/Geolife%20Trajectories%201.3.zip",
            "archive_bytes": 313164406,
            "archive_entries_plt": 18670,
            "note": "The details page returns HTML; the zip comes from the "
            "official Microsoft CDN direct link (same file, 298.7 MB). "
            "Archive holds 18,670 .plt files vs 17,621 trajectories in the "
            "dataset description (difference not investigated).",
        },
        "subset": subset,
        "recipe": {
            "name": "geolife_v1",
            "grid_size_m": GRID_SIZE_M,
            "time_bin": TIME_BIN,
            "target_crs": "EPSG:32650",
            "point_level": points,
            "panel": panel_info,
        },
        "hand_checks": hand_checks,
        "experiments": experiments,
        "parquet": args.parquet,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    out_path = Path(args.json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(
        f"panel: {panel_info['rows']} rows x {len(panel_info['columns'])} cols, "
        f"count sum {panel_info['count_sum']}"
    )
    print(
        f"point features: {points['rows']} rows x {len(points['columns'])} cols -> {args.parquet}"
    )
    print(f"replay JSON: {out_path}")
    for name, outcome in experiments.items():
        print(f"  {name}: {outcome.get('status')}")


if __name__ == "__main__":
    main()
