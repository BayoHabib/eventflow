"""GeoLife recipe (v1): trajectory points as events, honest subset only.

Pipeline (point level, then a count panel):

1. Project WGS 84 fixes to UTM 50N (EPSG:32650, metres) — Beijing.
2. Per-point kinematics (generic step): step distance, elapsed time,
   speed; null speed on the first fix, on ``dt <= 0`` and across
   logging gaps longer than ``max_gap_s``.
3. Temporal components + cyclical calendar encodings of the fix time.
4. Inter-arrival times per trajectory (the GPS sampling interval,
   including rolling mean/std) — the sampling itself is irregular.
5. Row-mode lag / rolling mean of speed per trajectory (``mode="rows"``:
   the duration-mode lag would need a fix at exactly ``t - lag``,
   which irregular 1 s / 5 s sampling almost never provides — see the
   boundary statement).
6. 5-minute time bins + 100 m grid assignment, then a count of fixes
   per (cell, bin). This panel is a well-defined aggregation, but it is
   the end of the honest pipeline: the panel-family steps
   (CompletePanel, spatial lag, KDE/STKDE, near-repeat, Hawkes) are NOT
   wired here. They run mechanically on the panel yet describe one
   person's path, not an event field — they are exercised as
   experiments in ``scripts/replay_geolife.py`` and adjudicated in
   ``docs/design-notes/phase3-geolife-boundary.md``.
"""

from __future__ import annotations

import polars as pl

from eventflow.core.event_frame import EventFrame
from eventflow.core.pipeline import Pipeline, Step
from eventflow.core.registry import StepRegistry
from eventflow.core.schema import RecipeConfig
from eventflow.core.steps.spatial import AssignToGridStep, TransformCRSStep
from eventflow.core.steps.temporal import (
    CalendarEncodingStep,
    ExtractTemporalComponentsStep,
    InterArrivalStep,
    MovingAverageStep,
    TemporalLagStep,
    TimeBinsStep,
)
from eventflow.datasets.geolife.kinematics import TrajectoryKinematicsStep
from eventflow.datasets.geolife.schema import GEOLIFE_TARGET_CRS
from eventflow.recipes.base import BaseRecipe

TRAJECTORY_GROUP_COLS = ["user_id", "trajectory_id"]


class GeoLifeCountStep(Step):
    """Count fixes per (grid cell, time bin), keeping cell centres.

    Mirrors ``ChicagoCountStep``: the count is the number of GPS fixes
    of the moving entity in that cell during that bin. Cell-centre
    coordinates come from the grid spec stored in the frame metadata
    by :class:`AssignToGridStep`, so the centres are exact, not the
    first fix in the cell.
    """

    def run(self, ef: EventFrame) -> EventFrame:
        """Aggregate fixes while keeping the output schema usable downstream."""
        lon, lat = ef.schema.lon_col, ef.schema.lat_col
        if lon is None or lat is None:
            raise ValueError("GeoLife aggregation requires projected coordinate columns")
        grid_spec = ef.metadata.custom.get("grid_spec") or {}
        size_m = float(grid_spec.get("size_m", 0) or 0)
        origin = grid_spec.get("origin")
        if size_m <= 0 or not origin or len(origin) != 2:
            raise ValueError(
                "GeoLifeCountStep needs the grid spec in metadata.custom['grid_spec'] "
                "(run AssignToGridStep first)"
            )
        origin_x, origin_y = float(origin[0]), float(origin[1])
        lf = ef.lazy_frame.group_by(["grid_id", "time_bin"]).agg(
            pl.len().alias("count"),
            pl.col("grid_row").first(),
            pl.col("grid_col").first(),
            pl.col(lon).first(),
            pl.col(lat).first(),
        )
        lf = lf.with_columns(
            (((pl.col(lon) - origin_x) / size_m).floor() * size_m + origin_x + size_m / 2).alias(
                lon
            ),
            (((pl.col(lat) - origin_y) / size_m).floor() * size_m + origin_y + size_m / 2).alias(
                lat
            ),
            pl.col("time_bin").alias(ef.schema.timestamp_col),
        ).sort(["time_bin", "grid_id"])
        result = ef.with_lazy_frame(lf).with_schema(
            numeric_cols=["count", "grid_id", "grid_row", "grid_col", lon, lat],
            categorical_cols=[],
        )
        return result.register_feature(
            "count", {"source_step": "GeoLifeCountStep", "inputs": ["grid_id", "time_bin"]}
        )


class GeoLifeV1Recipe(BaseRecipe):
    """Baseline recipe for GeoLife GPS trajectories."""

    def __init__(
        self,
        config: RecipeConfig,
        *,
        step_registry: StepRegistry | None = None,
    ) -> None:
        super().__init__(config, step_registry=step_registry)
        self.name = "geolife_v1"

    def build_pipeline(self) -> Pipeline:
        """Build the transformation pipeline."""
        grid_size_m = float(self.config.grid.size_m)
        time_bin = self.config.temporal.time_bin
        steps: list[Step] = [
            TransformCRSStep(target_crs=GEOLIFE_TARGET_CRS),
            TrajectoryKinematicsStep(group_cols=TRAJECTORY_GROUP_COLS),
            ExtractTemporalComponentsStep(components=["hour_of_day", "day_of_week"]),
            CalendarEncodingStep(encoding_type="cyclical", components=["hour", "day_of_week"]),
            InterArrivalStep(group_cols=TRAJECTORY_GROUP_COLS),
            TemporalLagStep(
                value_cols=["speed_m_s"],
                lag_periods=[1],
                mode="rows",
                group_cols=TRAJECTORY_GROUP_COLS,
            ),
            MovingAverageStep(
                value_cols=["speed_m_s"],
                windows=[5, 10],
                mode="rows",
                group_cols=TRAJECTORY_GROUP_COLS,
                closed="left",
            ),
            TimeBinsStep(bin_size=time_bin),
            AssignToGridStep(grid_size_m=grid_size_m),
            GeoLifeCountStep(),
        ]
        return Pipeline(steps)
