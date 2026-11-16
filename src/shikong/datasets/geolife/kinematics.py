"""Per-point kinematics for moving-entity (trajectory) frames.

This step is generic over any point frame with timestamp + projected
coordinates — it knows nothing about GeoLife. It derives, per
consecutive pair of points within each group (one trajectory):

- ``step_distance_m``: Euclidean distance between consecutive fixes,
  in metres (the frame must already be in a projected metric CRS;
  use :class:`TransformCRSStep` first).
- ``dt_s``: elapsed seconds between consecutive fixes.
- ``speed_m_s``: ``step_distance_m / dt_s``.

Honesty guards (a speed is a measurement, not a fill-in):

- the first fix of each trajectory has null kinematics (no previous fix);
- ``dt_s <= 0`` (duplicate timestamps, out-of-order fixes) yields a
  null speed, never a division by zero or a negative speed;
- ``dt_s > max_gap_s`` yields a null speed: the straight-line distance
  across a long logging gap is not a speed, it is a displacement
  between two unrelated fixes. The distance and ``dt_s`` themselves are
  still reported — only the ratio is withheld.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import polars as pl

from shikong.core.grid import require_metric_crs
from shikong.core.pipeline import Step
from shikong.core.schema import FeatureProvenance
from shikong.core.utils import get_logger

if TYPE_CHECKING:
    from shikong.core.event_frame import EventFrame

logger = get_logger(__name__)


class TrajectoryKinematicsStep(Step):
    """Derive step distance, elapsed time and speed for trajectory points.

    Inputs:
        - timestamp, lat/lon columns from the EventSchema (projected
          metres; a metric CRS is required)

    Outputs:
        - ``step_distance_m``, ``dt_s``, ``speed_m_s`` (Float64)
    """

    def __init__(
        self,
        group_cols: Sequence[str] | None = None,
        *,
        max_gap_s: float = 300.0,
    ) -> None:
        if not max_gap_s > 0:
            raise ValueError(f"max_gap_s must be positive, got {max_gap_s!r}")
        self.group_cols = list(group_cols) if group_cols else ["user_id", "trajectory_id"]
        self.max_gap_s = float(max_gap_s)

    def run(self, event_frame: EventFrame) -> EventFrame:
        """Execute kinematics derivation."""
        require_metric_crs(event_frame.metadata.crs)
        schema = event_frame.schema
        timestamp_col = schema.timestamp_col
        if schema.lat_col is None or schema.lon_col is None:
            raise ValueError("TrajectoryKinematicsStep needs lat/lon columns")
        lon_col, lat_col = schema.lon_col, schema.lat_col

        lf = event_frame.lazy_frame.sort(self.group_cols + [timestamp_col], maintain_order=True)
        prev_lon = pl.col(lon_col).shift(1).over(self.group_cols)
        prev_lat = pl.col(lat_col).shift(1).over(self.group_cols)
        prev_ts = pl.col(timestamp_col).shift(1).over(self.group_cols)
        dt_s = (pl.col(timestamp_col) - prev_ts).dt.total_seconds().alias("dt_s")
        step_distance_m = (
            ((pl.col(lon_col) - prev_lon) ** 2 + (pl.col(lat_col) - prev_lat) ** 2)
            .sqrt()
            .alias("step_distance_m")
        )
        speed_m_s = (
            pl.when((pl.col("dt_s") > 0) & (pl.col("dt_s") <= self.max_gap_s))
            .then(pl.col("step_distance_m") / pl.col("dt_s"))
            .otherwise(None)
            .alias("speed_m_s")
        )
        # Two with_columns calls: speed_m_s reads dt_s / step_distance_m,
        # which do not exist until the first call materialises them.
        lf = lf.with_columns([dt_s, step_distance_m])
        lf = lf.with_columns([speed_m_s])

        result = event_frame.with_lazy_frame(lf)
        for col, description in (
            ("step_distance_m", "Euclidean distance to the previous fix (m)"),
            ("dt_s", "Elapsed seconds since the previous fix"),
            ("speed_m_s", "step_distance_m / dt_s; null on first fix, dt<=0, or gap > max_gap_s"),
        ):
            result = result.register_feature(
                col,
                {"source_step": "TrajectoryKinematicsStep"},
                provenance=FeatureProvenance(
                    produced_by="TrajectoryKinematicsStep",
                    inputs=[timestamp_col, lon_col, lat_col],
                    tags={"temporal", "spatial"},
                    description=description,
                    metadata={"group_cols": self.group_cols, "max_gap_s": self.max_gap_s},
                ),
            )
        return result
