"""Baseline Chicago Crime recipe (v1)."""

import polars as pl

from eventflow.core.event_frame import EventFrame
from eventflow.core.pipeline import Pipeline, Step
from eventflow.core.registry import StepRegistry
from eventflow.core.schema import RecipeConfig
from eventflow.core.steps.categorical import CategoryCountsStep
from eventflow.core.steps.density import KernelDensityStep, SpaceTimeKernelDensityStep
from eventflow.core.steps.dynamics import (
    BurstinessStep,
    EWMAMeanStep,
    TimeSinceLastEventStep,
    TrendStep,
    ZeroFeaturesStep,
)
from eventflow.core.steps.hawkes import HawkesExcitationStep
from eventflow.core.steps.near_repeat import NearRepeatStep
from eventflow.core.steps.panel import CompletePanelStep
from eventflow.core.steps.spatial import (
    AssignToGridStep,
    MultiScaleSpatialLagStep,
    TransformCRSStep,
)
from eventflow.core.steps.temporal import (
    CalendarEncodingStep,
    ExtractTemporalComponentsStep,
    FillNullStep,
    MovingAverageStep,
    TemporalLagStep,
    TimeBinsStep,
)
from eventflow.datasets.chicago_crime.schema import CHICAGO_CRIME_SCHEMA
from eventflow.recipes.base import BaseRecipe

# Columns produced by CategoryCountsStep (naming convention: count_<col>_<value>,
# share_<col>_<value>, <col>_entropy). ChicagoCountStep carries them through
# the aggregation with first() — they are already per (grid_id, time_bin),
# so first() is exact, not a lossy collapse.
_CATEGORY_COLUMN_PREFIXES = ("count_", "share_")
_CATEGORY_COLUMN_SUFFIXES = ("_entropy",)


class ChicagoCountStep(Step):
    """Count events per cell/bin and retain cell centers and bin-start timestamps."""

    def __init__(self, size_m: float, origin: tuple[float, float]) -> None:
        self.size_m = size_m
        self.origin = origin

    def run(self, ef: EventFrame) -> EventFrame:
        """Aggregate events while keeping the output schema usable downstream."""
        lon, lat = ef.schema.lon_col, ef.schema.lat_col
        if lon is None or lat is None:
            raise ValueError("Chicago aggregation requires projected coordinate columns")
        # Per-category columns attached upstream (CategoryCountsStep) are
        # already per (grid_id, time_bin); carry them through with first().
        category_cols = [
            c
            for c in ef.lazy_frame.collect_schema().names()
            if c.startswith(_CATEGORY_COLUMN_PREFIXES) or c.endswith(_CATEGORY_COLUMN_SUFFIXES)
        ]
        lf = ef.lazy_frame.group_by(["grid_id", "time_bin"]).agg(
            pl.len().alias("count"),
            pl.col("grid_row").first(),
            pl.col("grid_col").first(),
            pl.col(lon).first(),
            pl.col(lat).first(),
            *[pl.col(c).first() for c in category_cols],
        )
        lf = lf.with_columns(
            (
                ((pl.col(lon) - self.origin[0]) / self.size_m).floor() * self.size_m
                + self.origin[0]
                + self.size_m / 2
            ).alias(lon),
            (
                ((pl.col(lat) - self.origin[1]) / self.size_m).floor() * self.size_m
                + self.origin[1]
                + self.size_m / 2
            ).alias(lat),
            pl.col("time_bin").alias(ef.schema.timestamp_col),
        ).sort(["time_bin", "grid_id"])
        result = ef.with_lazy_frame(lf).with_schema(
            numeric_cols=["count", "grid_id", "grid_row", "grid_col", lon, lat] + category_cols,
            categorical_cols=[],
        )
        return result.register_feature(
            "count", {"source_step": "ChicagoCountStep", "inputs": ["grid_id", "time_bin"]}
        )


class ChicagoCrimeV1Recipe(BaseRecipe):
    """
    Baseline recipe for Chicago Crime dataset.

    Creates spatial-temporal features without external context:
    - 300m x 300m spatial grid
    - 6-hour time bins
    - Temporal components (hour, day of week, etc.)
    - Count aggregations per grid cell and time bin
    - Phase 1 generic features (all opt-in via the features section):
      per-category counts/shares/entropy, explicit-zero panel completion,
      lags and rolling means over the declared aggregation windows, EWMA,
      cyclical calendar encodings, queen spatial lag of the shortest lag,
      trend, time since last event, burstiness and zero features.
    - Phase 2 space-time features (all opt-in via the features section):
      multi-scale queen-ring spatial lag of the shortest lag, Gaussian KDE
      of the shortest lag, separable space-time KDE of the count,
      near-repeat ring counts, and fixed-kernel Hawkes excitation of the
      count. See docs/design-notes/phase2-spacetime-v2.md.

    Causality convention (docs/design-notes/phase1-causality.md): a feature
    row at bin ``t`` predicts bin ``t + 1``; no feature may use data after
    ``t``. The panel is completed with explicit zeros before any window or
    lag, so trailing statistics see a dense panel, not a sparse one.
    """

    def __init__(
        self,
        config: RecipeConfig,
        *,
        step_registry: StepRegistry | None = None,
    ) -> None:
        """
        Initialize recipe.

        Args:
            config: Recipe configuration
        """
        super().__init__(config, step_registry=step_registry)
        self.name = "chicago_crime_v1"

    def build_pipeline(self) -> Pipeline:
        """Build the transformation pipeline."""
        grid_config = self.config.grid
        temporal_config = self.config.temporal
        features = self.config.features

        # Phase 0 honesty, kept in phase 1: no declared feature may be
        # silently dropped. require_implemented() now enforces the
        # cross-key rules (spatial_lag/trend need windows, recency /
        # zero_features / EWMA need a completed panel).
        features.require_implemented("chicago_crime_v1")
        if not features.count_features:
            raise ValueError(
                "chicago_crime_v1 always counts events per cell/bin; "
                "count_features: false is not supported by this recipe"
            )
        # NB: grid.buffer_m is rejected by GridConfig itself (extra="forbid").
        for source, enabled in (self.config.context or {}).items():
            if isinstance(enabled, dict) and enabled.get("enabled", False):
                raise ValueError(
                    f"chicago_crime_v1 does not wire context enrichment "
                    f"(context.{source}.enabled is true); context sources are "
                    "not implemented for this recipe"
                )

        size_m = float(grid_config.size_m)
        origin_config = grid_config.origin
        if len(origin_config) != 2:
            raise ValueError("grid.origin must contain x and y in projected meters")
        origin = (float(origin_config[0]), float(origin_config[1]))
        timestamp_col = CHICAGO_CRIME_SCHEMA.timestamp_col
        window_days = features.window_days()

        steps: list[Step] = [
            TransformCRSStep(target_crs="EPSG:26971"),
            AssignToGridStep(grid_size_m=size_m, origin=origin),
            TimeBinsStep(bin_size=temporal_config.time_bin),
        ]
        if features.categorical_features:
            steps.append(
                CategoryCountsStep(
                    category_cols=features.categorical_features,
                    unit_col="grid_id",
                    time_col="time_bin",
                    top_n=features.categorical_top_n,
                )
            )
        steps.append(ChicagoCountStep(size_m, origin))
        if features.complete_panel:
            steps.append(
                CompletePanelStep(
                    unit_col="grid_id",
                    value_cols=["count"],
                    value_prefixes=["count"],
                    min_periods=features.panel_min_periods,
                )
            )
        steps.append(ExtractTemporalComponentsStep(components=temporal_config.components))
        if features.cyclical_encoding:
            steps.append(
                CalendarEncodingStep(
                    encoding_type="cyclical",
                    components=["hour", "day_of_week", "month"],
                )
            )
        if window_days:
            steps.append(
                TemporalLagStep(
                    value_cols=["count"],
                    lag_periods=window_days,
                    time_unit="1d",
                    group_cols=["grid_id"],
                )
            )
            steps.append(
                MovingAverageStep(
                    value_cols=["count"],
                    windows=window_days,
                    time_unit="1d",
                    group_cols=["grid_id"],
                    closed="left",
                )
            )
        if features.ewma_halflife_days is not None:
            steps.append(
                EWMAMeanStep(
                    value_cols=["count"],
                    halflife_days=features.ewma_halflife_days,
                    group_cols=["grid_id"],
                )
            )
        if features.trend:
            shortest, longest = window_days[0], window_days[-1]
            steps.append(
                TrendStep(
                    recent_col=f"count_ma_{shortest}",
                    long_col=f"count_ma_{longest}",
                )
            )
        if features.spatial_lag or features.kde_bandwidth_m is not None:
            shortest = window_days[0]
            lag_col = f"count_lag_{shortest}"
            # Boundary policy: before the panel start there is no earlier
            # bin, so the lagged count is treated as 0 ("no events
            # recorded") instead of null — the spatial lag and the KDE
            # require finite non-null values. Only the lag feeding those
            # steps is filled; the other lags keep their nulls.
            steps.append(FillNullStep(value_cols=[lag_col], fill_value=0))
        if features.spatial_lag:
            shortest = window_days[0]
            lag_col = f"count_lag_{shortest}"
            # Phase 2: multi-scale queen-ring lag (rings 1-3 by default);
            # ring 1 reproduces the phase-1 single-scale lag exactly.
            steps.append(
                MultiScaleSpatialLagStep(
                    value_cols=[lag_col],
                    neighbor_col="grid_id",
                    rings=features.spatial_lag_rings,
                    group_cols=[timestamp_col],
                )
            )
        if features.kde_bandwidth_m is not None:
            shortest = window_days[0]
            lag_col = f"count_lag_{shortest}"
            # Phase 2: Gaussian KDE of the lagged count (same convention
            # as the spatial lag — the density at bin t describes
            # neighbours' counts from the previous day).
            steps.append(
                KernelDensityStep(
                    value_col=lag_col,
                    bandwidth_m=features.kde_bandwidth_m,
                    group_cols=[timestamp_col],
                )
            )
        if features.stkde_bandwidth_m is not None:
            # Phase 2: separable space-time KDE of the count (Gaussian in
            # space, exponential decay in time, past bins including the
            # current one — same precedent as the EWMA).
            # require_implemented() enforces both-or-neither for the
            # STKDE pair, so the half-life is not None here.
            stkde_halflife = features.stkde_halflife_days
            assert stkde_halflife is not None
            steps.append(
                SpaceTimeKernelDensityStep(
                    value_col="count",
                    bandwidth_m=features.stkde_bandwidth_m,
                    halflife_days=stkde_halflife,
                    # The canonical bin datetime: the pre-completion
                    # "time_bin" column is null on generated (explicit-
                    # zero) panel rows, so grouping/sorting by it would
                    # silently corrupt the time smoothing (phase 2).
                    time_col=timestamp_col,
                )
            )
        if features.near_repeat:
            # Phase 2: near-repeat ring counts over trailing 7 d and
            # 7-14 d windows, closed-left (strictly before t). Same
            # time-column caveat as STKDE above: use the canonical bin
            # datetime, not the pre-completion "time_bin" column.
            steps.append(NearRepeatStep(value_col="count", time_col=timestamp_col))
        if features.hawkes_halflife_days is not None:
            # Phase 2: fixed-kernel Hawkes excitation of the count,
            # strictly past bins only (no fitted Hawkes process).
            steps.append(
                HawkesExcitationStep(
                    value_col="count",
                    halflife_days=features.hawkes_halflife_days,
                    group_cols=["grid_id"],
                )
            )
        if features.recency:
            steps.append(TimeSinceLastEventStep(value_col="count", group_cols=["grid_id"]))
            steps.append(BurstinessStep(value_col="count", window="28d", group_cols=["grid_id"]))
        if features.zero_features:
            steps.append(ZeroFeaturesStep(value_col="count", group_cols=["grid_id"]))
        return Pipeline(steps)
