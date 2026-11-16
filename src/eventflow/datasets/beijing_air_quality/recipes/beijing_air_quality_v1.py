"""Baseline Beijing Multi-Site Air Quality recipe (v1).

Panel: station x hour. The station is the unit, the hour is the bin —
no grid, no crime columns, no spatial step: the 12 stations are
treated as separate entities and no feature ever mixes stations.

Pipeline:
- 1h time bins (the data are already hourly; binning validates the
  alignment and records the bin size in metadata).
- Panel completion with explicit zeros for the observation count only;
  the 11 measurement columns are declared as ``measure_cols``, so a
  missing reading stays null on generated pairs and on observed rows.
- Temporal components (hour of day, day of week, day of month, month,
  weekend) and cyclical calendar encodings from the timestamp.
- Temporal lags (1h, 24h, 168h) and trailing rolling means (24h, 168h)
  of the six pollutants, grouped by station. Both are duration-based
  exact joins / windows: a lag at t reads the same station's value at
  t - lag, and a missing value there stays null — absence stays
  unknown, it is never borrowed from another station or filled with 0.

Causality convention (docs/design-notes/phase1-causality.md): a
feature row at hour ``t`` predicts hour ``t + 1``; no feature may use
data after ``t``. Rolling means are closed-left (strictly before t);
lags read strictly earlier hours.
"""

from eventflow.core.pipeline import Pipeline, Step
from eventflow.core.registry import StepRegistry
from eventflow.core.schema import RecipeConfig
from eventflow.core.steps.panel import CompletePanelStep
from eventflow.core.steps.temporal import (
    CalendarEncodingStep,
    ExtractTemporalComponentsStep,
    MovingAverageStep,
    TemporalLagStep,
    TimeBinsStep,
)
from eventflow.datasets.beijing_air_quality.schema import (
    MEASUREMENT_COLS,
    POLLUTANT_COLS,
)
from eventflow.recipes.base import BaseRecipe

# Duration-mode lags and rolling windows, in hours: last hour,
# yesterday's same hour, last week's same hour.
LAG_HOURS: list[int] = [1, 24, 168]
ROLLING_WINDOW_HOURS: list[int] = [24, 168]


class BeijingAirQualityV1Recipe(BaseRecipe):
    """
    Baseline recipe for the Beijing Multi-Site Air Quality dataset.

    Builds station-hour features without any Chicago-specific
    assumption: the panel unit is the monitoring station, the bin is
    the hour, and only temporal steps apply (lags, rolling means,
    calendar encoding). See docs/design-notes/phase3-beijing.md for
    how the sensor panel maps onto the observation-support/completion
    contract and for the honest limit (regular hourly timestamps prove
    gaps / missing-stays-missing and per-station separation, not
    irregular sampling).
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
            config: Recipe configuration (``temporal.time_bin`` must be
                "1h": the panel is hourly by construction).
        """
        super().__init__(config, step_registry=step_registry)
        self.name = "beijing_air_quality_v1"

    def build_pipeline(self) -> Pipeline:
        """Build the transformation pipeline."""
        temporal_config = self.config.temporal
        if temporal_config.time_bin != "1h":
            raise ValueError(
                "beijing_air_quality_v1 is an hourly panel recipe: "
                f"temporal.time_bin must be '1h', got {temporal_config.time_bin!r}"
            )
        for source, enabled in (self.config.context or {}).items():
            if isinstance(enabled, dict) and enabled.get("enabled", False):
                raise ValueError(
                    "beijing_air_quality_v1 does not wire context enrichment "
                    f"(context.{source}.enabled is true); context sources are "
                    "not implemented for this recipe"
                )

        steps: list[Step] = [
            TimeBinsStep(bin_size="1h"),
            CompletePanelStep(
                unit_col="station",
                value_cols=["count"],
                measure_cols=MEASUREMENT_COLS,
                min_periods=1,
            ),
            ExtractTemporalComponentsStep(
                components=["hour_of_day", "day_of_week", "day_of_month", "month", "is_weekend"]
            ),
            CalendarEncodingStep(
                encoding_type="cyclical",
                components=["hour", "day_of_week", "month"],
            ),
            TemporalLagStep(
                value_cols=POLLUTANT_COLS,
                lag_periods=LAG_HOURS,
                time_unit="1h",
                group_cols=["station"],
            ),
            MovingAverageStep(
                value_cols=POLLUTANT_COLS,
                windows=ROLLING_WINDOW_HOURS,
                time_unit="1h",
                group_cols=["station"],
                closed="left",
            ),
        ]
        return Pipeline(steps)
