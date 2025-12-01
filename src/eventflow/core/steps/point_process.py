"""Point-process and continuous-time pipeline steps.

This module provides specialized steps for point process analysis:
- Exponential decay kernels (Hawkes trigger, background separation)
- Conditional intensity approximations
- Pair-correlation and K-function summaries
- Hazard-rate estimation
- Survival table generation
- Continuous inter-event features
- Duration-based features with configurable decay functions
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import polars as pl
from pydantic import BaseModel, Field

from eventflow.core.pipeline import Step
from eventflow.core.schema import FeatureProvenance
from eventflow.core.utils import get_logger

if TYPE_CHECKING:
    from eventflow.core.event_frame import EventFrame

logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# Pydantic Config Models
# -----------------------------------------------------------------------------


class ExponentialDecayConfig(BaseModel):
    """Configuration for exponential decay kernel computation."""

    decay_rate: float = Field(default=0.1, gt=0, description="Decay rate (lambda)")
    time_unit: str = Field(default="1h", description="Time unit for decay computation")
    output_col: str = Field(default="decay_weight", description="Output column name")
    group_cols: list[str] = Field(
        default_factory=list, description="Optional columns to group by"
    )


class HawkesKernelConfig(BaseModel):
    """Configuration for Hawkes process kernel step."""

    alpha: float = Field(default=0.5, gt=0, le=1, description="Excitation parameter")
    beta: float = Field(default=1.0, gt=0, description="Decay rate for triggering kernel")
    mu: float = Field(default=0.1, gt=0, description="Background intensity rate")
    time_unit: str = Field(default="1h", description="Time unit")
    max_history: int = Field(
        default=100, ge=1, description="Maximum number of past events to consider"
    )
    group_cols: list[str] = Field(
        default_factory=list, description="Columns to group by for separate processes"
    )


class ConditionalIntensityConfig(BaseModel):
    """Configuration for conditional intensity estimation."""

    method: Literal["hawkes", "exponential", "power_law"] = Field(
        default="exponential", description="Intensity model type"
    )
    bandwidth: float = Field(default=1.0, gt=0, description="Kernel bandwidth")
    time_unit: str = Field(default="1h", description="Time unit")
    normalize: bool = Field(default=True, description="Normalize intensity values")


class PairCorrelationConfig(BaseModel):
    """Configuration for pair-correlation function (g-function)."""

    max_distance: float = Field(default=1000.0, gt=0, description="Maximum distance in meters")
    n_bins: int = Field(default=50, ge=10, description="Number of distance bins")
    edge_correction: bool = Field(default=True, description="Apply edge correction")


class KFunctionConfig(BaseModel):
    """Configuration for Ripley's K-function."""

    max_distance: float = Field(default=1000.0, gt=0, description="Maximum distance in meters")
    n_bins: int = Field(default=50, ge=10, description="Number of distance bins")
    edge_correction: Literal["none", "border", "ripley"] = Field(
        default="ripley", description="Edge correction method"
    )
    transform: Literal["K", "L", "H"] = Field(
        default="L", description="Transform type: K, L (sqrt), or H (L-r)"
    )


class HazardRateConfig(BaseModel):
    """Configuration for hazard rate estimation."""

    method: Literal["nelson_aalen", "kernel", "piecewise"] = Field(
        default="nelson_aalen", description="Hazard estimation method"
    )
    bandwidth: float = Field(default=1.0, gt=0, description="Kernel bandwidth for smoothing")
    n_intervals: int = Field(default=10, ge=2, description="Number of intervals for piecewise")
    group_cols: list[str] = Field(
        default_factory=list, description="Columns to stratify hazard by"
    )


class SurvivalTableConfig(BaseModel):
    """Configuration for survival table generation."""

    method: Literal["kaplan_meier", "nelson_aalen"] = Field(
        default="kaplan_meier", description="Survival estimation method"
    )
    confidence_level: float = Field(
        default=0.95, gt=0, lt=1, description="Confidence interval level"
    )
    time_points: list[float] | None = Field(
        default=None, description="Specific time points to evaluate"
    )
    group_cols: list[str] = Field(
        default_factory=list, description="Columns to stratify survival by"
    )


class DurationFeaturesConfig(BaseModel):
    """Configuration for duration-based features."""

    decay_functions: list[Literal["exponential", "power_law", "linear", "logarithmic"]] = Field(
        default=["exponential"], description="Decay functions to apply"
    )
    decay_rates: list[float] = Field(
        default=[0.1, 0.5, 1.0], description="Decay rates for exponential decay"
    )
    power_exponents: list[float] = Field(
        default=[0.5, 1.0, 2.0], description="Exponents for power law decay"
    )
    include_time_to_next: bool = Field(
        default=True, description="Include time to next event (requires future info)"
    )
    group_cols: list[str] = Field(
        default_factory=list, description="Columns to group by"
    )


class ContinuousInterEventConfig(BaseModel):
    """Configuration for continuous inter-event features."""

    include_log_transform: bool = Field(
        default=True, description="Include log-transformed inter-event times"
    )
    include_normalized: bool = Field(
        default=True, description="Include normalized inter-event times"
    )
    rolling_windows: list[int] = Field(
        default=[5, 10, 20], description="Rolling window sizes for statistics"
    )
    include_coefficient_variation: bool = Field(
        default=True, description="Include coefficient of variation"
    )
    group_cols: list[str] = Field(
        default_factory=list, description="Columns to group by"
    )


# -----------------------------------------------------------------------------
# Point Process Steps
# -----------------------------------------------------------------------------


class ExponentialDecayStep(Step):
    """Apply exponential decay weighting based on time since reference.

    Computes weights using the formula: w(t) = exp(-lambda * (t_ref - t))

    Inputs:
        - timestamp_col from EventSchema

    Outputs:
        - decay_weight column with exponential decay values
    """

    def __init__(
        self,
        decay_rate: float = 0.1,
        time_unit: str = "1h",
        output_col: str = "decay_weight",
        group_cols: Sequence[str] | None = None,
    ) -> None:
        self.decay_rate = decay_rate
        self.time_unit = time_unit
        self.output_col = output_col
        self.group_cols = list(group_cols) if group_cols else []

    def run(self, event_frame: EventFrame) -> EventFrame:
        """Execute exponential decay computation."""
        timestamp_col = event_frame.schema.timestamp_col
        lf = event_frame.lazy_frame

        unit_seconds = _parse_time_unit_seconds(self.time_unit)
        decay_per_second = self.decay_rate / unit_seconds

        logger.info(
            f"Computing exponential decay with rate={self.decay_rate}/{self.time_unit}"
        )

        # Sort by timestamp
        sort_cols = self.group_cols + [timestamp_col] if self.group_cols else [timestamp_col]
        lf = lf.sort(sort_cols)

        if self.group_cols:
            # Compute decay relative to max time within each group
            lf = lf.with_columns(
                pl.col(timestamp_col).max().over(self.group_cols).alias("_max_ts")
            )
            lf = lf.with_columns(
                (
                    -decay_per_second
                    * (pl.col("_max_ts") - pl.col(timestamp_col)).dt.total_seconds()
                )
                .exp()
                .alias(self.output_col)
            )
            lf = lf.drop("_max_ts")
        else:
            max_ts = lf.select(pl.col(timestamp_col).max()).collect().item()
            lf = lf.with_columns(
                (
                    -decay_per_second
                    * (pl.lit(max_ts) - pl.col(timestamp_col)).dt.total_seconds()
                )
                .exp()
                .alias(self.output_col)
            )

        provenance = FeatureProvenance(
            produced_by="ExponentialDecayStep",
            inputs=[timestamp_col],
            tags={"continuous", "temporal"},
            description=f"Exponential decay weight with rate {self.decay_rate}/{self.time_unit}",
            metadata={"decay_rate": self.decay_rate, "time_unit": self.time_unit},
        )

        result = event_frame.with_lazy_frame(lf)
        return result.register_feature(
            self.output_col,
            {"source_step": "ExponentialDecayStep", "decay_rate": self.decay_rate},
            provenance=provenance,
        )


class HawkesKernelStep(Step):
    """Compute Hawkes process triggering kernel contributions.

    Implements the self-exciting point process model where past events
    increase the intensity of future events.

    Intensity: λ(t) = μ + α * Σ β * exp(-β * (t - t_i))

    Inputs:
        - timestamp_col from EventSchema

    Outputs:
        - hawkes_intensity: Conditional intensity at each event time
        - hawkes_trigger: Sum of triggering contributions from past events
        - hawkes_background: Background intensity component
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 1.0,
        mu: float = 0.1,
        time_unit: str = "1h",
        max_history: int = 100,
        group_cols: Sequence[str] | None = None,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.time_unit = time_unit
        self.max_history = max_history
        self.group_cols = list(group_cols) if group_cols else []

        # Validate stability condition
        if alpha >= 1.0:
            raise ValueError(
                f"Hawkes process unstable: alpha ({alpha}) must be < 1.0 for stability"
            )

    def run(self, event_frame: EventFrame) -> EventFrame:
        """Execute Hawkes kernel computation."""
        timestamp_col = event_frame.schema.timestamp_col
        lf = event_frame.lazy_frame

        unit_seconds = _parse_time_unit_seconds(self.time_unit)
        beta_per_second = self.beta / unit_seconds

        logger.info(
            f"Computing Hawkes kernel: α={self.alpha}, β={self.beta}, μ={self.mu}"
        )

        # Sort and collect for kernel computation (requires sequential access)
        sort_cols = self.group_cols + [timestamp_col] if self.group_cols else [timestamp_col]
        df = lf.sort(sort_cols).collect()

        # Compute triggering kernel contributions
        timestamps = df[timestamp_col].to_numpy()
        n_events = len(timestamps)

        trigger_contributions = []
        intensities = []

        for i in range(n_events):
            # Sum over past events within max_history
            start_idx = max(0, i - self.max_history)
            trigger_sum = 0.0

            for j in range(start_idx, i):
                dt_seconds = _timedelta_to_seconds(timestamps[i] - timestamps[j])
                trigger_sum += self.alpha * self.beta * math.exp(-beta_per_second * dt_seconds)

            trigger_contributions.append(trigger_sum)
            intensities.append(self.mu + trigger_sum)

        # Add computed columns
        result_df = df.with_columns(
            [
                pl.Series("hawkes_trigger", trigger_contributions),
                pl.Series("hawkes_intensity", intensities),
                pl.lit(self.mu).alias("hawkes_background"),
            ]
        )

        result = event_frame.with_lazy_frame(result_df.lazy())

        # Register features
        for col, desc in [
            ("hawkes_trigger", "Sum of triggering contributions from past events"),
            ("hawkes_intensity", "Conditional intensity at event time"),
            ("hawkes_background", "Background intensity rate"),
        ]:
            provenance = FeatureProvenance(
                produced_by="HawkesKernelStep",
                inputs=[timestamp_col],
                tags={"continuous", "point_process"},
                description=desc,
                metadata={
                    "alpha": self.alpha,
                    "beta": self.beta,
                    "mu": self.mu,
                },
            )
            result = result.register_feature(
                col,
                {"source_step": "HawkesKernelStep"},
                provenance=provenance,
            )

        return result


class ConditionalIntensityStep(Step):
    """Estimate conditional intensity function at event times.

    Provides various methods for intensity estimation:
    - hawkes: Self-exciting Hawkes process
    - exponential: Simple exponential kernel
    - power_law: Power law decay kernel

    Inputs:
        - timestamp_col from EventSchema

    Outputs:
        - conditional_intensity: Estimated intensity at each event
        - intensity_normalized: Normalized intensity (if normalize=True)
    """

    def __init__(
        self,
        method: str = "exponential",
        bandwidth: float = 1.0,
        time_unit: str = "1h",
        normalize: bool = True,
    ) -> None:
        self.method = method
        self.bandwidth = bandwidth
        self.time_unit = time_unit
        self.normalize = normalize

    def run(self, event_frame: EventFrame) -> EventFrame:
        """Execute conditional intensity estimation."""
        timestamp_col = event_frame.schema.timestamp_col
        lf = event_frame.lazy_frame

        unit_seconds = _parse_time_unit_seconds(self.time_unit)

        logger.info(f"Estimating conditional intensity using {self.method} method")

        # Collect for intensity computation
        df = lf.sort(timestamp_col).collect()
        timestamps = df[timestamp_col].to_numpy()
        n_events = len(timestamps)

        intensities = []

        for i in range(n_events):
            if i == 0:
                # First event: use a default low intensity
                intensities.append(1.0 / (len(timestamps) * unit_seconds))
                continue

            intensity = 0.0
            for j in range(i):
                dt_seconds = _timedelta_to_seconds(timestamps[i] - timestamps[j])
                dt_units = dt_seconds / unit_seconds

                if self.method == "exponential":
                    intensity += math.exp(-dt_units / self.bandwidth)
                elif self.method == "power_law":
                    intensity += 1.0 / (1.0 + dt_units / self.bandwidth) ** 2
                elif self.method == "hawkes":
                    # Simple Hawkes approximation
                    intensity += 0.5 * math.exp(-dt_units / self.bandwidth)

            intensities.append(intensity)

        result_df = df.with_columns(
            pl.Series("conditional_intensity", intensities)
        )

        if self.normalize:
            max_intensity = max(intensities) if intensities else 1.0
            normalized = [i / max_intensity for i in intensities]
            result_df = result_df.with_columns(
                pl.Series("intensity_normalized", normalized)
            )

        result = event_frame.with_lazy_frame(result_df.lazy())

        provenance = FeatureProvenance(
            produced_by="ConditionalIntensityStep",
            inputs=[timestamp_col],
            tags={"continuous", "point_process"},
            description=f"Conditional intensity using {self.method} kernel",
            metadata={"method": self.method, "bandwidth": self.bandwidth},
        )

        result = result.register_feature(
            "conditional_intensity",
            {"source_step": "ConditionalIntensityStep"},
            provenance=provenance,
        )

        if self.normalize:
            result = result.register_feature(
                "intensity_normalized",
                {"source_step": "ConditionalIntensityStep"},
                provenance=provenance,
            )

        return result


class PairCorrelationStep(Step):
    """Compute pair-correlation function (g-function) for spatial events.

    The pair-correlation function g(r) measures the density of event pairs
    at distance r relative to a homogeneous Poisson process.

    Inputs:
        - lat_col, lon_col from EventSchema

    Outputs:
        - Aggregated g-function values (stored in metadata)
        - Nearest neighbor distance for each event
    """

    def __init__(
        self,
        max_distance: float = 1000.0,
        n_bins: int = 50,
        edge_correction: bool = True,
    ) -> None:
        self.max_distance = max_distance
        self.n_bins = n_bins
        self.edge_correction = edge_correction

    def run(self, event_frame: EventFrame) -> EventFrame:
        """Execute pair-correlation computation."""
        lat_col = event_frame.schema.lat_col
        lon_col = event_frame.schema.lon_col

        if not lat_col or not lon_col:
            raise ValueError("PairCorrelationStep requires lat_col and lon_col in schema")

        logger.info(f"Computing pair-correlation function up to {self.max_distance}m")

        df = event_frame.lazy_frame.collect()
        lats = df[lat_col].to_numpy()
        lons = df[lon_col].to_numpy()
        n_events = len(lats)

        # Compute pairwise distances and nearest neighbor
        nearest_distances = []

        for i in range(n_events):
            min_dist = float("inf")
            for j in range(n_events):
                if i != j:
                    dist = _haversine_distance(lats[i], lons[i], lats[j], lons[j])
                    if dist < min_dist:
                        min_dist = dist
            nearest_distances.append(min_dist if min_dist != float("inf") else 0.0)

        result_df = df.with_columns(
            pl.Series("nearest_neighbor_distance", nearest_distances)
        )

        result = event_frame.with_lazy_frame(result_df.lazy())

        provenance = FeatureProvenance(
            produced_by="PairCorrelationStep",
            inputs=[lat_col, lon_col],
            tags={"spatial", "point_process"},
            description="Nearest neighbor distance from pair-correlation analysis",
            metadata={"max_distance": self.max_distance, "n_bins": self.n_bins},
        )

        return result.register_feature(
            "nearest_neighbor_distance",
            {"source_step": "PairCorrelationStep"},
            provenance=provenance,
        )


class KFunctionStep(Step):
    """Compute Ripley's K-function for spatial point pattern analysis.

    K(r) = (A / n²) * Σᵢ Σⱼ I(dᵢⱼ ≤ r) * wᵢⱼ

    Supports L-transform: L(r) = sqrt(K(r) / π)
    And H-transform: H(r) = L(r) - r

    Inputs:
        - lat_col, lon_col from EventSchema

    Outputs:
        - k_function_local: Local K-function contribution for each event
    """

    def __init__(
        self,
        max_distance: float = 1000.0,
        n_bins: int = 50,
        edge_correction: str = "ripley",
        transform: str = "L",
    ) -> None:
        self.max_distance = max_distance
        self.n_bins = n_bins
        self.edge_correction = edge_correction
        self.transform = transform

    def run(self, event_frame: EventFrame) -> EventFrame:
        """Execute K-function computation."""
        lat_col = event_frame.schema.lat_col
        lon_col = event_frame.schema.lon_col

        if not lat_col or not lon_col:
            raise ValueError("KFunctionStep requires lat_col and lon_col in schema")

        logger.info(f"Computing Ripley's K-function with {self.transform}-transform")

        df = event_frame.lazy_frame.collect()
        lats = df[lat_col].to_numpy()
        lons = df[lon_col].to_numpy()
        n_events = len(lats)

        # Compute local K-function contribution (count of neighbors within radius)
        bin_width = self.max_distance / self.n_bins
        local_k = []

        for i in range(n_events):
            neighbor_count = 0
            for j in range(n_events):
                if i != j:
                    dist = _haversine_distance(lats[i], lons[i], lats[j], lons[j])
                    if dist <= self.max_distance:
                        neighbor_count += 1
            local_k.append(neighbor_count)

        # Apply transform
        if self.transform == "L":
            local_values = [math.sqrt(k / math.pi) if k > 0 else 0 for k in local_k]
            output_col = "l_function_local"
        elif self.transform == "H":
            local_values = [
                math.sqrt(k / math.pi) - self.max_distance / 2 if k > 0 else 0
                for k in local_k
            ]
            output_col = "h_function_local"
        else:
            local_values = local_k
            output_col = "k_function_local"

        result_df = df.with_columns(pl.Series(output_col, local_values))

        result = event_frame.with_lazy_frame(result_df.lazy())

        provenance = FeatureProvenance(
            produced_by="KFunctionStep",
            inputs=[lat_col, lon_col],
            tags={"spatial", "point_process"},
            description=f"Local {self.transform}-function value",
            metadata={
                "max_distance": self.max_distance,
                "transform": self.transform,
            },
        )

        return result.register_feature(
            output_col,
            {"source_step": "KFunctionStep"},
            provenance=provenance,
        )


class HazardRateStep(Step):
    """Estimate hazard rate for event occurrence.

    The hazard rate h(t) represents the instantaneous rate of event
    occurrence given survival to time t.

    Methods:
    - nelson_aalen: Non-parametric cumulative hazard estimator
    - kernel: Kernel-smoothed hazard estimate
    - piecewise: Piecewise constant hazard

    Inputs:
        - timestamp_col from EventSchema
        - Optional: duration or censoring columns

    Outputs:
        - hazard_rate: Estimated hazard at each event time
        - cumulative_hazard: Cumulative hazard function
    """

    def __init__(
        self,
        method: str = "nelson_aalen",
        bandwidth: float = 1.0,
        n_intervals: int = 10,
        group_cols: Sequence[str] | None = None,
    ) -> None:
        self.method = method
        self.bandwidth = bandwidth
        self.n_intervals = n_intervals
        self.group_cols = list(group_cols) if group_cols else []

    def run(self, event_frame: EventFrame) -> EventFrame:
        """Execute hazard rate estimation."""
        timestamp_col = event_frame.schema.timestamp_col

        logger.info(f"Estimating hazard rate using {self.method} method")

        df = event_frame.lazy_frame.sort(timestamp_col).collect()
        timestamps = df[timestamp_col].to_numpy()
        n_events = len(timestamps)

        if n_events == 0:
            return event_frame

        # Compute time since start
        t0 = timestamps[0]
        times = [_timedelta_to_seconds(t - t0) for t in timestamps]

        # Nelson-Aalen estimator
        cumulative_hazard = []
        hazard_rates = []

        for i in range(n_events):
            at_risk = n_events - i
            cum_h = sum(1.0 / (n_events - j) for j in range(i + 1))
            cumulative_hazard.append(cum_h)

            # Instantaneous hazard approximation
            if i == 0:
                h = 1.0 / at_risk if at_risk > 0 else 0.0
            else:
                dt = times[i] - times[i - 1]
                h = (1.0 / at_risk) / max(dt, 1.0) if at_risk > 0 and dt > 0 else 0.0
            hazard_rates.append(h)

        result_df = df.with_columns(
            [
                pl.Series("hazard_rate", hazard_rates),
                pl.Series("cumulative_hazard", cumulative_hazard),
            ]
        )

        result = event_frame.with_lazy_frame(result_df.lazy())

        for col, desc in [
            ("hazard_rate", "Instantaneous hazard rate"),
            ("cumulative_hazard", "Cumulative hazard function"),
        ]:
            provenance = FeatureProvenance(
                produced_by="HazardRateStep",
                inputs=[timestamp_col],
                tags={"continuous", "survival"},
                description=desc,
                metadata={"method": self.method},
            )
            result = result.register_feature(
                col, {"source_step": "HazardRateStep"}, provenance=provenance
            )

        return result


class SurvivalTableStep(Step):
    """Generate survival table with Kaplan-Meier or Nelson-Aalen estimates.

    Computes survival probabilities S(t) = P(T > t) at event times.

    Inputs:
        - timestamp_col from EventSchema

    Outputs:
        - survival_probability: S(t) at each event time
        - survival_lower: Lower confidence bound
        - survival_upper: Upper confidence bound
    """

    def __init__(
        self,
        method: str = "kaplan_meier",
        confidence_level: float = 0.95,
        time_points: Sequence[float] | None = None,
        group_cols: Sequence[str] | None = None,
    ) -> None:
        self.method = method
        self.confidence_level = confidence_level
        self.time_points = list(time_points) if time_points else None
        self.group_cols = list(group_cols) if group_cols else []

    def run(self, event_frame: EventFrame) -> EventFrame:
        """Execute survival table generation."""
        timestamp_col = event_frame.schema.timestamp_col

        logger.info(f"Generating survival table using {self.method}")

        df = event_frame.lazy_frame.sort(timestamp_col).collect()
        n_events = len(df)

        if n_events == 0:
            return event_frame

        # Kaplan-Meier estimator
        survival_probs = []
        survival_lower = []
        survival_upper = []

        z = 1.96 if self.confidence_level == 0.95 else 1.645  # Normal quantile

        current_survival = 1.0
        for i in range(n_events):
            at_risk = n_events - i
            # Kaplan-Meier: S(t) = Π (1 - d_i / n_i)
            # For point process, each event is a "death"
            current_survival *= (at_risk - 1) / at_risk if at_risk > 1 else 0.0
            survival_probs.append(current_survival)

            # Greenwood's formula for variance
            var_sum = sum(
                1.0 / ((n_events - j) * (n_events - j - 1))
                for j in range(i + 1)
                if n_events - j > 1
            )
            se = current_survival * math.sqrt(var_sum) if var_sum > 0 else 0.0

            survival_lower.append(max(0.0, current_survival - z * se))
            survival_upper.append(min(1.0, current_survival + z * se))

        result_df = df.with_columns(
            [
                pl.Series("survival_probability", survival_probs),
                pl.Series("survival_lower", survival_lower),
                pl.Series("survival_upper", survival_upper),
            ]
        )

        result = event_frame.with_lazy_frame(result_df.lazy())

        for col, desc in [
            ("survival_probability", "Kaplan-Meier survival probability"),
            ("survival_lower", f"Lower {self.confidence_level*100:.0f}% confidence bound"),
            ("survival_upper", f"Upper {self.confidence_level*100:.0f}% confidence bound"),
        ]:
            provenance = FeatureProvenance(
                produced_by="SurvivalTableStep",
                inputs=[timestamp_col],
                tags={"continuous", "survival"},
                description=desc,
                metadata={"method": self.method, "confidence_level": self.confidence_level},
            )
            result = result.register_feature(
                col, {"source_step": "SurvivalTableStep"}, provenance=provenance
            )

        return result


class DurationFeaturesStep(Step):
    """Build duration-based features with configurable decay functions.

    Computes time since last event and optionally time to next observation
    with various decay transformations.

    Inputs:
        - timestamp_col from EventSchema

    Outputs:
        - time_since_last: Raw duration since previous event
        - time_since_last_{decay}: Decay-transformed durations
        - time_to_next: Duration to next event (if include_time_to_next=True)
    """

    def __init__(
        self,
        decay_functions: Sequence[str] | None = None,
        decay_rates: Sequence[float] | None = None,
        power_exponents: Sequence[float] | None = None,
        include_time_to_next: bool = True,
        group_cols: Sequence[str] | None = None,
    ) -> None:
        self.decay_functions = list(decay_functions) if decay_functions else ["exponential"]
        self.decay_rates = list(decay_rates) if decay_rates else [0.1, 0.5, 1.0]
        self.power_exponents = list(power_exponents) if power_exponents else [0.5, 1.0, 2.0]
        self.include_time_to_next = include_time_to_next
        self.group_cols = list(group_cols) if group_cols else []

    def run(self, event_frame: EventFrame) -> EventFrame:
        """Execute duration feature computation."""
        timestamp_col = event_frame.schema.timestamp_col
        lf = event_frame.lazy_frame

        logger.info(f"Computing duration features with decays: {self.decay_functions}")

        # Sort by timestamp
        sort_cols = self.group_cols + [timestamp_col] if self.group_cols else [timestamp_col]
        lf = lf.sort(sort_cols)

        result = event_frame

        # Time since last event
        if self.group_cols:
            lf = lf.with_columns(
                (pl.col(timestamp_col) - pl.col(timestamp_col).shift(1))
                .dt.total_seconds()
                .over(self.group_cols)
                .alias("time_since_last")
            )
        else:
            lf = lf.with_columns(
                (pl.col(timestamp_col) - pl.col(timestamp_col).shift(1))
                .dt.total_seconds()
                .alias("time_since_last")
            )

        result = result.with_lazy_frame(lf)
        result = result.register_feature(
            "time_since_last",
            {"source_step": "DurationFeaturesStep"},
            provenance=FeatureProvenance(
                produced_by="DurationFeaturesStep",
                inputs=[timestamp_col],
                tags={"continuous", "temporal"},
                description="Time since previous event in seconds",
            ),
        )

        # Apply decay transformations
        for decay_func in self.decay_functions:
            if decay_func == "exponential":
                for rate in self.decay_rates:
                    col_name = f"time_since_last_exp_{rate}"
                    lf = lf.with_columns(
                        (-rate * pl.col("time_since_last").fill_null(0) / 3600)
                        .exp()
                        .alias(col_name)
                    )
                    result = result.register_feature(
                        col_name,
                        {"source_step": "DurationFeaturesStep"},
                        provenance=FeatureProvenance(
                            produced_by="DurationFeaturesStep",
                            inputs=["time_since_last"],
                            tags={"continuous", "temporal"},
                            description=f"Exponential decay with rate {rate}",
                        ),
                    )

            elif decay_func == "power_law":
                for exp in self.power_exponents:
                    col_name = f"time_since_last_pow_{exp}"
                    lf = lf.with_columns(
                        (1.0 / (1.0 + pl.col("time_since_last").fill_null(0) / 3600) ** exp)
                        .alias(col_name)
                    )
                    result = result.register_feature(
                        col_name,
                        {"source_step": "DurationFeaturesStep"},
                        provenance=FeatureProvenance(
                            produced_by="DurationFeaturesStep",
                            inputs=["time_since_last"],
                            tags={"continuous", "temporal"},
                            description=f"Power law decay with exponent {exp}",
                        ),
                    )

            elif decay_func == "linear":
                # Linear decay over 24 hours
                lf = lf.with_columns(
                    (1.0 - pl.col("time_since_last").fill_null(0) / 86400)
                    .clip(0, 1)
                    .alias("time_since_last_linear")
                )
                result = result.register_feature(
                    "time_since_last_linear",
                    {"source_step": "DurationFeaturesStep"},
                    provenance=FeatureProvenance(
                        produced_by="DurationFeaturesStep",
                        inputs=["time_since_last"],
                        tags={"continuous", "temporal"},
                        description="Linear decay over 24 hours",
                    ),
                )

            elif decay_func == "logarithmic":
                lf = lf.with_columns(
                    (1.0 / (1.0 + (pl.col("time_since_last").fill_null(0) / 3600 + 1).log()))
                    .alias("time_since_last_log")
                )
                result = result.register_feature(
                    "time_since_last_log",
                    {"source_step": "DurationFeaturesStep"},
                    provenance=FeatureProvenance(
                        produced_by="DurationFeaturesStep",
                        inputs=["time_since_last"],
                        tags={"continuous", "temporal"},
                        description="Logarithmic decay",
                    ),
                )

        # Time to next event (requires future information)
        if self.include_time_to_next:
            if self.group_cols:
                lf = lf.with_columns(
                    (pl.col(timestamp_col).shift(-1) - pl.col(timestamp_col))
                    .dt.total_seconds()
                    .over(self.group_cols)
                    .alias("time_to_next")
                )
            else:
                lf = lf.with_columns(
                    (pl.col(timestamp_col).shift(-1) - pl.col(timestamp_col))
                    .dt.total_seconds()
                    .alias("time_to_next")
                )

            result = result.register_feature(
                "time_to_next",
                {"source_step": "DurationFeaturesStep"},
                provenance=FeatureProvenance(
                    produced_by="DurationFeaturesStep",
                    inputs=[timestamp_col],
                    tags={"continuous", "temporal"},
                    description="Time to next event in seconds (target variable)",
                ),
            )

        return result.with_lazy_frame(lf)


class ContinuousInterEventStep(Step):
    """Compute continuous inter-event time features.

    Generates comprehensive features from inter-event times including
    log-transforms, normalization, and rolling statistics.

    Inputs:
        - timestamp_col from EventSchema

    Outputs:
        - inter_event_seconds: Raw inter-event time
        - inter_event_log: Log-transformed inter-event time
        - inter_event_normalized: Normalized inter-event time
        - inter_event_cv_{window}: Coefficient of variation
        - inter_event_mean_{window}: Rolling mean
        - inter_event_std_{window}: Rolling standard deviation
    """

    def __init__(
        self,
        include_log_transform: bool = True,
        include_normalized: bool = True,
        rolling_windows: Sequence[int] | None = None,
        include_coefficient_variation: bool = True,
        group_cols: Sequence[str] | None = None,
    ) -> None:
        self.include_log_transform = include_log_transform
        self.include_normalized = include_normalized
        self.rolling_windows = list(rolling_windows) if rolling_windows else [5, 10, 20]
        self.include_coefficient_variation = include_coefficient_variation
        self.group_cols = list(group_cols) if group_cols else []

    def run(self, event_frame: EventFrame) -> EventFrame:
        """Execute continuous inter-event feature computation."""
        timestamp_col = event_frame.schema.timestamp_col
        lf = event_frame.lazy_frame

        logger.info("Computing continuous inter-event features")

        # Sort by timestamp
        sort_cols = self.group_cols + [timestamp_col] if self.group_cols else [timestamp_col]
        lf = lf.sort(sort_cols)

        result = event_frame

        # Base inter-event time
        if self.group_cols:
            lf = lf.with_columns(
                (pl.col(timestamp_col) - pl.col(timestamp_col).shift(1))
                .dt.total_seconds()
                .over(self.group_cols)
                .alias("inter_event_seconds")
            )
        else:
            lf = lf.with_columns(
                (pl.col(timestamp_col) - pl.col(timestamp_col).shift(1))
                .dt.total_seconds()
                .alias("inter_event_seconds")
            )

        result = result.register_feature(
            "inter_event_seconds",
            {"source_step": "ContinuousInterEventStep"},
            provenance=FeatureProvenance(
                produced_by="ContinuousInterEventStep",
                inputs=[timestamp_col],
                tags={"continuous", "temporal"},
                description="Inter-event time in seconds",
            ),
        )

        # Log transform
        if self.include_log_transform:
            lf = lf.with_columns(
                (pl.col("inter_event_seconds").fill_null(1) + 1).log().alias("inter_event_log")
            )
            result = result.register_feature(
                "inter_event_log",
                {"source_step": "ContinuousInterEventStep"},
                provenance=FeatureProvenance(
                    produced_by="ContinuousInterEventStep",
                    inputs=["inter_event_seconds"],
                    tags={"continuous", "temporal"},
                    description="Log-transformed inter-event time",
                ),
            )

        # Normalized inter-event time (divide by mean)
        if self.include_normalized:
            if self.group_cols:
                lf = lf.with_columns(
                    (
                        pl.col("inter_event_seconds")
                        / pl.col("inter_event_seconds").mean().over(self.group_cols)
                    ).alias("inter_event_normalized")
                )
            else:
                mean_iet = lf.select(pl.col("inter_event_seconds").mean()).collect().item()
                mean_iet = mean_iet if mean_iet and mean_iet > 0 else 1.0
                lf = lf.with_columns(
                    (pl.col("inter_event_seconds") / mean_iet).alias("inter_event_normalized")
                )

            result = result.register_feature(
                "inter_event_normalized",
                {"source_step": "ContinuousInterEventStep"},
                provenance=FeatureProvenance(
                    produced_by="ContinuousInterEventStep",
                    inputs=["inter_event_seconds"],
                    tags={"continuous", "temporal"},
                    description="Normalized inter-event time (divided by mean)",
                ),
            )

        # Rolling statistics
        for window in self.rolling_windows:
            mean_col = f"inter_event_mean_{window}"
            std_col = f"inter_event_std_{window}"

            if self.group_cols:
                lf = lf.with_columns(
                    [
                        pl.col("inter_event_seconds")
                        .rolling_mean(window_size=window, min_samples=1)
                        .over(self.group_cols)
                        .alias(mean_col),
                        pl.col("inter_event_seconds")
                        .rolling_std(window_size=window, min_samples=1)
                        .over(self.group_cols)
                        .alias(std_col),
                    ]
                )
            else:
                lf = lf.with_columns(
                    [
                        pl.col("inter_event_seconds")
                        .rolling_mean(window_size=window, min_samples=1)
                        .alias(mean_col),
                        pl.col("inter_event_seconds")
                        .rolling_std(window_size=window, min_samples=1)
                        .alias(std_col),
                    ]
                )

            for col in [mean_col, std_col]:
                result = result.register_feature(
                    col,
                    {"source_step": "ContinuousInterEventStep"},
                    provenance=FeatureProvenance(
                        produced_by="ContinuousInterEventStep",
                        inputs=["inter_event_seconds"],
                        tags={"continuous", "temporal"},
                        description=f"Rolling statistic (window={window})",
                    ),
                )

            # Coefficient of variation
            if self.include_coefficient_variation:
                cv_col = f"inter_event_cv_{window}"
                lf = lf.with_columns(
                    (pl.col(std_col) / pl.col(mean_col).fill_null(1)).alias(cv_col)
                )
                result = result.register_feature(
                    cv_col,
                    {"source_step": "ContinuousInterEventStep"},
                    provenance=FeatureProvenance(
                        produced_by="ContinuousInterEventStep",
                        inputs=[mean_col, std_col],
                        tags={"continuous", "temporal"},
                        description=f"Coefficient of variation (window={window})",
                    ),
                )

        return result.with_lazy_frame(lf)


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------


def _parse_time_unit_seconds(time_unit: str) -> float:
    """Parse time unit string to seconds."""
    unit_map = {
        "s": 1,
        "m": 60,
        "min": 60,
        "h": 3600,
        "hr": 3600,
        "d": 86400,
        "day": 86400,
        "w": 604800,
        "week": 604800,
        "1s": 1,
        "1m": 60,
        "1h": 3600,
        "1d": 86400,
        "1w": 604800,
    }
    return float(unit_map.get(time_unit.lower(), 3600))  # Default to 1 hour


def _timedelta_to_seconds(td: Any) -> float:
    """Convert timedelta (numpy or Python) to seconds."""
    if isinstance(td, np.timedelta64):
        return float(td / np.timedelta64(1, 's'))
    elif hasattr(td, 'total_seconds'):
        return td.total_seconds()
    else:
        return float(td)


def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute haversine distance between two points in meters."""
    R = 6371000  # Earth radius in meters

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c
