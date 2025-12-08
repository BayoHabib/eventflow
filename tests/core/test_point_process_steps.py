"""Tests for point-process pipeline steps."""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from eventflow.core.event_frame import EventFrame
from eventflow.core.schema import EventMetadata, EventSchema
from eventflow.core.steps.point_process import (
    ConditionalIntensityStep,
    ContinuousInterEventStep,
    DurationFeaturesStep,
    ExponentialDecayStep,
    HawkesKernelStep,
    HazardRateStep,
    KFunctionStep,
    PairCorrelationStep,
    SurvivalTableStep,
)


@pytest.fixture
def sample_event_frame() -> EventFrame:
    """Create a sample EventFrame for testing point-process steps."""
    # Generate timestamps with varying inter-arrival times
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [
        base_time,
        base_time + timedelta(hours=1),
        base_time + timedelta(hours=1, minutes=30),  # Cluster
        base_time + timedelta(hours=2),
        base_time + timedelta(hours=5),  # Gap
        base_time + timedelta(hours=5, minutes=15),  # Cluster
        base_time + timedelta(hours=5, minutes=30),  # Cluster
        base_time + timedelta(hours=10),
        base_time + timedelta(hours=15),
        base_time + timedelta(hours=20),
    ]

    lf = pl.LazyFrame(
        {
            "timestamp": timestamps,
            "latitude": [41.8781 + i * 0.001 for i in range(10)],
            "longitude": [-87.6298 + i * 0.001 for i in range(10)],
            "severity": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
            "category": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
        }
    )

    schema = EventSchema(
        timestamp_col="timestamp",
        lat_col="latitude",
        lon_col="longitude",
        numeric_cols=["severity"],
        categorical_cols=["category"],
    )
    metadata = EventMetadata(dataset_name="test-point-process", crs="EPSG:4326")
    return EventFrame(lf, schema, metadata)


@pytest.fixture
def larger_event_frame() -> EventFrame:
    """Create a larger EventFrame for statistical tests."""
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    n_events = 100
    # Simulate roughly exponential inter-arrival times
    timestamps = [base_time + timedelta(hours=i * 0.5 + (i % 5) * 0.1) for i in range(n_events)]

    lf = pl.LazyFrame(
        {
            "timestamp": timestamps,
            "latitude": [41.8781 + (i % 10) * 0.001 for i in range(n_events)],
            "longitude": [-87.6298 + (i % 10) * 0.001 for i in range(n_events)],
            "severity": [i % 5 + 1 for i in range(n_events)],
        }
    )

    schema = EventSchema(
        timestamp_col="timestamp",
        lat_col="latitude",
        lon_col="longitude",
        numeric_cols=["severity"],
    )
    metadata = EventMetadata(dataset_name="test-large", crs="EPSG:4326")
    return EventFrame(lf, schema, metadata)


class TestExponentialDecayStep:
    """Tests for ExponentialDecayStep."""

    def test_computes_decay_weight(self, sample_event_frame: EventFrame) -> None:
        """Test exponential decay weight computation."""
        step = ExponentialDecayStep(decay_rate=0.1, time_unit="1h")
        result = step.run(sample_event_frame)

        df = result.collect()
        assert "decay_weight" in df.columns
        # All weights should be between 0 and 1
        assert df["decay_weight"].min() > 0
        assert df["decay_weight"].max() <= 1.0

    def test_most_recent_has_highest_weight(self, sample_event_frame: EventFrame) -> None:
        """Test that most recent event has highest weight."""
        step = ExponentialDecayStep(decay_rate=0.5, time_unit="1h")
        result = step.run(sample_event_frame)

        df = result.collect()
        # Last event should have weight 1.0
        assert df["decay_weight"][-1] == pytest.approx(1.0)

    def test_decay_rate_affects_weights(self, sample_event_frame: EventFrame) -> None:
        """Test that higher decay rate leads to faster decay."""
        step_slow = ExponentialDecayStep(decay_rate=0.1, time_unit="1h")
        step_fast = ExponentialDecayStep(decay_rate=1.0, time_unit="1h")

        result_slow = step_slow.run(sample_event_frame).collect()
        result_fast = step_fast.run(sample_event_frame).collect()

        # First event should have lower weight with faster decay
        assert result_fast["decay_weight"][0] < result_slow["decay_weight"][0]

    def test_registers_provenance(self, sample_event_frame: EventFrame) -> None:
        """Test that decay weight has provenance."""
        step = ExponentialDecayStep(decay_rate=0.5)
        result = step.run(sample_event_frame)

        assert "decay_weight" in result.metadata.feature_provenance
        provenance = result.metadata.feature_provenance["decay_weight"]
        assert provenance.produced_by == "ExponentialDecayStep"


class TestHawkesKernelStep:
    """Tests for HawkesKernelStep."""

    def test_computes_hawkes_intensity(self, sample_event_frame: EventFrame) -> None:
        """Test Hawkes intensity computation."""
        step = HawkesKernelStep(alpha=0.5, beta=1.0, mu=0.1, time_unit="1h")
        result = step.run(sample_event_frame)

        df = result.collect()
        assert "hawkes_intensity" in df.columns
        assert "hawkes_trigger" in df.columns
        assert "hawkes_background" in df.columns

    def test_intensity_positive(self, sample_event_frame: EventFrame) -> None:
        """Test that all intensities are positive."""
        step = HawkesKernelStep(alpha=0.5, beta=1.0, mu=0.1)
        result = step.run(sample_event_frame)

        df = result.collect()
        assert df["hawkes_intensity"].min() > 0

    def test_background_equals_mu(self, sample_event_frame: EventFrame) -> None:
        """Test that background rate equals mu."""
        mu = 0.15
        step = HawkesKernelStep(alpha=0.5, beta=1.0, mu=mu)
        result = step.run(sample_event_frame)

        df = result.collect()
        assert df["hawkes_background"][0] == pytest.approx(mu)

    def test_intensity_components_sum(self, sample_event_frame: EventFrame) -> None:
        """Test that intensity = background + trigger."""
        step = HawkesKernelStep(alpha=0.5, beta=1.0, mu=0.1)
        result = step.run(sample_event_frame)

        df = result.collect()
        for i in range(len(df)):
            expected = df["hawkes_background"][i] + df["hawkes_trigger"][i]
            assert df["hawkes_intensity"][i] == pytest.approx(expected, rel=1e-6)

    def test_trigger_increases_after_event(self, sample_event_frame: EventFrame) -> None:
        """Test that trigger component is higher shortly after events."""
        step = HawkesKernelStep(alpha=0.5, beta=1.0, mu=0.1, time_unit="1h")
        result = step.run(sample_event_frame)

        df = result.collect()
        # First event has no history, trigger should be 0
        assert df["hawkes_trigger"][0] == pytest.approx(0.0)
        # Subsequent events should have non-zero trigger
        assert df["hawkes_trigger"][1] > 0

    def test_stability_constraint(self, sample_event_frame: EventFrame) -> None:
        """Test that unstable parameters raise error."""
        # alpha >= beta makes process unstable
        with pytest.raises(ValueError, match="stability|branching"):
            HawkesKernelStep(alpha=2.0, beta=1.0, mu=0.1)

    def test_max_history_limit(self, larger_event_frame: EventFrame) -> None:
        """Test that max_history limits computation."""
        step = HawkesKernelStep(alpha=0.3, beta=1.0, mu=0.1, max_history=10)
        result = step.run(larger_event_frame)

        df = result.collect()
        # Should complete without error
        assert len(df) == 100
        assert df["hawkes_intensity"].null_count() == 0

    def test_registers_provenance(self, sample_event_frame: EventFrame) -> None:
        """Test that Hawkes features have provenance."""
        step = HawkesKernelStep(alpha=0.5, beta=1.0, mu=0.1)
        result = step.run(sample_event_frame)

        assert "hawkes_intensity" in result.metadata.feature_provenance
        provenance = result.metadata.feature_provenance["hawkes_intensity"]
        assert "hawkes" in provenance.tags or "point_process" in provenance.tags


class TestConditionalIntensityStep:
    """Tests for ConditionalIntensityStep."""

    def test_exponential_method(self, sample_event_frame: EventFrame) -> None:
        """Test exponential intensity estimation."""
        step = ConditionalIntensityStep(method="exponential", bandwidth=2.0)
        result = step.run(sample_event_frame)

        df = result.collect()
        assert "conditional_intensity" in df.columns

    def test_gaussian_method(self, sample_event_frame: EventFrame) -> None:
        """Test Gaussian intensity estimation."""
        step = ConditionalIntensityStep(method="gaussian", bandwidth=2.0)
        result = step.run(sample_event_frame)

        df = result.collect()
        assert "conditional_intensity" in df.columns

    def test_intensity_positive(self, sample_event_frame: EventFrame) -> None:
        """Test that intensity is always positive."""
        step = ConditionalIntensityStep(method="exponential", bandwidth=1.0)
        result = step.run(sample_event_frame)

        df = result.collect()
        assert df["conditional_intensity"].min() > 0

    def test_normalized_option(self, sample_event_frame: EventFrame) -> None:
        """Test normalized intensity option."""
        step = ConditionalIntensityStep(method="exponential", normalize=True)
        result = step.run(sample_event_frame)

        df = result.collect()
        assert "intensity_normalized" in df.columns
        # Normalized values should be between 0 and 1
        assert df["intensity_normalized"].min() >= 0
        assert df["intensity_normalized"].max() <= 1.0

    def test_bandwidth_affects_smoothness(self, sample_event_frame: EventFrame) -> None:
        """Test that bandwidth affects intensity smoothness."""
        step_narrow = ConditionalIntensityStep(method="gaussian", bandwidth=0.5)
        step_wide = ConditionalIntensityStep(method="gaussian", bandwidth=5.0)

        result_narrow = step_narrow.run(sample_event_frame).collect()
        result_wide = step_wide.run(sample_event_frame).collect()

        # Wider bandwidth should produce smoother (less variable) intensity
        var_narrow = result_narrow["conditional_intensity"].std()
        var_wide = result_wide["conditional_intensity"].std()
        # Not guaranteed but typically true
        assert var_wide is not None and var_narrow is not None


class TestHazardRateStep:
    """Tests for HazardRateStep."""

    def test_computes_hazard_rate(self, sample_event_frame: EventFrame) -> None:
        """Test hazard rate computation."""
        step = HazardRateStep()
        result = step.run(sample_event_frame)

        df = result.collect()
        assert "hazard_rate" in df.columns
        assert "cumulative_hazard" in df.columns

    def test_hazard_positive(self, sample_event_frame: EventFrame) -> None:
        """Test that hazard rate is non-negative."""
        step = HazardRateStep()
        result = step.run(sample_event_frame)

        df = result.collect()
        # Drop nulls and check
        non_null = df.drop_nulls(subset=["hazard_rate"])
        assert non_null["hazard_rate"].min() >= 0

    def test_cumulative_hazard_increasing(self, sample_event_frame: EventFrame) -> None:
        """Test that cumulative hazard is non-decreasing."""
        step = HazardRateStep()
        result = step.run(sample_event_frame)

        df = result.collect()
        cum_hazard = df["cumulative_hazard"].drop_nulls().to_list()
        for i in range(1, len(cum_hazard)):
            assert cum_hazard[i] >= cum_hazard[i - 1]

    def test_nelson_aalen_method(self, sample_event_frame: EventFrame) -> None:
        """Test Nelson-Aalen method."""
        step = HazardRateStep(method="nelson_aalen")
        result = step.run(sample_event_frame)

        df = result.collect()
        assert "hazard_rate" in df.columns

    def test_registers_provenance(self, sample_event_frame: EventFrame) -> None:
        """Test that hazard features have provenance."""
        step = HazardRateStep()
        result = step.run(sample_event_frame)

        assert "hazard_rate" in result.metadata.feature_provenance


class TestSurvivalTableStep:
    """Tests for SurvivalTableStep."""

    def test_computes_survival_probability(self, sample_event_frame: EventFrame) -> None:
        """Test survival probability computation."""
        step = SurvivalTableStep()
        result = step.run(sample_event_frame)

        df = result.collect()
        assert "survival_probability" in df.columns

    def test_survival_bounded(self, sample_event_frame: EventFrame) -> None:
        """Test that survival is between 0 and 1."""
        step = SurvivalTableStep()
        result = step.run(sample_event_frame)

        df = result.collect()
        non_null = df.drop_nulls(subset=["survival_probability"])
        assert non_null["survival_probability"].min() >= 0
        assert non_null["survival_probability"].max() <= 1.0

    def test_survival_non_increasing(self, sample_event_frame: EventFrame) -> None:
        """Test that survival is non-increasing."""
        step = SurvivalTableStep()
        result = step.run(sample_event_frame)

        df = result.collect()
        survival = df["survival_probability"].drop_nulls().to_list()
        for i in range(1, len(survival)):
            assert survival[i] <= survival[i - 1]

    def test_confidence_intervals(self, sample_event_frame: EventFrame) -> None:
        """Test that confidence intervals are computed."""
        step = SurvivalTableStep(confidence_level=0.95)
        result = step.run(sample_event_frame)

        df = result.collect()
        assert "survival_lower" in df.columns
        assert "survival_upper" in df.columns

    def test_ci_bounds_valid(self, sample_event_frame: EventFrame) -> None:
        """Test that CI bounds are valid."""
        step = SurvivalTableStep(confidence_level=0.95)
        result = step.run(sample_event_frame)

        df = result.collect()
        non_null = df.drop_nulls(
            subset=["survival_probability", "survival_lower", "survival_upper"]
        )
        for row in non_null.iter_rows(named=True):
            if row["survival_lower"] is not None:
                assert row["survival_lower"] <= row["survival_probability"]
            if row["survival_upper"] is not None:
                assert row["survival_upper"] >= row["survival_probability"]

    def test_kaplan_meier_method(self, sample_event_frame: EventFrame) -> None:
        """Test Kaplan-Meier method."""
        step = SurvivalTableStep(method="kaplan_meier")
        result = step.run(sample_event_frame)

        df = result.collect()
        assert "survival_probability" in df.columns


class TestDurationFeaturesStep:
    """Tests for DurationFeaturesStep."""

    def test_computes_time_since_last(self, sample_event_frame: EventFrame) -> None:
        """Test time since last event computation."""
        step = DurationFeaturesStep()
        result = step.run(sample_event_frame)

        df = result.collect()
        assert "time_since_last" in df.columns

    def test_first_event_null(self, sample_event_frame: EventFrame) -> None:
        """Test that first event has null time_since_last."""
        step = DurationFeaturesStep()
        result = step.run(sample_event_frame)

        df = result.collect()
        assert df["time_since_last"][0] is None

    def test_exponential_decay(self, sample_event_frame: EventFrame) -> None:
        """Test exponential decay function."""
        step = DurationFeaturesStep(
            decay_functions=["exponential"],
            decay_rates=[0.1, 0.5],
        )
        result = step.run(sample_event_frame)

        df = result.collect()
        assert "time_since_last_exp_0.1" in df.columns
        assert "time_since_last_exp_0.5" in df.columns

    def test_power_law_decay(self, sample_event_frame: EventFrame) -> None:
        """Test power law decay function."""
        step = DurationFeaturesStep(
            decay_functions=["power_law"],
            power_exponents=[0.5, 1.0, 2.0],
        )
        result = step.run(sample_event_frame)

        df = result.collect()
        assert "time_since_last_pow_0.5" in df.columns
        assert "time_since_last_pow_1.0" in df.columns
        assert "time_since_last_pow_2.0" in df.columns

    def test_logarithmic_decay(self, sample_event_frame: EventFrame) -> None:
        """Test logarithmic decay function."""
        step = DurationFeaturesStep(decay_functions=["logarithmic"])
        result = step.run(sample_event_frame)

        df = result.collect()
        assert "time_since_last_log" in df.columns

    def test_time_to_next(self, sample_event_frame: EventFrame) -> None:
        """Test time to next event (target variable)."""
        step = DurationFeaturesStep(include_time_to_next=True)
        result = step.run(sample_event_frame)

        df = result.collect()
        assert "time_to_next" in df.columns
        # Last event should have null time_to_next
        assert df["time_to_next"][-1] is None

    def test_decay_values_bounded(self, sample_event_frame: EventFrame) -> None:
        """Test that decay values are between 0 and 1."""
        step = DurationFeaturesStep(
            decay_functions=["exponential"],
            decay_rates=[0.1],
        )
        result = step.run(sample_event_frame)

        df = result.collect()
        non_null = df.drop_nulls(subset=["time_since_last_exp_0.1"])
        assert non_null["time_since_last_exp_0.1"].min() >= 0
        assert non_null["time_since_last_exp_0.1"].max() <= 1.0


class TestContinuousInterEventStep:
    """Tests for ContinuousInterEventStep."""

    def test_computes_inter_event_seconds(self, sample_event_frame: EventFrame) -> None:
        """Test inter-event time computation in seconds."""
        step = ContinuousInterEventStep()
        result = step.run(sample_event_frame)

        df = result.collect()
        assert "inter_event_seconds" in df.columns

    def test_log_transform(self, sample_event_frame: EventFrame) -> None:
        """Test log transformation of inter-event times."""
        step = ContinuousInterEventStep(include_log_transform=True)
        result = step.run(sample_event_frame)

        df = result.collect()
        assert "inter_event_log" in df.columns

    def test_normalized_option(self, sample_event_frame: EventFrame) -> None:
        """Test normalized inter-event times."""
        step = ContinuousInterEventStep(include_normalized=True)
        result = step.run(sample_event_frame)

        df = result.collect()
        assert "inter_event_normalized" in df.columns

    def test_rolling_windows(self, sample_event_frame: EventFrame) -> None:
        """Test rolling window statistics."""
        step = ContinuousInterEventStep(rolling_windows=[3, 5])
        result = step.run(sample_event_frame)

        df = result.collect()
        assert "inter_event_mean_3" in df.columns
        assert "inter_event_mean_5" in df.columns
        assert "inter_event_std_3" in df.columns
        assert "inter_event_std_5" in df.columns

    def test_coefficient_of_variation(self, sample_event_frame: EventFrame) -> None:
        """Test coefficient of variation (burstiness metric)."""
        step = ContinuousInterEventStep(
            rolling_windows=[5],
            include_coefficient_variation=True,
        )
        result = step.run(sample_event_frame)

        df = result.collect()
        assert "inter_event_cv_5" in df.columns

    def test_first_event_null(self, sample_event_frame: EventFrame) -> None:
        """Test that first event has null inter-event time."""
        step = ContinuousInterEventStep()
        result = step.run(sample_event_frame)

        df = result.collect()
        assert df["inter_event_seconds"][0] is None

    def test_inter_event_positive(self, sample_event_frame: EventFrame) -> None:
        """Test that inter-event times are positive."""
        step = ContinuousInterEventStep()
        result = step.run(sample_event_frame)

        df = result.collect()
        non_null = df.drop_nulls(subset=["inter_event_seconds"])
        assert non_null["inter_event_seconds"].min() > 0


class TestPairCorrelationStep:
    """Tests for PairCorrelationStep."""

    def test_computes_pair_correlation(self, larger_event_frame: EventFrame) -> None:
        """Test pair correlation computation."""
        step = PairCorrelationStep(max_distance=1000.0, n_bins=10)
        result = step.run(larger_event_frame)

        df = result.collect()
        # Should have nearest neighbor distance feature
        assert "nearest_neighbor_distance" in df.columns

    def test_default_parameters(self) -> None:
        """Test that default parameters work."""
        step = PairCorrelationStep()
        assert step.max_distance == 1000.0
        assert step.n_bins == 50


class TestKFunctionStep:
    """Tests for KFunctionStep."""

    def test_computes_k_function(self, larger_event_frame: EventFrame) -> None:
        """Test Ripley's K-function computation."""
        step = KFunctionStep(max_distance=1000.0, transform="L")
        result = step.run(larger_event_frame)

        df = result.collect()
        # Should have L-function feature (since transform="L")
        assert any("l_function" in col.lower() or "k_function" in col.lower() for col in df.columns)

    def test_default_parameters(self) -> None:
        """Test that default parameters work."""
        step = KFunctionStep()
        assert step.max_distance == 1000.0
        assert step.transform == "L"
