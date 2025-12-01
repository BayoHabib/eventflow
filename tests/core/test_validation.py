"""Tests for point-process validation utilities."""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from eventflow.core.event_frame import EventFrame
from eventflow.core.schema import EventMetadata, EventSchema
from eventflow.core.steps.validation import (
    ValidationReport,
    ValidationResult,
    validate_hawkes_features,
    validate_hawkes_intensity,
    validate_hawkes_stability,
    validate_hazard_positivity,
    validate_intensity_integrability,
    validate_intensity_positivity,
    validate_inter_event_positivity,
    validate_point_process_features,
    validate_probability_bounds,
    validate_survival_monotonicity,
    validate_temporal_ordering,
)


@pytest.fixture
def sample_event_frame() -> EventFrame:
    """Create a sample EventFrame for validation tests."""
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [base_time + timedelta(hours=i) for i in range(10)]

    lf = pl.LazyFrame(
        {
            "timestamp": timestamps,
            "latitude": [41.8781] * 10,
            "longitude": [-87.6298] * 10,
            "intensity": [0.5, 0.6, 0.55, 0.7, 0.65, 0.8, 0.75, 0.85, 0.9, 0.95],
            "survival": [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55],
            "hazard": [0.05, 0.05, 0.06, 0.06, 0.07, 0.07, 0.08, 0.08, 0.09, 0.09],
            "probability": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
            "inter_event": [None, 3600.0, 3600.0, 3600.0, 3600.0, 3600.0, 3600.0, 3600.0, 3600.0, 3600.0],
        }
    )

    schema = EventSchema(
        timestamp_col="timestamp",
        lat_col="latitude",
        lon_col="longitude",
        numeric_cols=["intensity", "survival", "hazard", "probability"],
    )
    metadata = EventMetadata(dataset_name="test-validation", crs="EPSG:4326")
    return EventFrame(lf, schema, metadata)


@pytest.fixture
def invalid_event_frame() -> EventFrame:
    """Create EventFrame with invalid values for testing."""
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [base_time + timedelta(hours=i) for i in range(5)]

    lf = pl.LazyFrame(
        {
            "timestamp": timestamps,
            "latitude": [41.8781] * 5,
            "longitude": [-87.6298] * 5,
            "negative_intensity": [-0.1, 0.5, -0.2, 0.6, 0.7],
            "invalid_survival": [1.0, 0.8, 0.9, 0.7, 0.6],  # Non-monotonic
            "invalid_probability": [0.0, 0.5, 1.2, -0.1, 0.8],  # Out of bounds
            "negative_hazard": [0.1, -0.05, 0.2, 0.15, -0.1],
        }
    )

    schema = EventSchema(
        timestamp_col="timestamp",
        lat_col="latitude",
        lon_col="longitude",
        numeric_cols=["negative_intensity", "invalid_survival", "invalid_probability", "negative_hazard"],
    )
    metadata = EventMetadata(dataset_name="test-invalid", crs="EPSG:4326")
    return EventFrame(lf, schema, metadata)


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_create_valid_result(self) -> None:
        """Test creating a valid result."""
        result = ValidationResult(
            is_valid=True,
            check_name="test_check",
            message="Validation passed",
            details={"count": 100},
        )
        assert result.is_valid is True
        assert "passed" in result.message

    def test_create_invalid_result(self) -> None:
        """Test creating an invalid result."""
        result = ValidationResult(
            is_valid=False,
            check_name="test_check",
            message="Validation failed: negative values found",
            details={"invalid_count": 5},
        )
        assert result.is_valid is False
        assert "failed" in result.message

    def test_result_with_severity(self) -> None:
        """Test result with severity level."""
        result = ValidationResult(
            is_valid=True,
            check_name="test_check",
            message="Validation passed with note",
            details={},
            severity="warning",
        )
        assert result.is_valid is True
        assert result.severity == "warning"


class TestValidationReport:
    """Tests for ValidationReport class."""

    def test_create_empty_report(self) -> None:
        """Test creating an empty report."""
        report = ValidationReport()
        assert report.is_valid is True
        assert len(report.results) == 0

    def test_add_result(self) -> None:
        """Test adding a result to report."""
        report = ValidationReport()
        result = ValidationResult(is_valid=True, check_name="test_check", message="Test passed")
        report.add(result)
        assert len(report.results) == 1

    def test_report_invalid_if_any_invalid(self) -> None:
        """Test that report is invalid if any check fails."""
        report = ValidationReport()
        report.add(ValidationResult(is_valid=True, check_name="check1", message="OK"))
        report.add(ValidationResult(is_valid=False, check_name="check2", message="Failed"))
        assert report.is_valid is False

    def test_summary(self) -> None:
        """Test report summary generation."""
        report = ValidationReport()
        report.add(ValidationResult(is_valid=True, check_name="check1", message="OK"))
        report.add(ValidationResult(is_valid=True, check_name="check2", message="OK"))
        summary = report.summary()
        assert "2/2" in summary  # 2 out of 2 checks
        assert "passed" in summary.lower()


class TestValidateIntensityPositivity:
    """Tests for validate_intensity_positivity function."""

    def test_valid_positive_intensity(self, sample_event_frame: EventFrame) -> None:
        """Test validation of positive intensity values."""
        result = validate_intensity_positivity(sample_event_frame, intensity_col="intensity")
        assert result.is_valid is True

    def test_invalid_negative_intensity(self, invalid_event_frame: EventFrame) -> None:
        """Test detection of negative intensity values."""
        result = validate_intensity_positivity(invalid_event_frame, intensity_col="negative_intensity")
        assert result.is_valid is False
        assert result.details.get("n_negative", 0) > 0 or "non-positive" in result.message.lower()


class TestValidateIntensityIntegrability:
    """Tests for validate_intensity_integrability function."""

    def test_integrable_intensity(self, sample_event_frame: EventFrame) -> None:
        """Test validation of integrable intensity."""
        result = validate_intensity_integrability(sample_event_frame, intensity_col="intensity")
        # Should pass for bounded values
        assert result.is_valid is True

    def test_reports_estimate_in_details(self, sample_event_frame: EventFrame) -> None:
        """Test that estimates are reported in details."""
        result = validate_intensity_integrability(sample_event_frame, intensity_col="intensity")
        # Should have some details about the intensity values
        assert len(result.details) > 0


class TestValidateProbabilityBounds:
    """Tests for validate_probability_bounds function."""

    def test_valid_probability(self, sample_event_frame: EventFrame) -> None:
        """Test validation of valid probabilities."""
        result = validate_probability_bounds(sample_event_frame, probability_col="probability")
        assert result.is_valid is True

    def test_invalid_probability_bounds(self, invalid_event_frame: EventFrame) -> None:
        """Test detection of out-of-bounds probabilities."""
        result = validate_probability_bounds(invalid_event_frame, probability_col="invalid_probability")
        assert result.is_valid is False


class TestValidateSurvivalMonotonicity:
    """Tests for validate_survival_monotonicity function."""

    def test_valid_monotonic_survival(self, sample_event_frame: EventFrame) -> None:
        """Test validation of monotonically decreasing survival."""
        result = validate_survival_monotonicity(sample_event_frame, survival_col="survival")
        assert result.is_valid is True

    def test_invalid_non_monotonic_survival(self, invalid_event_frame: EventFrame) -> None:
        """Test detection of non-monotonic survival."""
        result = validate_survival_monotonicity(invalid_event_frame, survival_col="invalid_survival")
        assert result.is_valid is False


class TestValidateHazardPositivity:
    """Tests for validate_hazard_positivity function."""

    def test_valid_positive_hazard(self, sample_event_frame: EventFrame) -> None:
        """Test validation of positive hazard rate."""
        result = validate_hazard_positivity(sample_event_frame, hazard_col="hazard")
        assert result.is_valid is True

    def test_invalid_negative_hazard(self, invalid_event_frame: EventFrame) -> None:
        """Test detection of negative hazard rate."""
        result = validate_hazard_positivity(invalid_event_frame, hazard_col="negative_hazard")
        assert result.is_valid is False


class TestValidateHawkesStability:
    """Tests for validate_hawkes_stability function."""

    def test_stable_parameters(self) -> None:
        """Test validation of stable Hawkes parameters."""
        result = validate_hawkes_stability(alpha=0.5, beta=1.0)
        assert result.is_valid is True
        assert result.details["branching_ratio"] == pytest.approx(0.5)

    def test_unstable_parameters(self) -> None:
        """Test detection of unstable Hawkes parameters."""
        result = validate_hawkes_stability(alpha=1.5, beta=1.0)
        assert result.is_valid is False
        assert result.details["branching_ratio"] > 1.0

    def test_critical_parameters(self) -> None:
        """Test critical Hawkes parameters (alpha = beta)."""
        result = validate_hawkes_stability(alpha=1.0, beta=1.0)
        # Critical case may have warnings
        assert result.details["branching_ratio"] == pytest.approx(1.0)

    def test_branching_ratio_calculation(self) -> None:
        """Test correct branching ratio calculation."""
        alpha, beta = 0.3, 0.6
        result = validate_hawkes_stability(alpha=alpha, beta=beta)
        assert result.details["branching_ratio"] == pytest.approx(alpha / beta)


class TestValidateHawkesIntensity:
    """Tests for validate_hawkes_intensity function."""

    def test_valid_hawkes_intensity(self) -> None:
        """Test validation of valid Hawkes intensity."""
        lf = pl.LazyFrame({
            "timestamp": [datetime(2024, 1, 1, i) for i in range(5)],
            "latitude": [41.8781] * 5,
            "longitude": [-87.6298] * 5,
            "hawkes_intensity": [0.5, 0.6, 0.55, 0.7, 0.65],
            "hawkes_background": [0.1, 0.1, 0.1, 0.1, 0.1],
        })
        schema = EventSchema(
            timestamp_col="timestamp",
            lat_col="latitude",
            lon_col="longitude",
        )
        metadata = EventMetadata(dataset_name="test-hawkes", crs="EPSG:4326")
        ef = EventFrame(lf, schema, metadata)

        result = validate_hawkes_intensity(
            ef,
            intensity_col="hawkes_intensity",
            background_col="hawkes_background",
        )
        assert result.is_valid is True

    def test_invalid_intensity_below_background(self) -> None:
        """Test detection of intensity below background."""
        lf = pl.LazyFrame({
            "timestamp": [datetime(2024, 1, 1, i) for i in range(5)],
            "latitude": [41.8781] * 5,
            "longitude": [-87.6298] * 5,
            "hawkes_intensity": [0.05, 0.6, 0.55, 0.7, 0.65],  # First one below background
            "hawkes_background": [0.1, 0.1, 0.1, 0.1, 0.1],
        })
        schema = EventSchema(
            timestamp_col="timestamp",
            lat_col="latitude",
            lon_col="longitude",
        )
        metadata = EventMetadata(dataset_name="test-hawkes", crs="EPSG:4326")
        ef = EventFrame(lf, schema, metadata)

        result = validate_hawkes_intensity(
            ef,
            intensity_col="hawkes_intensity",
            background_col="hawkes_background",
        )
        # Should detect intensity < background violation
        assert result.is_valid is False


class TestValidateInterEventPositivity:
    """Tests for validate_inter_event_positivity function."""

    def test_valid_inter_event(self, sample_event_frame: EventFrame) -> None:
        """Test validation of positive inter-event times."""
        result = validate_inter_event_positivity(sample_event_frame, inter_event_col="inter_event")
        assert result.is_valid is True

    def test_allows_null_first_event(self, sample_event_frame: EventFrame) -> None:
        """Test that null first event is allowed."""
        result = validate_inter_event_positivity(
            sample_event_frame,
            inter_event_col="inter_event",
        )
        # Should pass - nulls are expected for the first event
        assert result.is_valid is True


class TestValidateTemporalOrdering:
    """Tests for validate_temporal_ordering function."""

    def test_valid_ordering(self, sample_event_frame: EventFrame) -> None:
        """Test validation of temporally ordered events."""
        result = validate_temporal_ordering(sample_event_frame)
        assert result.is_valid is True

    def test_invalid_ordering(self) -> None:
        """Test detection of out-of-order events."""
        lf = pl.LazyFrame({
            "timestamp": [
                datetime(2024, 1, 1, 3),
                datetime(2024, 1, 1, 1),  # Out of order
                datetime(2024, 1, 1, 2),
            ],
            "latitude": [41.8781] * 3,
            "longitude": [-87.6298] * 3,
        })
        schema = EventSchema(
            timestamp_col="timestamp",
            lat_col="latitude",
            lon_col="longitude",
        )
        metadata = EventMetadata(dataset_name="test", crs="EPSG:4326")
        ef = EventFrame(lf, schema, metadata)

        result = validate_temporal_ordering(ef)
        assert result.is_valid is False


class TestValidatePointProcessFeatures:
    """Tests for validate_point_process_features comprehensive validator."""

    def _has_check(self, report: ValidationReport, name_fragment: str) -> bool:
        """Check if any result's check_name contains the fragment."""
        return any(name_fragment in r.check_name for r in report.results)

    def test_validates_intensity(self, sample_event_frame: EventFrame) -> None:
        """Test validation of intensity in comprehensive check."""
        report = validate_point_process_features(
            sample_event_frame,
            intensity_col="intensity",
        )
        assert self._has_check(report, "intensity") or self._has_check(report, "positivity")

    def test_validates_survival(self, sample_event_frame: EventFrame) -> None:
        """Test validation of survival in comprehensive check."""
        report = validate_point_process_features(
            sample_event_frame,
            survival_col="survival",
        )
        assert self._has_check(report, "survival") or self._has_check(report, "bounds")

    def test_validates_hazard(self, sample_event_frame: EventFrame) -> None:
        """Test validation of hazard in comprehensive check."""
        report = validate_point_process_features(
            sample_event_frame,
            hazard_col="hazard",
        )
        assert self._has_check(report, "hazard") or self._has_check(report, "positivity")

    def test_validates_multiple_features(self, sample_event_frame: EventFrame) -> None:
        """Test validation of multiple features at once."""
        report = validate_point_process_features(
            sample_event_frame,
            intensity_col="intensity",
            survival_col="survival",
            hazard_col="hazard",
        )
        # At least temporal ordering + checks for each feature type
        assert len(report.results) >= 3
        assert report.is_valid is True

    def test_returns_invalid_report(self, invalid_event_frame: EventFrame) -> None:
        """Test that invalid features produce invalid report."""
        report = validate_point_process_features(
            invalid_event_frame,
            intensity_col="negative_intensity",
        )
        assert report.is_valid is False


class TestValidateHawkesFeatures:
    """Tests for validate_hawkes_features comprehensive validator."""

    def _has_check(self, report: ValidationReport, name_fragment: str) -> bool:
        """Check if any result's check_name contains the fragment."""
        return any(name_fragment in r.check_name for r in report.results)

    def _get_check(self, report: ValidationReport, name_fragment: str) -> ValidationResult | None:
        """Get a result by name fragment."""
        for r in report.results:
            if name_fragment in r.check_name:
                return r
        return None

    def test_validates_hawkes_output(self) -> None:
        """Test comprehensive Hawkes validation."""
        lf = pl.LazyFrame({
            "timestamp": [datetime(2024, 1, 1, i) for i in range(5)],
            "latitude": [41.8781] * 5,
            "longitude": [-87.6298] * 5,
            "hawkes_intensity": [0.15, 0.25, 0.20, 0.30, 0.28],
            "hawkes_background": [0.1, 0.1, 0.1, 0.1, 0.1],
            "hawkes_trigger": [0.05, 0.15, 0.10, 0.20, 0.18],
        })
        schema = EventSchema(
            timestamp_col="timestamp",
            lat_col="latitude",
            lon_col="longitude",
            numeric_cols=["hawkes_intensity", "hawkes_background", "hawkes_trigger"],
        )
        metadata = EventMetadata(dataset_name="test-hawkes", crs="EPSG:4326")
        ef = EventFrame(lf, schema, metadata)

        report = validate_hawkes_features(
            ef,
            alpha=0.5,
            beta=1.0,
            intensity_col="hawkes_intensity",
            background_col="hawkes_background",
        )
        assert self._has_check(report, "stability")
        assert self._has_check(report, "intensity") or self._has_check(report, "positivity")

    def test_includes_stability_check(self) -> None:
        """Test that stability check is included."""
        lf = pl.LazyFrame({
            "timestamp": [datetime(2024, 1, 1)],
            "latitude": [41.8781],
            "longitude": [-87.6298],
            "hawkes_intensity": [0.5],
        })
        schema = EventSchema(
            timestamp_col="timestamp",
            lat_col="latitude",
            lon_col="longitude",
        )
        metadata = EventMetadata(dataset_name="test", crs="EPSG:4326")
        ef = EventFrame(lf, schema, metadata)

        report = validate_hawkes_features(
            ef,
            alpha=0.5,
            beta=1.0,
            intensity_col="hawkes_intensity",
        )
        assert self._has_check(report, "stability")
        stability_result = self._get_check(report, "stability")
        assert stability_result is not None
        assert stability_result.details.get("branching_ratio") == pytest.approx(0.5)
