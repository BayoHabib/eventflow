"""Validation utilities for point-process and continuous-time features.

This module provides validation checks for:
- Intensity positivity
- Normalization constraints
- Boundary conditions
- Stability conditions for point processes
- Feature consistency checks
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from eventflow.core.utils import get_logger

if TYPE_CHECKING:
    from eventflow.core.event_frame import EventFrame

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check."""

    is_valid: bool
    check_name: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    severity: str = "error"  # error, warning, info


@dataclass
class ValidationReport:
    """Collection of validation results."""

    results: list[ValidationResult] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Check if all validations passed (no errors)."""
        return all(r.is_valid or r.severity != "error" for r in self.results)

    @property
    def errors(self) -> list[ValidationResult]:
        """Get all error-level failures."""
        return [r for r in self.results if not r.is_valid and r.severity == "error"]

    @property
    def warnings(self) -> list[ValidationResult]:
        """Get all warning-level issues."""
        return [r for r in self.results if not r.is_valid and r.severity == "warning"]

    def add(self, result: ValidationResult) -> None:
        """Add a validation result."""
        self.results.append(result)

    def summary(self) -> str:
        """Generate a summary string."""
        n_passed = sum(1 for r in self.results if r.is_valid)
        n_errors = len(self.errors)
        n_warnings = len(self.warnings)

        lines = [
            f"Validation Summary: {n_passed}/{len(self.results)} passed",
            f"  Errors: {n_errors}",
            f"  Warnings: {n_warnings}",
        ]

        if self.errors:
            lines.append("\nErrors:")
            for e in self.errors:
                lines.append(f"  - {e.check_name}: {e.message}")

        if self.warnings:
            lines.append("\nWarnings:")
            for w in self.warnings:
                lines.append(f"  - {w.check_name}: {w.message}")

        return "\n".join(lines)


# -----------------------------------------------------------------------------
# Intensity Validators
# -----------------------------------------------------------------------------


def validate_intensity_positivity(
    event_frame: EventFrame,
    intensity_col: str = "conditional_intensity",
    allow_zero: bool = True,
) -> ValidationResult:
    """Validate that intensity values are non-negative.

    Intensity functions in point processes must be non-negative
    (λ(t) ≥ 0 for all t).

    Args:
        event_frame: EventFrame containing intensity column
        intensity_col: Name of the intensity column
        allow_zero: Whether to allow zero values

    Returns:
        ValidationResult indicating pass/fail
    """
    df = event_frame.lazy_frame.collect()

    if intensity_col not in df.columns:
        return ValidationResult(
            is_valid=False,
            check_name="intensity_positivity",
            message=f"Column '{intensity_col}' not found",
            severity="error",
        )

    intensities = df[intensity_col]

    if allow_zero:
        n_negative = (intensities < 0).sum()
        min_val = intensities.min()
    else:
        n_negative = (intensities <= 0).sum()
        min_val = intensities.min()

    is_valid = n_negative == 0

    return ValidationResult(
        is_valid=bool(is_valid),
        check_name="intensity_positivity",
        message=(
            "All intensity values are positive"
            if is_valid
            else f"Found {n_negative} non-positive intensity values (min={min_val!r})"
        ),
        details={
            "n_negative": n_negative,
            "min_value": min_val,
            "allow_zero": allow_zero,
        },
        severity="error" if not is_valid else "info",
    )


def validate_intensity_integrability(
    event_frame: EventFrame,
    intensity_col: str = "conditional_intensity",
    time_col: str | None = None,
    max_integral: float = 1e10,
) -> ValidationResult:
    """Validate that intensity is integrable (finite total mass).

    For valid point processes, the integral of intensity over the
    observation period should be finite.

    Args:
        event_frame: EventFrame containing intensity column
        intensity_col: Name of the intensity column
        time_col: Timestamp column (uses schema default if None)
        max_integral: Maximum allowed integral value

    Returns:
        ValidationResult indicating pass/fail
    """
    df = event_frame.lazy_frame.sort(time_col or event_frame.schema.timestamp_col).collect()

    if intensity_col not in df.columns:
        return ValidationResult(
            is_valid=False,
            check_name="intensity_integrability",
            message=f"Column '{intensity_col}' not found",
            severity="error",
        )

    ts_col = time_col or event_frame.schema.timestamp_col
    timestamps = df[ts_col].to_list()
    intensities = df[intensity_col].to_list()

    # Approximate integral using trapezoidal rule
    total_integral = 0.0
    for i in range(1, len(timestamps)):
        dt = (timestamps[i] - timestamps[i - 1]).total_seconds()
        avg_intensity = (intensities[i] + intensities[i - 1]) / 2
        total_integral += dt * avg_intensity

    is_valid = total_integral < max_integral and not (total_integral != total_integral)  # NaN check

    return ValidationResult(
        is_valid=is_valid,
        check_name="intensity_integrability",
        message=(
            f"Intensity integral is finite: {total_integral:.2f}"
            if is_valid
            else f"Intensity integral exceeds limit: {total_integral:.2e} > {max_integral:.2e}"
        ),
        details={
            "total_integral": total_integral,
            "max_allowed": max_integral,
        },
        severity="error" if not is_valid else "info",
    )


# -----------------------------------------------------------------------------
# Normalization Validators
# -----------------------------------------------------------------------------


def validate_probability_bounds(
    event_frame: EventFrame,
    probability_col: str,
    tolerance: float = 1e-6,
) -> ValidationResult:
    """Validate that probability values are in [0, 1].

    Args:
        event_frame: EventFrame containing probability column
        probability_col: Name of the probability column
        tolerance: Tolerance for boundary checks

    Returns:
        ValidationResult indicating pass/fail
    """
    df = event_frame.lazy_frame.collect()

    if probability_col not in df.columns:
        return ValidationResult(
            is_valid=False,
            check_name="probability_bounds",
            message=f"Column '{probability_col}' not found",
            severity="error",
        )

    probs = df[probability_col]
    min_val = probs.min()
    max_val = probs.max()

    n_below = (probs < -tolerance).sum()
    n_above = (probs > 1 + tolerance).sum()

    is_valid = n_below == 0 and n_above == 0

    return ValidationResult(
        is_valid=bool(is_valid),
        check_name="probability_bounds",
        message=(
            "All probabilities are in [0, 1]"
            if is_valid
            else f"Found {n_below + n_above} values outside [0, 1] (range: [{min_val!r}, {max_val!r}])"
        ),
        details={
            "min_value": min_val,
            "max_value": max_val,
            "n_below_zero": n_below,
            "n_above_one": n_above,
        },
        severity="error" if not is_valid else "info",
    )


def validate_survival_monotonicity(
    event_frame: EventFrame,
    survival_col: str = "survival_probability",
    tolerance: float = 1e-6,
) -> ValidationResult:
    """Validate that survival function is non-increasing.

    Survival functions must satisfy S(t1) ≥ S(t2) for t1 < t2.

    Args:
        event_frame: EventFrame containing survival column
        survival_col: Name of the survival probability column
        tolerance: Tolerance for monotonicity check

    Returns:
        ValidationResult indicating pass/fail
    """
    df = event_frame.lazy_frame.sort(event_frame.schema.timestamp_col).collect()

    if survival_col not in df.columns:
        return ValidationResult(
            is_valid=False,
            check_name="survival_monotonicity",
            message=f"Column '{survival_col}' not found",
            severity="error",
        )

    survival = df[survival_col].to_list()

    violations = 0
    max_increase = 0.0
    for i in range(1, len(survival)):
        if survival[i] > survival[i - 1] + tolerance:
            violations += 1
            increase = survival[i] - survival[i - 1]
            if increase > max_increase:
                max_increase = increase

    is_valid = violations == 0

    return ValidationResult(
        is_valid=is_valid,
        check_name="survival_monotonicity",
        message=(
            "Survival function is monotonically non-increasing"
            if is_valid
            else f"Found {violations} monotonicity violations (max increase: {max_increase:.6f})"
        ),
        details={
            "violations": violations,
            "max_increase": max_increase,
        },
        severity="error" if not is_valid else "info",
    )


def validate_hazard_positivity(
    event_frame: EventFrame,
    hazard_col: str = "hazard_rate",
    allow_zero: bool = True,
) -> ValidationResult:
    """Validate that hazard rate is non-negative.

    Args:
        event_frame: EventFrame containing hazard column
        hazard_col: Name of the hazard rate column
        allow_zero: Whether to allow zero values

    Returns:
        ValidationResult indicating pass/fail
    """
    return validate_intensity_positivity(
        event_frame,
        intensity_col=hazard_col,
        allow_zero=allow_zero,
    )


# -----------------------------------------------------------------------------
# Boundary Condition Validators
# -----------------------------------------------------------------------------


def validate_cumulative_hazard_start(
    event_frame: EventFrame,
    cumulative_hazard_col: str = "cumulative_hazard",
    tolerance: float = 1e-6,
) -> ValidationResult:
    """Validate that cumulative hazard starts near zero.

    The cumulative hazard function should satisfy H(0) = 0.

    Args:
        event_frame: EventFrame containing cumulative hazard column
        cumulative_hazard_col: Name of the cumulative hazard column
        tolerance: Tolerance for zero check

    Returns:
        ValidationResult indicating pass/fail
    """
    df = event_frame.lazy_frame.sort(event_frame.schema.timestamp_col).collect()

    if cumulative_hazard_col not in df.columns:
        return ValidationResult(
            is_valid=False,
            check_name="cumulative_hazard_start",
            message=f"Column '{cumulative_hazard_col}' not found",
            severity="error",
        )

    first_value = df[cumulative_hazard_col][0]

    # First cumulative hazard should be small (near zero or first contribution)
    is_valid = first_value is not None and first_value >= -tolerance

    return ValidationResult(
        is_valid=is_valid,
        check_name="cumulative_hazard_start",
        message=(
            f"Cumulative hazard starts at {first_value:.6f}"
            if is_valid
            else f"Invalid initial cumulative hazard: {first_value}"
        ),
        details={"first_value": first_value},
        severity="warning" if not is_valid else "info",
    )


def validate_survival_boundaries(
    event_frame: EventFrame,
    survival_col: str = "survival_probability",
    tolerance: float = 1e-6,
) -> ValidationResult:
    """Validate survival function boundary conditions.

    Survival function should satisfy:
    - S(0) ≈ 1 (or S at first event ≤ 1)
    - S(∞) → 0 (or S at last event ≥ 0)

    Args:
        event_frame: EventFrame containing survival column
        survival_col: Name of the survival probability column
        tolerance: Tolerance for boundary checks

    Returns:
        ValidationResult indicating pass/fail
    """
    df = event_frame.lazy_frame.sort(event_frame.schema.timestamp_col).collect()

    if survival_col not in df.columns:
        return ValidationResult(
            is_valid=False,
            check_name="survival_boundaries",
            message=f"Column '{survival_col}' not found",
            severity="error",
        )

    survival = df[survival_col]
    first_value = survival[0]
    last_value = survival[-1]

    issues = []
    if first_value is not None and first_value > 1 + tolerance:
        issues.append(f"S(0) = {first_value:.6f} > 1")
    if last_value is not None and last_value < -tolerance:
        issues.append(f"S(T) = {last_value:.6f} < 0")

    is_valid = len(issues) == 0

    return ValidationResult(
        is_valid=is_valid,
        check_name="survival_boundaries",
        message=(
            f"Survival boundaries valid: S(0)={first_value:.4f}, S(T)={last_value:.4f}"
            if is_valid
            else f"Boundary violations: {'; '.join(issues)}"
        ),
        details={
            "first_value": first_value,
            "last_value": last_value,
        },
        severity="error" if not is_valid else "info",
    )


# -----------------------------------------------------------------------------
# Hawkes Process Validators
# -----------------------------------------------------------------------------


def validate_hawkes_stability(
    alpha: float,
    beta: float | None = None,
    branching_ratio: float | None = None,
) -> ValidationResult:
    """Validate Hawkes process stability condition.

    A Hawkes process is stable (stationary) if the branching ratio
    α/β < 1 (or equivalently, the spectral radius < 1).

    Args:
        alpha: Excitation parameter
        beta: Decay rate (optional if branching_ratio provided)
        branching_ratio: Pre-computed branching ratio (optional)

    Returns:
        ValidationResult indicating pass/fail
    """
    if branching_ratio is None:
        if beta is None or beta <= 0:
            return ValidationResult(
                is_valid=False,
                check_name="hawkes_stability",
                message="Beta must be positive for stability check",
                severity="error",
            )
        branching_ratio = alpha / beta

    is_valid = branching_ratio < 1.0

    return ValidationResult(
        is_valid=is_valid,
        check_name="hawkes_stability",
        message=(
            f"Hawkes process is stable (branching ratio = {branching_ratio:.4f} < 1)"
            if is_valid
            else f"Hawkes process is unstable (branching ratio = {branching_ratio:.4f} ≥ 1)"
        ),
        details={
            "alpha": alpha,
            "beta": beta,
            "branching_ratio": branching_ratio,
        },
        severity="error" if not is_valid else "info",
    )


def validate_hawkes_intensity(
    event_frame: EventFrame,
    intensity_col: str = "hawkes_intensity",
    background_col: str = "hawkes_background",
) -> ValidationResult:
    """Validate Hawkes intensity properties.

    Checks:
    - Intensity is always ≥ background rate
    - Intensity is positive

    Args:
        event_frame: EventFrame containing Hawkes columns
        intensity_col: Name of the intensity column
        background_col: Name of the background rate column

    Returns:
        ValidationResult indicating pass/fail
    """
    df = event_frame.lazy_frame.collect()

    missing = []
    for col in [intensity_col, background_col]:
        if col not in df.columns:
            missing.append(col)

    if missing:
        return ValidationResult(
            is_valid=False,
            check_name="hawkes_intensity",
            message=f"Missing columns: {missing}",
            severity="error",
        )

    intensity = df[intensity_col]
    background = df[background_col]

    # Check intensity >= background
    violations = int((intensity < background).sum())
    _min_intensity = intensity.min()
    _min_background = background.min()
    min_intensity = cast(float, _min_intensity) if _min_intensity is not None else 0.0
    min_background = cast(float, _min_background) if _min_background is not None else 0.0

    is_valid = violations == 0 and min_intensity >= 0

    issues = []
    if violations > 0:
        issues.append(f"{violations} events where intensity < background")
    if min_intensity < 0:
        issues.append(f"Negative intensity found: {min_intensity}")

    return ValidationResult(
        is_valid=is_valid,
        check_name="hawkes_intensity",
        message=(
            "Hawkes intensity is valid"
            if is_valid
            else f"Hawkes intensity issues: {'; '.join(issues)}"
        ),
        details={
            "violations": violations,
            "min_intensity": min_intensity,
            "min_background": min_background,
        },
        severity="error" if not is_valid else "info",
    )


# -----------------------------------------------------------------------------
# Inter-Event Time Validators
# -----------------------------------------------------------------------------


def validate_inter_event_positivity(
    event_frame: EventFrame,
    inter_event_col: str = "inter_event_seconds",
    allow_zero: bool = False,
) -> ValidationResult:
    """Validate that inter-event times are positive.

    Args:
        event_frame: EventFrame containing inter-event column
        inter_event_col: Name of the inter-event time column
        allow_zero: Whether to allow zero (simultaneous events)

    Returns:
        ValidationResult indicating pass/fail
    """
    df = event_frame.lazy_frame.collect()

    if inter_event_col not in df.columns:
        return ValidationResult(
            is_valid=False,
            check_name="inter_event_positivity",
            message=f"Column '{inter_event_col}' not found",
            severity="error",
        )

    inter_times = df[inter_event_col].drop_nulls()  # First event has null

    if allow_zero:
        n_negative = (inter_times < 0).sum()
    else:
        n_negative = (inter_times <= 0).sum()

    min_val = inter_times.min()

    is_valid = n_negative == 0

    return ValidationResult(
        is_valid=bool(is_valid),
        check_name="inter_event_positivity",
        message=(
            "All inter-event times are positive"
            if is_valid
            else f"Found {n_negative} non-positive inter-event times (min={min_val!r})"
        ),
        details={
            "n_negative": n_negative,
            "min_value": min_val,
        },
        severity="error" if not is_valid else "info",
    )


def validate_temporal_ordering(
    event_frame: EventFrame,
    timestamp_col: str | None = None,
) -> ValidationResult:
    """Validate that events are properly ordered in time.

    Args:
        event_frame: EventFrame to validate
        timestamp_col: Timestamp column (uses schema default if None)

    Returns:
        ValidationResult indicating pass/fail
    """
    ts_col = timestamp_col or event_frame.schema.timestamp_col
    df = event_frame.lazy_frame.collect()

    if ts_col not in df.columns:
        return ValidationResult(
            is_valid=False,
            check_name="temporal_ordering",
            message=f"Column '{ts_col}' not found",
            severity="error",
        )

    timestamps = df[ts_col].to_list()

    violations = 0
    for i in range(1, len(timestamps)):
        if timestamps[i] < timestamps[i - 1]:
            violations += 1

    is_valid = violations == 0

    return ValidationResult(
        is_valid=is_valid,
        check_name="temporal_ordering",
        message=(
            "Events are properly ordered in time"
            if is_valid
            else f"Found {violations} temporal ordering violations"
        ),
        details={"violations": violations},
        severity="error" if not is_valid else "info",
    )


# -----------------------------------------------------------------------------
# Comprehensive Validation
# -----------------------------------------------------------------------------


def validate_point_process_features(
    event_frame: EventFrame,
    intensity_col: str | None = None,
    survival_col: str | None = None,
    hazard_col: str | None = None,
    inter_event_col: str | None = None,
) -> ValidationReport:
    """Run comprehensive validation on point-process features.

    Args:
        event_frame: EventFrame to validate
        intensity_col: Intensity column to validate
        survival_col: Survival probability column to validate
        hazard_col: Hazard rate column to validate
        inter_event_col: Inter-event time column to validate

    Returns:
        ValidationReport with all check results
    """
    report = ValidationReport()

    # Always check temporal ordering
    report.add(validate_temporal_ordering(event_frame))

    # Intensity checks
    if intensity_col:
        report.add(validate_intensity_positivity(event_frame, intensity_col))
        report.add(validate_intensity_integrability(event_frame, intensity_col))

    # Survival checks
    if survival_col:
        report.add(validate_probability_bounds(event_frame, survival_col))
        report.add(validate_survival_monotonicity(event_frame, survival_col))
        report.add(validate_survival_boundaries(event_frame, survival_col))

    # Hazard checks
    if hazard_col:
        report.add(validate_hazard_positivity(event_frame, hazard_col))

    # Inter-event checks
    if inter_event_col:
        report.add(validate_inter_event_positivity(event_frame, inter_event_col))

    logger.info(report.summary())
    return report


def validate_hawkes_features(
    event_frame: EventFrame,
    alpha: float,
    beta: float,
    intensity_col: str = "hawkes_intensity",
    background_col: str = "hawkes_background",
) -> ValidationReport:
    """Run comprehensive validation on Hawkes process features.

    Args:
        event_frame: EventFrame to validate
        alpha: Hawkes excitation parameter
        beta: Hawkes decay rate
        intensity_col: Intensity column name
        background_col: Background rate column name

    Returns:
        ValidationReport with all check results
    """
    report = ValidationReport()

    # Stability check
    report.add(validate_hawkes_stability(alpha, beta))

    # Intensity properties
    report.add(validate_hawkes_intensity(event_frame, intensity_col, background_col))

    # General intensity positivity
    report.add(validate_intensity_positivity(event_frame, intensity_col))

    logger.info(report.summary())
    return report
