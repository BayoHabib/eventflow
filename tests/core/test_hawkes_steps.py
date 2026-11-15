"""Tests for the fixed-kernel Hawkes excitation step (Phase 2, Decision 5).

Reference values come from an independent brute-force Python loop
(never from the step itself). A dedicated identity test class proves
the closed form the implementation uses — and proves that the
``adjust=False`` identity proposed in the design note does NOT hold
in general.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import polars as pl
import pytest
from pydantic import ValidationError

from eventflow.core.event_frame import EventFrame
from eventflow.core.schema import EventMetadata, EventSchema
from eventflow.core.steps.hawkes import HawkesExcitationConfig, HawkesExcitationStep

# Hand-chosen counts including zeros, 2 units x 10 daily bins.
UNIT_A_COUNTS = [0, 2, 0, 4, 1, 0, 0, 3, 0, 2]
UNIT_B_COUNTS = [5, 0, 0, 0, 1, 2, 0, 0, 4, 0]
DAYS = [f"2024-01-{day:02d}T00:00:00" for day in range(1, 11)]
TOL = 1e-9


@pytest.fixture
def panel_frame() -> EventFrame:
    """Small synthetic completed panel: 2 units x 10 daily bins, explicit zeros."""
    lf = pl.LazyFrame(
        {
            "timestamp": DAYS * 2,
            "unit": ["A"] * 10 + ["B"] * 10,
            "latitude": [41.8781] * 20,
            "longitude": [-87.6298] * 20,
            "count": UNIT_A_COUNTS + UNIT_B_COUNTS,
        }
    ).with_columns(pl.col("timestamp").str.to_datetime())
    lf = lf.sort(["unit", "timestamp"], maintain_order=True)

    schema = EventSchema(
        timestamp_col="timestamp",
        lat_col="latitude",
        lon_col="longitude",
        numeric_cols=["count"],
        categorical_cols=["unit"],
    )
    metadata = EventMetadata(dataset_name="test-hawkes", crs="EPSG:4326", time_bin="1d")
    return EventFrame(lf, schema, metadata)


def make_frame(units_counts: dict[str, list[float | None]], time_bin: str = "1d") -> EventFrame:
    """Build a panel frame from {unit: counts} over consecutive daily bins."""
    n_bins = len(next(iter(units_counts.values())))
    days = [f"2024-01-{day:02d}T00:00:00" for day in range(1, n_bins + 1)]
    lf = pl.LazyFrame(
        {
            "timestamp": days * len(units_counts),
            "unit": [u for u in units_counts for _ in range(n_bins)],
            "latitude": [41.8781] * (n_bins * len(units_counts)),
            "longitude": [-87.6298] * (n_bins * len(units_counts)),
            "count": [c for u in units_counts for c in units_counts[u]],
        }
    ).with_columns(pl.col("timestamp").str.to_datetime())
    lf = lf.sort(["unit", "timestamp"], maintain_order=True)
    schema = EventSchema(
        timestamp_col="timestamp",
        lat_col="latitude",
        lon_col="longitude",
        numeric_cols=["count"],
        categorical_cols=["unit"],
    )
    metadata = EventMetadata(dataset_name="test-hawkes", crs="EPSG:4326", time_bin=time_bin)
    return EventFrame(lf, schema, metadata)


def brute_force(counts: Sequence[float], halflife_bins: float) -> list[float]:
    """Independent reference: E(t) = sum_{s<t} count(s) * d**(t-s)."""
    d = 0.5 ** (1.0 / halflife_bins)
    return [sum(counts[s] * d ** (t - s) for s in range(t)) for t in range(len(counts))]


def assert_close(
    actual: Sequence[float | None], expected: Sequence[float | None], tol: float = TOL
) -> None:
    """Maximum absolute deviation must stay below *tol* (no relative slack)."""
    actual_vals = list(actual)
    expected_vals = list(expected)
    assert len(actual_vals) == len(expected_vals)
    assert all(v is not None for v in actual_vals)
    assert all(v is not None for v in expected_vals)
    numeric = [float(v) for v in actual_vals if v is not None]
    reference = [float(v) for v in expected_vals if v is not None]
    assert max(abs(a - e) for a, e in zip(numeric, reference, strict=True)) <= tol


def rows_for(df: pl.DataFrame, unit: str, col: str) -> list[float | None]:
    """Values of *col* for one unit, in timestamp order."""
    return df.filter(pl.col("unit") == unit).sort("timestamp")[col].to_list()


class TestHawkesExcitationStep:
    """Reference tests vs the brute-force loop."""

    def test_values_match_brute_force_7d(self, panel_frame: EventFrame) -> None:
        """7d half-life on a daily panel (7 bins) for both units."""
        step = HawkesExcitationStep(value_col="count", halflife_days=7.0, group_cols=["unit"])
        result = step.run(panel_frame)

        df = result.collect()
        assert "count_hawkes_7d" in df.columns
        assert_close(rows_for(df, "A", "count_hawkes_7d"), brute_force(UNIT_A_COUNTS, 7.0))
        assert_close(rows_for(df, "B", "count_hawkes_7d"), brute_force(UNIT_B_COUNTS, 7.0))

    def test_first_bin_is_zero(self, panel_frame: EventFrame) -> None:
        """No past bins: E = 0 at the first bin of every unit, even with events."""
        step = HawkesExcitationStep(value_col="count", halflife_days=7.0, group_cols=["unit"])
        df = step.run(panel_frame).collect()
        # Unit B opens with 5 events; its excitation at bin 0 is still 0.
        assert rows_for(df, "A", "count_hawkes_7d")[0] == 0.0
        assert rows_for(df, "B", "count_hawkes_7d")[0] == 0.0

    def test_single_impulse_exact_decay_shape(self) -> None:
        """One impulse decays by exactly d per bin: hand-computed values.

        Counts [5, 0, 0, 0, 0], half-life 1d (d = 0.5):
        E = [0, 2.5, 1.25, 0.625, 0.3125].
        """
        frame = make_frame({"A": [5, 0, 0, 0, 0]})
        step = HawkesExcitationStep(value_col="count", halflife_days=1.0, group_cols=["unit"])
        df = step.run(frame).collect()
        assert rows_for(df, "A", "count_hawkes_1d") == pytest.approx(
            [0.0, 2.5, 1.25, 0.625, 0.3125], abs=1e-9
        )

    def test_halflife_shorter_than_one_bin(self, panel_frame: EventFrame) -> None:
        """Half-life 0.5d on a daily panel (0.5 bins, d = 0.25)."""
        step = HawkesExcitationStep(value_col="count", halflife_days=0.5, group_cols=["unit"])
        df = step.run(panel_frame).collect()
        assert "count_hawkes_0.5d" in df.columns
        assert_close(rows_for(df, "A", "count_hawkes_0.5d"), brute_force(UNIT_A_COUNTS, 0.5))
        assert_close(rows_for(df, "B", "count_hawkes_0.5d"), brute_force(UNIT_B_COUNTS, 0.5))

    def test_halflife_much_longer_than_panel(self, panel_frame: EventFrame) -> None:
        """Half-life 90d on a 10-bin panel: decay is near-flat, weights ~1."""
        step = HawkesExcitationStep(value_col="count", halflife_days=90.0, group_cols=["unit"])
        df = step.run(panel_frame).collect()
        assert_close(rows_for(df, "A", "count_hawkes_90d"), brute_force(UNIT_A_COUNTS, 90.0))
        assert_close(rows_for(df, "B", "count_hawkes_90d"), brute_force(UNIT_B_COUNTS, 90.0))
        # Spot-check by hand: d = 0.5 ** (1/90); E_A(2) = 2 * d.
        d = 0.5 ** (1.0 / 90.0)
        assert rows_for(df, "A", "count_hawkes_90d")[2] == pytest.approx(2 * d, abs=1e-9)

    def test_halflife_1d_values(self, panel_frame: EventFrame) -> None:
        """1d half-life (d = 0.5), hand-checked first values for unit A.

        Unit A [0, 2, 0, 4, 1, ...]: E = [0, 0, 1, 0.5, 2.25, ...].
        """
        step = HawkesExcitationStep(value_col="count", halflife_days=1.0, group_cols=["unit"])
        df = step.run(panel_frame).collect()
        assert rows_for(df, "A", "count_hawkes_1d")[:5] == pytest.approx(
            [0.0, 0.0, 1.0, 0.5, 2.25], abs=1e-9
        )
        assert_close(rows_for(df, "A", "count_hawkes_1d"), brute_force(UNIT_A_COUNTS, 1.0))

    def test_uses_time_bin_metadata(self, panel_frame: EventFrame) -> None:
        """A 6h panel converts a 1d half-life to 4 bins."""
        frame_6h = panel_frame.with_metadata(time_bin="6h")
        step = HawkesExcitationStep(value_col="count", halflife_days=1.0, group_cols=["unit"])
        df = step.run(frame_6h).collect()
        assert_close(rows_for(df, "A", "count_hawkes_1d"), brute_force(UNIT_A_COUNTS, 4.0))

    def test_no_group_cols_single_unit(self) -> None:
        """Without group_cols a single-unit panel still matches brute force."""
        frame = make_frame({"A": [3, 2, 0, 4, 1, 0, 0, 3, 0, 2]})
        step = HawkesExcitationStep(value_col="count", halflife_days=7.0)
        df = step.run(frame).collect()
        assert_close(
            rows_for(df, "A", "count_hawkes_7d"),
            brute_force([3, 2, 0, 4, 1, 0, 0, 3, 0, 2], 7.0),
        )

    def test_three_units_are_independent(self) -> None:
        """Excitation never leaks across units: each unit matches its own loop."""
        frame = make_frame(
            {
                "A": [0, 2, 0, 4, 1, 0, 0, 3, 0, 2],
                "B": [5, 0, 0, 0, 1, 2, 0, 0, 4, 0],
                "C": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            }
        )
        step = HawkesExcitationStep(value_col="count", halflife_days=3.0, group_cols=["unit"])
        df = step.run(frame).collect()
        for unit, counts in [
            ("A", [0, 2, 0, 4, 1, 0, 0, 3, 0, 2]),
            ("B", [5, 0, 0, 0, 1, 2, 0, 0, 4, 0]),
        ]:
            assert_close(rows_for(df, unit, "count_hawkes_3d"), brute_force(counts, 3.0))
        # A unit with no events at all stays exactly 0.
        assert rows_for(df, "C", "count_hawkes_3d") == [0.0] * 10

    def test_fractional_halflife_label(self, panel_frame: EventFrame) -> None:
        """A 2.5d half-life labels the column 'count_hawkes_2.5d'."""
        step = HawkesExcitationStep(value_col="count", halflife_days=2.5, group_cols=["unit"])
        assert "count_hawkes_2.5d" in step.run(panel_frame).collect().columns

    def test_multiple_value_cols(self, panel_frame: EventFrame) -> None:
        """The value_cols alias computes one column per value column."""
        step = HawkesExcitationStep(value_cols=["count"], halflife_days=7.0, group_cols=["unit"])
        df = step.run(panel_frame).collect()
        assert_close(rows_for(df, "A", "count_hawkes_7d"), brute_force(UNIT_A_COUNTS, 7.0))

    def test_registers_provenance(self, panel_frame: EventFrame) -> None:
        """The output column carries provenance with halflife, tau, strictly-past."""
        step = HawkesExcitationStep(value_col="count", halflife_days=7.0, group_cols=["unit"])
        result = step.run(panel_frame)

        provenance = result.metadata.feature_provenance["count_hawkes_7d"]
        assert provenance.produced_by == "HawkesExcitationStep"
        assert "temporal" in provenance.tags
        assert provenance.metadata["halflife_days"] == 7.0
        assert provenance.metadata["tau_days"] == pytest.approx(7.0 / math.log(2))
        assert provenance.metadata["halflife_bins"] == 7.0
        assert provenance.metadata["strictly_past"] is True
        assert provenance.description is not None
        assert "strictly past" in provenance.description

    def test_requires_time_bin_metadata(self, panel_frame: EventFrame) -> None:
        """Without metadata.time_bin the halflife cannot be converted to bins."""
        frame = panel_frame.with_metadata(time_bin=None)
        step = HawkesExcitationStep(value_col="count", halflife_days=1.0)
        with pytest.raises(ValueError, match="time_bin"):
            step.run(frame)

    def test_rejects_unparseable_time_bin(self, panel_frame: EventFrame) -> None:
        """An unparseable metadata.time_bin must fail with a clear error."""
        frame = panel_frame.with_metadata(time_bin="bogus")
        step = HawkesExcitationStep(value_col="count", halflife_days=1.0)
        with pytest.raises(ValueError, match="bogus"):
            step.run(frame)

    def test_rejects_missing_column(self, panel_frame: EventFrame) -> None:
        """A missing value column must fail, not silently no-op."""
        step = HawkesExcitationStep(value_col="nope", halflife_days=1.0)
        with pytest.raises(ValueError, match="nope"):
            step.run(panel_frame)

    def test_rejects_null_value(self) -> None:
        """A null count is a blocking error (same policy as SpatialLagStep)."""
        frame = make_frame({"A": [1, 2, None, 4, 1]})
        step = HawkesExcitationStep(value_col="count", halflife_days=1.0, group_cols=["unit"])
        with pytest.raises(ValueError, match="null"):
            step.run(frame)

    def test_rejects_inf_value(self) -> None:
        """An infinite count is a blocking error."""
        frame = make_frame({"A": [1.0, float("inf"), 0.0, 4.0, 1.0]})
        step = HawkesExcitationStep(value_col="count", halflife_days=1.0, group_cols=["unit"])
        with pytest.raises(ValueError, match="non-finite"):
            step.run(frame)

    def test_rejects_nan_value(self) -> None:
        """A NaN count is a blocking error."""
        frame = make_frame({"A": [1.0, float("nan"), 0.0, 4.0, 1.0]})
        step = HawkesExcitationStep(value_col="count", halflife_days=1.0, group_cols=["unit"])
        with pytest.raises(ValueError, match="non-finite"):
            step.run(frame)

    def test_rejects_nonpositive_halflife(self) -> None:
        """halflife_days <= 0 must fail at construction."""
        with pytest.raises(ValueError, match="halflife_days"):
            HawkesExcitationStep(value_col="count", halflife_days=0)
        with pytest.raises(ValueError, match="halflife_days"):
            HawkesExcitationStep(value_col="count", halflife_days=-3.0)

    def test_rejects_nonfinite_halflife(self) -> None:
        """halflife_days = inf / nan must fail at construction."""
        with pytest.raises(ValueError, match="halflife_days"):
            HawkesExcitationStep(value_col="count", halflife_days=float("inf"))
        with pytest.raises(ValueError, match="halflife_days"):
            HawkesExcitationStep(value_col="count", halflife_days=float("nan"))

    def test_rejects_empty_value_cols(self) -> None:
        """An empty value_cols list (or empty value_col) must fail, not no-op."""
        with pytest.raises(ValueError, match="value_cols"):
            HawkesExcitationStep(value_cols=[])
        with pytest.raises(ValueError, match="value_cols"):
            HawkesExcitationStep(value_col="")
        with pytest.raises(ValueError, match="value_cols"):
            HawkesExcitationStep(value_cols=["count", ""])

    def test_rejects_both_value_col_and_value_cols(self) -> None:
        """Passing both spellings with different values is ambiguous."""
        with pytest.raises(ValueError, match="value_col"):
            HawkesExcitationStep(value_col="other", value_cols=["count"])

    def test_config_model_roundtrip(self, panel_frame: EventFrame) -> None:
        """The registry path (config -> model_dump -> step kwargs) works."""
        config = HawkesExcitationConfig(value_col="count", halflife_days=7.0, group_cols=["unit"])
        step = HawkesExcitationStep(**config.model_dump())
        result = step.run(panel_frame)
        assert "count_hawkes_7d" in result.collect().columns

    def test_config_model_rejects_bad_values(self) -> None:
        """Pydantic guards the registry path too."""
        with pytest.raises(ValidationError):
            HawkesExcitationConfig(value_col="count", halflife_days=0)
        with pytest.raises(ValidationError):
            HawkesExcitationConfig(value_col="count", halflife_days=float("inf"))
        with pytest.raises(ValidationError):
            HawkesExcitationConfig(value_col="count", halflife_days=float("nan"))


class TestHawkesEwmIdentity:
    """The implementation's correctness argument, as executable tests.

    The design note proposes ``d / (1 - d) * shift(ewm_mean(adjust=False), 1)``.
    That identity is false in general: Polars' ``adjust=False`` EWMA is
    initialised at ``y(0) = count(0)`` rather than ``(1 - d) * count(0)``,
    so the first bin's contribution is over-counted by exactly
    ``d**t * count(0) * d / (1 - d)``. It only holds when ``count(0) == 0``.
    The implementation instead uses the ``adjust=True`` closed form
    ``d * shift(ewm_mean(adjust=True), 1) * (1 - d**t) / (1 - d)``.
    """

    def _design_note_formula(self, counts: list[float], halflife_bins: float) -> list[float]:
        """The design note's adjust=False formula, computed in Polars."""
        d = 0.5 ** (1.0 / halflife_bins)
        df = pl.DataFrame({"count": counts})
        return df.with_columns(
            (
                (d / (1.0 - d))
                * pl.col("count")
                .ewm_mean(half_life=halflife_bins, adjust=False)
                .shift(1)
                .fill_null(0.0)
            ).alias("e")
        )["e"].to_list()

    def _adjust_true_formula(self, counts: list[float], halflife_bins: float) -> list[float]:
        """The adjust=True closed form the implementation uses, in Polars."""
        d = 0.5 ** (1.0 / halflife_bins)
        df = pl.DataFrame({"count": counts})
        idx = (pl.col("count").cum_count() - 1).cast(pl.Float64)
        return df.with_columns(
            (
                pl.lit(d)
                * pl.col("count").ewm_mean(half_life=halflife_bins, adjust=True).shift(1)
                * (1.0 - pl.lit(d).pow(idx))
                / (1.0 - d)
            )
            .fill_null(0.0)
            .alias("e")
        )["e"].to_list()

    def test_design_note_identity_fails_with_nonzero_first_bin(self) -> None:
        """The adjust=False formula is wrong by the analytic error term.

        Counts [3, 2, 0, 4, 1, ...], half-life 1 bin (d = 0.5): at t = 1
        the formula returns 3.0 where the brute force gives 1.5.
        """
        counts = [3.0, 2.0, 0.0, 4.0, 1.0, 0.0, 0.0, 3.0, 0.0, 2.0]
        for halflife_bins in (0.5, 1.0, 7.0, 28.0, 100.0):
            d = 0.5 ** (1.0 / halflife_bins)
            expected = brute_force(counts, halflife_bins)
            claimed = self._design_note_formula(counts, halflife_bins)
            # The error is exactly d**t * count(0) * d / (1 - d) at every
            # t >= 1; at t = 0 the shifted EWMA is null (filled with 0),
            # so the formula and the brute force agree there.
            error = [0.0] + [d**t * counts[0] * d / (1.0 - d) for t in range(1, len(counts))]
            assert_close([c - e for c, e in zip(claimed, expected, strict=True)], error, tol=1e-9)
            assert max(abs(c - e) for c, e in zip(claimed, expected, strict=True)) > 0.1

    def test_design_note_identity_holds_only_when_first_bin_is_zero(self) -> None:
        """With count(0) == 0 the adjust=False formula accidentally matches.

        This is why the identity looks right on panels whose first bin
        is empty — the error term vanishes with count(0).
        """
        counts = [0.0, 2.0, 0.0, 4.0, 1.0, 0.0, 0.0, 3.0, 0.0, 2.0]
        for halflife_bins in (1.0, 7.0):
            assert_close(
                self._design_note_formula(counts, halflife_bins),
                brute_force(counts, halflife_bins),
            )

    def test_adjust_true_closed_form_matches_brute_force(self) -> None:
        """The implemented closed form matches brute force to 1e-9."""
        counts = [3.0, 2.0, 0.0, 4.0, 1.0, 0.0, 0.0, 3.0, 0.0, 2.0]
        for halflife_bins in (0.5, 1.0, 7.0, 28.0, 100.0):
            assert_close(
                self._adjust_true_formula(counts, halflife_bins),
                brute_force(counts, halflife_bins),
            )


class TestHawkesLeakage:
    """No excitation at bin t may depend on bins after t, or on count(t)."""

    def _modified_frame(
        self, panel_frame: EventFrame, unit: str, day: int, count: int
    ) -> EventFrame:
        """Copy of the panel with one bin's count changed."""
        df = panel_frame.collect()
        target = f"2024-01-{day:02d}T00:00:00"
        df = df.with_columns(
            pl.when(
                (pl.col("unit") == unit) & (pl.col("timestamp") == pl.lit(target).str.to_datetime())
            )
            .then(count)
            .otherwise(pl.col("count"))
            .alias("count")
        )
        return EventFrame(df.lazy(), panel_frame.schema, panel_frame.metadata)

    def test_ignores_future_bins(self, panel_frame: EventFrame) -> None:
        """Changing counts at bins > t leaves E at bins <= t unchanged."""
        step = HawkesExcitationStep(value_col="count", halflife_days=7.0, group_cols=["unit"])
        before = rows_for(step.run(panel_frame).collect(), "A", "count_hawkes_7d")
        changed = panel_frame
        for day, count in [(8, 99), (9, 99), (10, 99)]:
            changed = self._modified_frame(changed, "A", day, count)
        after = rows_for(step.run(changed).collect(), "A", "count_hawkes_7d")
        assert_close(before[:5], after[:5])
        assert before[7:] != after[7:]  # sanity: later bins really did change

    def test_strictly_past_ignores_current_bin(self, panel_frame: EventFrame) -> None:
        """Changing count(t) itself leaves E(t) unchanged, but moves E(t+1)."""
        step = HawkesExcitationStep(value_col="count", halflife_days=7.0, group_cols=["unit"])
        before = rows_for(step.run(panel_frame).collect(), "A", "count_hawkes_7d")
        # Change unit A day 5 (index 4) from 1 to 99.
        changed = self._modified_frame(panel_frame, "A", 5, 99)
        after = rows_for(step.run(changed).collect(), "A", "count_hawkes_7d")
        assert before[4] == pytest.approx(after[4], abs=1e-9)
        assert before[5] != after[5]

    def test_other_unit_change_is_isolated(self, panel_frame: EventFrame) -> None:
        """Changing unit B's counts leaves unit A's excitation untouched."""
        step = HawkesExcitationStep(value_col="count", halflife_days=7.0, group_cols=["unit"])
        before = rows_for(step.run(panel_frame).collect(), "A", "count_hawkes_7d")
        changed = panel_frame
        for day, count in [(2, 99), (5, 99)]:
            changed = self._modified_frame(changed, "B", day, count)
        after = rows_for(step.run(changed).collect(), "A", "count_hawkes_7d")
        assert_close(before, after)
