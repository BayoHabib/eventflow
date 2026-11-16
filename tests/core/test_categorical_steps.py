"""Tests for category-count pipeline steps (Phase 1, group 4)."""

from __future__ import annotations

import datetime
import math
import re

import polars as pl
import pytest
from pydantic import ValidationError

from eventflow.core.event_frame import EventFrame
from eventflow.core.schema import EventMetadata, EventSchema
from eventflow.core.steps.categorical import (
    CategoryCountsConfig,
    CategoryCountsStep,
    sanitize_category_value,
)

BIN_1 = "2024-01-01T00:00:00"
BIN_2 = "2024-01-02T00:00:00"
BIN_3 = "2024-01-03T00:00:00"

BIN_1_DT = datetime.datetime(2024, 1, 1)
BIN_2_DT = datetime.datetime(2024, 1, 2)
BIN_3_DT = datetime.datetime(2024, 1, 3)


def _make_frame(
    grid_ids: list[str | None],
    time_bins: list[str],
    categories: dict[str, list[str | None]],
    *,
    timestamp: str = BIN_1,
) -> EventFrame:
    """Build a small binned EventFrame with the given category columns."""
    data: dict[str, list[object]] = {
        "timestamp": [timestamp] * len(grid_ids),
        "latitude": [41.8781] * len(grid_ids),
        "longitude": [-87.6298] * len(grid_ids),
        "grid_id": grid_ids,
        "time_bin": time_bins,
    }
    data.update(categories)
    lf = pl.LazyFrame(data).with_columns(
        pl.col("timestamp").str.to_datetime(),
        pl.col("time_bin").str.to_datetime(),
    )
    schema = EventSchema(
        timestamp_col="timestamp",
        lat_col="latitude",
        lon_col="longitude",
        categorical_cols=list(categories),
    )
    metadata = EventMetadata(dataset_name="test-categorical", crs="EPSG:4326")
    return EventFrame(lf, schema, metadata)


@pytest.fixture
def reference_frame() -> EventFrame:
    """2 units x 3 bins, categories A/B/C with global counts A=7, B=7, C=4.

    With top_n=2 the kept values are A and B; C falls into "other".
    Per-group (A, B, other) counts:
        (g1, d1): (2, 1, 1)   (g1, d2): (1, 2, 0)   (g1, d3): (0, 1, 1)
        (g2, d1): (1, 1, 0)   (g2, d2): (2, 1, 0)   (g2, d3): (1, 1, 2)
    """
    grid_ids = ["g1"] * 9 + ["g2"] * 9
    time_bins = [BIN_1] * 4 + [BIN_2] * 3 + [BIN_3] * 2 + [BIN_1] * 2 + [BIN_2] * 3 + [BIN_3] * 4
    cats = [
        "A",
        "A",
        "B",
        "C",  # g1 d1
        "A",
        "B",
        "B",  # g1 d2
        "B",
        "C",  # g1 d3
        "A",
        "B",  # g2 d1
        "A",
        "A",
        "B",  # g2 d2
        "A",
        "B",
        "C",
        "C",  # g2 d3
    ]
    return _make_frame(grid_ids, time_bins, {"primary_type": cats})


def _expected_entropy(counts: list[int]) -> float:
    """Shannon entropy (natural log) computed independently of the step."""
    total = sum(counts)
    return -sum((p := c / total) * math.log(p) for c in counts if c > 0)


def _group_row(df: pl.DataFrame, grid_id: str, time_bin: datetime.datetime) -> dict[str, object]:
    """First row of one (unit, time) group, as a dict."""
    rows = df.filter((pl.col("grid_id") == grid_id) & (pl.col("time_bin") == time_bin)).to_dicts()
    assert rows, f"no rows for group {(grid_id, time_bin)}"
    return rows[0]


class TestCategoryCountsStepReference:
    """Hand-computed counts/shares/entropy on the reference frame."""

    def test_counts_shares_entropy(self, reference_frame: EventFrame) -> None:
        """Exact counts, shares summing to 1, entropy vs math.log."""
        step = CategoryCountsStep(category_cols=["primary_type"], top_n=2)
        result = step.run(reference_frame)

        df = result.collect()
        assert len(df) == 18  # every event row kept, features attached
        assert "count_primary_type_other" in df.columns
        assert "count_primary_type_C" not in df.columns  # C is folded into other

        expected = {
            ("g1", BIN_1_DT): [2, 1, 1],
            ("g1", BIN_2_DT): [1, 2, 0],
            ("g1", BIN_3_DT): [0, 1, 1],
            ("g2", BIN_1_DT): [1, 1, 0],
            ("g2", BIN_2_DT): [2, 1, 0],
            ("g2", BIN_3_DT): [1, 1, 2],
        }
        for (grid_id, time_bin), counts in expected.items():
            row = _group_row(df, grid_id, time_bin)
            total = sum(counts)
            assert row["count_primary_type_A"] == counts[0]
            assert row["count_primary_type_B"] == counts[1]
            assert row["count_primary_type_other"] == counts[2]
            shares = [
                row["share_primary_type_A"],
                row["share_primary_type_B"],
                row["share_primary_type_other"],
            ]
            assert shares == pytest.approx([c / total for c in counts])
            assert sum(shares) == pytest.approx(1.0)
            assert row["primary_type_entropy"] == pytest.approx(_expected_entropy(counts))

    def test_top_n_tie_break_is_deterministic(self, reference_frame: EventFrame) -> None:
        """A and B tie globally (7 each); both are kept, C (4) is not."""
        step = CategoryCountsStep(category_cols=["primary_type"], top_n=2)
        df = step.run(reference_frame).collect()
        assert "count_primary_type_A" in df.columns
        assert "count_primary_type_B" in df.columns
        assert "count_primary_type_C" not in df.columns

    def test_registers_provenance(self, reference_frame: EventFrame) -> None:
        """Every produced column is registered with provenance."""
        step = CategoryCountsStep(category_cols=["primary_type"], top_n=2)
        result = step.run(reference_frame)

        expected_cols = {
            "count_primary_type_A",
            "count_primary_type_B",
            "count_primary_type_other",
            "share_primary_type_A",
            "share_primary_type_B",
            "share_primary_type_other",
            "primary_type_entropy",
        }
        assert expected_cols <= set(result.metadata.feature_provenance)
        for col in expected_cols:
            provenance = result.metadata.feature_provenance[col]
            assert provenance.produced_by == "CategoryCountsStep"
            assert "categorical" in provenance.tags
            assert provenance.metadata["top_n"] == 2
            assert provenance.metadata["category_col"] == "primary_type"

    def test_original_columns_preserved(self, reference_frame: EventFrame) -> None:
        """The step attaches features; it does not drop event columns."""
        step = CategoryCountsStep(category_cols=["primary_type"], top_n=2)
        df = step.run(reference_frame).collect()
        for col in ("timestamp", "latitude", "longitude", "grid_id", "time_bin", "primary_type"):
            assert col in df.columns


class TestCategoryCountsGuardrails:
    """Top-N cap, sanitization, nulls, and leakage."""

    def test_high_cardinality_capped_at_top_n_plus_one(self) -> None:
        """25 distinct values with top_n=10 yield exactly 11 count columns."""
        values = [f"V{i:02d}" for i in range(25)]
        frame = _make_frame(["g1"] * 25, [BIN_1] * 25, {"primary_type": values})
        step = CategoryCountsStep(category_cols=["primary_type"], top_n=10)
        df = step.run(frame).collect()

        count_cols = sorted(c for c in df.columns if c.startswith("count_primary_type_"))
        assert count_cols == [f"count_primary_type_V{i:02d}" for i in range(10)] + [
            "count_primary_type_other"
        ]
        assert len(count_cols) == 11
        row = df.row(0, named=True)
        assert row["count_primary_type_other"] == 15  # 25 values - 10 kept
        share_cols = [c for c in df.columns if c.startswith("share_primary_type_")]
        assert len(share_cols) == 11
        assert "primary_type_entropy" in df.columns

    def test_sanitization_produces_valid_column_names(self) -> None:
        """Spaces and punctuation collapse to single underscores."""
        values = ["THEFT / ASSAULT", "BATTERY!", "  SPACED  ", "a.b.c"]
        frame = _make_frame(["g1"] * 4, [BIN_1] * 4, {"primary_type": values})
        step = CategoryCountsStep(category_cols=["primary_type"], top_n=10)
        df = step.run(frame).collect()

        assert "count_primary_type_THEFT_ASSAULT" in df.columns
        assert "count_primary_type_BATTERY" in df.columns
        assert "count_primary_type_SPACED" in df.columns
        assert "count_primary_type_a_b_c" in df.columns
        produced = [c for c in df.columns if c.startswith(("count_", "share_"))]
        assert all(re.fullmatch(r"[A-Za-z0-9_]+", c) for c in produced)
        row = df.row(0, named=True)
        assert row["count_primary_type_THEFT_ASSAULT"] == 1
        assert row["count_primary_type_BATTERY"] == 1

    def test_sanitize_helper(self) -> None:
        """The sanitizer collapses runs and strips edge underscores."""
        assert sanitize_category_value("THEFT / ASSAULT") == "THEFT_ASSAULT"
        assert sanitize_category_value("a...b") == "a_b"
        assert sanitize_category_value("  x  ") == "x"
        assert sanitize_category_value("!!!") == ""

    def test_sanitization_collision_raises(self) -> None:
        """Two kept values sanitizing to the same name are an error."""
        frame = _make_frame(["g1"] * 2, [BIN_1] * 2, {"primary_type": ["A B", "A-B"]})
        step = CategoryCountsStep(category_cols=["primary_type"], top_n=10)
        with pytest.raises(ValueError, match="both sanitize"):
            step.run(frame)

    def test_empty_sanitization_raises(self) -> None:
        """A value with no alphanumeric characters is an error, not a column."""
        frame = _make_frame(["g1"], [BIN_1], {"primary_type": ["!!!"]})
        step = CategoryCountsStep(category_cols=["primary_type"], top_n=10)
        with pytest.raises(ValueError, match="empty"):
            step.run(frame)

    def test_value_sanitizing_to_other_raises(self) -> None:
        """A literal 'other' value would collide with the other bucket."""
        frame = _make_frame(["g1"], [BIN_1], {"primary_type": ["other"]})
        step = CategoryCountsStep(category_cols=["primary_type"], top_n=10)
        with pytest.raises(ValueError, match="other"):
            step.run(frame)

    def test_null_categories_land_in_other(self) -> None:
        """Nulls are counted in 'other' and never occupy a top-N slot."""
        # Nulls are the most frequent value (5x) but must not displace A/B.
        frame = _make_frame(
            ["g1"] * 7,
            [BIN_1] * 7,
            {"primary_type": [None, None, None, None, None, "A", "B"]},
        )
        step = CategoryCountsStep(category_cols=["primary_type"], top_n=2)
        df = step.run(frame).collect()

        assert "count_primary_type_A" in df.columns
        assert "count_primary_type_B" in df.columns
        row = df.row(0, named=True)
        assert row["count_primary_type_A"] == 1
        assert row["count_primary_type_B"] == 1
        assert row["count_primary_type_other"] == 5
        assert row["share_primary_type_other"] == pytest.approx(5 / 7)
        assert row["primary_type_entropy"] == pytest.approx(_expected_entropy([1, 1, 5]))

    def test_no_leakage_from_future_bins(self) -> None:
        """Adding events at later bins leaves earlier bins' features unchanged."""
        base = _make_frame(
            ["g1", "g1", "g1", "g1", "g1", "g1", "g2", "g2", "g2", "g2"],
            [BIN_1, BIN_1, BIN_1, BIN_2, BIN_2, BIN_2, BIN_1, BIN_1, BIN_2, BIN_2],
            {
                "primary_type": [
                    "A",
                    "A",
                    "B",  # g1 d1
                    "A",
                    "B",
                    "C",  # g1 d2 (C -> other)
                    "B",
                    "B",  # g2 d1
                    "A",
                    "C",  # g2 d2 (C -> other)
                ]
            },
        )
        # Future events: many A's (already top-N) plus one new rare value D.
        # Global top-2 stays {A, B}, so any change at d1/d2 would be leakage.
        extended = _make_frame(
            ["g1"] * 6 + ["g2"] * 4 + ["g1"] * 11 + ["g2"] * 3,
            [BIN_1] * 3 + [BIN_2] * 3 + [BIN_1] * 2 + [BIN_2] * 2 + [BIN_3] * 11 + [BIN_3] * 3,
            {
                "primary_type": [
                    "A",
                    "A",
                    "B",
                    "A",
                    "B",
                    "C",
                    "B",
                    "B",
                    "A",
                    "C",
                    "A",
                    "A",
                    "A",
                    "A",
                    "A",
                    "A",
                    "A",
                    "A",
                    "A",
                    "A",
                    "D",
                    "A",
                    "A",
                    "A",
                ]
            },
        )
        step = CategoryCountsStep(category_cols=["primary_type"], top_n=2)
        feature_cols = [
            "count_primary_type_A",
            "count_primary_type_B",
            "count_primary_type_other",
            "share_primary_type_A",
            "share_primary_type_B",
            "share_primary_type_other",
            "primary_type_entropy",
        ]
        sort_keys = ["grid_id", "time_bin", "primary_type"]

        def _early_features(frame: EventFrame) -> pl.DataFrame:
            return (
                step.run(frame)
                .collect()
                .filter(pl.col("time_bin") < datetime.datetime(2024, 1, 3))
                .select([*sort_keys, *feature_cols])
                .sort(sort_keys)
            )

        assert _early_features(base).equals(_early_features(extended))


class TestCategoryCountsErrors:
    """Construction-time and run-time rejections (nothing chosen silently)."""

    def test_empty_category_cols_rejected(self) -> None:
        """An empty category list must fail, not produce zero features."""
        with pytest.raises(ValueError, match="at least one"):
            CategoryCountsStep(category_cols=[])

    def test_duplicate_category_cols_rejected(self) -> None:
        """The same column twice would produce duplicate output columns."""
        with pytest.raises(ValueError, match="duplicates"):
            CategoryCountsStep(category_cols=["primary_type", "primary_type"])

    def test_top_n_below_one_rejected(self) -> None:
        """top_n must be an integer >= 1."""
        with pytest.raises(ValueError, match="top_n"):
            CategoryCountsStep(category_cols=["primary_type"], top_n=0)
        with pytest.raises(ValueError, match="top_n"):
            CategoryCountsStep(category_cols=["primary_type"], top_n=-3)

    def test_empty_unit_or_time_col_rejected(self) -> None:
        """Empty grouping-key names and unit == time are rejected."""
        with pytest.raises(ValueError, match="unit_col"):
            CategoryCountsStep(category_cols=["primary_type"], unit_col="")
        with pytest.raises(ValueError, match="time_col"):
            CategoryCountsStep(category_cols=["primary_type"], time_col="")
        with pytest.raises(ValueError, match="different columns"):
            CategoryCountsStep(category_cols=["primary_type"], unit_col="x", time_col="x")

    def test_produced_name_collision_across_columns_raises(self) -> None:
        """Two category columns must not produce the same output name.

        Column ``a`` with value ``"entropy"`` produces ``count_a_entropy``,
        and column ``count_a`` produces an entropy column of the same name.
        """
        frame = _make_frame(["g1"], [BIN_1], {"a": ["entropy"], "count_a": ["X"]})
        step = CategoryCountsStep(category_cols=["a", "count_a"], top_n=10)
        with pytest.raises(ValueError, match="produced column name collision"):
            step.run(frame)

    def test_produced_column_already_in_frame_raises(self, reference_frame: EventFrame) -> None:
        """A produced name that already exists must fail, not overwrite."""
        frame = reference_frame.with_lazy_frame(
            reference_frame.lazy_frame.with_columns(pl.lit(0).alias("count_primary_type_A"))
        )
        step = CategoryCountsStep(category_cols=["primary_type"], top_n=2)
        with pytest.raises(ValueError, match="already present"):
            step.run(frame)

    def test_missing_category_column_rejected(self, reference_frame: EventFrame) -> None:
        """A category column absent from the frame is an error at run time."""
        step = CategoryCountsStep(category_cols=["no_such_column"], top_n=2)
        with pytest.raises(ValueError, match="missing column"):
            step.run(reference_frame)

    def test_missing_unit_or_time_column_rejected(self, reference_frame: EventFrame) -> None:
        """A missing grouping key is an error at run time."""
        with pytest.raises(ValueError, match="missing column"):
            CategoryCountsStep(
                category_cols=["primary_type"], unit_col="no_such_unit", top_n=2
            ).run(reference_frame)
        with pytest.raises(ValueError, match="missing column"):
            CategoryCountsStep(category_cols=["primary_type"], time_col="no_such_bin", top_n=2).run(
                reference_frame
            )

    def test_null_grouping_key_rejected(self) -> None:
        """Null unit keys would silently join to null features; reject them."""
        frame = _make_frame(["g1", None], [BIN_1, BIN_1], {"primary_type": ["A", "B"]})
        step = CategoryCountsStep(category_cols=["primary_type"], top_n=2)
        with pytest.raises(ValueError, match="nulls"):
            step.run(frame)


class TestCategoryCountsConfig:
    """Pydantic config model: validation and registry-style construction."""

    def test_config_rejects_empty_category_cols(self) -> None:
        """Pydantic enforces at least one category column."""
        with pytest.raises(ValidationError):
            CategoryCountsConfig(category_cols=[])

    def test_config_rejects_top_n_below_one(self) -> None:
        """Pydantic enforces top_n >= 1 before the step is built."""
        with pytest.raises(ValidationError):
            CategoryCountsConfig(category_cols=["primary_type"], top_n=0)

    def test_config_round_trip_builds_step(self, reference_frame: EventFrame) -> None:
        """model_dump() kwargs must match the step constructor (registry path)."""
        config = CategoryCountsConfig(category_cols=["primary_type"], top_n=2)
        step = CategoryCountsStep(**config.model_dump())
        df = step.run(reference_frame).collect()
        assert "count_primary_type_A" in df.columns
        assert "count_primary_type_other" in df.columns

    def test_config_defaults(self) -> None:
        """Defaults are grid_id / time_bin / top_n=10."""
        config = CategoryCountsConfig(category_cols=["primary_type"])
        assert config.unit_col == "grid_id"
        assert config.time_col == "time_bin"
        assert config.top_n == 10


class TestCategoryCountsMultipleColumns:
    """Two category columns produce disjoint, prefixed feature sets."""

    def test_two_category_columns(self) -> None:
        """Each column gets its own count/share/entropy set."""
        frame = _make_frame(
            ["g1"] * 3,
            [BIN_1] * 3,
            {
                "primary_type": ["THEFT", "THEFT", "BATTERY"],
                "location_description": ["STREET", "ALLEY", "STREET"],
            },
        )
        step = CategoryCountsStep(category_cols=["primary_type", "location_description"], top_n=10)
        df = step.run(frame).collect()

        row = df.row(0, named=True)
        assert row["count_primary_type_THEFT"] == 2
        assert row["count_primary_type_BATTERY"] == 1
        assert row["count_primary_type_other"] == 0
        assert row["count_location_description_STREET"] == 2
        assert row["count_location_description_ALLEY"] == 1
        assert row["count_location_description_other"] == 0
        assert "primary_type_entropy" in df.columns
        assert "location_description_entropy" in df.columns
        assert row["primary_type_entropy"] == pytest.approx(_expected_entropy([2, 1, 0]))
        assert row["location_description_entropy"] == pytest.approx(_expected_entropy([2, 1, 0]))

    def test_custom_unit_and_time_columns(self) -> None:
        """Non-default grouping keys work when named explicitly."""
        lf = pl.LazyFrame(
            {
                "timestamp": ["2024-01-01T00:00:00"] * 3,
                "latitude": [41.8781] * 3,
                "longitude": [-87.6298] * 3,
                "zone": ["z1", "z1", "z2"],
                "day": ["2024-01-01T00:00:00"] * 3,
                "primary_type": ["A", "A", "B"],
            }
        ).with_columns(
            pl.col("timestamp").str.to_datetime(),
            pl.col("day").str.to_datetime(),
        )
        schema = EventSchema(
            timestamp_col="timestamp",
            lat_col="latitude",
            lon_col="longitude",
            categorical_cols=["primary_type"],
        )
        frame = EventFrame(
            lf, schema, EventMetadata(dataset_name="test-categorical", crs="EPSG:4326")
        )
        step = CategoryCountsStep(
            category_cols=["primary_type"], unit_col="zone", time_col="day", top_n=2
        )
        df = step.run(frame).collect()
        z1 = df.filter(pl.col("zone") == "z1").row(0, named=True)
        assert z1["count_primary_type_A"] == 2
        assert z1["count_primary_type_other"] == 0
        z2 = df.filter(pl.col("zone") == "z2").row(0, named=True)
        assert z2["count_primary_type_B"] == 1


class TestCategoryCountsFitFreeze:
    """fit/freeze semantics (Phase 4, criterion 5).

    ``fit(train)`` learns the top-N vocabulary from the train slice only;
    ``run()`` afterwards uses the frozen vocabulary and never re-learns
    from the run frame.
    """

    @pytest.fixture
    def train_frame(self) -> EventFrame:
        """One unit/bin: A x6, B x3, C x1 -> top-2 vocabulary {A, B}."""
        return _make_frame(
            ["g1"] * 10,
            [BIN_1] * 10,
            {"primary_type": ["A"] * 6 + ["B"] * 3 + ["C"]},
        )

    @pytest.fixture
    def test_frame(self) -> EventFrame:
        """Same unit/bin: C x8 (would be top-1 globally), A x1, D x1.

        C is frequent here but rare in train; D is entirely new.
        """
        return _make_frame(
            ["g1"] * 10,
            [BIN_1] * 10,
            {"primary_type": ["C"] * 8 + ["A", "D"]},
        )

    def test_is_fitted_flag(self, train_frame: EventFrame) -> None:
        """is_fitted is False before fit and True after."""
        step = CategoryCountsStep(category_cols=["primary_type"], top_n=2)
        assert step.is_fitted is False
        step.fit(train_frame)
        assert step.is_fitted is True

    def test_fit_returns_self(self, train_frame: EventFrame) -> None:
        """fit() returns the step itself, for harness chaining."""
        step = CategoryCountsStep(category_cols=["primary_type"], top_n=2)
        assert step.fit(train_frame) is step

    def test_fitted_values(self, train_frame: EventFrame) -> None:
        """fitted_values() returns the frozen (value, sanitized) pairs."""
        step = CategoryCountsStep(category_cols=["primary_type"], top_n=2)
        step.fit(train_frame)
        assert step.fitted_values("primary_type") == [("A", "A"), ("B", "B")]

    def test_fitted_values_requires_fit(self, train_frame: EventFrame) -> None:
        """fitted_values() on an unfitted step raises; unknown column too."""
        step = CategoryCountsStep(category_cols=["primary_type"], top_n=2)
        with pytest.raises(ValueError, match="not fitted"):
            step.fitted_values("primary_type")
        step.fit(train_frame)
        with pytest.raises(KeyError, match="no_such_column"):
            step.fitted_values("no_such_column")

    def test_fitted_values_returns_a_copy(self, train_frame: EventFrame) -> None:
        """Mutating the returned list cannot change the frozen vocabulary."""
        step = CategoryCountsStep(category_cols=["primary_type"], top_n=2)
        step.fit(train_frame)
        values = step.fitted_values("primary_type")
        values.append(("C", "C"))
        values.clear()
        assert step.fitted_values("primary_type") == [("A", "A"), ("B", "B")]

    def test_unseen_test_categories_land_in_other(
        self, train_frame: EventFrame, test_frame: EventFrame
    ) -> None:
        """A category frequent in test but rare in train goes to ``other``.

        C would have been top-N had the vocabulary been learned from the
        test slice (or from train+test); frozen on train it is ``other``,
        and no column for C or D is produced.
        """
        step = CategoryCountsStep(category_cols=["primary_type"], top_n=2)
        step.fit(train_frame)
        df = step.run(test_frame).collect()

        assert "count_primary_type_A" in df.columns
        assert "count_primary_type_B" in df.columns
        assert "count_primary_type_C" not in df.columns
        assert "count_primary_type_D" not in df.columns
        row = df.row(0, named=True)
        assert row["count_primary_type_A"] == 1
        assert row["count_primary_type_B"] == 0
        assert row["count_primary_type_other"] == 9  # C x8 + D x1
        assert row["share_primary_type_other"] == pytest.approx(0.9)
        assert row["primary_type_entropy"] == pytest.approx(_expected_entropy([1, 0, 9]))

    def test_held_out_slice_cannot_change_vocabulary(
        self, train_frame: EventFrame, test_frame: EventFrame
    ) -> None:
        """Mutating the held-out slice leaves the produced columns identical.

        Twenty new rows of a frequent unseen category ZZZ are appended
        to the test slice between two runs: no ZZZ column appears and
        the full produced column set is unchanged.
        """
        step = CategoryCountsStep(category_cols=["primary_type"], top_n=2)
        step.fit(train_frame)

        first = step.run(test_frame).collect()
        mutated = _make_frame(
            ["g1"] * 30,
            [BIN_1] * 30,
            {"primary_type": ["C"] * 8 + ["A", "D"] + ["ZZZ"] * 20},
        )
        second = step.run(mutated).collect()

        assert "count_primary_type_ZZZ" not in second.columns
        assert sorted(second.columns) == sorted(first.columns)
        # The frozen vocabulary itself is untouched by either run.
        assert step.fitted_values("primary_type") == [("A", "A"), ("B", "B")]

    def test_refit_replaces_vocabulary(
        self, train_frame: EventFrame, test_frame: EventFrame
    ) -> None:
        """Fitting again replaces the frozen vocabulary cleanly.

        A harness refits the same step instance per fold; no trace of
        the previous vocabulary may survive.
        """
        step = CategoryCountsStep(category_cols=["primary_type"], top_n=2)
        step.fit(train_frame)
        assert step.fitted_values("primary_type") == [("A", "A"), ("B", "B")]

        step.fit(test_frame)
        assert step.is_fitted is True
        assert step.fitted_values("primary_type") == [("C", "C"), ("A", "A")]
        df = step.run(test_frame).collect()
        assert "count_primary_type_C" in df.columns
        assert "count_primary_type_B" not in df.columns  # B was in the old vocabulary
        row = df.row(0, named=True)
        assert row["count_primary_type_C"] == 8
        assert row["count_primary_type_other"] == 1  # D only

    def test_failed_refit_keeps_previous_vocabulary(self, train_frame: EventFrame) -> None:
        """A fit that raises leaves the previously frozen vocabulary intact.

        The new vocabulary is assigned only after every column validates.
        """
        step = CategoryCountsStep(category_cols=["primary_type"], top_n=2)
        step.fit(train_frame)
        assert step.fitted_values("primary_type") == [("A", "A"), ("B", "B")]

        bad = _make_frame(["g1"] * 2, [BIN_1] * 2, {"primary_type": ["A B", "A-B"]})
        with pytest.raises(ValueError, match="both sanitize"):
            step.fit(bad)
        assert step.is_fitted is True
        assert step.fitted_values("primary_type") == [("A", "A"), ("B", "B")]

    def test_fit_missing_category_column_raises(self, reference_frame: EventFrame) -> None:
        """fit() on a frame missing a category column raises like run()."""
        step = CategoryCountsStep(category_cols=["no_such_column"], top_n=2)
        with pytest.raises(ValueError, match="missing column"):
            step.fit(reference_frame)

    def test_fit_reuses_top_n_validation_errors(self) -> None:
        """fit() reuses _select_top_values, including its collision errors."""
        frame = _make_frame(["g1"] * 2, [BIN_1] * 2, {"primary_type": ["A B", "A-B"]})
        step = CategoryCountsStep(category_cols=["primary_type"], top_n=10)
        with pytest.raises(ValueError, match="both sanitize"):
            step.fit(frame)
        assert step.is_fitted is False  # a failed fit leaves nothing frozen

    def test_fit_with_multiple_category_columns(self) -> None:
        """Each category column gets its own frozen vocabulary."""
        frame = _make_frame(
            ["g1"] * 4,
            [BIN_1] * 4,
            {
                "primary_type": ["THEFT", "THEFT", "BATTERY", "ASSAULT"],
                "location_description": ["STREET", "ALLEY", "STREET", "STREET"],
            },
        )
        step = CategoryCountsStep(category_cols=["primary_type", "location_description"], top_n=2)
        step.fit(frame)
        assert step.fitted_values("primary_type") == [("THEFT", "THEFT"), ("ASSAULT", "ASSAULT")]
        assert step.fitted_values("location_description") == [
            ("STREET", "STREET"),
            ("ALLEY", "ALLEY"),
        ]
        df = step.run(frame).collect()
        assert "count_primary_type_BATTERY" not in df.columns  # 3rd, folded into other
        assert df.row(0, named=True)["count_primary_type_other"] == 1

    def test_unfitted_run_matches_fit_on_same_frame(self, reference_frame: EventFrame) -> None:
        """Unfitted run() is the old code path: same output as fit-then-run.

        The unfitted branch of run() learns the top-N from the run frame
        exactly as before Phase 4, so fitting on the same frame and then
        running must produce an identical frame.
        """
        unfitted = CategoryCountsStep(category_cols=["primary_type"], top_n=2)
        fitted = CategoryCountsStep(category_cols=["primary_type"], top_n=2)
        fitted.fit(reference_frame)

        assert unfitted.run(reference_frame).collect().equals(fitted.run(reference_frame).collect())

    def test_unfitted_run_produces_pre_phase4_column_set(self, reference_frame: EventFrame) -> None:
        """The unfitted column set is pinned to the pre-Phase-4 output.

        The full Chicago replay (1,908,088 rows) is not re-run in a unit
        test; the recipe-level check in tests/core/test_fitted_statistics.py
        pins the Chicago step itself, and this test pins the exact
        unfitted column set on the reference frame.
        """
        step = CategoryCountsStep(category_cols=["primary_type"], top_n=2)
        assert step.is_fitted is False
        df = step.run(reference_frame).collect()
        produced = sorted(
            c
            for c in df.columns
            if c.startswith(("count_primary_type_", "share_primary_type_"))
            or c == "primary_type_entropy"
        )
        assert produced == [
            "count_primary_type_A",
            "count_primary_type_B",
            "count_primary_type_other",
            "primary_type_entropy",
            "share_primary_type_A",
            "share_primary_type_B",
            "share_primary_type_other",
        ]
