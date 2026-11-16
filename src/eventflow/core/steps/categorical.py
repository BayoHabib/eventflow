"""Per-category count, share and entropy features per (unit, time) group.

This step replaces the declared-but-ignored ``categorical_features`` of the
Chicago recipe (see ``docs/design-notes/chicago-declared-features.md``) and
the experimental ``core/features.py::aggregate_by_category``. Unlike that
function it:

- keeps only the top-N categories per column by total frequency (ties
  broken by sorted value, for determinism) and folds every other value
  into an ``other`` bucket, so a high-cardinality column cannot explode
  the output schema;
- sanitizes category values before using them in column names, and raises
  instead of silently colliding;
- registers every produced column with provenance.

Missing-data policy: null category values are counted in the ``other``
bucket. They never occupy a top-N slot.

Shares and entropy are null when the group total is 0. That cannot happen
on a pre-aggregation event frame (every (unit, time) group holds at least
one event by construction), but the rule is implemented and documented
rather than left implicit.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import polars as pl
from pydantic import BaseModel, Field

from eventflow.core.pipeline import Step
from eventflow.core.schema import FeatureProvenance
from eventflow.core.utils import get_logger

if TYPE_CHECKING:
    from eventflow.core.event_frame import EventFrame

logger = get_logger(__name__)

# Runs of characters outside ASCII letters/digits collapse to one "_".
_SANITIZE_RE = re.compile(r"[^0-9A-Za-z]+")

# Internal name for the per-group event total inside the aggregation frame.
# It is dropped before the join, so it never reaches the output.
_TOTAL_COL = "_category_total"


def sanitize_category_value(value: Any) -> str:
    """Sanitize a category value for use in a column name.

    Every run of non-alphanumeric characters is replaced by a single
    ``_`` and leading/trailing ``_`` are stripped. The result may be
    empty (for example for ``"!!!"``); callers must reject that instead
    of emitting an invalid column name.
    """
    return _SANITIZE_RE.sub("_", str(value)).strip("_")


# -----------------------------------------------------------------------------
# Pydantic Config Model for Step Parameters
# -----------------------------------------------------------------------------


class CategoryCountsConfig(BaseModel):
    """Configuration for per-category count features."""

    category_cols: list[str] = Field(
        ..., min_length=1, description="Category columns to count per (unit, time) group"
    )
    unit_col: str = Field(default="grid_id", description="Spatial unit column (e.g. grid_id)")
    time_col: str = Field(default="time_bin", description="Time bin column")
    top_n: int = Field(
        default=10,
        ge=1,
        description="Keep the top-N categories per column by total frequency; "
        "all other values go into an 'other' bucket",
    )


# -----------------------------------------------------------------------------
# Step
# -----------------------------------------------------------------------------


class CategoryCountsStep(Step):
    """Count, share and entropy of categories per (unit, time) group.

    Inputs:
        - ``unit_col`` (default ``grid_id``) and ``time_col`` (default
          ``time_bin``): the grouping keys. Both must be present and
          non-null; a null key would silently produce null features on the
          join, so it is rejected instead.
        - ``category_cols``: one or more category columns, all present in
          the frame.

    Outputs (per category column ``<col>``, attached to every event row
    via a left join on ``(unit_col, time_col)`` with ``validate="m:1"``):

        - ``count_<col>_<sanitized value>`` for each of the top-N values
          and ``count_<col>_other`` for all other values (nulls included).
        - ``share_<col>_<sanitized value>`` and ``share_<col>_other``:
          the category count divided by the group total; null when the
          total is 0.
        - ``<col>_entropy``: Shannon entropy ``-sum(p * ln(p))`` over the
          top-N + ``other`` buckets, using the natural logarithm (base e);
          null when the total is 0. Buckets with a zero count contribute 0.

    Top-N selection uses total frequency over the whole frame; ties are
    broken by sorted value so the selection is deterministic. Null values
    are excluded from the ranking and counted in ``other``.

    Fit/freeze (Phase 4): :meth:`fit` learns the top-N vocabulary from one
    frame only and freezes it; :meth:`run` after :meth:`fit` uses the
    frozen vocabulary and never re-learns from the run frame, so a
    category that is frequent in a later slice but absent (or rare) in
    the fit frame lands in ``other``. This is the anti-leakage contract
    for train/test evaluation: fit on the train fold, run on the test
    fold. Calling :meth:`fit` again replaces the frozen vocabulary
    cleanly, so a harness can refit the same step instance per fold.
    Without :meth:`fit`, :meth:`run` keeps the original behaviour of
    learning the top-N from the run frame (single-run recipes).
    """

    def __init__(
        self,
        category_cols: Sequence[str],
        unit_col: str = "grid_id",
        time_col: str = "time_bin",
        top_n: int = 10,
    ) -> None:
        self.category_cols = list(category_cols)
        if not self.category_cols:
            raise ValueError("category_cols must contain at least one column")
        duplicates = sorted({c for c in self.category_cols if self.category_cols.count(c) > 1})
        if duplicates:
            raise ValueError(
                "category_cols must not contain duplicates: " + ", ".join(duplicates) + "."
            )
        if not unit_col:
            raise ValueError("unit_col must be a non-empty column name")
        if not time_col:
            raise ValueError("time_col must be a non-empty column name")
        if unit_col == time_col:
            raise ValueError("unit_col and time_col must be different columns")
        if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
            raise ValueError(f"top_n must be an integer >= 1, got {top_n!r}")
        self.unit_col = unit_col
        self.time_col = time_col
        self.top_n = top_n
        # Frozen top-N vocabulary per category column, set by fit().
        # None means "unfitted": run() learns the vocabulary from the run
        # frame, as before Phase 4.
        self._fitted: dict[str, list[tuple[Any, str]]] | None = None

    def fit(self, event_frame: EventFrame) -> CategoryCountsStep:
        """Learn the top-N vocabulary from ``event_frame`` only and freeze it.

        Reuses :meth:`_select_top_values`, including its
        collision/validation errors. A frame missing a category column
        raises the same ``ValueError`` as :meth:`run`. Calling ``fit``
        again replaces the frozen vocabulary cleanly.
        """
        lf = event_frame.lazy_frame
        self._require_columns(lf.collect_schema().names())
        fitted = {cat_col: self._select_top_values(lf, cat_col) for cat_col in self.category_cols}
        self._fitted = fitted
        logger.info(
            "Fitted CategoryCountsStep vocabulary (top_n=%d): %s",
            self.top_n,
            {cat_col: [value for value, _ in pairs] for cat_col, pairs in fitted.items()},
        )
        return self

    @property
    def is_fitted(self) -> bool:
        """Whether fitted statistics (a frozen top-N vocabulary) are held."""
        return self._fitted is not None

    def fitted_values(self, cat_col: str) -> list[tuple[Any, str]]:
        """Return the frozen (value, sanitized) pairs for ``cat_col``.

        Raises ``ValueError`` when the step is not fitted and ``KeyError``
        when ``cat_col`` is not one of the configured category columns.
        The returned list is a copy: mutating it cannot change the frozen
        vocabulary.
        """
        if self._fitted is None:
            raise ValueError("CategoryCountsStep is not fitted; call fit(event_frame) first.")
        if cat_col not in self._fitted:
            raise KeyError(f"CategoryCountsStep has no fitted vocabulary for column {cat_col!r}.")
        return list(self._fitted[cat_col])

    def _require_columns(self, columns: Sequence[str]) -> None:
        """Raise the missing-column error shared by fit() and run()."""
        missing = [
            c for c in [self.unit_col, self.time_col, *self.category_cols] if c not in columns
        ]
        if missing:
            raise ValueError(
                "CategoryCountsStep: missing column(s) in the frame: " + ", ".join(missing) + "."
            )

    def run(self, event_frame: EventFrame) -> EventFrame:
        """Execute per-category count computation."""
        lf = event_frame.lazy_frame
        columns = lf.collect_schema().names()

        self._require_columns(columns)
        self._reject_null_keys(lf)

        # Select the top-N values per column first: sanitization collisions
        # can only be judged once the kept values are known, and every
        # produced column name must be known before building the aggregation
        # so a collision with the frame (or among produced names) is an
        # error instead of a silent overwrite.
        if self._fitted is not None:
            # Frozen vocabulary: never re-learn from the run frame.
            kept: dict[str, list[tuple[Any, str]]] = {
                cat_col: list(pairs) for cat_col, pairs in self._fitted.items()
            }
        else:
            # Unfitted: the pre-Phase-4 behaviour — learn top-N from the
            # run frame.
            kept = {cat_col: self._select_top_values(lf, cat_col) for cat_col in self.category_cols}
        produced: list[str] = []
        for cat_col in self.category_cols:
            for _, sanitized in kept[cat_col]:
                produced.append(f"count_{cat_col}_{sanitized}")
                produced.append(f"share_{cat_col}_{sanitized}")
            produced.append(f"count_{cat_col}_other")
            produced.append(f"share_{cat_col}_other")
            produced.append(f"{cat_col}_entropy")

        duplicates = sorted({name for name in produced if produced.count(name) > 1})
        if duplicates:
            raise ValueError(
                "CategoryCountsStep: produced column name collision: " + ", ".join(duplicates) + "."
            )
        clashes = sorted(set(produced) & set(columns))
        if clashes:
            raise ValueError(
                "CategoryCountsStep: produced column(s) already present in the frame: "
                + ", ".join(clashes)
                + "."
            )

        logger.info(
            "Computing category counts for %s (top_n=%d)",
            ", ".join(self.category_cols),
            self.top_n,
        )
        agg_lf = self._build_aggregation(lf, kept)
        joined_lf = lf.join(
            agg_lf,
            on=[self.unit_col, self.time_col],
            how="left",
            validate="m:1",
            maintain_order="left",
        )

        result = event_frame.with_lazy_frame(joined_lf)
        for cat_col in self.category_cols:
            for _, sanitized in kept[cat_col]:
                result = self._register(result, cat_col, sanitized, is_other=False)
            result = self._register(result, cat_col, "other", is_other=True)
            result = self._register_entropy(result, cat_col)
        return result

    def _reject_null_keys(self, lf: pl.LazyFrame) -> None:
        """Reject null grouping keys: they would join to null features."""
        null_counts = (
            lf.select(
                pl.col(self.unit_col).null_count().alias("unit_nulls"),
                pl.col(self.time_col).null_count().alias("time_nulls"),
            )
            .collect()
            .row(0, named=True)
        )
        null_key_cols = [
            col
            for col, count in (
                (self.unit_col, null_counts["unit_nulls"]),
                (self.time_col, null_counts["time_nulls"]),
            )
            if count > 0
        ]
        if null_key_cols:
            raise ValueError(
                "CategoryCountsStep: grouping key column(s) contain nulls: "
                + ", ".join(null_key_cols)
                + "; assign every event to a unit and a time bin first."
            )

    def _select_top_values(self, lf: pl.LazyFrame, cat_col: str) -> list[tuple[Any, str]]:
        """Return the top-N (value, sanitized name) pairs for one column.

        Nulls are excluded from the ranking: they are counted in ``other``.
        A value that sanitizes to an empty string, to ``"other"`` (which
        would collide with the ``other`` bucket), or to the same name as
        another kept value raises ValueError.
        """
        freq = (
            lf.select(pl.col(cat_col))
            .drop_nulls()
            .group_by(cat_col)
            .agg(pl.len().alias("_count"))
            .collect()
        )
        pairs: list[tuple[Any, int]] = [(value, int(count)) for value, count in freq.iter_rows()]
        try:
            ranked = sorted(pairs, key=lambda p: (-p[1], p[0]))
        except TypeError:
            # Mixed-type category values have no single ordering; fall back
            # to ordering by their string form so the ranking stays defined.
            ranked = sorted(pairs, key=lambda p: (-p[1], str(p[0])))
        kept: list[tuple[Any, str]] = []
        seen: dict[str, Any] = {}
        for value, _ in ranked[: self.top_n]:
            sanitized = sanitize_category_value(value)
            if not sanitized:
                raise ValueError(
                    f"CategoryCountsStep: category value {value!r} in column {cat_col!r} "
                    "sanitizes to an empty column-name fragment; "
                    "rename the value or drop it before counting."
                )
            if sanitized == "other":
                raise ValueError(
                    f"CategoryCountsStep: category value {value!r} in column {cat_col!r} "
                    'sanitizes to "other", which collides with the other bucket.'
                )
            if sanitized in seen:
                raise ValueError(
                    f"CategoryCountsStep: category values {seen[sanitized]!r} and {value!r} "
                    f"in column {cat_col!r} both sanitize to {sanitized!r}; "
                    "rename one of the values before counting."
                )
            seen[sanitized] = value
            kept.append((value, sanitized))
        return kept

    def _build_aggregation(
        self, lf: pl.LazyFrame, kept: dict[str, list[tuple[Any, str]]]
    ) -> pl.LazyFrame:
        """Aggregate per-(unit, time) counts, shares and entropy."""
        count_exprs: list[pl.Expr] = []
        for cat_col, top_values in kept.items():
            top_raw = [value for value, _ in top_values]
            for value, sanitized in top_values:
                name = f"count_{cat_col}_{sanitized}"
                count_exprs.append(pl.col(cat_col).eq(value).cast(pl.Int64).sum().alias(name))
            # Anything that is not a kept value — including nulls — is other.
            other_name = f"count_{cat_col}_other"
            count_exprs.append(
                pl.when(pl.col(cat_col).is_null() | ~pl.col(cat_col).is_in(top_raw))
                .then(1)
                .otherwise(0)
                .cast(pl.Int64)
                .sum()
                .alias(other_name)
            )

        agg_lf = lf.group_by([self.unit_col, self.time_col]).agg(
            [*count_exprs, pl.len().alias(_TOTAL_COL)]
        )

        total = pl.col(_TOTAL_COL)
        derived: list[pl.Expr] = []
        for cat_col, top_values in kept.items():
            buckets = [f"count_{cat_col}_{sanitized}" for _, sanitized in top_values]
            buckets.append(f"count_{cat_col}_other")
            for count_name in buckets:
                share_name = count_name.replace("count_", "share_", 1)
                derived.append(
                    pl.when(total > 0)
                    .then(pl.col(count_name).cast(pl.Float64) / total.cast(pl.Float64))
                    .otherwise(None)
                    .alias(share_name)
                )
            # Shannon entropy, natural log; zero-count buckets contribute 0.
            terms = [
                pl.when(pl.col(name).cast(pl.Float64) / total.cast(pl.Float64) > 0)
                .then(
                    -(
                        pl.col(name).cast(pl.Float64)
                        / total.cast(pl.Float64)
                        * (pl.col(name).cast(pl.Float64) / total.cast(pl.Float64)).log()
                    )
                )
                .otherwise(0.0)
                for name in buckets
            ]
            entropy = sum(terms, start=pl.lit(0.0))
            derived.append(
                pl.when(total > 0).then(entropy).otherwise(None).alias(f"{cat_col}_entropy")
            )

        return agg_lf.with_columns(derived).drop(_TOTAL_COL)

    def _register(
        self, event_frame: EventFrame, cat_col: str, sanitized: str, *, is_other: bool
    ) -> EventFrame:
        """Register the count and share columns for one bucket."""
        result = event_frame
        for prefix in ("count", "share"):
            feat_name = f"{prefix}_{cat_col}_{sanitized}"
            bucket = "other" if is_other else sanitized
            provenance = FeatureProvenance(
                produced_by="CategoryCountsStep",
                inputs=[cat_col, self.unit_col, self.time_col],
                tags={"categorical"},
                description=f"Per-(unit, time) {prefix} of category {bucket!r} in {cat_col!r}",
                metadata={
                    "top_n": self.top_n,
                    "category_col": cat_col,
                    "bucket": bucket,
                },
            )
            result = result.register_feature(
                feat_name,
                {
                    "source_step": "CategoryCountsStep",
                    "category_col": cat_col,
                    "bucket": bucket,
                    "top_n": self.top_n,
                },
                provenance=provenance,
            )
        return result

    def _register_entropy(self, event_frame: EventFrame, cat_col: str) -> EventFrame:
        """Register the entropy column for one category column."""
        feat_name = f"{cat_col}_entropy"
        provenance = FeatureProvenance(
            produced_by="CategoryCountsStep",
            inputs=[cat_col, self.unit_col, self.time_col],
            tags={"categorical"},
            description=(
                f"Shannon entropy (natural log) of the top-{self.top_n} + other "
                f"category distribution in {cat_col!r} per (unit, time) group"
            ),
            metadata={"top_n": self.top_n, "category_col": cat_col},
        )
        return event_frame.register_feature(
            feat_name,
            {
                "source_step": "CategoryCountsStep",
                "category_col": cat_col,
                "top_n": self.top_n,
            },
            provenance=provenance,
        )
