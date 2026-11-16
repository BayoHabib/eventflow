"""Generic external-table connector: load a local tabular file as context.

Phase 3 (criterion 5) replaces the pattern of hand-rolling one loader
per dataset (see ``datasets/chicago_crime/context/weather.py``) with a
single declared-schema adapter: point it at a local CSV or Parquet
file, declare which column is the timestamp, which columns are join
keys, and which columns are values, and it hands the existing context
join steps (``TemporalContextJoinStep``, ``SpatialContextJoinStep``,
``SpatioTemporalContextJoinStep``, ``EnricherStep``) a validated table.

Validation discipline — the same "nothing silently ignored" rule as
the rest of the library:

- The declaration is validated at construction (missing file, unknown
  suffix, empty/duplicate/overlapping declarations, missing declared
  columns, non-temporal timestamp dtype) so a bad declaration fails
  before any pipeline runs.
- The data are validated at :meth:`TabularFileSource.load` (unparseable
  or null timestamps, duplicate ``(key, timestamp)`` pairs — an asof
  join over duplicate context keys would match ambiguously).
- :meth:`TabularFileSource.load` reads the file once and returns a
  ``LazyFrame`` with exactly the declared columns, sorted by
  ``key_cols + [timestamp_col]``. Unmatched events stay null after a
  join; the connector never invents a value.

Out of scope, by design: no live API clients, no network fetching, no
POI / rates / geocoding providers. A future provider plugs in by
subclassing :class:`eventflow.core.context.sources.BaseContextSource`
(or by writing its payload to a local file and reusing this class) —
see ``docs/design-notes/phase3-connectors.md``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import polars as pl

from eventflow.core.context.sources import DynamicTemporalSource
from eventflow.core.schema import ContextSchema
from eventflow.core.utils import get_logger

logger = get_logger(__name__)

_SUPPORTED_SUFFIXES = {".csv", ".parquet"}


class TabularFileSource(DynamicTemporalSource):
    """A local CSV/Parquet file as an external context table.

    Args:
        path: Path to a ``.csv`` or ``.parquet`` file.
        timestamp_col: Column holding the observation time. Must be a
            ``Datetime``/``Date`` column, or a string column parseable
            as datetimes (see ``timestamp_format``).
        value_cols: Columns to join onto events. At least one is
            required; every name must exist in the file.
        key_cols: Optional join-key columns (e.g. a station id) used
            by spatio-temporal joins to keep groups isolated. Must
            exist in the file.
        timestamp_format: ``strptime`` format for string timestamps
            (e.g. ``"%Y-%m-%d %H:%M:%S"``). When ``None`` the format
            is inferred strictly — an unparseable value fails the
            load instead of becoming null.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        timestamp_col: str,
        value_cols: Sequence[str],
        key_cols: Sequence[str] = (),
        timestamp_format: str | None = None,
    ) -> None:
        super().__init__(str(path))
        self.path = Path(path)
        self.timestamp_col = timestamp_col
        self.value_cols = list(value_cols)
        self.key_cols = list(key_cols)
        self.timestamp_format = timestamp_format
        self._validate_declaration()

    # -- declaration validation (construction) ---------------------------

    def _scan(self) -> pl.LazyFrame:
        """Lazy scan of the file, by suffix."""
        if self.path.suffix.lower() == ".csv":
            return pl.scan_csv(self.path)
        return pl.scan_parquet(self.path)

    def _validate_declaration(self) -> None:
        """Validate everything checkable without reading the rows."""
        if not self.path.is_file():
            raise FileNotFoundError(f"External context table not found: {self.path}")
        if self.path.suffix.lower() not in _SUPPORTED_SUFFIXES:
            raise ValueError(
                f"Unsupported external context table {self.path}: "
                f"expected one of {sorted(_SUPPORTED_SUFFIXES)}, "
                f"got {self.path.suffix!r}"
            )
        if not self.timestamp_col:
            raise ValueError("timestamp_col must be a non-empty column name")
        if not self.value_cols:
            raise ValueError(f"{self.path}: value_cols must name at least one column to join")
        declared = [self.timestamp_col, *self.key_cols, *self.value_cols]
        if len(set(declared)) != len(declared):
            duplicates = sorted({c for c in declared if declared.count(c) > 1})
            raise ValueError(
                f"{self.path}: timestamp/key/value columns must be distinct, "
                f"duplicated: {duplicates}"
            )
        schema = self._scan().collect_schema()
        missing = [c for c in declared if c not in schema]
        if missing:
            raise ValueError(
                f"{self.path}: declared column(s) missing from the file: {missing} "
                f"(file columns: {list(schema.names())})"
            )
        ts_dtype = schema[self.timestamp_col]
        if ts_dtype not in (pl.Datetime, pl.Date, pl.String):
            raise ValueError(
                f"{self.path}: timestamp column {self.timestamp_col!r} has dtype "
                f"{ts_dtype}; expected Datetime, Date, or a parseable string column"
            )

    # -- BaseContextSource contract --------------------------------------

    @property
    def schema(self) -> ContextSchema:
        """Context schema: timestamp + first key col as spatial key."""
        return ContextSchema(
            timestamp_col=self.timestamp_col,
            spatial_col=self.key_cols[0] if self.key_cols else None,
            attribute_cols=list(self.value_cols),
        )

    def load(self) -> pl.LazyFrame:
        """Load, validate and sort the declared table.

        Returns:
            LazyFrame with exactly ``key_cols + [timestamp_col] +
            value_cols``, sorted by ``key_cols + [timestamp_col]``.

        Raises:
            ValueError: On null or unparseable timestamps, or on
                duplicate ``(key_cols, timestamp)`` pairs.
        """
        cols = [*self.key_cols, self.timestamp_col, *self.value_cols]
        frame = self._scan().select(cols).collect()
        ts_dtype = frame.schema[self.timestamp_col]
        if ts_dtype == pl.String:
            try:
                frame = frame.with_columns(
                    pl.col(self.timestamp_col)
                    .str.to_datetime(self.timestamp_format, strict=True)
                    .alias(self.timestamp_col)
                )
            except Exception as exc:
                raise ValueError(
                    f"{self.path}: timestamp column {self.timestamp_col!r} could not "
                    f"be parsed as datetimes"
                    + (
                        f" with format {self.timestamp_format!r}"
                        if self.timestamp_format
                        else " (no timestamp_format given)"
                    )
                    + f": {exc}"
                ) from exc
        elif ts_dtype == pl.Date:
            frame = frame.with_columns(
                pl.col(self.timestamp_col).cast(pl.Datetime).alias(self.timestamp_col)
            )
        null_ts = frame[self.timestamp_col].null_count()
        if null_ts:
            raise ValueError(
                f"{self.path}: timestamp column {self.timestamp_col!r} has "
                f"{null_ts} null value(s); context timestamps must all be known"
            )
        key = [*self.key_cols, self.timestamp_col]
        n_dup = int(frame.select(key).is_duplicated().sum())
        if n_dup:
            raise ValueError(
                f"{self.path}: {n_dup} row(s) share a duplicate "
                f"({', '.join(key)}) pair; an asof join over duplicate "
                "context keys would match ambiguously"
            )
        logger.info(
            "Loaded external context table %s: %d rows, %d value column(s)",
            self.path,
            frame.height,
            len(self.value_cols),
        )
        return frame.sort(key).lazy()

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TabularFileSource(path={self.path}, "
            f"timestamp_col={self.timestamp_col!r}, "
            f"key_cols={self.key_cols}, value_cols={self.value_cols})"
        )
