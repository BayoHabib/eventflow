"""Shared metric grid identity, independent of batch extent and ID ordering."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from pyproj import CRS

GRID_ID_SCHEME = "signed_cantor_v1"
MAX_GRID_INDEX = 1_000_000_000


def require_metric_crs(crs: str) -> None:
    """Reject meter-based operations on geographic or non-meter coordinates."""
    definition = CRS.from_user_input(crs)
    if not definition.is_projected or not all(
        math.isclose(axis.unit_conversion_factor, 1.0) for axis in definition.axis_info[:2]
    ):
        raise ValueError("Grid operations require a projected CRS in meters; transform CRS first")


def grid_cell_id(row: int, column: int) -> int:
    """Pair signed indices injectively in Int64, rejecting unsupported magnitudes."""
    if not all(-MAX_GRID_INDEX <= v <= MAX_GRID_INDEX for v in (row, column)):
        raise ValueError(f"Grid indices must lie within +/-{MAX_GRID_INDEX}")
    a = 2 * row if row >= 0 else -2 * row - 1
    b = 2 * column if column >= 0 else -2 * column - 1
    return (a + b) * (a + b + 1) // 2 + b


def _encode_indices(series: pl.Series) -> pl.Series:
    rows, cols = series.struct.field("grid_row"), series.struct.field("grid_col")
    if rows.null_count() or cols.null_count():
        raise ValueError("Grid assignment requires finite, non-null coordinates")
    r, c = rows.to_numpy(), cols.to_numpy()
    if any(np.any((v < -MAX_GRID_INDEX) | (v > MAX_GRID_INDEX)) for v in (r, c)):
        raise ValueError(f"Grid indices must lie within +/-{MAX_GRID_INDEX}")
    a, b = np.where(r >= 0, 2 * r, -2 * r - 1), np.where(c >= 0, 2 * c, -2 * c - 1)
    total = a + b
    # Divide before multiplying to avoid an intermediate Int64 overflow.
    triangular = np.where(total % 2 == 0, (total // 2) * (total + 1), total * ((total + 1) // 2))
    return pl.Series(triangular + b, dtype=pl.Int64)


def grid_id_expr() -> pl.Expr:
    """Build canonical IDs from grid_row/grid_col without a fixed column stride."""
    return (
        pl.struct("grid_row", "grid_col")
        .map_batches(_encode_indices, return_dtype=pl.Int64, is_elementwise=True)
        .alias("grid_id")
    )


@dataclass(frozen=True)
class GridSpec:
    """Identity of an anchored square lattice; cell IDs are scoped to this spec.

    Fixed origin, size and CRS are required for stable identities across batches.
    Bounds deliberately do not participate in the lattice identity.
    """

    size_m: float
    origin: tuple[float, float]
    crs: str

    def __post_init__(self) -> None:
        require_metric_crs(self.crs)
        if not math.isfinite(self.size_m) or self.size_m <= 0:
            raise ValueError("Grid cell size must be positive and finite")
        if len(self.origin) != 2 or not all(math.isfinite(v) for v in self.origin):
            raise ValueError("Grid origin must contain two finite coordinates")
        object.__setattr__(self, "size_m", float(self.size_m))
        object.__setattr__(self, "origin", tuple(0.0 if v == 0 else float(v) for v in self.origin))
        object.__setattr__(self, "crs", CRS.from_user_input(self.crs).to_string())

    def metadata(self) -> dict[str, Any]:
        """Return a serializable, versioned identity for persistence and joins."""
        data: dict[str, Any] = {
            "size_m": self.size_m,
            "origin": list(self.origin),
            "crs": self.crs,
            "id_scheme": GRID_ID_SCHEME,
        }
        data["fingerprint"] = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
        return data

    def assign(self, lf: pl.LazyFrame, x_col: str, y_col: str) -> pl.LazyFrame:
        """Assign half-open cells; invalid or excessive indices raise at collection."""
        return lf.with_columns(
            ((pl.col(x_col) - self.origin[0]) / self.size_m)
            .floor()
            .cast(pl.Int64)
            .alias("grid_col"),
            ((pl.col(y_col) - self.origin[1]) / self.size_m)
            .floor()
            .cast(pl.Int64)
            .alias("grid_row"),
        ).with_columns(grid_id_expr())
