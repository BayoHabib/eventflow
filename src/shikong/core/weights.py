"""Binary spatial neighborhoods with explicit unit identity and no self edges.

Graphs may encode domain barriers or network connectivity through caller-supplied
edges. Geometry and distance builders never infer neighbors from ID ordering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
import shapely

from shikong.core.spatial import require_metric_crs


@dataclass(frozen=True)
class SpatialWeights:
    """Immutable adjacency lists. Lag operations row-standardize binary edges."""

    unit_ids: tuple[Any, ...]
    neighbors: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "unit_ids", tuple(self.unit_ids))
        object.__setattr__(self, "neighbors", tuple(tuple(row) for row in self.neighbors))
        n = len(self.unit_ids)
        if len(set(self.unit_ids)) != n or any(x is None for x in self.unit_ids):
            raise ValueError("Spatial unit IDs must be unique and non-null")
        if len(self.neighbors) != n:
            raise ValueError("One neighbor list is required per spatial unit")
        for i, row in enumerate(self.neighbors):
            if len(set(row)) != len(row) or any(
                not isinstance(j, (int, np.integer)) or j == i or not 0 <= j < n for j in row
            ):
                raise ValueError("Neighbors must be unique in-range indices without self edges")

    @classmethod
    def from_edges(cls, unit_ids: list[Any], edges: list[tuple[Any, Any]]) -> SpatialWeights:
        """Build directed binary edges; supply both directions for an undirected graph."""
        lookup = {unit: i for i, unit in enumerate(unit_ids)}
        rows: list[set[int]] = [set() for _ in unit_ids]
        for source, target in edges:
            if source not in lookup or target not in lookup:
                raise ValueError("Edge references an unknown spatial unit")
            rows[lookup[source]].add(lookup[target])
        return cls(tuple(unit_ids), tuple(tuple(sorted(row)) for row in rows))

    @classmethod
    def from_units(
        cls,
        units: pl.DataFrame,
        id_col: str,
        *,
        method: str = "queen",
        crs: str | None = None,
        k: int = 4,
        distance_threshold: float | None = None,
        x_col: str = "x",
        y_col: str = "y",
        geometry_col: str = "geometry",
    ) -> SpatialWeights:
        """Build contiguity from grid indices/polygons or metric point distances.

        KNN is directed; equal-distance ties follow reference-table order.
        Distance bands are binary and inclusive. Isolated units retain empty lists.
        """
        ids = tuple(units[id_col].to_list())
        if len(set(ids)) != len(ids) or units[id_col].null_count():
            raise ValueError("Spatial reference must contain one row per non-null unit")
        n = len(ids)
        rows: list[tuple[int, ...]] = []
        if method in {"queen", "rook"}:
            if {"grid_row", "grid_col"} <= set(units.columns):
                pairs = list(
                    zip(units["grid_row"].to_list(), units["grid_col"].to_list(), strict=True)
                )
                if any(r is None or c is None or int(r) != r or int(c) != c for r, c in pairs):
                    raise ValueError("Grid indices must be non-null integers")
                if len(set(pairs)) != n:
                    raise ValueError("Each spatial unit requires a distinct grid location")
                lookup = {p: i for i, p in enumerate(pairs)}
                offsets = [
                    (r, c)
                    for r in (-1, 0, 1)
                    for c in (-1, 0, 1)
                    if (r or c) and (method == "queen" or abs(r) + abs(c) == 1)
                ]
                for r, c in pairs:
                    rows.append(
                        tuple(
                            sorted(
                                lookup[(r + dr, c + dc)]
                                for dr, dc in offsets
                                if (r + dr, c + dc) in lookup
                            )
                        )
                    )
            elif geometry_col in units.columns:
                geoms = shapely.from_wkt(units[geometry_col].to_numpy())
                if not np.all(
                    shapely.is_valid(geoms)
                    & ~shapely.is_empty(geoms)
                    & np.isin(shapely.get_type_id(geoms), [3, 6])
                ):
                    raise ValueError("Contiguity requires valid non-empty polygon geometries")
                tree = shapely.STRtree(geoms)
                for i, geom in enumerate(geoms):
                    candidates = [int(j) for j in tree.query(geom, predicate="touches") if j != i]
                    if method == "rook":
                        candidates = [
                            j for j in candidates if geom.intersection(geoms[j]).length > 0
                        ]
                    rows.append(tuple(sorted(candidates)))
            else:
                raise ValueError(
                    "Contiguity requires grid_row/grid_col or polygon geometry; supply spatial_units"
                )
        elif method in {"distance", "knn"}:
            if crs is None:
                raise ValueError("Distance-based weights require a metric CRS")
            require_metric_crs(crs)
            if method == "knn" and (not isinstance(k, int) or isinstance(k, bool) or k < 1):
                raise ValueError("k must be a positive integer")
            if method == "distance" and (
                distance_threshold is None
                or not np.isfinite(distance_threshold)
                or distance_threshold <= 0
            ):
                raise ValueError("Distance weights require a positive distance_threshold in meters")
            coords = units.select(x_col, y_col).to_numpy().astype(float)
            if not np.isfinite(coords).all():
                raise ValueError("Spatial coordinates must be finite and non-null")
            for i in range(n):
                distances = np.linalg.norm(coords - coords[i], axis=1)
                distances[i] = np.inf
                if method == "knn":
                    rows.append(
                        tuple(int(j) for j in np.argsort(distances, kind="stable")[: min(k, n - 1)])
                    )
                else:
                    rows.append(
                        tuple(int(j) for j in np.flatnonzero(distances <= distance_threshold))
                    )
        else:
            raise ValueError(f"Unknown spatial weights: {method!r}")
        return cls(ids, tuple(rows))

    def aligned(self, unit_ids: list[Any]) -> tuple[tuple[int, ...], ...]:
        """Align by identity; missing reference units are not silently treated as zero."""
        if set(unit_ids) != set(self.unit_ids) or len(unit_ids) != len(self.unit_ids):
            raise ValueError(
                "Spatial weights and data must contain the same units; complete the observation panel first"
            )
        positions = {unit: i for i, unit in enumerate(unit_ids)}
        source = {unit: i for i, unit in enumerate(self.unit_ids)}
        return tuple(
            tuple(positions[self.unit_ids[j]] for j in self.neighbors[source[unit]])
            for unit in unit_ids
        )
