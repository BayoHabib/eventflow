"""Dataset-independent spatial domains and geometric grid coverage."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import polars as pl
import shapely
from pyproj import CRS, Transformer
from shapely.geometry.base import BaseGeometry

from eventflow.core.spatial import require_metric_crs


@dataclass(frozen=True)
class SpatialDomain:
    """A caller-defined 2D footprint, with explicit CRS and identity.

    Holes and disjoint parts are supported. The domain carries no assumptions
    about land, water, population, accessibility, or the observed phenomenon.
    """

    geometry: BaseGeometry
    crs: str
    name: str = "domain"
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        geometry = shapely.force_2d(self.geometry)
        if geometry.geom_type not in {"Polygon", "MultiPolygon"} or not geometry.is_valid:
            raise ValueError("Spatial domains require valid Polygon or MultiPolygon geometry")
        if not np.isfinite(geometry.area):
            raise ValueError("Domain geometry must have finite area")
        crs = CRS.from_user_input(self.crs).to_string()
        object.__setattr__(self, "geometry", geometry)
        object.__setattr__(self, "crs", crs)
        object.__setattr__(
            self, "fingerprint", hashlib.sha256(crs.encode() + b"\0" + geometry.wkb).hexdigest()
        )

    @classmethod
    def from_wkt(
        cls,
        allowed_wkt: Sequence[str],
        crs: str,
        excluded_wkt: Sequence[str] = (),
        name: str = "domain",
    ) -> SpatialDomain:
        """Union supplied polygons and subtract caller-supplied exclusions."""
        if not allowed_wkt:
            raise ValueError("Provide at least one allowed polygon")
        allowed = [cls(shapely.from_wkt(wkt), crs).geometry for wkt in allowed_wkt]
        excluded = [cls(shapely.from_wkt(wkt), crs).geometry for wkt in excluded_wkt]
        geometry = shapely.union_all(allowed)
        if excluded:
            geometry = geometry.difference(shapely.union_all(excluded))
        return cls(geometry, crs, name)

    def to_crs(self, crs: str) -> SpatialDomain:
        """Reproject synchronously; no PROJ transformer is retained by lazy workers."""
        target = CRS.from_user_input(crs)
        if target == CRS.from_user_input(self.crs):
            return self
        if self.geometry.is_empty:
            return SpatialDomain(self.geometry, target.to_string(), self.name)
        transformer = Transformer.from_crs(self.crs, target, always_xy=True)
        geometry = shapely.transform(
            self.geometry,
            lambda coordinates: np.column_stack(
                transformer.transform(coordinates[:, 0], coordinates[:, 1], errcheck=True)
            ),
        )
        return SpatialDomain(geometry, target.to_string(), self.name)

    def metadata(self) -> dict[str, str]:
        """Return reproducible domain descriptors for feature provenance."""
        return {"name": self.name, "crs": self.crs, "sha256": self.fingerprint}


def domain_membership_expr(
    domain: SpatialDomain,
    *,
    x_col: str | None = None,
    y_col: str | None = None,
    geometry_col: str | None = None,
    include_boundary: bool = True,
) -> pl.Expr:
    """Classify points or complete WKT geometries; missing locations yield null.

    Inputs and domain must already share a CRS. Native GEOS objects are created
    inside each executing batch. Finite coordinates are evaluated in 2D.
    """
    if (x_col is None) != (y_col is None):
        raise ValueError("Provide both x_col and y_col")
    point_mode = x_col is not None and y_col is not None
    if not point_mode and geometry_col is None:
        raise ValueError("Provide coordinate columns or a WKT geometry column")
    columns = [x_col, y_col] if point_mode else [geometry_col]
    inputs = [name for name in columns if name is not None]
    wkb = domain.geometry.wkb

    def classify(batch: Sequence[pl.Series]) -> pl.Series:
        boundary = shapely.from_wkb(wkb)
        result = np.full(len(batch[0]), None, dtype=object)
        if point_mode:
            x = batch[0].cast(pl.Float64, strict=False).to_numpy()
            y = batch[1].cast(pl.Float64, strict=False).to_numpy()
            valid = np.isfinite(x) & np.isfinite(y)
            locations = shapely.points(x[valid], y[valid])
        else:
            raw = batch[0].cast(pl.String).to_numpy()
            parsed = shapely.from_wkt(raw)
            valid = ~shapely.is_missing(parsed) & ~shapely.is_empty(parsed)
            locations = parsed[valid]
            if not np.all(shapely.is_valid(locations)):
                raise ValueError("Input WKT geometries must be valid")
        predicate = shapely.covers if include_boundary else shapely.contains_properly
        result[valid] = predicate(boundary, locations)
        return pl.Series(result.tolist(), dtype=pl.Boolean)

    return pl.map_batches(inputs, classify, return_dtype=pl.Boolean)


def annotate_grid_coverage(
    grid: pl.DataFrame,
    domain: SpatialDomain,
    *,
    crs: str,
    geometry_col: str = "geometry",
    prefix: str = "domain",
) -> pl.DataFrame:
    """Add covered area (m²), cell-area fraction, and positive-coverage flag.

    Grid rows contain polygon WKT. Zero coverage is retained explicitly, so
    callers can choose a threshold. Areas are geometric, not population or
    observation exposure. Input geometries and row order are preserved.
    """
    require_metric_crs(crs)
    if geometry_col not in grid.columns:
        raise ValueError(f"Missing grid geometry column: {geometry_col}")
    output_cols = [f"{prefix}_area_m2", f"{prefix}_fraction", f"{prefix}_covered"]
    if set(output_cols) & set(grid.columns):
        raise ValueError("Coverage output columns already exist; choose another prefix")
    cells = shapely.from_wkt(grid[geometry_col].cast(pl.String).to_numpy())
    areas = shapely.area(cells)
    if not np.all(
        shapely.is_valid(cells)
        & np.isin(shapely.get_type_id(cells), [3, 6])
        & np.isfinite(areas)
        & (areas > 0)
    ):
        raise ValueError("Grid cells must be valid polygons with positive finite area")
    boundary = domain.to_crs(crs)
    covered = shapely.area(shapely.intersection(cells, boundary.geometry))
    fractions = np.clip(covered / areas, 0.0, 1.0)
    return grid.with_columns(
        pl.Series(output_cols[0], covered, dtype=pl.Float64),
        pl.Series(output_cols[1], fractions, dtype=pl.Float64),
        pl.Series(output_cols[2], covered > 0, dtype=pl.Boolean),
    )
