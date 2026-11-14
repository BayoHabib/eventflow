"""Composable spatial-domain preprocessing without dataset-specific policies."""

from __future__ import annotations

from typing import Any, Literal

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from eventflow.core.domain import SpatialDomain, annotate_grid_coverage, domain_membership_expr
from eventflow.core.event_frame import EventFrame
from eventflow.core.pipeline import Step
from eventflow.core.schema import FeatureProvenance


class SpatialDomainConfig(BaseModel):
    """Serializable, explicit domain definition for configured pipelines."""

    model_config = ConfigDict(extra="forbid")
    allowed_wkt: list[str] = Field(min_length=1)
    excluded_wkt: list[str] = Field(default_factory=list)
    crs: str
    name: str = "domain"

    def build(self) -> SpatialDomain:
        """Validate and construct the footprint."""
        return SpatialDomain.from_wkt(**self.model_dump())


class SpatialDomainMaskConfig(BaseModel):
    """Choose annotation or explicit filtering, and boundary policy."""

    model_config = ConfigDict(extra="forbid")
    domain: SpatialDomainConfig
    mode: Literal["annotate", "filter"] = "annotate"
    include_boundary: bool = True
    output_col: str = "inside_domain"


class SpatialDomainMaskStep(Step):
    """Annotate membership by default; filtering is an explicit caller choice."""

    def __init__(
        self,
        domain: SpatialDomain | dict[str, Any],
        mode: Literal["annotate", "filter"] = "annotate",
        include_boundary: bool = True,
        output_col: str = "inside_domain",
    ) -> None:
        self.domain = (
            domain if isinstance(domain, SpatialDomain) else SpatialDomainConfig(**domain).build()
        )
        if mode not in {"annotate", "filter"}:
            raise ValueError("mode must be 'annotate' or 'filter'")
        self.mode, self.include_boundary, self.output_col = mode, include_boundary, output_col

    def run(self, event_frame: EventFrame) -> EventFrame:
        """Apply the domain in the frame's CRS without changing its schema coordinates."""
        if self.output_col in event_frame.lazy_frame.collect_schema():
            raise ValueError(f"Output column already exists: {self.output_col}")
        boundary = self.domain.to_crs(event_frame.metadata.crs)
        schema = event_frame.schema
        mask = domain_membership_expr(
            boundary,
            x_col=schema.lon_col,
            y_col=schema.lat_col,
            geometry_col=schema.geometry_col,
            include_boundary=self.include_boundary,
        )
        lf = event_frame.lazy_frame.with_columns(mask.alias(self.output_col))
        if self.mode == "filter":
            lf = lf.filter(pl.col(self.output_col).fill_null(False))
        inputs = (
            [schema.lon_col, schema.lat_col]
            if schema.lon_col is not None and schema.lat_col is not None
            else [schema.geometry_col]
        )
        provenance = FeatureProvenance(
            produced_by="SpatialDomainMaskStep",
            inputs=[name for name in inputs if name],
            tags={"spatial"},
            metadata={
                "domain": self.domain.metadata(),
                "mode": self.mode,
                "include_boundary": self.include_boundary,
            },
        )
        return (
            event_frame.with_lazy_frame(lf)
            .register_feature(
                self.output_col, {"source_step": "SpatialDomainMaskStep"}, provenance=provenance
            )
            .require_context(
                context_tags={"spatial_domain"}, notes={self.output_col: self.domain.metadata()}
            )
        )


class GridDomainCoverageStep(Step):
    """Join caller-defined cell coverage onto event rows using explicit cell IDs.

    Unknown cell IDs keep null coverage, distinct from known cells with zero
    intersection. The supplied grid IDs must be unique and match the input.
    """

    def __init__(
        self,
        domain: SpatialDomain,
        grid: pl.DataFrame,
        crs: str,
        grid_id_col: str = "grid_id",
        geometry_col: str = "geometry",
        prefix: str = "domain",
    ) -> None:
        if grid_id_col not in grid.columns or grid[grid_id_col].null_count():
            raise ValueError("Grid requires non-null cell identifiers")
        if grid[grid_id_col].n_unique() != grid.height:
            raise ValueError("Grid cell identifiers must be unique")
        self.columns = [f"{prefix}_area_m2", f"{prefix}_fraction", f"{prefix}_covered"]
        self.coverage = annotate_grid_coverage(
            grid, domain, crs=crs, geometry_col=geometry_col, prefix=prefix
        ).select([grid_id_col, *self.columns])
        self.domain, self.crs, self.grid_id_col = domain, crs, grid_id_col

    def run(self, event_frame: EventFrame) -> EventFrame:
        """Enrich observed rows; defining a complete observation panel is separate."""
        if set(self.columns) & set(event_frame.lazy_frame.collect_schema().names()):
            raise ValueError("Coverage output columns already exist; choose another prefix")
        lf = event_frame.lazy_frame.join(
            self.coverage.lazy(),
            on=self.grid_id_col,
            how="left",
            validate="m:1",
            maintain_order="left",
        )
        result = event_frame.with_lazy_frame(lf).with_schema(
            numeric_cols=list(dict.fromkeys([*event_frame.schema.numeric_cols, *self.columns[:2]]))
        )
        for column in self.columns:
            result = result.register_feature(
                column,
                {"source_step": "GridDomainCoverageStep"},
                provenance=FeatureProvenance(
                    produced_by="GridDomainCoverageStep",
                    inputs=[self.grid_id_col],
                    tags={"spatial"},
                    metadata={
                        "domain": self.domain.metadata(),
                        "coverage_crs": self.crs,
                        "area_unit": "m2",
                    },
                ),
            )
        return result.require_context(
            context_tags={"spatial_domain"}, notes={self.columns[1]: self.domain.metadata()}
        )
