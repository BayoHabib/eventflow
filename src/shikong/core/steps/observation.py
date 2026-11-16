"""Reduce point-event records onto explicitly declared observation support."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, cast

import polars as pl
from pyproj import CRS

from shikong.core.cadence import CadenceRule, cadence_columns
from shikong.core.event_frame import EventFrame
from shikong.core.observation import ObservationPanel
from shikong.core.pipeline import Step
from shikong.core.schema import FeatureProvenance


class ObservationPanelStep(Step):
    """Count raw point-event rows; preserve unknown totals and audit exclusions.

    All panel rows are returned in reference order, including rows with no
    events. Source measurements are not imputed or aggregated by this step.
    Existing provenance is retained as lineage; active schema columns describe
    only the resulting support and count/rate fields.
    """

    def __init__(
        self,
        panel: ObservationPanel,
        count_col: str = "event_count",
        recorded_col: str = "recorded_count",
        rate_col: str = "event_rate",
        offset_col: str = "log_exposure",
        events_outside: Literal["error", "ignore"] = "error",
        event_id_col: str | None = None,
        cadence_rule: CadenceRule | dict[str, Any] | None = None,
        cadence_prefix: str = "cadence",
        include_missing_times: bool = False,
        on_gap: Literal["report", "raise"] = "report",
    ) -> None:
        names = [count_col, recorded_col, rate_col, offset_col]
        if len(set(names)) != len(names) or any(not isinstance(c, str) or not c for c in names):
            raise ValueError("Panel output names must be distinct non-empty strings")
        self.panel = panel
        self.count_col, self.recorded_col = count_col, recorded_col
        self.rate_col, self.offset_col = rate_col, offset_col
        self.events_outside, self.event_id_col = events_outside, event_id_col
        self.cadence_rule = (
            CadenceRule(**cadence_rule) if isinstance(cadence_rule, dict) else cadence_rule
        )
        self.cadence_prefix, self.include_missing_times = cadence_prefix, include_missing_times
        if on_gap not in {"report", "raise"}:
            raise ValueError("on_gap must be report or raise")
        self.on_gap = on_gap

    def run(self, event_frame: EventFrame) -> EventFrame:
        """Materialize counts, attach support semantics and preserve async compatibility."""
        if (
            event_frame.metadata.time_bin
            or event_frame.schema.context_requirements.temporal_resolution
        ):
            raise ValueError(
                "Observation panels require unbinned point-event records; count before binning"
            )
        if "observation_panel" in event_frame.metadata.custom:
            raise ValueError("Cannot count an observation panel as raw point-event records")
        if CRS.from_user_input(event_frame.metadata.crs) != CRS.from_user_input(self.panel.crs):
            raise ValueError(
                "Event and panel CRS must agree; prepare the spatial reference explicitly"
            )
        timestamp = event_frame.schema.timestamp_col
        reference = self.panel.cells
        if timestamp in reference.columns and timestamp != self.panel.start_col:
            raise ValueError(
                "Event timestamp name conflicts with a panel column; rename the reference column"
            )
        output_names = [self.count_col, self.recorded_col, self.rate_col, self.offset_col]
        if timestamp in output_names:
            raise ValueError("Panel output names cannot replace the event timestamp")
        data, diagnostics = self.panel.count_events(
            event_frame.lazy_frame,
            timestamp,
            count_col=self.count_col,
            recorded_col=self.recorded_col,
            rate_col=self.rate_col,
            offset_col=self.offset_col,
            events_outside=self.events_outside,
            event_id_col=self.event_id_col,
            cadence_rule=self.cadence_rule,
            cadence_prefix=self.cadence_prefix,
            include_missing_times=self.include_missing_times,
            on_gap=self.on_gap,
        )
        if timestamp != self.panel.start_col:
            data = data.with_columns(pl.col(self.panel.start_col).alias(timestamp))
        identifiers = {*self.panel.key_cols, self.panel.x_col, self.panel.y_col}
        numeric = [
            c for c, dtype in data.schema.items() if dtype.is_numeric() and c not in identifiers
        ]
        categorical = [
            c
            for c in [
                *event_frame.schema.categorical_cols,
                *self.panel.key_cols,
                *self.panel.quality_columns,
            ]
            if c in data.columns and c not in numeric
        ]
        support = {
            **self.panel.metadata(),
            **diagnostics,
            "events_outside": self.events_outside,
            "event_id_col": self.event_id_col,
            "count_col": self.count_col,
            "recorded_col": self.recorded_col,
            "rate_col": self.rate_col if self.panel.exposure_col else None,
            "offset_col": self.offset_col if self.panel.exposure_col else None,
            "timestamp_role": "interval start",
            "spatial_role": "support reference",
            "source_numeric_cols": list(event_frame.schema.numeric_cols),
            "source_provenance_retained_as_lineage": True,
            "cadence_rule": (
                self.cadence_rule.model_dump(mode="json") if self.cadence_rule else None
            ),
            "cadence_on_gap": self.on_gap,
            "cadence_prefix": self.cadence_prefix if self.cadence_rule else None,
            "cadence_include_missing_times": (
                self.include_missing_times if self.cadence_rule else False
            ),
            "cadence_check": "evaluated" if self.cadence_rule else "unknown",
            "cadence_does_not_certify_collection_or_completeness": True,
        }
        custom = {**event_frame.metadata.custom, "observation_panel": support}
        if reference.is_empty():
            date_range = None
        else:
            date_range = (
                cast(datetime, reference[self.panel.start_col].min()).isoformat(),
                cast(datetime, reference[self.panel.end_col].max()).isoformat(),
            )
        result = (
            event_frame.with_lazy_frame(data.lazy())
            .with_schema(
                lon_col=self.panel.x_col,
                lat_col=self.panel.y_col,
                geometry_col=self.panel.geometry_col,
                numeric_cols=numeric,
                categorical_cols=list(dict.fromkeys(categorical)),
            )
            .with_metadata(custom=custom, time_bin=None, bounds=None, date_range=date_range)
        )
        dtype = self.panel.time_dtype
        if dtype.time_zone:
            result = result.with_metadata(time_zone=dtype.time_zone)
        generated = [self.count_col, self.recorded_col]
        if self.panel.exposure_col:
            generated += [self.rate_col, self.offset_col]
        if self.cadence_rule is not None:
            generated += cadence_columns(self.cadence_prefix, self.include_missing_times)
        for name in generated:
            meaning = {
                self.count_col: "Record count only for declared count-eligible support",
                self.recorded_col: "Accepted input records, including incomplete coverage",
                self.rate_col: "Complete record count divided by known positive exposure",
                self.offset_col: "Natural log of known positive exposure for observed support",
            }.get(
                name,
                "Configured cadence diagnostic; does not certify collection or batch completeness",
            )
            result = result.register_feature(
                name,
                {"source_step": "ObservationPanelStep"},
                provenance=FeatureProvenance(
                    produced_by="ObservationPanelStep",
                    inputs=[*self.panel.key_cols, timestamp],
                    tags={"spatial", "temporal", "observation"},
                    description=meaning,
                    metadata={
                        "panel_fingerprint": self.panel.fingerprint,
                        "exposure_unit": self.panel.exposure_unit,
                        "bounds": "[start, end)",
                        "missing_policy": "preserve unknown totals",
                        "quality_mode": "legacy" if self.panel.status_col else "independent",
                        "cadence_rule": (
                            self.cadence_rule.model_dump(mode="json") if self.cadence_rule else None
                        ),
                    },
                ),
            )
        return result.require_context(
            context_tags={"observation_panel"}, notes={"observation_panel": support}
        )
