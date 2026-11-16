"""Explicit observation support for count data, independent of any dataset."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import polars as pl
import shapely
from pyproj import CRS

from shikong.core.cadence import CadenceRule, cadence_columns, cadence_summary


@dataclass(frozen=True, init=False, eq=False)
class ObservationPanel:
    """A caller-supplied table of non-overlapping half-open observation intervals.

    Every row defines one spatial unit/stratum and [start, end). Domain,
    collection activity and input-batch completeness can be declared separately;
    unspecified quality stays unknown. A consolidated legacy status is accepted
    for compatibility, without expanding it into independent declarations.
    Exposure is optional and never inferred from geometry. Reference coordinates
    describe the support, not individual event locations.
    """

    key_cols: tuple[str, ...]
    start_col: str
    end_col: str
    status_col: str | None
    domain_col: str | None
    collection_col: str | None
    completeness_col: str | None
    exposure_col: str | None
    exposure_unit: str | None
    x_col: str | None
    y_col: str | None
    geometry_col: str | None
    crs: str
    time_dtype: pl.Datetime
    fingerprint: str
    _cells: pl.DataFrame = field(repr=False)

    def __init__(
        self,
        cells: pl.DataFrame,
        crs: str,
        key_cols: Sequence[str] = ("unit_id",),
        start_col: str = "start",
        end_col: str = "end",
        status_col: str | None = None,
        exposure_col: str | None = None,
        exposure_unit: str | None = None,
        x_col: str | None = "x",
        y_col: str | None = "y",
        geometry_col: str | None = None,
        domain_col: str | None = None,
        collection_col: str | None = None,
        completeness_col: str | None = None,
    ) -> None:
        if isinstance(key_cols, str):
            raise ValueError("key_cols must be a sequence of column names, not a string")
        independent = any(c is not None for c in (domain_col, collection_col, completeness_col))
        if independent and (status_col is not None or "observation_status" in cells.columns):
            raise ValueError(
                "Do not mix legacy observation_status with independent quality declarations"
            )
        if status_col is None and not independent and "observation_status" in cells.columns:
            status_col = "observation_status"  # Compatibility only; no new quality inference.
        config = {
            "key_cols": tuple(key_cols),
            "start_col": start_col,
            "end_col": end_col,
            "status_col": status_col,
            "domain_col": domain_col,
            "collection_col": collection_col,
            "completeness_col": completeness_col,
            "exposure_col": exposure_col,
            "exposure_unit": exposure_unit,
            "x_col": x_col,
            "y_col": y_col,
            "geometry_col": geometry_col,
            "crs": CRS.from_user_input(crs).to_string(),
        }
        for name, value in config.items():
            object.__setattr__(self, name, value)
        quality = [
            c for c in (status_col, domain_col, collection_col, completeness_col) if c is not None
        ]
        roles = [*self.key_cols, start_col, end_col, *quality]
        if not self.key_cols or len(set(roles)) != len(roles):
            raise ValueError("Panel keys and temporal/status columns must be distinct")
        if exposure_col is not None:
            if exposure_col in roles or not exposure_unit or not exposure_unit.strip():
                raise ValueError(
                    "Exposure requires a distinct column and an explicit non-empty unit"
                )
            roles.append(exposure_col)
        elif exposure_unit is not None:
            raise ValueError("exposure_unit requires exposure_col")
        if (x_col is None) != (y_col is None):
            raise ValueError("Provide both panel coordinate columns or neither")
        spatial = [x_col, y_col] if x_col is not None and y_col is not None else []
        if geometry_col is not None:
            spatial.append(geometry_col)
        if not spatial or len(set(spatial)) != len(spatial) or set(spatial) & set(roles):
            raise ValueError("Panel needs distinct coordinate or geometry columns")
        required = [*roles, *spatial]
        if any(not isinstance(c, str) or not c for c in required):
            raise ValueError("Panel column names must be non-empty strings")
        if any(name not in cells.columns for name in required):
            raise ValueError(f"Missing panel columns: {sorted(set(required) - set(cells.columns))}")
        data = cells.clone()
        mandatory = [*self.key_cols, start_col, end_col, *([status_col] if status_col else [])]
        if any(data[c].null_count() for c in mandatory):
            raise ValueError("Panel keys, times and observation status cannot contain nulls")
        _validate_finite_keys(data, self.key_cols)
        start_dtype, end_dtype = data.schema[start_col], data.schema[end_col]
        if not isinstance(start_dtype, pl.Datetime) or start_dtype != end_dtype:
            raise ValueError("Panel bounds require identical Datetime precision and time zone")
        object.__setattr__(self, "time_dtype", start_dtype)
        if data.filter(pl.col(start_col) >= pl.col(end_col)).height:
            raise ValueError("Every panel interval must have start < end")
        if status_col is not None and not set(data[status_col].to_list()) <= {
            "observed",
            "missing",
            "outside",
        }:
            raise ValueError("Observation status must be observed, missing or outside")
        if domain_col is not None:
            if data.schema[domain_col] not in (pl.Boolean, pl.Null):
                raise ValueError("Domain membership must be Boolean with null for unknown")
            data = data.with_columns(pl.col(domain_col).cast(pl.Boolean))
        for column, allowed in [
            (collection_col, {"active", "inactive", "unknown", None}),
            (completeness_col, {"complete", "incomplete", "unknown", None}),
        ]:
            if column is not None and not set(data[column].to_list()) <= allowed:
                raise ValueError(f"Invalid quality declaration in {column!r}")
        # Sort within each stratum. Adjacent [a,b), [b,c) intervals are valid.
        ordered = data.sort([*self.key_cols, start_col])
        previous_end = pl.col(end_col).shift(1).over(list(self.key_cols))
        if ordered.select((pl.col(start_col) < previous_end).any()).item():
            raise ValueError("Panel intervals must not overlap within the same keys")
        if x_col is not None and y_col is not None:
            if (
                not all(data[c].dtype.is_numeric() for c in (x_col, y_col))
                or not np.isfinite(data.select(x_col, y_col).to_numpy().astype(float)).all()
            ):
                raise ValueError("Panel coordinates must be finite numeric values")
        if geometry_col is not None:
            geometries = shapely.from_wkt(data[geometry_col].to_numpy())
            if not np.all(shapely.is_valid(geometries) & ~shapely.is_empty(geometries)):
                raise ValueError("Panel geometries must be valid and non-empty")
        if exposure_col is not None:
            if not data[exposure_col].dtype.is_numeric():
                raise ValueError("Exposure must be numeric with explicit nulls for unknown values")
            exposure = pl.col(exposure_col)
            numeric = exposure.cast(pl.Float64)
            invalid = exposure.is_not_null() & (
                ~numeric.is_finite() | (exposure < 0) | ((exposure > 0) & (numeric == 0))
            )
            if data.filter(invalid).height:
                raise ValueError(
                    "Exposure must be finite and non-negative, representable in Float64, or null"
                )
            if data.filter(self.outside_expr() & (exposure > 0)).height:
                raise ValueError("Outside-domain rows cannot have positive exposure")
        object.__setattr__(self, "_cells", data)
        signature = json.dumps(
            {
                "keys": self.key_cols,
                "start": start_col,
                "end": end_col,
                "status": status_col,
                "quality_contract": 2,
                "domain": domain_col,
                "collection": collection_col,
                "completeness": completeness_col,
                "exposure": exposure_col,
                "unit": exposure_unit,
                "crs": self.crs,
                "x": x_col,
                "y": y_col,
                "geometry": geometry_col,
                "schema": str(data.schema),
                "cells": data.write_json(),
            },
            sort_keys=True,
        )
        object.__setattr__(self, "fingerprint", hashlib.sha256(signature.encode()).hexdigest())

    @property
    def cells(self) -> pl.DataFrame:
        """Return a copy; mutating a caller's DataFrame does not alter the panel."""
        return self._cells.clone()

    def metadata(self) -> dict[str, Any]:
        """Serializable observation support and exposure conventions."""
        return {
            "fingerprint": self.fingerprint,
            "key_cols": list(self.key_cols),
            "start_col": self.start_col,
            "end_col": self.end_col,
            "status_col": self.status_col,
            "quality_mode": "legacy" if self.status_col else "independent",
            "domain_col": self.domain_col,
            "collection_col": self.collection_col,
            "completeness_col": self.completeness_col,
            "unspecified_quality": "unknown",
            "quality_contract": 2,
            "bounds": "[start, end)",
            "exposure_col": self.exposure_col,
            "exposure_unit": self.exposure_unit,
            "crs": self.crs,
            "x_col": self.x_col,
            "y_col": self.y_col,
            "geometry_col": self.geometry_col,
            "timestamp_precision": self.time_dtype.time_unit,
            "time_zone": self.time_dtype.time_zone,
            "rows": self._cells.height,
            "status_meaning": (
                {
                    "observed": "legacy consolidated eligibility declaration",
                    "missing": "unknown or incomplete collection",
                    "outside": "excluded from the observation domain",
                }
                if self.status_col
                else {}
            ),
            "independent_quality_meaning": {
                "domain": "explicit reference membership, nullable when supplied",
                "collection": "declared active/inactive/unknown collection",
                "completeness": "declared complete/incomplete/unknown input batch",
            },
        }

    def count_events(
        self,
        events: pl.LazyFrame,
        timestamp_col: str,
        *,
        count_col: str = "event_count",
        recorded_col: str = "recorded_count",
        rate_col: str = "event_rate",
        offset_col: str = "log_exposure",
        events_outside: Literal["error", "ignore"] = "error",
        event_id_col: str | None = None,
        cadence_rule: CadenceRule | None = None,
        cadence_prefix: str = "cadence",
        include_missing_times: bool = False,
        on_gap: Literal["report", "raise"] = "report",
    ) -> tuple[pl.DataFrame, dict[str, int]]:
        """Count input records on explicit support; only eligible support gets zeros.

        Missing rows can have recorded_count > 0 while their total remains null.
        Outside-support records error by default; ignoring is explicit and audited.
        The returned diagnostics count accepted/excluded input rows, not events
        inferred from unobserved intervals. No input measurements are summed.
        """
        if on_gap not in {"report", "raise"}:
            raise ValueError("on_gap must be report or raise")
        output_cols = [count_col, recorded_col]
        if self.exposure_col is not None:
            output_cols += [rate_col, offset_col]
        if len(set(output_cols)) != len(output_cols) or set(output_cols) & set(self._cells.columns):
            raise ValueError("Panel output names must be distinct and absent from the reference")
        if cadence_rule is not None:
            extra = cadence_columns(cadence_prefix, include_missing_times)
            if set(extra) & set(output_cols):
                raise ValueError("Cadence output columns collide with count output names")
        reference, accepted, diagnostics, row_id = self._match_events(
            events,
            timestamp_col,
            events_outside=events_outside,
            event_id_col=event_id_col,
            reserved_names=output_cols,
        )
        counts = accepted.group_by(row_id).agg(pl.len().cast(pl.Int64).alias(recorded_col))
        output = reference.join(
            counts, on=row_id, how="left", validate="1:1", maintain_order="left"
        )
        output = output.with_columns(pl.col(recorded_col).fill_null(0))
        output = output.with_columns(
            pl.when(self.complete_expr())
            .then(pl.col(recorded_col))
            .otherwise(None)
            .cast(pl.Int64)
            .alias(count_col)
        )
        if self.exposure_col is not None:
            exposure = pl.col(self.exposure_col).cast(pl.Float64)
            if output.filter((exposure == 0) & (pl.col(recorded_col) > 0)).height:
                raise ValueError("Recorded events cannot have known zero exposure")
            known = self.complete_expr() & (exposure > 0)
            output = output.with_columns(
                pl.when(known).then(pl.col(count_col) / exposure).otherwise(None).alias(rate_col),
                pl.when(known).then(exposure.log()).otherwise(None).alias(offset_col),
            )
            if output.filter(pl.col(rate_col).is_not_null() & ~pl.col(rate_col).is_finite()).height:
                raise ValueError("Normalized event rate is not finite; check the exposure scale")
        if cadence_rule is not None:
            summary = self._cadence_summary(
                reference,
                accepted,
                row_id,
                timestamp_col,
                cadence_rule,
                cadence_prefix,
                include_missing_times,
                on_gap,
            )
            output = output.join(
                summary, on=row_id, how="left", validate="1:1", maintain_order="left"
            )
        return output.drop(row_id), diagnostics

    def complete_expr(self) -> pl.Expr:
        """Count eligibility: independent declarations are never filled from records."""
        if self.status_col is not None:
            return pl.col(self.status_col) == "observed"
        domain = pl.col(self.domain_col) if self.domain_col else pl.lit(True)
        active = pl.col(self.collection_col) == "active" if self.collection_col else pl.lit(False)
        complete = (
            pl.col(self.completeness_col) == "complete" if self.completeness_col else pl.lit(False)
        )
        return (domain & active & complete).fill_null(False)

    def outside_expr(self) -> pl.Expr:
        if self.status_col is not None:
            return pl.col(self.status_col) == "outside"
        return (~pl.col(self.domain_col)).fill_null(False) if self.domain_col else pl.lit(False)

    def applicable_expr(self) -> pl.Expr:
        """Schedule applicability follows declared domain, never record presence."""
        if self.status_col is not None:
            return pl.col(self.status_col) != "outside"
        return pl.col(self.domain_col) if self.domain_col else pl.lit(True)

    @property
    def quality_columns(self) -> list[str]:
        return [
            c
            for c in (self.status_col, self.domain_col, self.collection_col, self.completeness_col)
            if c is not None
        ]

    def check_cadence(
        self,
        events: pl.LazyFrame,
        timestamp_col: str,
        rule: CadenceRule | None = None,
        *,
        prefix: str = "cadence",
        include_missing_times: bool = False,
        events_outside: Literal["error", "ignore"] = "error",
        on_gap: Literal["report", "raise"] = "report",
    ) -> tuple[pl.DataFrame, dict[str, int]]:
        """Report scheduled gaps without adding rows or changing declared quality."""
        if on_gap not in {"report", "raise"}:
            raise ValueError("on_gap must be report or raise")
        reference, accepted, diagnostics, row_id = self._match_events(
            events,
            timestamp_col,
            events_outside=events_outside,
            event_id_col=None,
            reserved_names=[],
        )
        summary = self._cadence_summary(
            reference, accepted, row_id, timestamp_col, rule, prefix, include_missing_times, on_gap
        )
        return (
            reference.join(summary, on=row_id, how="left", maintain_order="left").drop(row_id),
            diagnostics,
        )

    def _cadence_summary(
        self,
        reference: pl.DataFrame,
        accepted: pl.DataFrame,
        row_id: str,
        timestamp_col: str,
        rule: CadenceRule | None,
        prefix: str,
        include_missing_times: bool,
        on_gap: str,
    ) -> pl.DataFrame:
        summary = cadence_summary(
            reference,
            accepted,
            row_id=row_id,
            timestamp_col=timestamp_col,
            start_col=self.start_col,
            end_col=self.end_col,
            applicable=self.applicable_expr(),
            time_dtype=self.time_dtype,
            rule=rule,
            prefix=prefix,
            include_missing_times=include_missing_times,
        )
        if on_gap == "raise" and summary.filter(pl.col(f"{prefix}_status") == "gaps").height:
            raise ValueError("Cadence check found missing scheduled slots")
        return summary

    def _match_events(
        self,
        events: pl.LazyFrame,
        timestamp_col: str,
        *,
        events_outside: Literal["error", "ignore"],
        event_id_col: str | None,
        reserved_names: Sequence[str],
    ) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, int], str]:
        if events_outside not in {"error", "ignore"}:
            raise ValueError("events_outside must be error or ignore")
        if timestamp_col in self.key_cols:
            raise ValueError("Event timestamp cannot also be a panel key")
        needed = list(
            dict.fromkeys(
                [*self.key_cols, timestamp_col, *([event_id_col] if event_id_col else [])]
            )
        )
        data = events.select(needed).collect()
        if any(data[c].null_count() for c in needed):
            raise ValueError("Event keys, timestamps and optional identifiers cannot contain nulls")
        for key in self.key_cols:
            if data.schema[key] != self._cells.schema[key]:
                raise ValueError(f"Event/panel key dtypes differ for {key!r}; normalize explicitly")
        _validate_finite_keys(data, self.key_cols)
        dtype = data.schema[timestamp_col]
        target_dtype = self.time_dtype
        if not isinstance(dtype, pl.Datetime) or dtype.time_unit != target_dtype.time_unit:
            raise ValueError("Event/panel timestamps require the same Datetime precision")
        if (dtype.time_zone is None) != (target_dtype.time_zone is None):
            raise ValueError("Cannot mix naive and time-zone-aware event/panel timestamps")
        if target_dtype.time_zone is not None and dtype.time_zone != target_dtype.time_zone:
            data = data.with_columns(
                pl.col(timestamp_col).dt.convert_time_zone(target_dtype.time_zone)
            )
        if event_id_col is not None and data[event_id_col].n_unique() != data.height:
            raise ValueError("Duplicate event identifiers: choose a deduplication policy first")
        taken = set(data.columns) | set(self._cells.columns) | set(reserved_names)
        row_id = _internal_name(taken, "_panel_row")
        start = _internal_name(taken, "_panel_start")
        end = _internal_name(taken, "_panel_end")
        status = _internal_name(taken, "_panel_status")
        reference = self._cells.with_row_index(row_id)
        support = reference.select(
            *self.key_cols,
            row_id,
            pl.col(self.start_col).alias(start),
            pl.col(self.end_col).alias(end),
            self.outside_expr().alias(status),
        )
        # Global time sorting guarantees within-key sorting as well.
        matched = data.sort(timestamp_col).join_asof(
            support.sort(start),
            left_on=timestamp_col,
            right_on=start,
            by=list(self.key_cols),
            strategy="backward",
            check_sortedness=False,
        )
        valid = (
            pl.col(row_id).is_not_null() & (pl.col(timestamp_col) < pl.col(end)) & ~pl.col(status)
        )
        accepted = matched.filter(valid)
        excluded = data.height - accepted.height
        if excluded and events_outside == "error":
            raise ValueError(
                f"{excluded} input records lie outside the declared observation support"
            )
        diagnostics = {
            "input_records": data.height,
            "accepted_records": accepted.height,
            "excluded_records": excluded,
        }
        return reference, accepted, diagnostics, row_id


def _internal_name(taken: set[str], stem: str) -> str:
    while stem in taken:
        stem += "_"
    taken.add(stem)
    return stem


def _validate_finite_keys(data: pl.DataFrame, keys: Sequence[str]) -> None:
    for key in keys:
        if data.schema[key].is_float() and not data[key].is_finite().all():
            raise ValueError("Panel/event numeric keys must be finite")
