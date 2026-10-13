"""Versioned, provenance-aware event exchange contract.

This contract is deliberately independent of any particular surveillance,
OSINT, or predictive system. It does not authorize collecting or disclosing
underlying data: adapters must enforce access control separately.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class AssertionKind(str, Enum):
    OBSERVATION = "observation"
    INFERENCE = "inference"
    PREDICTION = "prediction"
    HYPOTHESIS = "hypothesis"
    VERIFICATION = "verification"


class VerificationStatus(str, Enum):
    UNVERIFIED = "unverified"
    CORROBORATED = "corroborated"
    REFUTED = "refuted"
    INCONCLUSIVE = "inconclusive"


class DataClassification(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    RESTRICTED = "restricted"


class Provenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(min_length=1, max_length=256)
    source_record_id: str | None = Field(default=None, max_length=512)
    collected_at: datetime
    transformation_id: str | None = Field(default=None, max_length=256)
    upstream_assertion_ids: list[str] = Field(default_factory=list)

    @field_validator("collected_at")
    @classmethod
    def require_timezone(cls, dt: datetime) -> datetime:
        if dt.tzinfo is None or dt.utcoffset() is None:
            raise ValueError("timestamp must include timezone")
        return dt


class EventLocation(BaseModel):
    """A location need not have exact geographical coordinates."""
    model_config = ConfigDict(extra="forbid")

    reference_system: Literal["EPSG:4326", "site_local", "unknown"]
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    site_id: str | None = None
    zone_id: str | None = None
    spatial_precision_m: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def check_location(self) -> "EventLocation":
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError("latitude and longitude must be provided together")
        if self.latitude is not None and self.reference_system != "EPSG:4326":
            raise ValueError("geographic coordinates require EPSG:4326")
        if self.reference_system == "site_local" and not self.site_id:
            raise ValueError("site_local requires site_id")
        return self


class UnifiedEvent(BaseModel):
    """One assertion about an event; derived assertions reference their inputs.

    The envelope is not itself proof that an assertion is true.
    Verification is represented as a distinct assertion.
    """
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["0.1.0"] = "0.1.0"
    assertion_id: str = Field(min_length=1, max_length=256)
    assertion_kind: AssertionKind
    event_type: str = Field(min_length=1, max_length=256)
    occurred_at: datetime
    valid_until: datetime | None = None
    provenance: Provenance
    location: EventLocation | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED
    classification: DataClassification = DataClassification.INTERNAL
    model_id: str | None = None
    target_assertion_id: str | None = None

    @field_validator("occurred_at", "valid_until")
    @classmethod
    def require_timezone(cls, dt: datetime | None) -> datetime | None:
        if dt is not None and (dt.tzinfo is None or dt.utcoffset() is None):
            raise ValueError("timestamp must include timezone")
        return dt

    @model_validator(mode="after")
    def check_semantics(self) -> "UnifiedEvent":
        if self.valid_until is not None and self.valid_until <= self.occurred_at:
            raise ValueError("valid_until must follow occurred_at")
        if self.assertion_kind in {AssertionKind.PREDICTION, AssertionKind.INFERENCE}:
            if not self.model_id:
                raise ValueError("derived predictions and inferences require model_id")
        if self.assertion_kind == AssertionKind.VERIFICATION and not self.target_assertion_id:
            raise ValueError("verification requires target_assertion_id")
        if self.assertion_kind == AssertionKind.OBSERVATION and self.model_id:
            raise ValueError("model outputs must be represented as inferences, not observations")
        if self.assertion_id in self.provenance.upstream_assertion_ids:
            raise ValueError("an assertion cannot directly depend on itself")
        return self
