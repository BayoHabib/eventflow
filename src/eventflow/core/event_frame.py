"""EventFrame: Central abstraction for event data."""

from typing import Any

import polars as pl

from eventflow.core.schema import (
    ContextRequirementState,
    EventMetadata,
    EventSchema,
    FeatureProvenance,
    OutputModality,
)
from eventflow.core.utils import get_logger

logger = get_logger(__name__)


class EventFrame:
    """
    Central abstraction for event data.

    Wraps a Polars LazyFrame with schema and metadata, providing a rich API
    for spatio-temporal transformations while maintaining lazy evaluation.

    Attributes:
        lazy_frame: The underlying Polars LazyFrame
        schema: The event schema describing column structure
        metadata: Metadata about the dataset (CRS, time zone, etc.)
    """

    def __init__(
        self,
        lazy_frame: pl.LazyFrame,
        schema: EventSchema,
        metadata: EventMetadata,
    ) -> None:
        """
        Initialize an EventFrame.

        Args:
            lazy_frame: Polars LazyFrame containing the event data
            schema: Schema describing the event structure
            metadata: Metadata about the dataset
        """
        self.lazy_frame = lazy_frame

        # Synchronize schema/metadata enhanced descriptors to prevent drift.
        combined_modalities = set(schema.output_modalities)
        combined_modalities.update(OutputModality(mod) for mod in metadata.output_modalities)

        if combined_modalities != set(schema.output_modalities):
            schema = schema.model_copy(update={"output_modalities": combined_modalities})

        metadata_modalities = {mod.value for mod in combined_modalities}
        if metadata_modalities != set(metadata.output_modalities):
            metadata = metadata.model_copy(update={"output_modalities": metadata_modalities})

        combined_provenance = dict(metadata.feature_provenance)
        combined_provenance.update(schema.feature_provenance)

        if combined_provenance != metadata.feature_provenance:
            metadata = metadata.model_copy(update={"feature_provenance": combined_provenance})
        if combined_provenance != schema.feature_provenance:
            schema = schema.model_copy(update={"feature_provenance": combined_provenance})

        combined_requirements = _merge_context_requirements(
            metadata.context_requirements, schema.context_requirements
        )
        if combined_requirements != metadata.context_requirements:
            metadata = metadata.model_copy(update={"context_requirements": combined_requirements})
        if combined_requirements != schema.context_requirements:
            schema = schema.model_copy(update={"context_requirements": combined_requirements})

        self.schema = schema
        self.metadata = metadata
        logger.debug(
            f"Created EventFrame for dataset '{metadata.dataset_name}' "
            f"with schema timestamp_col='{schema.timestamp_col}'"
        )

    def with_lazy_frame(self, lazy_frame: pl.LazyFrame) -> "EventFrame":
        """
        Create a new EventFrame with a different LazyFrame.

        Args:
            lazy_frame: New LazyFrame to wrap

        Returns:
            New EventFrame with updated LazyFrame
        """
        return self._spawn(lazy_frame=lazy_frame)

    def with_metadata(self, **updates: Any) -> "EventFrame":
        """
        Create a new EventFrame with updated metadata.

        Args:
            **updates: Metadata fields to update

        Returns:
            New EventFrame with updated metadata
        """
        new_metadata = self.metadata.model_copy(update=updates)
        return self._spawn(metadata=new_metadata)

    def with_schema(self, **updates: Any) -> "EventFrame":
        """Return a new EventFrame with schema updates applied immutably."""

        new_schema = self.schema.model_copy(update=updates)
        return self._spawn(schema=new_schema)

    def _spawn(
        self,
        *,
        lazy_frame: pl.LazyFrame | None = None,
        schema: EventSchema | None = None,
        metadata: EventMetadata | None = None,
    ) -> "EventFrame":
        """Internal helper to create new EventFrame instances preserving invariants."""

        return EventFrame(
            lazy_frame or self.lazy_frame,
            schema or self.schema,
            metadata or self.metadata,
        )

    def add_output_modality(self, modality: str) -> "EventFrame":
        """Return a new EventFrame with the given output modality registered."""
        mod_enum = _coerce_modality(modality)

        schema_modalities = set(self.schema.output_modalities)
        if mod_enum not in schema_modalities:
            schema_modalities.add(mod_enum)

        metadata_modalities = set(self.metadata.output_modalities)
        metadata_modalities.add(mod_enum.value)

        return self._spawn(
            schema=self.schema.model_copy(update={"output_modalities": schema_modalities}),
            metadata=self.metadata.model_copy(update={"output_modalities": metadata_modalities}),
        )

    def register_feature(
        self,
        name: str,
        info: dict[str, Any],
        modality: str | None = None,
        *,
        provenance: FeatureProvenance | None = None,
    ) -> "EventFrame":
        """Return a new EventFrame with feature catalog updated.

        Args:
            name: Feature identifier to register.
            info: Arbitrary metadata describing the feature.
            modality: Optional modality hint to add to metadata.
            provenance: Optional provenance record; inferred from *info* when omitted.

        Returns:
            EventFrame whose metadata includes the registered feature.
        """
        catalog = dict(self.metadata.feature_catalog)
        catalog[name] = info

        if provenance is None:
            provenance = FeatureProvenance(
                produced_by=info.get("source_step"),
                inputs=list(info.get("inputs", [])),
                tags=set(info.get("tags", [])),
                description=info.get("description"),
                metadata={
                    k: v
                    for k, v in info.items()
                    if k not in {"source_step", "inputs", "tags", "description"}
                },
            )

        metadata_provenance = dict(self.metadata.feature_provenance)
        metadata_provenance[name] = provenance

        schema_provenance = dict(self.schema.feature_provenance)
        schema_provenance[name] = provenance

        schema_modalities = set(self.schema.output_modalities)
        metadata_modalities = set(self.metadata.output_modalities)
        if modality:
            mod_enum = _coerce_modality(modality)
            schema_modalities.add(mod_enum)
            metadata_modalities.add(mod_enum.value)

        logger.debug(
            "Registering feature '%s' (modality=%s) on dataset '%s'",
            name,
            modality or "<unchanged>",
            self.metadata.dataset_name,
        )

        return self._spawn(
            schema=self.schema.model_copy(
                update={
                    "feature_provenance": schema_provenance,
                    "output_modalities": schema_modalities,
                }
            ),
            metadata=self.metadata.model_copy(
                update={
                    "feature_catalog": catalog,
                    "feature_provenance": metadata_provenance,
                    "output_modalities": metadata_modalities,
                }
            ),
        )

    def require_context(
        self,
        *,
        spatial_crs: str | None = None,
        temporal_resolution: str | None = None,
        context_tags: set[str] | None = None,
        notes: dict[str, Any] | None = None,
    ) -> "EventFrame":
        """Return a new EventFrame with updated context requirements."""

        metadata_req = self.metadata.context_requirements.model_copy(deep=True)
        schema_req = self.schema.context_requirements.model_copy(deep=True)

        for requirement in (metadata_req, schema_req):
            if spatial_crs:
                requirement.spatial_crs = spatial_crs
            if temporal_resolution:
                requirement.temporal_resolution = temporal_resolution
            if context_tags:
                requirement.required_context.update(context_tags)
            if notes:
                requirement.notes.update(notes)

        merged = _merge_context_requirements(metadata_req, schema_req)

        return self._spawn(
            schema=self.schema.model_copy(update={"context_requirements": merged}),
            metadata=self.metadata.model_copy(update={"context_requirements": merged}),
        )

    def collect(self) -> pl.DataFrame:
        """Materialize the lazy frame into a DataFrame."""

        logger.debug("Collecting EventFrame for dataset '%s'", self.metadata.dataset_name)
        df = self.lazy_frame.collect()
        logger.info("Collected %s rows, %s columns", len(df), len(df.columns))
        return df

    def head(self, n: int = 5) -> pl.DataFrame:
        """Collect the first *n* rows."""

        return self.lazy_frame.head(n).collect()

    def describe(self) -> pl.DataFrame:
        """Return descriptive statistics about the event data."""

        return self.lazy_frame.describe()

    def select(self, *exprs: pl.Expr | str) -> "EventFrame":
        """Return a new EventFrame selecting the provided expressions."""

        return self.with_lazy_frame(self.lazy_frame.select(*exprs))

    def filter(self, *predicates: pl.Expr) -> "EventFrame":
        """Return a new EventFrame filtered by the predicates."""

        return self.with_lazy_frame(self.lazy_frame.filter(*predicates))

    def with_columns(self, *exprs: pl.Expr, **named_exprs: pl.Expr) -> "EventFrame":
        """Return a new EventFrame with additional or transformed columns."""

        return self.with_lazy_frame(self.lazy_frame.with_columns(*exprs, **named_exprs))

    def sort(
        self,
        by: str | pl.Expr | list[str | pl.Expr],
        descending: bool = False,
    ) -> "EventFrame":
        """Return a new EventFrame sorted by *by*."""

        return self.with_lazy_frame(self.lazy_frame.sort(by, descending=descending))

    def count(self) -> int:
        """Return the number of events."""

        df = self.lazy_frame.select(pl.len().alias("_count"))
        result = df.collect()
        rows = result.rows()
        return int(rows[0][0]) if rows else 0

    def __repr__(self) -> str:
        """String representation of the EventFrame."""

        return (
            "EventFrame(\n"
            f"  dataset={self.metadata.dataset_name},\n"
            f"  schema={self.schema.timestamp_col},\n"
            f"  crs={self.metadata.crs},\n"
            f"  time_zone={self.metadata.time_zone}\n"
            ")"
        )

    def __len__(self) -> int:
        """Return the number of events."""

        return self.count()


def _coerce_modality(value: str | OutputModality) -> OutputModality:
    """Normalise modality input into an OutputModality enum value."""

    if isinstance(value, OutputModality):
        return value
    try:
        return OutputModality(value)
    except ValueError as exc:
        raise ValueError(f"Unsupported output modality: {value!r}") from exc


def _merge_context_requirements(
    left: ContextRequirementState, right: ContextRequirementState
) -> ContextRequirementState:
    """Combine two requirement states, preferring explicit entries."""

    merged = ContextRequirementState(
        spatial_crs=right.spatial_crs or left.spatial_crs,
        temporal_resolution=right.temporal_resolution or left.temporal_resolution,
        required_context=set(left.required_context) | set(right.required_context),
        notes={**left.notes, **right.notes},
    )

    # If both provide spatial CRS but conflict, keep the right-hand value while logging.
    if left.spatial_crs and right.spatial_crs and left.spatial_crs != right.spatial_crs:
        logger.warning(
            "Context requirement spatial CRS conflict detected: %s vs %s; keeping %s",
            left.spatial_crs,
            right.spatial_crs,
            right.spatial_crs,
        )
        merged.spatial_crs = right.spatial_crs

    if (
        left.temporal_resolution
        and right.temporal_resolution
        and left.temporal_resolution != right.temporal_resolution
    ):
        logger.warning(
            "Context requirement temporal resolution conflict detected: %s vs %s; keeping %s",
            left.temporal_resolution,
            right.temporal_resolution,
            right.temporal_resolution,
        )
        merged.temporal_resolution = right.temporal_resolution

    return merged
