"""EventFrame: Central abstraction for event data."""

from typing import Any
import polars as pl
from eventflow.core.schema import EventSchema, EventMetadata
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
        return EventFrame(lazy_frame, self.schema, self.metadata)

    def with_metadata(self, **updates: Any) -> "EventFrame":
        """
        Create a new EventFrame with updated metadata.
        
        Args:
            **updates: Metadata fields to update
            
        Returns:
            New EventFrame with updated metadata
        """
        new_metadata = self.metadata.model_copy(update=updates)
        return EventFrame(self.lazy_frame, self.schema, new_metadata)

    def register_feature(
        self,
        name: str,
        info: dict[str, Any],
        modality: str | None = None,
    ) -> "EventFrame":
        """Return a new EventFrame with feature catalog updated.

        Args:
            name: Feature identifier to register.
            info: Arbitrary metadata describing the feature.
            modality: Optional modality hint to add to metadata.

        Returns:
            EventFrame whose metadata includes the registered feature.
        """
        catalog = dict(self.metadata.feature_catalog)
        catalog[name] = info

        modalities = set(self.metadata.output_modalities)
        if modality:
            modalities.add(modality)

        logger.debug(
            "Registering feature '%s' (modality=%s) on dataset '%s'",
            name,
            modality or "<unchanged>",
            self.metadata.dataset_name,
        )

        return self.with_metadata(
            feature_catalog=catalog,
            output_modalities=modalities,
        )

    def collect(self) -> pl.DataFrame:
        """
        Materialize the lazy frame into a DataFrame.
        
        Returns:
            Collected Polars DataFrame
        """
        logger.debug(f"Collecting EventFrame for dataset '{self.metadata.dataset_name}'")
        df = self.lazy_frame.collect()
        logger.info(f"Collected {len(df)} rows, {len(df.columns)} columns")
        return df

    def head(self, n: int = 5) -> pl.DataFrame:
        """
        Collect the first n rows.
        
        Args:
            n: Number of rows to collect
            
        Returns:
            DataFrame with first n rows
        """
        return self.lazy_frame.head(n).collect()

    def describe(self) -> pl.DataFrame:
        """
        Get descriptive statistics about the event data.
        
        Returns:
            DataFrame with statistics
        """
        return self.lazy_frame.describe()

    def select(self, *exprs: pl.Expr | str) -> "EventFrame":
        """
        Select columns from the event frame.
        
        Args:
            *exprs: Column expressions or names to select
            
        Returns:
            New EventFrame with selected columns
        """
        return self.with_lazy_frame(self.lazy_frame.select(*exprs))

    def filter(self, *predicates: pl.Expr) -> "EventFrame":
        """
        Filter rows based on predicates.
        
        Args:
            *predicates: Boolean expressions for filtering
            
        Returns:
            New EventFrame with filtered rows
        """
        return self.with_lazy_frame(self.lazy_frame.filter(*predicates))

    def with_columns(self, *exprs: pl.Expr, **named_exprs: pl.Expr) -> "EventFrame":
        """
        Add or transform columns.
        
        Args:
            *exprs: Column expressions
            **named_exprs: Named column expressions
            
        Returns:
            New EventFrame with added/transformed columns
        """
        return self.with_lazy_frame(
            self.lazy_frame.with_columns(*exprs, **named_exprs)
        )

    def sort(self, by: str | pl.Expr | list[str | pl.Expr], descending: bool = False) -> "EventFrame":
        """
        Sort the event frame.
        
        Args:
            by: Column(s) to sort by
            descending: Sort in descending order
            
        Returns:
            New sorted EventFrame
        """
        return self.with_lazy_frame(self.lazy_frame.sort(by, descending=descending))

    def count(self) -> int:
        """
        Count the number of events.
        
        Returns:
            Number of events
        """
        return self.lazy_frame.select(pl.len()).collect().item()

    def __repr__(self) -> str:
        """String representation of the EventFrame."""
        return (
            f"EventFrame(\n"
            f"  dataset={self.metadata.dataset_name},\n"
            f"  schema={self.schema.timestamp_col},\n"
            f"  crs={self.metadata.crs},\n"
            f"  time_zone={self.metadata.time_zone}\n"
            f")"
        )

    def __len__(self) -> int:
        """Get the number of events."""
        return self.count()
