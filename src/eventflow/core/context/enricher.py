"""Enricher step for pipeline integration."""

from eventflow.core.event_frame import EventFrame
from eventflow.core.pipeline import Step
from eventflow.core.context.sources import BaseContextSource
from eventflow.core.context.joiners import TemporalJoin, SpatialJoin, SpatioTemporalJoin
from eventflow.core.utils import get_logger

logger = get_logger(__name__)


class EnricherStep(Step):
    """
    Pipeline step that enriches events with context data.
    
    This step loads a context source and joins it with the event data
    using the specified join strategy.
    """

    def __init__(
        self,
        source: BaseContextSource,
        joiner: TemporalJoin | SpatialJoin | SpatioTemporalJoin,
    ) -> None:
        """
        Initialize enricher step.
        
        Args:
            source: Context data source
            joiner: Join strategy
        """
        self.source = source
        self.joiner = joiner

    def run(self, event_frame: EventFrame) -> EventFrame:
        """
        Enrich event data with context.
        
        Args:
            event_frame: Input EventFrame
            
        Returns:
            Enriched EventFrame
        """
        logger.info(f"Enriching with context source: {self.source.__class__.__name__}")
        # Load context data
        context_frame = self.source.load()
        context_schema = self.source.schema
        logger.debug(f"Context schema attributes: {context_schema.attribute_cols}")
        
        # Apply join
        logger.debug(f"Applying join strategy: {self.joiner}")
        enriched = self.joiner.join(event_frame, context_frame, context_schema)
        logger.info("Context enrichment completed")
        
        return enriched

    def __repr__(self) -> str:
        """String representation."""
        return f"EnricherStep(source={self.source}, joiner={self.joiner})"
