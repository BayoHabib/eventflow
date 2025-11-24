"""Base classes for context data sources."""

from abc import ABC, abstractmethod
import polars as pl
from eventflow.core.schema import ContextSchema


class BaseContextSource(ABC):
    """
    Base class for all context data sources.
    
    A context source provides external data that can be joined with events
    (e.g., weather, demographics, special events).
    """

    @abstractmethod
    def load(self) -> pl.LazyFrame:
        """
        Load the context data.
        
        Returns:
            LazyFrame with context data
        """
        pass

    @property
    @abstractmethod
    def schema(self) -> ContextSchema:
        """
        Get the schema of the context data.
        
        Returns:
            ContextSchema describing the structure
        """
        pass

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}()"


class StaticSpatialSource(BaseContextSource):
    """
    Context source with static spatial data (no temporal dimension).
    
    Examples: Census demographics, neighborhood boundaries, POIs.
    """

    def __init__(self, data_path: str) -> None:
        """
        Initialize static spatial source.
        
        Args:
            data_path: Path to data file(s)
        """
        self.data_path = data_path

    @abstractmethod
    def load(self) -> pl.LazyFrame:
        """Load static spatial data."""
        pass

    @property
    @abstractmethod
    def schema(self) -> ContextSchema:
        """Schema must have spatial_col but not timestamp_col."""
        pass


class StaticTemporalSource(BaseContextSource):
    """
    Context source with static temporal data (no spatial dimension).
    
    Examples: Global economic indicators, holidays.
    """

    def __init__(self, data_path: str) -> None:
        """
        Initialize static temporal source.
        
        Args:
            data_path: Path to data file(s)
        """
        self.data_path = data_path

    @abstractmethod
    def load(self) -> pl.LazyFrame:
        """Load static temporal data."""
        pass

    @property
    @abstractmethod
    def schema(self) -> ContextSchema:
        """Schema must have timestamp_col but not spatial_col."""
        pass


class DynamicTemporalSource(BaseContextSource):
    """
    Context source with dynamic temporal data (time series).
    
    Examples: Weather data, stock prices.
    """

    def __init__(self, data_path: str) -> None:
        """
        Initialize dynamic temporal source.
        
        Args:
            data_path: Path to data file(s)
        """
        self.data_path = data_path

    @abstractmethod
    def load(self) -> pl.LazyFrame:
        """Load dynamic temporal data."""
        pass

    @property
    @abstractmethod
    def schema(self) -> ContextSchema:
        """Schema must have timestamp_col."""
        pass


class SpatioTemporalSource(BaseContextSource):
    """
    Context source with both spatial and temporal dimensions.
    
    Examples: Regional weather, traffic sensors, mobile network data.
    """

    def __init__(self, data_path: str) -> None:
        """
        Initialize spatio-temporal source.
        
        Args:
            data_path: Path to data file(s)
        """
        self.data_path = data_path

    @abstractmethod
    def load(self) -> pl.LazyFrame:
        """Load spatio-temporal data."""
        pass

    @property
    @abstractmethod
    def schema(self) -> ContextSchema:
        """Schema must have both timestamp_col and spatial_col."""
        pass
