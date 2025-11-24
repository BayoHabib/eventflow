"""Base recipe abstraction."""

from abc import ABC, abstractmethod
from eventflow.core.pipeline import Pipeline
from eventflow.core.event_frame import EventFrame
from eventflow.core.schema import RecipeConfig


class BaseRecipe(ABC):
    """
    Base class for all recipes.
    
    A recipe encapsulates a complete feature engineering pipeline
    for a specific dataset and use case.
    """

    def __init__(self, config: RecipeConfig) -> None:
        """
        Initialize recipe.
        
        Args:
            config: Recipe configuration
        """
        self.config = config
        self.name = config.recipe

    @abstractmethod
    def build_pipeline(self) -> Pipeline:
        """
        Build the transformation pipeline.
        
        Returns:
            Pipeline with all transformation steps
        """
        pass

    def run(self, event_frame: EventFrame) -> EventFrame:
        """
        Run the recipe on event data.
        
        Args:
            event_frame: Input EventFrame
            
        Returns:
            Transformed EventFrame with features
        """
        pipeline = self.build_pipeline()
        return pipeline.run(event_frame)

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name={self.name})"
