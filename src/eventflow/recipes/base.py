"""Base recipe abstraction."""

from abc import ABC, abstractmethod
from eventflow.core.pipeline import Pipeline
from eventflow.core.event_frame import EventFrame
from eventflow.core.registry import StepRegistry
from eventflow.core.schema import RecipeConfig


class BaseRecipe(ABC):
    """
    Base class for all recipes.
    
    A recipe encapsulates a complete feature engineering pipeline
    for a specific dataset and use case.
    """

    def __init__(
        self,
        config: RecipeConfig,
        *,
        step_registry: StepRegistry | None = None,
    ) -> None:
        """
        Initialize recipe.
        
        Args:
            config: Recipe configuration
        """
        self.config = config
        self.name = config.recipe
        self._step_registry = step_registry

    @abstractmethod
    def build_pipeline(self) -> Pipeline:
        """
        Build the transformation pipeline.
        
        Returns:
            Pipeline with all transformation steps
        """
        pass

    def get_pipeline(self) -> Pipeline:
        """Return a pipeline, using configured steps if provided."""
        feature_cfg = self.config.features
        steps_cfg = feature_cfg.get("steps") if isinstance(feature_cfg, dict) else None
        if steps_cfg:
            if self._step_registry is None:
                raise ValueError("Step registry required when features.steps is configured")
            if not isinstance(steps_cfg, (list, tuple)):
                raise TypeError("features.steps must be a sequence of step definitions")
            return self._step_registry.build_pipeline(steps_cfg)
        return self.build_pipeline()

    def run(self, event_frame: EventFrame) -> EventFrame:
        """
        Run the recipe on event data.

        Args:
            event_frame: Input EventFrame

        Returns:
            Transformed EventFrame with features
        """
        pipeline = self.get_pipeline()
        return pipeline.run(event_frame)

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name={self.name})"
