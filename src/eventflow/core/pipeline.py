"""Pipeline orchestration for composable transformations."""

from abc import ABC, abstractmethod
from typing import Any
from eventflow.core.event_frame import EventFrame
from eventflow.core.utils import get_logger

logger = get_logger(__name__)


class Step(ABC):
    """
    Base class for pipeline steps.
    
    A step is a stateless transformation that takes an EventFrame
    and returns a new EventFrame.
    """

    @abstractmethod
    def run(self, event_frame: EventFrame) -> EventFrame:
        """
        Execute the step transformation.
        
        Args:
            event_frame: Input EventFrame
            
        Returns:
            Transformed EventFrame
        """
        pass

    def __repr__(self) -> str:
        """String representation of the step."""
        return f"{self.__class__.__name__}()"


class Pipeline:
    """
    A pipeline is a sequence of steps applied to an EventFrame.
    
    Each step transforms the EventFrame and passes it to the next step.
    """

    def __init__(self, steps: list[Step]) -> None:
        """
        Initialize a pipeline.
        
        Args:
            steps: List of Step instances to apply in sequence
        """
        self.steps = steps
        logger.info(f"Created pipeline with {len(steps)} steps: {[s.__class__.__name__ for s in steps]}")

    def run(self, event_frame: EventFrame) -> EventFrame:
        """
        Run the pipeline on an EventFrame.
        
        Args:
            event_frame: Input EventFrame
            
        Returns:
            Transformed EventFrame after all steps
        """
        logger.info(f"Starting pipeline execution with {len(self.steps)} steps")
        current = event_frame
        
        for i, step in enumerate(self.steps, 1):
            step_name = step.__class__.__name__
            logger.info(f"Step {i}/{len(self.steps)}: Executing {step_name}")
            try:
                current = step.run(current)
                logger.debug(f"Step {i}/{len(self.steps)}: {step_name} completed successfully")
            except Exception as e:
                logger.error(f"Step {i}/{len(self.steps)}: {step_name} failed with error: {e}")
                raise
        
        logger.info(f"Pipeline execution completed successfully")
        return current

    def add_step(self, step: Step) -> "Pipeline":
        """
        Add a step to the pipeline.
        
        Args:
            step: Step to add
            
        Returns:
            Self for chaining
        """
        self.steps.append(step)
        return self

    def __repr__(self) -> str:
        """String representation of the pipeline."""
        step_names = [step.__class__.__name__ for step in self.steps]
        return f"Pipeline({' -> '.join(step_names)})"

    def __len__(self) -> int:
        """Get the number of steps."""
        return len(self.steps)


class LambdaStep(Step):
    """
    A step that wraps a function.
    
    Useful for quick transformations without defining a new class.
    """

    def __init__(self, fn: Any, name: str | None = None) -> None:
        """
        Initialize a lambda step.
        
        Args:
            fn: Function that takes and returns an EventFrame
            name: Optional name for the step
        """
        self.fn = fn
        self.name = name or "LambdaStep"

    def run(self, event_frame: EventFrame) -> EventFrame:
        """Execute the wrapped function."""
        return self.fn(event_frame)

    def __repr__(self) -> str:
        """String representation."""
        return f"LambdaStep({self.name})"


class ConditionalStep(Step):
    """
    A step that conditionally applies another step.
    """

    def __init__(
        self,
        condition: Any,
        if_true: Step,
        if_false: Step | None = None,
    ) -> None:
        """
        Initialize a conditional step.
        
        Args:
            condition: Function that takes EventFrame and returns bool
            if_true: Step to apply if condition is True
            if_false: Optional step to apply if condition is False
        """
        self.condition = condition
        self.if_true = if_true
        self.if_false = if_false

    def run(self, event_frame: EventFrame) -> EventFrame:
        """Execute conditional logic."""
        if self.condition(event_frame):
            return self.if_true.run(event_frame)
        elif self.if_false is not None:
            return self.if_false.run(event_frame)
        else:
            return event_frame

    def __repr__(self) -> str:
        """String representation."""
        return f"ConditionalStep({self.if_true} | {self.if_false})"


class ParallelSteps(Step):
    """
    Apply multiple steps in parallel and merge results.
    
    Note: This doesn't actually execute in parallel (Polars handles that),
    but it applies multiple transformations to the same input and merges them.
    """

    def __init__(self, steps: list[Step]) -> None:
        """
        Initialize parallel steps.
        
        Args:
            steps: Steps to apply in parallel
        """
        self.steps = steps

    def run(self, event_frame: EventFrame) -> EventFrame:
        """
        Apply all steps and merge results.
        
        The merge strategy is to collect all new columns from each step.
        """
        # Apply first step to get base
        result = self.steps[0].run(event_frame)
        
        # Apply remaining steps and collect new columns
        for step in self.steps[1:]:
            temp = step.run(event_frame)
            # Get columns that weren't in original
            new_cols = [
                col for col in temp.lazy_frame.columns
                if col not in event_frame.lazy_frame.columns
            ]
            # Add new columns to result
            if new_cols:
                result = result.with_columns([
                    temp.lazy_frame.select(new_cols)
                ])
        
        return result

    def __repr__(self) -> str:
        """String representation."""
        step_names = [step.__class__.__name__ for step in self.steps]
        return f"ParallelSteps([{', '.join(step_names)}])"
