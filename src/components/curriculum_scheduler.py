"""
Curriculum scheduler for dynamically adjusting recurrent_weight during training.
"""

import math
from typing import Optional


class RecurrentWeightCurriculumScheduler:
    """
    Scheduler for curriculum learning of recurrent_weight.
    
    Gradually increases (or decreases) the recurrent_weight from a starting value
    to an ending value over the course of training.
    """
    
    def __init__(
        self,
        start_value: float,
        end_value: float,
        total_steps: int,
        schedule: str = "linear",
        warmup_steps: Optional[int] = None,
    ):
        """
        Initialize the curriculum scheduler.
        
        Args:
            start_value: Initial recurrent_weight value
            end_value: Final recurrent_weight value
            total_steps: Total number of training steps
            schedule: Schedule type ('linear', 'cosine', 'exponential', 'step')
            warmup_steps: Number of steps to reach end_value (None = use total_steps)
        """
        self.start_value = start_value
        self.end_value = end_value
        self.total_steps = total_steps
        self.schedule = schedule.lower()
        self.warmup_steps = warmup_steps if warmup_steps is not None else total_steps
        
        # Validate parameters
        if self.warmup_steps > self.total_steps:
            raise ValueError(
                f"warmup_steps ({self.warmup_steps}) cannot exceed total_steps ({self.total_steps})"
            )
        
        if self.schedule not in ["linear", "cosine", "exponential", "step"]:
            raise ValueError(
                f"Invalid schedule '{self.schedule}'. Must be one of: linear, cosine, exponential, step"
            )
    
    def get_value(self, current_step: int) -> float:
        """
        Get the recurrent_weight value for the current training step.
        
        Args:
            current_step: Current training step (0-indexed)
            
        Returns:
            The recurrent_weight value for this step
        """
        if current_step >= self.warmup_steps:
            return self.end_value
        
        if current_step <= 0:
            return self.start_value
        
        # Calculate progress ratio (0 to 1)
        progress = current_step / self.warmup_steps
        
        # Apply schedule
        if self.schedule == "linear":
            alpha = progress
        elif self.schedule == "cosine":
            # Cosine annealing: smooth transition
            alpha = (1 - math.cos(progress * math.pi)) / 2
        elif self.schedule == "exponential":
            # Exponential growth: slow start, fast finish
            alpha = progress ** 2
        elif self.schedule == "step":
            # Step function: sudden change at halfway point
            alpha = 1.0 if progress >= 0.5 else 0.0
        else:
            alpha = progress  # fallback to linear
        
        # Interpolate between start and end values
        value = self.start_value + (self.end_value - self.start_value) * alpha
        
        return value
    
    def get_state_dict(self) -> dict:
        """Get the state dictionary for checkpointing."""
        return {
            "start_value": self.start_value,
            "end_value": self.end_value,
            "total_steps": self.total_steps,
            "schedule": self.schedule,
            "warmup_steps": self.warmup_steps,
        }
    
    def load_state_dict(self, state_dict: dict):
        """Load state from a checkpoint."""
        self.start_value = state_dict["start_value"]
        self.end_value = state_dict["end_value"]
        self.total_steps = state_dict["total_steps"]
        self.schedule = state_dict["schedule"]
        self.warmup_steps = state_dict["warmup_steps"]
    
    def __repr__(self) -> str:
        return (
            f"RecurrentWeightCurriculumScheduler("
            f"start={self.start_value}, end={self.end_value}, "
            f"schedule={self.schedule}, warmup_steps={self.warmup_steps})"
        )

