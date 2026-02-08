"""
Step registry for procedure steps.
Import step classes and register them by type name.
"""

from .base_step import BaseStep, StepResult
from .insertion_step import DiathermyInsertionStep
from .trajectory_step import TrajectoryStep
from .retraction_step import RetractionStep

# Registry maps step type names to classes
STEP_REGISTRY = {
    'insertion': DiathermyInsertionStep,
    'trajectory': TrajectoryStep,
    'retraction': RetractionStep,
}


def create_step(step_config, context, prev_result=None):
    """
    Factory function to create step instance from config.

    Args:
        step_config: Dict with 'name', 'type', 'params'
        context: Shared context dict
        prev_result: Previous step's StepResult

    Returns:
        Step instance

    Raises:
        ValueError: If step type is not registered
    """
    step_type = step_config['type']
    if step_type not in STEP_REGISTRY:
        raise ValueError(f"Unknown step type: {step_type}. "
                         f"Available types: {list(STEP_REGISTRY.keys())}")

    StepClass = STEP_REGISTRY[step_type]
    return StepClass(
        name=step_config['name'],
        params=step_config['params'],
        context=context,
        prev_result=prev_result
    )


__all__ = [
    'BaseStep',
    'StepResult',
    'DiathermyInsertionStep',
    'TrajectoryStep',
    'RetractionStep',
    'STEP_REGISTRY',
    'create_step',
]
