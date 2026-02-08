"""
Base step class for procedure steps.
All step implementations inherit from this class.
"""
import numpy as np
from abc import ABC, abstractmethod


class StepResult:
    """Result returned by each step after execution"""

    def __init__(self, end_position, end_orientation=None, xml_updates=None, metadata=None):
        self.end_position = np.array(end_position) if end_position is not None else None
        self.end_orientation = np.array(end_orientation) if end_orientation is not None else None
        self.xml_updates = xml_updates or {}
        self.metadata = metadata or {}

    def to_dict(self):
        return {
            'end_position': self.end_position.tolist() if self.end_position is not None else None,
            'end_orientation': self.end_orientation.tolist() if self.end_orientation is not None else None,
            'xml_updates': self.xml_updates,
            'metadata': self.metadata
        }


class BaseStep(ABC):
    """
    Base class for all procedure steps.

    Each step:
    - Receives params from JSON config
    - Receives context (MuJoCo model, RCM controller, painter, etc.)
    - Receives previous step's result (if any)
    - Generates trajectory in __init__
    - Returns StepResult after execution
    """

    def __init__(self, name, params, context, prev_result=None):
        """
        Initialize step.

        Args:
            name: Step name from config
            params: Step parameters from config
            context: Shared context dict containing:
                - model: MuJoCo model
                - data: MuJoCo data
                - rcm_controller: RCMController instance
                - rcm_world: RCM position in world coordinates
                - lens_geometry: Lens geometry dict
                - eye_pos: Eye assembly position
                - painter: TexturePainter instance (optional)
            prev_result: Previous step's StepResult (None for first step)
        """
        self.name = name
        self.params = params
        self.context = context
        self.prev_result = prev_result
        self.trajectory = []
        self.last_position = None
        self.last_orientation = None

        # Generate trajectory in subclass __init__
        self._generate_trajectory()

    @property
    def start_position(self):
        """Get start position from previous step or None"""
        if self.prev_result and self.prev_result.end_position is not None:
            return self.prev_result.end_position.copy()
        return None

    @abstractmethod
    def _generate_trajectory(self):
        """
        Generate trajectory points. Called in __init__.
        Must populate self.trajectory with list of positions.
        """
        raise NotImplementedError

    def should_paint(self):
        """Check if UV painting is enabled for this step"""
        return self.params.get('paint', False)

    def use_rcm(self):
        """Check if this step uses RCM constraint for orientation. Default True."""
        return self.params.get('use_rcm', True)

    def get_fixed_orientation(self):
        """Get fixed quaternion orientation (only valid if use_rcm=False). Default None."""
        return None

    def get_use_rcm_for_frame(self, frame_idx):
        """Get whether specific frame uses RCM. Override for per-frame control."""
        return self.use_rcm()

    def get_orientation_for_frame(self, frame_idx):
        """Get quaternion for specific frame (if fixed orientation). Override for per-frame control."""
        return self.get_fixed_orientation()

    def on_frame(self, frame_idx, position, orientation=None):
        """
        Called after each frame is executed.
        Can be overridden for custom per-frame behavior.

        Args:
            frame_idx: Current frame index within this step
            position: Current tool tip position
            orientation: Current tool orientation (quaternion)
        """
        self.last_position = position
        self.last_orientation = orientation

    def get_result(self):
        """
        Get step result after execution.

        Returns:
            StepResult with end state
        """
        end_pos = self.trajectory[-1] if self.trajectory else self.last_position
        return StepResult(
            end_position=end_pos,
            end_orientation=self.last_orientation,
            xml_updates={},
            metadata={
                'step_name': self.name,
                'num_frames': len(self.trajectory)
            }
        )

    def __len__(self):
        """Return number of frames in trajectory"""
        return len(self.trajectory)

    def __iter__(self):
        """Iterate over trajectory positions"""
        return iter(self.trajectory)
