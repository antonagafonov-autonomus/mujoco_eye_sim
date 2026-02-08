"""
Diathermy-specific insertion step.
Moves tool from outside the eye (+X offset) to the first trajectory point.
Uses fixed orientation (no RCM) with linear motion.
"""
import numpy as np
from .base_step import BaseStep, StepResult


def euler_to_quat(euler_deg):
    """Convert euler angles (degrees) to quaternion [w,x,y,z] using XYZ intrinsic convention"""
    ax, ay, az = np.radians(euler_deg)

    # Quaternion for each axis rotation
    qx = np.array([np.cos(ax/2), np.sin(ax/2), 0, 0])
    qy = np.array([np.cos(ay/2), 0, np.sin(ay/2), 0])
    qz = np.array([np.cos(az/2), 0, 0, np.sin(az/2)])

    def quat_mul(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    # Intrinsic XYZ = extrinsic ZYX, so multiply: qz * qy * qx
    return quat_mul(qz, quat_mul(qy, qx))


class DiathermyInsertionStep(BaseStep):
    """
    Insertion step for diathermy procedure.

    Starts at +X offset from RCM point (outside eye)
    Ends at the first point of the next step's trajectory (on lens surface)

    By default uses fixed orientation (use_rcm=false) with linear motion.
    Orientation stays constant throughout the insertion.

    Supports motion types:
    - 'linear': Simple linear interpolation (default)
    - 'stepped': Stepped motion based on eye geometry (NOT IMPLEMENTED)
    """

    def _generate_trajectory(self):
        """Generate insertion trajectory from outside to lens surface"""
        # Get initial position from config - this IS the start tip position
        initial_position = self.params.get('initial_position')
        if initial_position is None:
            raise ValueError("Insertion step requires 'initial_position' in params")
        initial_position = np.array(initial_position)

        # Start position is the initial_position (tip position at frame 0)
        start = initial_position.copy()

        # End position: from context (set by procedure_runner to match trajectory start)
        end = self.context.get('insertion_end_position')
        if end is None:
            # Fallback to lens center if not provided
            end = self._get_insertion_target()
        else:
            end = np.array(end)

        # Generate trajectory based on motion type
        motion_type = self.params.get('motion_type', 'linear')
        if motion_type == 'linear':
            self.trajectory = self._linear_interpolation(start, end)
        elif motion_type == 'stepped':
            raise NotImplementedError(
                "Stepped motion type not yet implemented. "
                "This will use eye geometry for stepped path through incision."
            )
        else:
            raise ValueError(f"Unknown motion type: {motion_type}")

        # Store fixed orientation if not using RCM
        if not self.use_rcm():
            initial_euler = self.params.get('initial_euler', [0, 45, -45])
            self.fixed_quaternion = euler_to_quat(initial_euler)

        print(f"  Insertion step: {len(self.trajectory)} frames")
        print(f"    Initial position (config): {initial_position}")
        print(f"    Initial euler (config): {self.params.get('initial_euler')}")
        print(f"    Start tip pos: [{start[0]:.4f}, {start[1]:.4f}, {start[2]:.4f}]")
        print(f"    End tip pos: [{end[0]:.4f}, {end[1]:.4f}, {end[2]:.4f}]")
        print(f"    Use RCM: {self.use_rcm()}")

    def _get_insertion_target(self):
        """
        Get the target position for insertion (where tool tip should end up).

        This is typically the first point of the diathermy trajectory.
        For now, we use the lens center projected to the mesh surface.
        """
        lens_geometry = self.context.get('lens_geometry')
        eye_pos = self.context.get('eye_pos', np.array([0, 0, 0.1]))

        if lens_geometry is None:
            raise ValueError("Insertion step requires 'lens_geometry' in context")

        # Get lens center in world coordinates
        lens_center_local = lens_geometry['center_local']
        lens_center_world = lens_center_local + np.array(eye_pos)

        return lens_center_world

    def _linear_interpolation(self, start, end):
        """
        Generate linear interpolation trajectory with optional pause at end.

        Args:
            start: Start position (outside eye)
            end: End position (on lens surface)

        Returns:
            List of positions
        """
        num_frames = self.params.get('num_frames', 50)
        pause_frames = self.params.get('pause_frames', 0)

        trajectory = []

        # Linear interpolation
        for i in range(num_frames):
            t = i / (num_frames - 1) if num_frames > 1 else 1.0
            pos = start + t * (end - start)
            trajectory.append(pos)

        # Add pause frames at end position
        for _ in range(pause_frames):
            trajectory.append(end.copy())

        return trajectory

    def use_rcm(self):
        """Check if this step uses RCM constraint for orientation"""
        return self.params.get('use_rcm', False)

    def get_fixed_orientation(self):
        """Get fixed quaternion orientation (only valid if use_rcm=False)"""
        if hasattr(self, 'fixed_quaternion'):
            return self.fixed_quaternion
        return None

    def get_result(self):
        """Get step result with insertion-specific metadata"""
        result = super().get_result()
        result.metadata.update({
            'motion_type': self.params.get('motion_type', 'linear'),
            'start_offset_x': self.params.get('start_offset_x', 0.020),
            'use_rcm': self.use_rcm(),
        })
        if not self.use_rcm():
            result.metadata['initial_euler'] = self.params.get('initial_euler', [0, 45, -45])
        return result
