"""
Trajectory step for following a path on the lens surface.
Currently supports ellipse shape trajectories.
"""
import numpy as np
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from .base_step import BaseStep, StepResult


class TrajectoryStep(BaseStep):
    """
    Trajectory following step.

    Generates a trajectory (e.g., ellipse) on the lens surface
    and follows it with the tool tip.

    Supports shapes:
    - 'ellipse': Elliptical path on lens surface (default)

    Parameters:
    - shape: Trajectory shape ('ellipse')
    - radius_x: Semi-major axis (meters)
    - radius_y: Semi-minor axis (meters)
    - radius_std: Std dev for radius randomization
    - center_std: Std dev for center offset randomization
    - num_points: Number of points in trajectory
    - offset: Offset along surface normal
    - paint: Whether to paint on UV texture
    """

    def _generate_trajectory(self):
        """Generate trajectory on lens surface"""
        shape = self.params.get('shape', 'ellipse')

        if shape == 'ellipse':
            self.trajectory, self.traj_params = self._generate_ellipse_trajectory()
        else:
            raise ValueError(f"Unknown trajectory shape: {shape}")

        print(f"  Trajectory step: {len(self.trajectory)} frames")
        print(f"    Shape: {shape}")
        if self.traj_params:
            print(f"    Radius X: {self.traj_params['radius_x']*1000:.2f} mm")
            print(f"    Radius Y: {self.traj_params['radius_y']*1000:.2f} mm")

    def _generate_ellipse_trajectory(self):
        """
        Generate ellipse trajectory with optional randomization.

        Returns:
            tuple: (trajectory_points, trajectory_params)
        """
        from utils import generate_ellipsoid_trajectory, generate_random_trajectory_params
        from analyze_lens import project_trajectory_to_mesh, transform_to_world_coordinates

        lens_geometry = self.context.get('lens_geometry')
        eye_pos = self.context.get('eye_pos', np.array([0, 0, 0.1]))

        if lens_geometry is None:
            raise ValueError("Trajectory step requires 'lens_geometry' in context")

        # Get parameters with randomization
        traj_params = generate_random_trajectory_params(
            base_radius_x=self.params.get('radius_x', 0.003),
            base_radius_y=self.params.get('radius_y', 0.003),
            radius_std=self.params.get('radius_std', 0.0002),
            center_std=self.params.get('center_std', 0.0002)
        )

        # Generate ellipse trajectory in local coordinates
        trajectory_local = generate_ellipsoid_trajectory(
            lens_geometry,
            radius_x=traj_params['radius_x'],
            radius_y=traj_params['radius_y'],
            num_points=self.params.get('num_points', 300),
            offset_meters=self.params.get('offset', 0.0),
            center_offset_x=traj_params['center_offset_x'],
            center_offset_y=traj_params['center_offset_y'],
            rotation_angle=traj_params['rotation_angle'],
            noise_std=traj_params['noise_std']
        )

        # Project to mesh surface
        print("    Projecting trajectory to mesh surface...")
        projected_local = project_trajectory_to_mesh(trajectory_local, lens_geometry)

        # Transform to world coordinates
        projected_world = transform_to_world_coordinates(projected_local, eye_pos)

        # Store local trajectory for potential UV painting use
        self.trajectory_local = trajectory_local
        self.projected_local = projected_local

        return list(projected_world), traj_params

    def get_result(self):
        """Get step result with trajectory-specific metadata"""
        result = super().get_result()
        result.metadata.update({
            'shape': self.params.get('shape', 'ellipse'),
            'trajectory_params': self.traj_params if hasattr(self, 'traj_params') else None,
        })
        return result
