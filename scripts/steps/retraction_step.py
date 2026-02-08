"""
Retraction step for withdrawing tool from the eye.
Two-phase motion:
1. RCM-constrained from lens surface to RCM point
2. Fixed orientation linear motion from RCM outward in +X direction
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


class RetractionStep(BaseStep):
    """
    Retraction step - withdraws tool from eye in two phases.

    Phase 1 (RCM-constrained):
        From previous step's end position (on lens surface) to RCM point
        Tool orientation follows RCM constraint

    Phase 2 (Fixed orientation):
        From RCM point outward in +X direction
        Uses same orientation as insertion (from config or matching insertion euler)
    """

    def _generate_trajectory(self):
        """Generate two-phase retraction trajectory"""
        # Get RCM world position
        rcm_world = self.context.get('rcm_world')
        if rcm_world is None:
            raise ValueError("Retraction step requires 'rcm_world' in context")

        # Start position: previous step's end position
        if self.prev_result is None or self.prev_result.end_position is None:
            raise ValueError("Retraction step requires previous step result with end_position")

        start = self.prev_result.end_position.copy()
        rcm_point = np.array(rcm_world)

        # End position: +X offset from RCM (outside eye)
        end_offset_x = self.params.get('end_offset_x', 0.020)  # 20mm default
        end = rcm_point + np.array([end_offset_x, 0, 0])

        # Small offset to avoid exact RCM point (causes RCM controller to fail)
        # Phase 1 ends slightly before RCM, phase 2 starts from there
        rcm_margin = 0.001  # 1mm margin from RCM point

        # Direction from start to RCM for phase 1 end point
        dir_to_rcm = rcm_point - start
        dist_to_rcm = np.linalg.norm(dir_to_rcm)
        if dist_to_rcm > rcm_margin:
            dir_to_rcm_normalized = dir_to_rcm / dist_to_rcm
            phase1_end = rcm_point - dir_to_rcm_normalized * rcm_margin
        else:
            phase1_end = start  # Already very close to RCM

        # Get frame counts for each phase
        num_frames_phase1 = self.params.get('num_frames_to_rcm', 25)
        num_frames_phase2 = self.params.get('num_frames_exit', 25)
        # Fallback to old param for backwards compatibility
        if 'num_frames' in self.params and 'num_frames_to_rcm' not in self.params:
            total = self.params.get('num_frames', 50)
            num_frames_phase1 = total // 2
            num_frames_phase2 = total - num_frames_phase1

        # Phase 1: RCM-constrained motion from lens to near-RCM
        phase1_trajectory = []
        for i in range(num_frames_phase1):
            t = i / (num_frames_phase1 - 1) if num_frames_phase1 > 1 else 1.0
            pos = start + t * (phase1_end - start)
            phase1_trajectory.append(pos)

        # Phase 2: Fixed orientation motion from near-RCM outward in +X
        # Start from phase1_end to maintain continuity
        phase2_trajectory = []
        for i in range(num_frames_phase2):
            t = i / (num_frames_phase2 - 1) if num_frames_phase2 > 1 else 1.0
            pos = phase1_end + t * (end - phase1_end)
            phase2_trajectory.append(pos)

        # Combine trajectories
        self.trajectory = phase1_trajectory + phase2_trajectory

        # Store phase boundary for per-frame RCM lookup
        self.phase1_frames = num_frames_phase1
        self.phase2_frames = num_frames_phase2

        # Get fixed orientation for phase 2 (matches insertion orientation)
        exit_euler = self.params.get('exit_euler')
        if exit_euler is None:
            # Default to insertion's initial euler if not specified
            exit_euler = self.params.get('initial_euler', [-25.9, -43.7, 101.2])
        self.exit_quaternion = euler_to_quat(exit_euler)

        print(f"  Retraction step: {len(self.trajectory)} frames")
        print(f"    Phase 1 (RCM): {num_frames_phase1} frames, lens -> near-RCM")
        print(f"    Phase 2 (fixed): {num_frames_phase2} frames, near-RCM -> exit")
        print(f"    Start: [{start[0]:.4f}, {start[1]:.4f}, {start[2]:.4f}]")
        print(f"    Phase1 end: [{phase1_end[0]:.4f}, {phase1_end[1]:.4f}, {phase1_end[2]:.4f}]")
        print(f"    RCM: [{rcm_point[0]:.4f}, {rcm_point[1]:.4f}, {rcm_point[2]:.4f}]")
        print(f"    End: [{end[0]:.4f}, {end[1]:.4f}, {end[2]:.4f}]")
        print(f"    Exit euler: {exit_euler}")

    def use_rcm(self):
        """Default RCM status (used for step-level reporting)"""
        # Report as mixed - actual per-frame status from get_use_rcm_for_frame
        return True

    def get_use_rcm_for_frame(self, frame_idx):
        """Per-frame RCM status: True for phase 1, False for phase 2"""
        return frame_idx < self.phase1_frames

    def get_fixed_orientation(self):
        """Default fixed orientation (used for step-level reporting)"""
        return self.exit_quaternion if hasattr(self, 'exit_quaternion') else None

    def get_orientation_for_frame(self, frame_idx):
        """Per-frame orientation: None for phase 1 (RCM), fixed quat for phase 2"""
        if frame_idx < self.phase1_frames:
            return None  # RCM will calculate orientation
        return self.exit_quaternion

    def get_result(self):
        """Get step result with retraction-specific metadata"""
        result = super().get_result()
        result.metadata.update({
            'motion_type': 'two_phase',
            'end_offset_x': self.params.get('end_offset_x', 0.020),
            'phase1_frames': self.phase1_frames,
            'phase2_frames': self.phase2_frames,
        })
        return result
