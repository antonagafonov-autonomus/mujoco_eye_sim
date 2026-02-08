#!/usr/bin/env python3
"""
Procedure Runner: Executes multi-step surgical procedures in MuJoCo.

Loads procedure definition from JSON and orchestrates step execution.
Each step generates its trajectory in __init__ and passes results to next step.
"""
import mujoco
import mujoco.viewer
import sys
import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image

from utils import (
    TexturePainter,
    RCMController,
    calculate_rcm_from_lens,
    create_side_by_side_video,
)

from analyze_lens import analyze_lens_geometry

from steps import create_step, StepResult


def quaternion_to_euler_deg(quat):
    """Convert quaternion [w,x,y,z] to euler angles in degrees"""
    w, x, y, z = quat
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.degrees([roll, pitch, yaw])


def quaternion_to_rotation_matrix(quat):
    """Convert quaternion [w,x,y,z] to 3x3 rotation matrix"""
    w, x, y, z = quat
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])


# Tip position in tool body local frame (must match TIP_LOCAL_OFFSET in utils.py)
TIP_LOCAL = np.array([0, 0.01, 0])


def calculate_body_pos_from_tip(tip_pos, quat):
    """
    Calculate body position from tip position and orientation.
    body_pos = tip_pos - R @ tip_local
    """
    R = quaternion_to_rotation_matrix(quat)
    return tip_pos - R @ TIP_LOCAL


def reset_texture(texture_path, color=(128, 128, 128), size=(512, 512)):
    """Reset texture to a solid color"""
    img = Image.new('RGB', size, color)
    img.save(texture_path)


def vary_lights(model, base_light_values):
    """
    Vary light parameters for per-frame variation.

    Args:
        model: MuJoCo model
        base_light_values: Dict with base values for each light
    """
    for light_name, base_values in base_light_values.items():
        light_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, light_name)
        if light_id < 0:
            continue

        if 'diffuse' in base_values:
            intensity = np.random.uniform(0.3, 0.7)
            model.light_diffuse[light_id] = base_values['diffuse'] * intensity

        if 'pos' in base_values:
            offset = np.array([
                np.random.uniform(-0.02, 0.02),
                np.random.uniform(-0.02, 0.02),
                np.random.uniform(-0.01, 0.01)
            ])
            model.light_pos[light_id] = base_values['pos'] + offset


def create_temp_scene(scene_path, temp_texture_name='lens_uv_map_temp.png'):
    """Create a temporary scene with modified assets to use temp texture."""
    scene_dir = Path(scene_path).parent

    assets_path = scene_dir / 'eye_assets.xml'
    with open(assets_path, 'r') as f:
        assets_content = f.read()

    modified_content = assets_content.replace(
        'file="lens_uv_map.png"',
        f'file="{temp_texture_name}"'
    )

    temp_assets_path = scene_dir / 'eye_assets_temp.xml'
    with open(temp_assets_path, 'w') as f:
        f.write(modified_content)

    with open(scene_path, 'r') as f:
        scene_content = f.read()

    modified_scene = scene_content.replace(
        'file="eye_assets.xml"',
        'file="eye_assets_temp.xml"'
    )

    temp_scene_path = scene_dir / 'eye_scene_temp.xml'
    with open(temp_scene_path, 'w') as f:
        f.write(modified_scene)

    return str(temp_scene_path)


def update_tool_xml(xml_path, pos, euler_deg):
    """Update eye_tool.xml with new tool position and orientation"""
    import re
    with open(xml_path, 'r') as f:
        content = f.read()

    pattern = r'(<body name="diathermic_tip" pos=")[^"]*(" euler=")[^"]*(")'
    pos_str = f"{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}"
    euler_str = f"{euler_deg[0]:.1f} {euler_deg[1]:.1f} {euler_deg[2]:.1f}"
    replacement = rf'\g<1>{pos_str}\g<2>{euler_str}\g<3>'
    new_content = re.sub(pattern, replacement, content)

    with open(xml_path, 'w') as f:
        f.write(new_content)

    return pos_str, euler_str


class ProcedureRunner:
    """
    Main procedure runner class.

    Loads configuration from JSON and orchestrates step execution.
    """

    def __init__(self, config_path):
        """
        Initialize procedure runner.

        Args:
            config_path: Path to procedure JSON config file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.context = {}
        self.model = None
        self.data = None

    def _load_config(self):
        """Load and validate procedure configuration"""
        with open(self.config_path, 'r') as f:
            config = json.load(f)

        required_keys = ['name', 'scene', 'steps']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")

        return config

    def _resolve_path(self, path):
        """Resolve relative paths from config file location"""
        if Path(path).is_absolute():
            return path
        return str(self.config_path.parent / path)

    def setup(self, capture=False):
        """
        Setup procedure context.

        Initializes MuJoCo model, RCM controller, painter, etc.
        """
        print("\n" + "="*60)
        print(f"PROCEDURE: {self.config['name'].upper()}")
        print("="*60)

        # Resolve paths
        scene_path = self._resolve_path(self.config['scene'])
        mesh_path = self._resolve_path(self.config.get('mesh', '../meshes/Lens_L_extracted.obj'))
        texture_path = self._resolve_path(self.config.get('texture', '../textures/lens_uv_map.png'))

        # Analyze lens geometry
        print("\nAnalyzing lens geometry...")
        lens_geometry = analyze_lens_geometry(mesh_path)

        # Get eye position
        eye_pos = np.array(self.config.get('eye_pos', [0, 0, 0.1]))

        # Setup RCM controller
        rcm_config = self.config.get('rcm', {})
        # Use fixed rotation angle for RCM position calculation
        rotation_angle = (3/2) * np.pi

        rcm_world = calculate_rcm_from_lens(
            lens_geometry,
            rotation_angle=rotation_angle,
            eye_pos=eye_pos,
            rcm_radius=rcm_config.get('radius', 0.005),
            rcm_height=rcm_config.get('height', 0.004)
        )
        rcm_controller = RCMController(
            rcm_world,
            tool_roll_deg=rcm_config.get('tool_roll', -45)
        )

        print(f"\nRCM parameters:")
        print(f"  Position: [{rcm_world[0]:.4f}, {rcm_world[1]:.4f}, {rcm_world[2]:.4f}]")
        print(f"  Radius: {rcm_config.get('radius', 0.005)*1000:.1f} mm")
        print(f"  Height: {rcm_config.get('height', 0.004)*1000:.1f} mm")
        print(f"  Tool roll: {rcm_config.get('tool_roll', -45):.1f} deg")

        # Setup texture painter if enabled
        painter = None
        painting_config = self.config.get('painting', {})
        if painting_config.get('enabled', False):
            print("\nInitializing texture painter...")
            painter = TexturePainter(
                texture_path=texture_path,
                mesh_path=mesh_path,
                eye_assembly_pos=eye_pos
            )

        # Create temp scene if capturing with painting
        actual_scene_path = scene_path
        if capture and painting_config.get('enabled', False):
            print("Creating temp scene for texture updates...")
            actual_scene_path = create_temp_scene(scene_path)

        # Store context
        self.context = {
            'lens_geometry': lens_geometry,
            'eye_pos': eye_pos,
            'rcm_world': rcm_world,
            'rcm_controller': rcm_controller,
            'painter': painter,
            'scene_path': actual_scene_path,
            'texture_path': texture_path,
            'mesh_path': mesh_path,
            'painting_config': painting_config,
            'lighting_config': self.config.get('lighting', {}),
            'capture_config': self.config.get('capture', {}),
        }

        return self.context

    def run(self, iterations=1, viewer=False, capture=False, debug=False, speed=1.0):
        """
        Run the procedure.

        Args:
            iterations: Number of procedure repetitions
            viewer: Use interactive viewer (no data saving)
            capture: Capture images
            debug: Print debug info per frame
            speed: Viewer playback speed
        """
        # Create output directory
        if capture:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_output_dir = Path('../captures') / f"run_{timestamp}"
            base_output_dir.mkdir(parents=True, exist_ok=True)
            print(f"\nOutput directory: {base_output_dir}")

            # Save config
            with open(base_output_dir / 'config.json', 'w') as f:
                json.dump(self.config, f, indent=2)
        else:
            base_output_dir = None

        # Run iterations
        for iteration in range(iterations):
            print(f"\n{'='*60}")
            print(f"ITERATION {iteration + 1}/{iterations}")
            print(f"{'='*60}")

            # Reset texture for each iteration
            painting_config = self.context.get('painting_config', {})
            if painting_config.get('enabled', False):
                reset_texture(self.context['texture_path'])

            # Create iteration output directory
            if capture:
                iter_output_dir = base_output_dir / f"iter_{iteration:04d}"
                iter_output_dir.mkdir(parents=True, exist_ok=True)
            else:
                iter_output_dir = None

            # Run steps for this iteration
            self._run_iteration(
                iteration,
                viewer=viewer,
                capture=capture,
                debug=debug,
                speed=speed,
                output_dir=iter_output_dir
            )

        print(f"\n{'='*60}")
        print("PROCEDURE COMPLETE")
        print("="*60)
        if base_output_dir:
            print(f"Output: {base_output_dir}")

    def _run_iteration(self, iteration_idx, viewer=False, capture=False,
                       debug=False, speed=1.0, output_dir=None):
        """Run a single iteration of the procedure"""

        # Build all steps first (trajectories generated in __init__)
        print("\nBuilding steps...")
        steps = []
        prev_result = None

        # First pass: create trajectory step to get first point (needed for insertion target)
        # Store it to reuse in second pass (avoids randomness creating different trajectory)
        trajectory_step = None
        trajectory_step_name = None
        for step_config in self.config['steps']:
            if step_config['type'] == 'trajectory':
                print(f"\n  Step: {step_config['name']} (building first to get start point)")
                trajectory_step = create_step(step_config, self.context, None)
                trajectory_step_name = step_config['name']
                break

        # Get first trajectory point for insertion end target
        first_traj_point = trajectory_step.trajectory[0] if trajectory_step else None

        # Second pass: create all steps in order with correct targets
        # IMPORTANT: Reuse the trajectory_step from first pass to avoid randomness issues
        prev_result = None
        for step_config in self.config['steps']:
            print(f"\n  Step: {step_config['name']}")

            # For insertion step, pass the first trajectory point as end target
            if step_config['type'] == 'insertion' and first_traj_point is not None:
                # Add end_position to context temporarily
                self.context['insertion_end_position'] = first_traj_point

            # Reuse trajectory step from first pass instead of creating new one
            if step_config['type'] == 'trajectory' and trajectory_step is not None:
                step = trajectory_step
                # Update prev_result connection for trajectory step
                if prev_result:
                    step.prev_result = prev_result
            else:
                step = create_step(step_config, self.context, prev_result)

            steps.append(step)
            prev_result = step.get_result()

            # Clean up temp context
            if 'insertion_end_position' in self.context:
                del self.context['insertion_end_position']

        # Combine all trajectories and build per-frame info
        full_trajectory = []
        step_boundaries = [0]
        paint_mask = []
        frame_step_info = []  # Store step info for each frame

        for step in steps:
            full_trajectory.extend(step.trajectory)
            step_boundaries.append(len(full_trajectory))
            paint_mask.extend([step.should_paint()] * len(step.trajectory))
            # Store step info for each frame - use per-frame methods if available
            for frame_idx in range(len(step.trajectory)):
                step_info = {
                    'use_rcm': step.get_use_rcm_for_frame(frame_idx),
                    'fixed_quat': step.get_orientation_for_frame(frame_idx),
                    'step_name': step.name
                }
                frame_step_info.append(step_info)

        print(f"\nTotal frames: {len(full_trajectory)}")
        for i, step in enumerate(steps):
            print(f"  {step.name}: frames {step_boundaries[i]}-{step_boundaries[i+1]-1} (use_rcm={step.use_rcm()})")

        # Store steps for later use
        self.steps = steps
        self.frame_step_info = frame_step_info

        # Load MuJoCo model
        print("\nLoading MuJoCo model...")
        scene_path = self.context['scene_path']
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        # Update tool XML with initial pose
        if full_trajectory:
            initial_tip_pos = full_trajectory[0]
            first_step_info = frame_step_info[0]

            if first_step_info['use_rcm']:
                rcm_controller = self.context['rcm_controller']
                initial_pose = rcm_controller.calculate_tool_pose(initial_tip_pos)
                initial_body_pos = initial_pose['body_position']
                initial_quat = initial_pose['quaternion']
            else:
                # Fixed orientation - calculate body pos using rotation
                initial_quat = first_step_info['fixed_quat']
                initial_body_pos = calculate_body_pos_from_tip(initial_tip_pos, initial_quat)

            initial_euler = quaternion_to_euler_deg(initial_quat)

            print(f"\n[DEBUG] Initial tool pose (frame 0, step: {first_step_info['step_name']}):")
            print(f"  Tip position: [{initial_tip_pos[0]:.4f}, {initial_tip_pos[1]:.4f}, {initial_tip_pos[2]:.4f}]")
            print(f"  Body position: [{initial_body_pos[0]:.4f}, {initial_body_pos[1]:.4f}, {initial_body_pos[2]:.4f}]")
            print(f"  Euler (deg): [{initial_euler[0]:.1f}, {initial_euler[1]:.1f}, {initial_euler[2]:.1f}]")
            print(f"  Quaternion: [{initial_quat[0]:.4f}, {initial_quat[1]:.4f}, {initial_quat[2]:.4f}, {initial_quat[3]:.4f}]")
            print(f"  Use RCM: {first_step_info['use_rcm']}")
            print(f"  TIP_LOCAL: {TIP_LOCAL}")

            tool_xml_path = Path(scene_path).parent / 'eye_tool.xml'
            if tool_xml_path.exists():
                update_tool_xml(tool_xml_path, initial_body_pos, initial_euler)
                print(f"✓ Updated tool XML with initial pose")

        # Execute
        if viewer:
            self._run_viewer(full_trajectory, paint_mask, speed, debug)
        elif capture:
            self._run_capture(full_trajectory, paint_mask, output_dir, debug)
        else:
            print("Warning: Neither viewer nor capture enabled")

    def _run_viewer(self, trajectory, paint_mask, speed, debug):
        """Run trajectory in interactive viewer"""
        rcm_controller = self.context['rcm_controller']
        painter = self.context.get('painter')
        paint_radius = self.context['painting_config'].get('radius', 3)

        tool_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'diathermic_tip')
        mocap_id = self.model.body_mocapid[tool_body_id]

        current_idx = [0]
        frame_time = [0.0]
        frame_delay = 0.02 / speed

        def on_key(keycode):
            pass

        with mujoco.viewer.launch_passive(self.model, self.data, key_callback=on_key) as viewer:
            while viewer.is_running() and current_idx[0] < len(trajectory):
                frame_time[0] += self.model.opt.timestep

                if frame_time[0] >= frame_delay:
                    frame_time[0] = 0.0
                    idx = current_idx[0]
                    tip_pos = trajectory[idx]
                    step_info = self.frame_step_info[idx]

                    # Calculate pose based on step settings
                    if step_info['use_rcm']:
                        pose = rcm_controller.calculate_tool_pose(tip_pos)
                        body_pos = pose['body_position']
                        quat = pose['quaternion']
                    else:
                        # Fixed orientation - calculate body pos using rotation
                        quat = step_info['fixed_quat']
                        body_pos = calculate_body_pos_from_tip(tip_pos, quat)

                    self.data.mocap_pos[mocap_id] = body_pos
                    self.data.mocap_quat[mocap_id] = quat

                    # Paint if enabled for this frame
                    if paint_mask[idx] and painter:
                        painter.paint_at_world_pos(tip_pos, radius_pixels=paint_radius)

                    if debug:
                        euler = quaternion_to_euler_deg(quat)
                        print(f"Frame {idx} ({step_info['step_name']}): pos=[{tip_pos[0]:.4f}, {tip_pos[1]:.4f}, {tip_pos[2]:.4f}] "
                              f"euler=[{euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}]")

                    current_idx[0] += 1

                mujoco.mj_step(self.model, self.data)
                viewer.sync()

    def _run_capture(self, trajectory, paint_mask, output_dir, debug):
        """Run trajectory with image capture"""
        from utils import run_with_capture

        rcm_controller = self.context['rcm_controller']
        painter = self.context.get('painter')
        paint_radius = self.context['painting_config'].get('radius', 3)
        capture_config = self.context['capture_config']
        lighting_config = self.context['lighting_config']
        scene_path = self.context['scene_path']

        # Custom capture loop that respects paint_mask
        tool_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'diathermic_tip')
        mocap_id = self.model.body_mocapid[tool_body_id]

        width = capture_config.get('width', 640)
        height = capture_config.get('height', 480)
        should_vary_lights = lighting_config.get('vary', False)

        # Setup renderers
        renderer_top = mujoco.Renderer(self.model, height, width)
        renderer_angle = mujoco.Renderer(self.model, height, width)
        renderer_tool = mujoco.Renderer(self.model, height, width)

        # Create output subdirectories
        top_dir = output_dir / 'top_view'
        angle_dir = output_dir / 'angle_view'
        tool_dir = output_dir / 'tool_view'
        top_dir.mkdir(exist_ok=True)
        angle_dir.mkdir(exist_ok=True)
        tool_dir.mkdir(exist_ok=True)

        # Get camera IDs
        top_cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, 'top_view')
        angle_cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, 'angle_view')
        tool_cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, 'tool_view')

        # Store base light values for variation
        base_light_values = {}
        if should_vary_lights:
            for light_name in ['top_light', 'side_light']:
                light_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_LIGHT, light_name)
                if light_id >= 0:
                    base_light_values[light_name] = {
                        'diffuse': self.model.light_diffuse[light_id].copy(),
                        'pos': self.model.light_pos[light_id].copy()
                    }

        log_data = []

        print(f"\nCapturing {len(trajectory)} frames...")

        for idx, tip_pos in enumerate(trajectory):
            step_info = self.frame_step_info[idx]

            # Vary lights if enabled
            if should_vary_lights and base_light_values:
                vary_lights(self.model, base_light_values)

            # Calculate pose based on step settings
            if step_info['use_rcm']:
                pose = rcm_controller.calculate_tool_pose(tip_pos)
                body_pos = pose['body_position']
                quat = pose['quaternion']
            else:
                # Fixed orientation - calculate body pos using rotation
                quat = step_info['fixed_quat']
                body_pos = calculate_body_pos_from_tip(tip_pos, quat)

            self.data.mocap_pos[mocap_id] = body_pos
            self.data.mocap_quat[mocap_id] = quat

            mujoco.mj_forward(self.model, self.data)

            # Paint if enabled for this frame
            if paint_mask[idx] and painter:
                painter.paint_at_world_pos(tip_pos, radius_pixels=paint_radius)
                # Save updated texture to temp file BEFORE model reload
                painter.save_texture(painter.temp_texture_path)
                # Reload texture in MuJoCo - need to recreate model, data, and renderers
                if scene_path:
                    # Close old renderers first to free OpenGL resources
                    renderer_top.close()
                    renderer_angle.close()
                    renderer_tool.close()

                    # Reload model with updated texture
                    self.model = mujoco.MjModel.from_xml_path(scene_path)
                    self.data = mujoco.MjData(self.model)

                    # Re-obtain IDs from new model
                    tool_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'diathermic_tip')
                    mocap_id = self.model.body_mocapid[tool_body_id]
                    top_cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, 'top_view')
                    angle_cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, 'angle_view')
                    tool_cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, 'tool_view')

                    # Re-apply mocap pose
                    self.data.mocap_pos[mocap_id] = body_pos
                    self.data.mocap_quat[mocap_id] = quat

                    # Forward to update physics
                    mujoco.mj_forward(self.model, self.data)

                    # Recreate renderers with new model
                    renderer_top = mujoco.Renderer(self.model, height, width)
                    renderer_angle = mujoco.Renderer(self.model, height, width)
                    renderer_tool = mujoco.Renderer(self.model, height, width)

            # Render and save
            renderer_top.update_scene(self.data, camera=top_cam_id)
            renderer_angle.update_scene(self.data, camera=angle_cam_id)
            renderer_tool.update_scene(self.data, camera=tool_cam_id)

            img_top = renderer_top.render()
            img_angle = renderer_angle.render()
            img_tool = renderer_tool.render()

            Image.fromarray(img_top).save(top_dir / f"frame_{idx:05d}.png")
            Image.fromarray(img_tool).save(tool_dir / f"frame_{idx:05d}.png")
            Image.fromarray(img_angle).save(angle_dir / f"frame_{idx:05d}.png")

            # Log
            euler = quaternion_to_euler_deg(quat)
            log_data.append({
                'frame': idx,
                'step': step_info['step_name'],
                'tip_position': tip_pos.tolist(),
                'body_position': body_pos.tolist(),
                'euler': euler.tolist(),
                'painting': paint_mask[idx],
                'use_rcm': step_info['use_rcm']
            })

            if debug:
                print(f"Frame {idx} ({step_info['step_name']}): pos=[{tip_pos[0]:.4f}, {tip_pos[1]:.4f}, {tip_pos[2]:.4f}] "
                      f"euler=[{euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}]")

            if (idx + 1) % 50 == 0:
                print(f"  Captured {idx + 1}/{len(trajectory)} frames")

        # Save log
        with open(output_dir / 'capture_log.json', 'w') as f:
            json.dump(log_data, f, indent=2)

        # Save final painted texture to output directory
        if painter:
            painted_texture_path = output_dir / 'painted_texture.png'
            painter.save_texture(painted_texture_path)
            print(f"✓ Painted texture saved: {painted_texture_path}")

        print(f"✓ Captured {len(trajectory)} frames")
        print(f"✓ Log saved to {output_dir / 'capture_log.json'}")

        # Generate video
        create_side_by_side_video(output_dir, fps=30)


def main():
    parser = argparse.ArgumentParser(
        description='Run multi-step surgical procedure in MuJoCo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--procedure', '-p', type=str, default='diathermy',
                        help='Procedure name (looks for parameters/procedures/<name>.json)')
    parser.add_argument('--config', '-c', type=str, default=None,
                        help='Direct path to procedure config JSON (overrides --procedure)')
    parser.add_argument('--iterations', '-i', type=int, default=1,
                        help='Number of procedure repetitions')
    parser.add_argument('--viewer', '-v', action='store_true',
                        help='View simulation interactively (no data saving)')
    parser.add_argument('--capture', action='store_true',
                        help='Capture images from cameras')
    parser.add_argument('--debug', '-d', action='store_true',
                        help='Print debug info per frame')
    parser.add_argument('--speed', '-s', type=float, default=1.0,
                        help='Viewer playback speed')

    args = parser.parse_args()

    # Resolve config path
    if args.config:
        config_path = Path(args.config)
    else:
        # Look in parameters/procedures/
        script_dir = Path(__file__).parent
        config_path = script_dir.parent / 'parameters' / 'procedures' / f'{args.procedure}.json'

    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    print(f"Loading procedure: {config_path}")

    # Create and run procedure
    runner = ProcedureRunner(config_path)
    runner.setup(capture=args.capture)
    runner.run(
        iterations=args.iterations,
        viewer=args.viewer,
        capture=args.capture,
        debug=args.debug,
        speed=args.speed
    )


if __name__ == "__main__":
    main()
