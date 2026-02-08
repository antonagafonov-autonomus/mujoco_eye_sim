#!/usr/bin/env python3
"""
Tool Controller: Moves tool along projected trajectory in MuJoCo
Uses pre-computed projected trajectory from trajectory_generator.py
Captures images from both cameras and logs tool positions
Optionally paints trajectory on lens UV texture
Supports multiple iterations with randomized ellipsoid trajectories
"""

import mujoco
import sys
import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image

from utils import (
    load_trajectory_file,
    run_with_capture,
    move_tool_along_trajectory,
    TexturePainter,
    generate_ellipsoid_trajectory,
    generate_random_trajectory_params,
    load_obj_with_uv,
    RCMController,
    calculate_rcm_from_lens
)

from analyze_lens import (
    analyze_lens_geometry,
    transform_to_world_coordinates,
    project_trajectory_to_mesh
)


def reset_texture(texture_path, color=(128, 128, 128), size=(512, 512)):
    """Reset texture to a solid color"""
    img = Image.new('RGB', size, color)


def quaternion_to_euler_deg(quat):
    """Convert quaternion [w,x,y,z] to euler angles in degrees"""
    w, x, y, z = quat
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.degrees([roll, pitch, yaw])


def update_tool_xml(xml_path, pos, euler_deg):
    """Update eye_tool.xml with new tool position and orientation"""
    import re
    with open(xml_path, 'r') as f:
        content = f.read()

    # Update position and euler in the diathermic_tip body
    pattern = r'(<body name="diathermic_tip" pos=")[^"]*(" euler=")[^"]*(")'
    pos_str = f"{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}"
    euler_str = f"{euler_deg[0]:.1f} {euler_deg[1]:.1f} {euler_deg[2]:.1f}"
    replacement = rf'\g<1>{pos_str}\g<2>{euler_str}\g<3>'
    new_content = re.sub(pattern, replacement, content)

    with open(xml_path, 'w') as f:
        f.write(new_content)

    return pos_str, euler_str
    img.save(texture_path)


def get_camera_params_from_model(model, camera_names):
    """
    Extract camera parameters from loaded MuJoCo model

    Args:
        model: MuJoCo model
        camera_names: List of camera names to extract

    Returns:
        Dict with camera parameters
    """
    cameras = {}
    for name in camera_names:
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if cam_id >= 0:
            pos = model.cam_pos[cam_id].tolist()
            quat = model.cam_quat[cam_id].tolist()
            fovy = float(model.cam_fovy[cam_id])
            cameras[name] = {
                'pos': pos,
                'quat': quat,
                'fovy': fovy
            }
    return cameras


def create_temp_scene(scene_path, temp_texture_name='lens_uv_map_temp.png'):
    """
    Create a temporary scene with modified assets to use temp texture.
    """
    scene_dir = Path(scene_path).parent

    # Read original assets file
    assets_path = scene_dir / 'eye_assets.xml'
    with open(assets_path, 'r') as f:
        assets_content = f.read()

    # Replace lens texture with temp texture
    modified_content = assets_content.replace(
        'file="lens_uv_map.png"',
        f'file="{temp_texture_name}"'
    )

    # Write to temp assets file
    temp_assets_path = scene_dir / 'eye_assets_temp.xml'
    with open(temp_assets_path, 'w') as f:
        f.write(modified_content)

    # Read original scene file
    with open(scene_path, 'r') as f:
        scene_content = f.read()

    # Replace assets include with temp assets
    modified_scene = scene_content.replace(
        'file="eye_assets.xml"',
        'file="eye_assets_temp.xml"'
    )

    # Write to temp scene file
    temp_scene_path = scene_dir / 'eye_scene_temp.xml'
    with open(temp_scene_path, 'w') as f:
        f.write(modified_scene)

    return str(temp_scene_path)


def save_trajectory(trajectory_local, trajectory_world, params,
                    projected_local, projected_world, output_path):
    """Save trajectory to JSON file"""
    data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'num_points': len(trajectory_local),
            'parameters': params,
            'has_projected': True
        },
        'trajectory_local': trajectory_local.tolist(),
        'trajectory_world': trajectory_world.tolist(),
        'projected_local': projected_local.tolist(),
        'projected_world': projected_world.tolist()
    }

    filepath = output_path / 'trajectory.json'
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    return str(filepath)


def run_iteration(iteration_idx, args, lens_geometry, scene_path, base_output_dir):
    """
    Run a single iteration with a randomized trajectory

    Args:
        iteration_idx: Current iteration number
        args: Parsed command line arguments
        lens_geometry: Lens geometry from analyze_lens_geometry()
        scene_path: Path to scene XML (temp scene if painting)
        base_output_dir: Base output directory

    Returns:
        Path to iteration output directory
    """
    print(f"\n{'='*60}")
    print(f"ITERATION {iteration_idx + 1}/{args.iterations}")
    print(f"{'='*60}")

    # Generate randomized trajectory parameters
    traj_params = generate_random_trajectory_params(
        base_radius_x=args.radius_x,
        base_radius_y=args.radius_y,
        radius_std=args.radius_std,
        center_std=args.center_std
    )

    print(f"\nTrajectory parameters:")
    print(f"  Radius X: {traj_params['radius_x']*1000:.2f} mm")
    print(f"  Radius Y: {traj_params['radius_y']*1000:.2f} mm")
    print(f"  Center offset: ({traj_params['center_offset_x']*1000:.2f}, {traj_params['center_offset_y']*1000:.2f}) mm")
    print(f"  Rotation: {np.degrees(traj_params['rotation_angle']):.1f} deg")
    print(f"  Noise std: {traj_params['noise_std']*1000:.3f} mm")

    # Calculate RCM position if RCM mode is enabled
    rcm_controller = None
    if args.rcm:
        rcm_world = calculate_rcm_from_lens(
            lens_geometry,
            rotation_angle=traj_params['rotation_angle'],
            eye_pos=args.eye_pos,
            rcm_radius=args.rcm_radius,
            rcm_height=args.rcm_height
        )
        rcm_controller = RCMController(rcm_world, tool_roll_deg=args.tool_roll)
        print(f"\nRCM parameters:")
        print(f"  RCM position: [{rcm_world[0]:.4f}, {rcm_world[1]:.4f}, {rcm_world[2]:.4f}]")
        print(f"  RCM radius: {args.rcm_radius*1000:.1f} mm")
        print(f"  RCM height: {args.rcm_height*1000:.1f} mm")
        print(f"  Tool roll: {args.tool_roll:.1f} deg")

    # Generate ellipsoid trajectory
    trajectory_local = generate_ellipsoid_trajectory(
        lens_geometry,
        radius_x=traj_params['radius_x'],
        radius_y=traj_params['radius_y'],
        num_points=args.num_points,
        offset_meters=args.offset,
        center_offset_x=traj_params['center_offset_x'],
        center_offset_y=traj_params['center_offset_y'],
        rotation_angle=traj_params['rotation_angle'],
        noise_std=traj_params['noise_std']
    )

    # Transform to world coordinates
    trajectory_world = transform_to_world_coordinates(trajectory_local, args.eye_pos)

    # Project to mesh surface
    print("\nProjecting trajectory to lens surface...")
    projected_local = project_trajectory_to_mesh(trajectory_local, lens_geometry)
    projected_world = transform_to_world_coordinates(projected_local, args.eye_pos)

    print(f"✓ Generated trajectory: {len(projected_world)} points")

    # Calculate and print initial tool pose (first trajectory point)
    initial_tip_pos = projected_world[0]
    print(f"\nInitial trajectory point (tip position):")
    print(f"  [{initial_tip_pos[0]:.4f}, {initial_tip_pos[1]:.4f}, {initial_tip_pos[2]:.4f}]")

    if rcm_controller:
        initial_pose = rcm_controller.calculate_tool_pose(initial_tip_pos)
        initial_body_pos = initial_pose['body_position']
        initial_quat = initial_pose['quaternion']
        initial_euler = quaternion_to_euler_deg(initial_quat)
    else:
        # Legacy: fixed orientation
        from utils import get_tool_offset
        initial_body_pos = initial_tip_pos - get_tool_offset()
        initial_euler = np.array([0, 0, 45])

    print(f"Initial tool body position:")
    print(f"  [{initial_body_pos[0]:.4f}, {initial_body_pos[1]:.4f}, {initial_body_pos[2]:.4f}]")
    print(f"Initial tool euler angles (degrees):")
    print(f"  [{initial_euler[0]:.1f}, {initial_euler[1]:.1f}, {initial_euler[2]:.1f}]")

    # Update eye_tool.xml with initial pose
    tool_xml_path = Path(args.scene).parent / 'eye_tool.xml'
    if tool_xml_path.exists():
        pos_str, euler_str = update_tool_xml(tool_xml_path, initial_body_pos, initial_euler)
        print(f"✓ Updated {tool_xml_path}:")
        print(f"  pos=\"{pos_str}\" euler=\"{euler_str}\"")

    # Create iteration output directory
    iter_output_dir = base_output_dir / f"iter_{iteration_idx:04d}"
    iter_output_dir.mkdir(parents=True, exist_ok=True)

    # Save trajectory
    save_trajectory(
        trajectory_local, trajectory_world,
        traj_params,
        projected_local, projected_world,
        iter_output_dir
    )
    print(f"✓ Trajectory saved to: {iter_output_dir / 'trajectory.json'}")

    # Reset texture
    if args.paint:
        print("\nResetting texture...")
        reset_texture(args.texture)

    # Initialize painter
    painter = None
    if args.paint:
        painter = TexturePainter(
            texture_path=args.texture,
            mesh_path=args.mesh,
            eye_assembly_pos=args.eye_pos
        )

    # Load model
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Run viewer or capture
    if args.viewer:
        print("\n✓ Starting interactive viewer (no data saving)...")
        move_tool_along_trajectory(
            model, data, projected_world,
            speed=args.speed,
            painter=painter, paint_radius=args.paint_radius,
            rcm_controller=rcm_controller,
            debug=args.debug
        )
    else:
        run_with_capture(
            model, data, projected_world, iter_output_dir,
            width=args.width, height=args.height,
            painter=painter, paint_radius=args.paint_radius,
            scene_path=scene_path,
            rcm_controller=rcm_controller,
            vary_lights=args.vary_lights,
            debug=args.debug
        )

    return iter_output_dir


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description='Tool trajectory controller with multi-iteration support',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Mode selection
    parser.add_argument('trajectory_file', type=str, nargs='?', default=None,
                        help='Path to trajectory JSON file (for single run mode)')
    parser.add_argument('--iterations', '-i', type=int, default=1,
                        help='Number of iterations (generates new trajectory each time)')

    # Trajectory generation parameters
    parser.add_argument('--radius-x', type=float, default=0.003,
                        help='Base semi-major axis in meters')
    parser.add_argument('--radius-y', type=float, default=0.002,
                        help='Base semi-minor axis in meters')
    parser.add_argument('--radius-std', type=float, default=0.0002,
                        help='Std dev for radius randomization')
    parser.add_argument('--center-std', type=float, default=0.0003,
                        help='Std dev for center offset randomization')
    parser.add_argument('--num-points', '-n', type=int, default=100,
                        help='Number of points in trajectory')
    parser.add_argument('--offset', type=float, default=0.0,
                        help='Offset along normal in meters')

    # Scene and capture
    parser.add_argument('--speed', '-s', type=float, default=1.0,
                        help='Animation speed multiplier')
    parser.add_argument('--scene', type=str, default='../scene/eye_scene.xml',
                        help='Path to scene XML file')
    parser.add_argument('--capture', '-c', action='store_true',
                        help='Capture images from both cameras')
    parser.add_argument('--viewer', '-v', action='store_true',
                        help='View simulation interactively (no data saving)')
    parser.add_argument('--output', '-o', type=str, default='../captures',
                        help='Output directory for captured data')
    parser.add_argument('--width', type=int, default=640,
                        help='Capture image width')
    parser.add_argument('--height', type=int, default=480,
                        help='Capture image height')

    # Painting options
    parser.add_argument('--paint', '-p', action='store_true',
                        help='Paint trajectory on lens texture')
    parser.add_argument('--paint-radius', type=int, default=3,
                        help='Radius in pixels for painting')
    parser.add_argument('--texture', type=str, default='../textures/lens_uv_map.png',
                        help='Path to lens UV texture')
    parser.add_argument('--mesh', type=str, default='../meshes/Lens_L_extracted.obj',
                        help='Path to lens mesh with UVs')
    parser.add_argument('--eye-pos', type=float, nargs=3, default=[0, 0, 0.1],
                        metavar=('X', 'Y', 'Z'),
                        help='Eye assembly position in world coordinates')

    # RCM (Remote Center of Motion) options
    parser.add_argument('--rcm', action='store_true',
                        help='Enable RCM constraint for tool orientation')
    parser.add_argument('--rcm-radius', type=float, default=0.005,
                        help='RCM distance from lens center in meters (default 5mm)')
    parser.add_argument('--rcm-height', type=float, default=0.006,
                        help='RCM height above lens surface in meters (default 6mm, tool reach is 8mm)')
    parser.add_argument('--tool-roll', type=float, default=0.0,
                        help='Tool roll around shaft axis in degrees (default 0)')

    # Lighting variation
    parser.add_argument('--vary-lights', action='store_true',
                        help='Enable per-frame lighting variation')

    # Debug
    parser.add_argument('--debug', action='store_true',
                        help='Print frame idx, tool pos and euler for each frame')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("TOOL TRAJECTORY CONTROLLER")
    print("="*60)

    # Determine mode: single trajectory file or multi-iteration
    if args.trajectory_file is not None and args.iterations == 1:
        # Single trajectory file mode
        if not Path(args.trajectory_file).exists():
            print(f"Error: Trajectory file not found: {args.trajectory_file}")
            sys.exit(1)

        print("\nMode: Single trajectory file")
        trajectory_world, metadata = load_trajectory_file(args.trajectory_file)

        # Initialize RCM controller if requested
        rcm_controller = None
        if args.rcm:
            print("\nInitializing RCM controller...")
            # Need lens geometry to calculate RCM position
            lens_geometry = analyze_lens_geometry(args.mesh)
            # Get rotation angle from trajectory metadata (default 0 if not present)
            rotation_angle = metadata.get('parameters', {}).get('rotation_angle', 0.0)
            rcm_world = calculate_rcm_from_lens(
                lens_geometry,
                rotation_angle=rotation_angle,
                eye_pos=args.eye_pos,
                rcm_radius=args.rcm_radius,
                rcm_height=args.rcm_height
            )
            rcm_controller = RCMController(rcm_world, tool_roll_deg=args.tool_roll)
            print(f"  RCM position: [{rcm_world[0]:.4f}, {rcm_world[1]:.4f}, {rcm_world[2]:.4f}]")
            print(f"  RCM radius: {args.rcm_radius*1000:.1f} mm")
            print(f"  RCM height: {args.rcm_height*1000:.1f} mm")
            print(f"  Tool roll: {args.tool_roll:.1f} deg")

        # Initialize painter if requested
        painter = None
        scene_path = args.scene

        if args.paint:
            print("\nResetting texture to clean state...")
            reset_texture(args.texture)
            print("\nInitializing texture painter...")
            painter = TexturePainter(
                texture_path=args.texture,
                mesh_path=args.mesh,
                eye_assembly_pos=args.eye_pos
            )
            if args.capture:
                print("Creating temp scene for texture updates...")
                scene_path = create_temp_scene(args.scene)

        print("\nLoading MuJoCo model...")
        model = mujoco.MjModel.from_xml_path(scene_path)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        print("✓ Model loaded")

        if args.capture:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(args.output) / f"capture_{timestamp}"
            run_with_capture(
                model, data, trajectory_world, output_dir,
                width=args.width, height=args.height,
                painter=painter, paint_radius=args.paint_radius,
                scene_path=scene_path,
                rcm_controller=rcm_controller,
                vary_lights=args.vary_lights,
                debug=args.debug
            )
        else:
            move_tool_along_trajectory(
                model, data, trajectory_world,
                speed=args.speed,
                painter=painter, paint_radius=args.paint_radius,
                rcm_controller=rcm_controller,
                debug=args.debug
            )

    else:
        # Multi-iteration mode
        print(f"\nMode: Multi-iteration ({args.iterations} iterations)")
        print("Generating randomized ellipsoid trajectories")

        # Analyze lens geometry once
        print("\nAnalyzing lens geometry...")
        lens_geometry = analyze_lens_geometry(args.mesh)

        # Prepare scene
        scene_path = args.scene
        if args.paint and args.capture:
            print("\nCreating temp scene for texture updates...")
            scene_path = create_temp_scene(args.scene)
            print(f"  Temp scene: {scene_path}")

        # Create base output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output_dir = Path(args.output) / f"run_{timestamp}"
        base_output_dir.mkdir(parents=True, exist_ok=True)

        # Load model to get camera parameters
        print("\nLoading model for camera parameters...")
        temp_model = mujoco.MjModel.from_xml_path(scene_path)
        camera_params = get_camera_params_from_model(
            temp_model, ['top_view', 'angle_view', 'tool_view']
        )

        # Save run configuration
        config = {
            'timestamp': timestamp,
            'iterations': args.iterations,
            'base_radius_x': args.radius_x,
            'base_radius_y': args.radius_y,
            'radius_std': args.radius_std,
            'center_std': args.center_std,
            'num_points': args.num_points,
            'eye_pos': args.eye_pos,
            'resolution': [args.width, args.height],
            'paint': args.paint,
            'paint_radius': args.paint_radius,
            'cameras': camera_params,
            'rcm': {
                'enabled': args.rcm,
                'radius': args.rcm_radius,
                'height': args.rcm_height
            }
        }
        with open(base_output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\nOutput directory: {base_output_dir}")
        print(f"Config saved to: {base_output_dir / 'config.json'}")

        # Run iterations
        for i in range(args.iterations):
            run_iteration(i, args, lens_geometry, scene_path, base_output_dir)

        print(f"\n{'='*60}")
        print(f"ALL ITERATIONS COMPLETE")
        print(f"{'='*60}")
        print(f"Output: {base_output_dir}")
        print(f"Iterations: {args.iterations}")

    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
