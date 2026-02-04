#!/usr/bin/env python3
"""
Trajectory Generator: Generates 3D trajectories for tool movement in MuJoCo simulation
Uses the same functions as analyze_lens.py for consistency.
"""

import numpy as np
import json
import argparse
from datetime import datetime
from pathlib import Path
from analyze_lens import (
    analyze_lens_geometry,
    generate_test_trajectory,
    transform_to_world_coordinates,
    project_trajectory_to_mesh,
    generate_xml_markers,
    generate_xml_projected_only,
    wrap_xml_for_mujoco
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate trajectory for tool movement on lens surface',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Trajectory parameters
    parser.add_argument('-r', '--radius', type=float, default=0.003,
                        help='Trajectory radius in meters')
    parser.add_argument('-n', '--num-points', type=int, default=600,
                        help='Number of points in trajectory')
    parser.add_argument('-o', '--offset', type=float, default=0.0,
                        help='Offset along normal in meters (0 = on surface)')

    # File paths
    parser.add_argument('--obj', type=str, default='../meshes/Lens_L_extracted.obj',
                        help='Path to lens OBJ file')
    parser.add_argument('--output-dir', type=str, default='../trajectories',
                        help='Output directory for trajectory JSON')
    parser.add_argument('--scene-dir', type=str, default='../scene',
                        help='Output directory for XML files')

    # Eye assembly position
    parser.add_argument('--eye-pos', type=float, nargs=3, default=[0, 0, 0.1],
                        metavar=('X', 'Y', 'Z'),
                        help='Eye assembly position in world coordinates')

    # Options
    parser.add_argument('--no-xml', action='store_true',
                        help='Skip generating XML visualization files')

    return parser.parse_args()


def save_trajectory(trajectory_local, trajectory_world, params,
                   projected_local=None, projected_world=None,
                   output_dir='../trajectories'):
    """
    Save trajectory to file with metadata

    Args:
        trajectory_local: Nx3 array in lens local coordinates (raw)
        trajectory_world: Nx3 array in world coordinates (raw)
        params: Parameters used to generate trajectory
        projected_local: Nx3 array of projected trajectory in local coords
        projected_world: Nx3 array of projected trajectory in world coords
        output_dir: Directory to save trajectory files

    Returns:
        Path to saved trajectory file
    """
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # Generate timestamp-based filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"trajectory_{timestamp}.json"
    filepath = Path(output_dir) / filename

    # Prepare data
    data = {
        'metadata': {
            'timestamp': timestamp,
            'num_points': len(trajectory_local),
            'parameters': params,
            'has_projected': projected_local is not None
        },
        'trajectory_local': trajectory_local.tolist(),
        'trajectory_world': trajectory_world.tolist()
    }

    # Add projected trajectory if available
    if projected_local is not None:
        data['projected_local'] = projected_local.tolist()
        data['projected_world'] = projected_world.tolist()

    # Save to JSON
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"✓ Trajectory saved to: {filepath}")
    return str(filepath)


def main():
    """
    Main execution: Analyze lens and generate projected trajectory
    Uses same generate_test_trajectory function as analyze_lens.py
    """
    args = parse_args()

    # Configuration from args
    OBJ_FILE = args.obj
    EYE_ASSEMBLY_POS = args.eye_pos
    RADIUS_METERS = args.radius
    NUM_POINTS = args.num_points
    OFFSET_METERS = args.offset
    OUTPUT_DIR = args.output_dir
    SCENE_DIR = args.scene_dir

    # Step 1: Analyze lens geometry
    print("="*60)
    print("TRAJECTORY GENERATOR")
    print("="*60)
    print("\nStep 1: Analyzing lens geometry...")
    lens_geometry = analyze_lens_geometry(OBJ_FILE)

    # Step 2: Generate trajectory using same function as analyze_lens.py
    print("\n" + "="*60)
    print("Step 2: Generating trajectory...")
    print("="*60)
    print(f"  Radius: {RADIUS_METERS*1000:.1f} mm")
    print(f"  Points: {NUM_POINTS}")
    print(f"  Offset: {OFFSET_METERS*1000:.1f} mm")

    trajectory_local = generate_test_trajectory(
        lens_geometry,
        radius_meters=RADIUS_METERS,
        num_points=NUM_POINTS,
        offset_meters=OFFSET_METERS
    )

    # Transform to world coordinates
    trajectory_world = transform_to_world_coordinates(trajectory_local, EYE_ASSEMBLY_POS)

    print(f"\n✓ Generated raw trajectory: {len(trajectory_local)} points")

    # Step 3: Project trajectory onto lens surface
    print("\n" + "="*60)
    print("Step 3: Projecting to lens surface...")
    print("="*60)

    projected_local = project_trajectory_to_mesh(trajectory_local, lens_geometry)
    projected_world = transform_to_world_coordinates(projected_local, EYE_ASSEMBLY_POS)

    print(f"✓ Projected trajectory: {len(projected_local)} points")

    # Step 4: Save trajectory
    print("\n" + "="*60)
    print("Step 4: Saving trajectory...")
    print("="*60)

    params = {
        'radius_meters': RADIUS_METERS,
        'num_points': NUM_POINTS,
        'offset_meters': OFFSET_METERS,
        'eye_assembly_pos': list(EYE_ASSEMBLY_POS)
    }

    trajectory_file = save_trajectory(
        trajectory_local,
        trajectory_world,
        params,
        projected_local=projected_local,
        projected_world=projected_world,
        output_dir=OUTPUT_DIR
    )

    # Step 5: Generate XML visualization files (optional)
    if not args.no_xml:
        print("\n" + "="*60)
        print("Step 5: Generating XML visualization files...")
        print("="*60)

        # Generate XML with both trajectories (raw=red, projected=purple)
        xml_both = generate_xml_markers(
            lens_geometry,
            trajectory_local,
            EYE_ASSEMBLY_POS,
            projected_trajectory_local=projected_local
        )

        # Generate XML with only projected trajectory
        xml_projected = generate_xml_projected_only(
            lens_geometry,
            projected_local,
            EYE_ASSEMBLY_POS
        )

        # Save both trajectories file
        output_file_both = f'{SCENE_DIR}/lens_visualization.xml'
        with open(output_file_both, 'w') as f:
            f.write(wrap_xml_for_mujoco(xml_both))
        print(f"✓ Both trajectories: {output_file_both}")
        print(f"    - Red markers: raw trajectory")
        print(f"    - Purple markers: projected onto surface")

        # Save projected only file
        output_file_projected = f'{SCENE_DIR}/lens_visualization_projected.xml'
        with open(output_file_projected, 'w') as f:
            f.write(wrap_xml_for_mujoco(xml_projected))
        print(f"✓ Projected only: {output_file_projected}")

    # Summary
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)
    print(f"\nTrajectory JSON: {trajectory_file}")
    if not args.no_xml:
        print(f"XML files:")
        print(f"  - {output_file_both} (both trajectories)")
        print(f"  - {output_file_projected} (projected only)")
    print(f"\nRun tool controller:")
    print(f"  python3 tool_controller.py {trajectory_file}")
    if not args.no_xml:
        print(f"\nTo visualize in scene, use:")
        print(f"  <include file=\"lens_visualization.xml\"/>")
    print("="*60 + "\n")

    return trajectory_file


if __name__ == "__main__":
    main()
