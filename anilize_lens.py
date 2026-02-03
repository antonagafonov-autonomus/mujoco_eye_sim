#!/usr/bin/env python3
"""
Phase 1: OBJ Parser & Lens Analysis
Analyzes Lens_L_extracted.obj to extract geometry and calculate lens plane
"""

import numpy as np
from pathlib import Path


def parse_obj_file(obj_path):
    """
    Parse OBJ file and extract vertices, UVs, and faces
    
    Returns:
        dict with:
            - vertices: Nx3 array of vertex positions
            - uvs: Mx2 array of texture coordinates
            - faces: List of face definitions (vertex indices)
            - face_uvs: List of UV indices per face
    """
    vertices = []
    uvs = []
    faces = []
    face_uvs = []
    
    print(f"Parsing OBJ file: {obj_path}")
    
    with open(obj_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith('v '):  # Vertex position
                parts = line.split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                vertices.append([x, y, z])
                
            elif line.startswith('vt '):  # Texture coordinate
                parts = line.split()
                u, v = float(parts[1]), float(parts[2])
                uvs.append([u, v])
                
            elif line.startswith('f '):  # Face
                parts = line.split()[1:]  # Skip 'f'
                face_vertices = []
                face_uv_indices = []
                
                for part in parts:
                    # Format can be: v, v/vt, v/vt/vn, v//vn
                    indices = part.split('/')
                    
                    # Vertex index (1-based in OBJ, convert to 0-based)
                    v_idx = int(indices[0]) - 1
                    face_vertices.append(v_idx)
                    
                    # UV index (if present)
                    if len(indices) > 1 and indices[1]:
                        uv_idx = int(indices[1]) - 1
                        face_uv_indices.append(uv_idx)
                
                faces.append(face_vertices)
                if face_uv_indices:
                    face_uvs.append(face_uv_indices)
    
    vertices = np.array(vertices)
    uvs = np.array(uvs) if uvs else np.array([])
    
    print(f"  ✓ Vertices: {len(vertices)}")
    print(f"  ✓ UVs: {len(uvs)}")
    print(f"  ✓ Faces: {len(faces)}")
    
    return {
        'vertices': vertices,
        'uvs': uvs,
        'faces': faces,
        'face_uvs': face_uvs
    }


def calculate_geometric_center(vertices):
    """Calculate the geometric center (centroid) of all vertices"""
    center = np.mean(vertices, axis=0)
    return center


def calculate_plane_normal(vertices, faces):
    """
    Calculate the average normal vector of the lens surface
    Uses face normals weighted by face area
    """
    face_normals = []
    face_areas = []
    
    for face in faces:
        if len(face) < 3:
            continue
            
        # Get first three vertices of face
        v0 = vertices[face[0]]
        v1 = vertices[face[1]]
        v2 = vertices[face[2]]
        
        # Calculate face normal using cross product
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        
        # Area is half the magnitude of cross product
        area = np.linalg.norm(normal) / 2.0
        
        if area > 1e-10:  # Avoid degenerate faces
            normal = normal / (2.0 * area)  # Normalize
            face_normals.append(normal)
            face_areas.append(area)
    
    # Weighted average of normals by face area
    face_normals = np.array(face_normals)
    face_areas = np.array(face_areas)
    
    weighted_normal = np.average(face_normals, axis=0, weights=face_areas)
    weighted_normal = weighted_normal / np.linalg.norm(weighted_normal)
    
    return weighted_normal


def calculate_bounding_info(vertices, center):
    """Calculate bounding radius and dimensions"""
    # Distance from center to each vertex
    distances = np.linalg.norm(vertices - center, axis=1)
    
    max_radius = np.max(distances)
    mean_radius = np.mean(distances)
    
    # Bounding box
    min_bounds = np.min(vertices, axis=0)
    max_bounds = np.max(vertices, axis=0)
    dimensions = max_bounds - min_bounds
    
    return {
        'max_radius': max_radius,
        'mean_radius': mean_radius,
        'min_bounds': min_bounds,
        'max_bounds': max_bounds,
        'dimensions': dimensions
    }


def create_local_coordinate_frame(normal, center):
    """
    Create a local coordinate frame for the lens plane
    
    Returns:
        - x_axis: tangent to plane
        - y_axis: tangent to plane (perpendicular to x_axis)
        - z_axis: normal to plane
        - origin: center point
    """
    z_axis = normal / np.linalg.norm(normal)
    
    # Create arbitrary perpendicular vector
    if abs(z_axis[0]) < 0.9:
        arbitrary = np.array([1, 0, 0])
    else:
        arbitrary = np.array([0, 1, 0])
    
    x_axis = np.cross(arbitrary, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    
    return {
        'origin': center,
        'x_axis': x_axis,
        'y_axis': y_axis,
        'z_axis': z_axis
    }


def analyze_lens_geometry(obj_path):
    """
    Complete analysis of lens geometry
    
    Returns comprehensive lens geometry information
    """
    print("\n" + "="*60)
    print("LENS GEOMETRY ANALYSIS")
    print("="*60)
    
    # Parse OBJ file
    obj_data = parse_obj_file(obj_path)
    
    # Calculate geometric properties
    print("\nCalculating geometric properties...")
    center = calculate_geometric_center(obj_data['vertices'])
    normal = calculate_plane_normal(obj_data['vertices'], obj_data['faces'])
    bounds = calculate_bounding_info(obj_data['vertices'], center)
    coord_frame = create_local_coordinate_frame(normal, center)
    
    # Display results
    print(f"\n  Geometric Center (local coords):")
    print(f"    X: {center[0]:.6f}")
    print(f"    Y: {center[1]:.6f}")
    print(f"    Z: {center[2]:.6f}")
    
    print(f"\n  Plane Normal Vector:")
    print(f"    X: {normal[0]:.6f}")
    print(f"    Y: {normal[1]:.6f}")
    print(f"    Z: {normal[2]:.6f}")
    
    print(f"\n  Bounding Information:")
    print(f"    Max radius: {bounds['max_radius']:.6f}")
    print(f"    Mean radius: {bounds['mean_radius']:.6f}")
    print(f"    Dimensions (XYZ): {bounds['dimensions']}")
    
    print(f"\n  Local Coordinate Frame:")
    print(f"    X-axis: {coord_frame['x_axis']}")
    print(f"    Y-axis: {coord_frame['y_axis']}")
    print(f"    Z-axis: {coord_frame['z_axis']}")
    
    # Combine all results
    lens_geometry = {
        'vertices': obj_data['vertices'],
        'uvs': obj_data['uvs'],
        'faces': obj_data['faces'],
        'face_uvs': obj_data['face_uvs'],
        'center_local': center,
        'normal': normal,
        'bounds': bounds,
        'coord_frame': coord_frame
    }
    
    return lens_geometry


def generate_test_trajectory(lens_geometry, radius_meters=0.003, num_points=36):
    """
    Generate a test circular trajectory on the lens plane
    
    Args:
        lens_geometry: Output from analyze_lens_geometry()
        radius_meters: Radius of circle in meters
        num_points: Number of points around circle
        
    Returns:
        Nx3 array of points in lens local coordinates
    """
    center = lens_geometry['center_local']
    coord_frame = lens_geometry['coord_frame']
    
    angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    trajectory = []
    
    for angle in angles:
        # Create point in local 2D plane coordinates
        x_local = radius_meters * np.cos(angle)
        y_local = radius_meters * np.sin(angle)
        
        # Transform to 3D lens coordinates
        point = (center + 
                 x_local * coord_frame['x_axis'] + 
                 y_local * coord_frame['y_axis'])
        
        trajectory.append(point)
    
    return np.array(trajectory)


def transform_to_world_coordinates(points_local, eye_assembly_pos):
    """
    Transform points from lens local coordinates to world coordinates
    
    Args:
        points_local: Nx3 array in lens local coordinates
        eye_assembly_pos: [x, y, z] position of eye_assembly body
        
    Returns:
        Nx3 array in world coordinates
    """
    eye_assembly_pos = np.array(eye_assembly_pos)
    points_world = points_local + eye_assembly_pos
    return points_world


def generate_xml_markers(lens_geometry, trajectory_local, eye_assembly_pos, 
                        marker_size=0.002):
    """
    Generate XML code to visualize lens center and trajectory in MuJoCo
    
    Returns:
        String containing XML code to add to eye_scene.xml
    """
    center_local = lens_geometry['center_local']
    center_world = center_local + np.array(eye_assembly_pos)
    
    trajectory_world = transform_to_world_coordinates(trajectory_local, eye_assembly_pos)
    
    xml_lines = []
    xml_lines.append("\n    <!-- LENS ANALYSIS VISUALIZATION -->")
    
    # Lens center marker
    xml_lines.append(f"    <!-- Lens geometric center -->")
    xml_lines.append(f"    <body name=\"lens_center_marker\" pos=\"{center_world[0]:.6f} {center_world[1]:.6f} {center_world[2]:.6f}\">")
    xml_lines.append(f"      <geom name=\"lens_center_sphere\" type=\"sphere\" size=\"{marker_size*2}\" rgba=\"0 1 0 0.8\" contype=\"0\" conaffinity=\"0\"/>")
    xml_lines.append(f"    </body>")
    
    # Trajectory markers
    xml_lines.append(f"\n    <!-- Test trajectory points ({len(trajectory_world)} points) -->")
    for i, point in enumerate(trajectory_world):
        xml_lines.append(f"    <body name=\"trajectory_point_{i}\" pos=\"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\">")
        xml_lines.append(f"      <geom name=\"traj_marker_{i}\" type=\"sphere\" size=\"{marker_size}\" rgba=\"1 0 1 0.6\" contype=\"0\" conaffinity=\"0\"/>")
        xml_lines.append(f"    </body>")
    
    xml_code = "\n".join(xml_lines)
    
    return xml_code


def main():
    """Main execution"""
    # Configuration
    OBJ_FILE = 'meshes/Lens_L_extracted.obj'
    EYE_ASSEMBLY_POS = [0, 0, 0.1]  # From eye_scene.xml
    
    # Analyze lens
    lens_geometry = analyze_lens_geometry(OBJ_FILE)
    
    # Generate test trajectory
    print("\n" + "="*60)
    print("GENERATING TEST TRAJECTORY")
    print("="*60)
    
    trajectory_local = generate_test_trajectory(
        lens_geometry, 
        radius_meters=0.003,  # 3mm radius
        num_points=36  # 36 points for visibility
    )
    
    print(f"  ✓ Generated {len(trajectory_local)} trajectory points")
    print(f"  ✓ Radius: 0.003 m (3 mm)")
    
    # Generate XML for visualization
    print("\n" + "="*60)
    print("GENERATING XML CODE")
    print("="*60)
    
    xml_code = generate_xml_markers(
        lens_geometry, 
        trajectory_local, 
        EYE_ASSEMBLY_POS
    )
    
    # Save to file
    output_file = 'lens_visualization.xml'
    with open(output_file, 'w') as f:
        f.write(xml_code)
    
    print(f"\n  ✓ XML code saved to: {output_file}")
    print(f"\n  To add to eye_scene.xml:")
    print(f"    1. Open eye_scene.xml")
    print(f"    2. Find the </worldbody> closing tag")
    print(f"    3. Paste the contents of {output_file} BEFORE </worldbody>")
    print(f"    4. Run: python3 render_eye.py")
    print(f"\n  You should see:")
    print(f"    - Green sphere at lens center")
    print(f"    - Magenta spheres forming a circle (trajectory)")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60 + "\n")
    
    return lens_geometry, trajectory_local


if __name__ == "__main__":
    lens_geometry, trajectory = main()