#!/usr/bin/env python3
"""
Phase 1: OBJ Parser & Lens Analysis
Analyzes Lens_L_extracted.obj to extract geometry and calculate lens plane
Uses trimesh for fast mesh operations.
"""

import numpy as np
from pathlib import Path

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    print("Warning: trimesh not installed. Using slow fallback. Install with: pip install trimesh")


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
    
    # Create trimesh object for fast operations
    trimesh_obj = None
    if TRIMESH_AVAILABLE:
        # Triangulate faces for trimesh
        triangles = []
        for face in obj_data['faces']:
            if len(face) >= 3:
                # Fan triangulation for polygons
                for i in range(1, len(face) - 1):
                    triangles.append([face[0], face[i], face[i + 1]])
        triangles = np.array(triangles)
        trimesh_obj = trimesh.Trimesh(vertices=obj_data['vertices'], faces=triangles)
        print(f"  ✓ Trimesh object created ({len(triangles)} triangles)")

    # Combine all results
    lens_geometry = {
        'vertices': obj_data['vertices'],
        'uvs': obj_data['uvs'],
        'faces': obj_data['faces'],
        'face_uvs': obj_data['face_uvs'],
        'center_local': center,
        'normal': normal,
        'bounds': bounds,
        'coord_frame': coord_frame,
        'trimesh': trimesh_obj  # Cached trimesh object for fast operations
    }

    return lens_geometry


def generate_test_trajectory(lens_geometry, radius_meters=0.012, num_points=36, offset_meters=-0.0015):
    """
    Generate a test circular trajectory on the lens plane
    
    Args:
        lens_geometry: Output from analyze_lens_geometry()
        radius_meters: Radius of circle in meters
        num_points: Number of points around circle
        offset_meters: Offset along normal (positive = outward)
        
    Returns:
        Nx3 array of points in lens local coordinates
    """
    center = lens_geometry['center_local']
    coord_frame = lens_geometry['coord_frame']
    
    # Offset center along the normal (outward from lens surface)
    center = center + offset_meters * coord_frame['z_axis']
    
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


def closest_point_on_triangle(point, v0, v1, v2):
    """
    Find the closest point on a triangle to a given point.

    Args:
        point: 3D point to project
        v0, v1, v2: Triangle vertices

    Returns:
        Closest point on the triangle surface
    """
    # Edge vectors
    edge0 = v1 - v0
    edge1 = v2 - v0
    v0_to_point = v0 - point

    # Compute dot products
    a = np.dot(edge0, edge0)
    b = np.dot(edge0, edge1)
    c = np.dot(edge1, edge1)
    d = np.dot(edge0, v0_to_point)
    e = np.dot(edge1, v0_to_point)

    det = a * c - b * b

    # Handle degenerate triangle (zero area)
    if abs(det) < 1e-12:
        # Return closest vertex
        d0 = np.linalg.norm(point - v0)
        d1 = np.linalg.norm(point - v1)
        d2 = np.linalg.norm(point - v2)
        if d0 <= d1 and d0 <= d2:
            return v0.copy()
        elif d1 <= d2:
            return v1.copy()
        else:
            return v2.copy()

    s = b * e - c * d
    t = b * d - a * e

    if s + t <= det:
        if s < 0:
            if t < 0:
                # Region 4
                if d < 0:
                    t = 0
                    s = np.clip(-d / a, 0, 1)
                else:
                    s = 0
                    t = np.clip(-e / c, 0, 1)
            else:
                # Region 3
                s = 0
                t = np.clip(-e / c, 0, 1)
        elif t < 0:
            # Region 5
            t = 0
            s = np.clip(-d / a, 0, 1)
        else:
            # Region 0 (inside triangle)
            inv_det = 1.0 / det
            s *= inv_det
            t *= inv_det
    else:
        if s < 0:
            # Region 2
            tmp0 = b + d
            tmp1 = c + e
            if tmp1 > tmp0:
                numer = tmp1 - tmp0
                denom = a - 2 * b + c
                s = np.clip(numer / denom, 0, 1)
                t = 1 - s
            else:
                s = 0
                t = np.clip(-e / c, 0, 1)
        elif t < 0:
            # Region 6
            tmp0 = b + e
            tmp1 = a + d
            if tmp1 > tmp0:
                numer = tmp1 - tmp0
                denom = a - 2 * b + c
                t = np.clip(numer / denom, 0, 1)
                s = 1 - t
            else:
                t = 0
                s = np.clip(-d / a, 0, 1)
        else:
            # Region 1
            numer = (c + e) - (b + d)
            if numer <= 0:
                s = 0
            else:
                denom = a - 2 * b + c
                s = np.clip(numer / denom, 0, 1)
            t = 1 - s

    return v0 + s * edge0 + t * edge1


def project_point_to_mesh(point, vertices, faces):
    """
    Project a single point onto the mesh surface (find closest point).

    Args:
        point: 3D point to project
        vertices: Mesh vertices
        faces: Mesh faces (triangle indices)

    Returns:
        Closest point on mesh surface
    """
    min_dist = float('inf')
    closest_point = None

    for face in faces:
        if len(face) < 3:
            continue

        # Get triangle vertices
        v0 = vertices[face[0]]
        v1 = vertices[face[1]]
        v2 = vertices[face[2]]

        # Find closest point on this triangle
        closest_on_tri = closest_point_on_triangle(point, v0, v1, v2)
        dist = np.linalg.norm(point - closest_on_tri)

        if dist < min_dist:
            min_dist = dist
            closest_point = closest_on_tri

        # Handle quads (split into two triangles)
        if len(face) == 4:
            v3 = vertices[face[3]]
            closest_on_tri = closest_point_on_triangle(point, v0, v2, v3)
            dist = np.linalg.norm(point - closest_on_tri)

            if dist < min_dist:
                min_dist = dist
                closest_point = closest_on_tri

    return closest_point


def project_trajectory_to_mesh(trajectory_local, lens_geometry):
    """
    Project trajectory points onto the mesh surface.

    Args:
        trajectory_local: Nx3 array of trajectory points in local coordinates
        lens_geometry: Output from analyze_lens_geometry()

    Returns:
        Nx3 array of projected points on mesh surface
    """
    print(f"\nProjecting {len(trajectory_local)} points onto mesh surface...")

    # Use trimesh for fast vectorized projection
    if TRIMESH_AVAILABLE and lens_geometry.get('trimesh') is not None:
        mesh = lens_geometry['trimesh']
        # nearest.on_surface returns (closest_points, distances, triangle_ids)
        projected, distances, _ = mesh.nearest.on_surface(trajectory_local)
        print(f"  ✓ Projection complete (using trimesh)")
        print(f"  ✓ Mean projection distance: {np.mean(distances)*1000:.4f} mm")
        print(f"  ✓ Max projection distance: {np.max(distances)*1000:.4f} mm")
        return projected

    # Fallback to slow method if trimesh not available
    print("  Using slow fallback (trimesh not available)...")
    vertices = lens_geometry['vertices']
    faces = lens_geometry['faces']

    projected = []
    for i, point in enumerate(trajectory_local):
        proj_point = project_point_to_mesh(point, vertices, faces)
        projected.append(proj_point)

        if (i + 1) % 10 == 0:
            print(f"  Projected {i + 1}/{len(trajectory_local)} points")

    projected = np.array(projected)

    # Calculate projection statistics
    distances = np.linalg.norm(trajectory_local - projected, axis=1)
    print(f"  ✓ Projection complete")
    print(f"  ✓ Mean projection distance: {np.mean(distances)*1000:.4f} mm")
    print(f"  ✓ Max projection distance: {np.max(distances)*1000:.4f} mm")

    return projected


def generate_xml_markers(lens_geometry, trajectory_local, eye_assembly_pos,
                        marker_size=0.0001, projected_trajectory_local=None):
    """
    Generate XML code to visualize lens center and trajectories in MuJoCo

    Args:
        lens_geometry: Output from analyze_lens_geometry()
        trajectory_local: Raw trajectory in local coordinates
        eye_assembly_pos: Position of eye assembly
        marker_size: Size of marker spheres
        projected_trajectory_local: Optional projected trajectory (if provided, both are shown)

    Returns:
        String containing XML code to add to ../scene/eye_scene.xml
    """
    center_local = lens_geometry['center_local']
    center_world = center_local + np.array(eye_assembly_pos)

    trajectory_world = transform_to_world_coordinates(trajectory_local, eye_assembly_pos)

    xml_lines = []
    xml_lines.append("    <!-- LENS ANALYSIS VISUALIZATION -->")

    # Lens center marker
    xml_lines.append(f"    <!-- Lens geometric center -->")
    xml_lines.append(f"    <body name=\"lens_center_marker\" pos=\"{center_world[0]:.6f} {center_world[1]:.6f} {center_world[2]:.6f}\">")
    xml_lines.append(f"      <geom name=\"lens_center_sphere\" type=\"sphere\" size=\"{marker_size*2}\" rgba=\"0 1 0 0.8\" contype=\"0\" conaffinity=\"0\"/>")
    xml_lines.append(f"    </body>")

    # Raw trajectory markers (RED)
    xml_lines.append(f"\n    <!-- Raw trajectory points ({len(trajectory_world)} points) - RED -->")
    for i, point in enumerate(trajectory_world):
        xml_lines.append(f"    <body name=\"raw_trajectory_point_{i}\" pos=\"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\">")
        xml_lines.append(f"      <geom name=\"raw_traj_marker_{i}\" type=\"sphere\" size=\"{marker_size}\" rgba=\"1 0 0 0.6\" contype=\"0\" conaffinity=\"0\"/>")
        xml_lines.append(f"    </body>")

    # Projected trajectory markers (PURPLE/MAGENTA) if provided
    if projected_trajectory_local is not None:
        projected_world = transform_to_world_coordinates(projected_trajectory_local, eye_assembly_pos)
        xml_lines.append(f"\n    <!-- Projected trajectory points ({len(projected_world)} points) - PURPLE -->")
        for i, point in enumerate(projected_world):
            xml_lines.append(f"    <body name=\"proj_trajectory_point_{i}\" pos=\"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\">")
            xml_lines.append(f"      <geom name=\"proj_traj_marker_{i}\" type=\"sphere\" size=\"{marker_size}\" rgba=\"1 0 1 0.8\" contype=\"0\" conaffinity=\"0\"/>")
            xml_lines.append(f"    </body>")

    xml_code = "\n".join(xml_lines)

    return xml_code


def generate_xml_projected_only(lens_geometry, projected_trajectory_local, eye_assembly_pos,
                                marker_size=0.0001):
    """
    Generate XML code with only the projected trajectory.

    Args:
        lens_geometry: Output from analyze_lens_geometry()
        projected_trajectory_local: Projected trajectory in local coordinates
        eye_assembly_pos: Position of eye assembly
        marker_size: Size of marker spheres

    Returns:
        String containing XML code
    """
    center_local = lens_geometry['center_local']
    center_world = center_local + np.array(eye_assembly_pos)

    projected_world = transform_to_world_coordinates(projected_trajectory_local, eye_assembly_pos)

    xml_lines = []
    xml_lines.append("    <!-- LENS VISUALIZATION - PROJECTED TRAJECTORY ONLY -->")

    # Lens center marker
    xml_lines.append(f"    <!-- Lens geometric center -->")
    xml_lines.append(f"    <body name=\"lens_center_marker\" pos=\"{center_world[0]:.6f} {center_world[1]:.6f} {center_world[2]:.6f}\">")
    xml_lines.append(f"      <geom name=\"lens_center_sphere\" type=\"sphere\" size=\"{marker_size*2}\" rgba=\"0 1 0 0.8\" contype=\"0\" conaffinity=\"0\"/>")
    xml_lines.append(f"    </body>")

    # Projected trajectory markers (PURPLE/MAGENTA)
    xml_lines.append(f"\n    <!-- Projected trajectory points ({len(projected_world)} points) -->")
    for i, point in enumerate(projected_world):
        xml_lines.append(f"    <body name=\"trajectory_point_{i}\" pos=\"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\">")
        xml_lines.append(f"      <geom name=\"traj_marker_{i}\" type=\"sphere\" size=\"{marker_size}\" rgba=\"1 0 1 0.8\" contype=\"0\" conaffinity=\"0\"/>")
        xml_lines.append(f"    </body>")

    xml_code = "\n".join(xml_lines)

    return xml_code


def wrap_xml_for_mujoco(xml_content):
    """Wrap XML content in MuJoCo structure for include."""
    return f"""<mujoco>
  <worldbody>
{xml_content}
  </worldbody>
</mujoco>
"""


def main():
    """Main execution"""
    # Configuration
    OBJ_FILE = '../meshes/Lens_L_extracted.obj'
    EYE_ASSEMBLY_POS = [0, 0, 0.1]  # From ../scene/eye_scene.xml

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

    # Project trajectory onto mesh surface
    print("\n" + "="*60)
    print("PROJECTING TRAJECTORY TO MESH SURFACE")
    print("="*60)

    projected_trajectory_local = project_trajectory_to_mesh(trajectory_local, lens_geometry)

    # Generate XML for visualization
    print("\n" + "="*60)
    print("GENERATING XML CODE")
    print("="*60)

    # Generate XML with both trajectories (raw=red, projected=purple)
    xml_both = generate_xml_markers(
        lens_geometry,
        trajectory_local,
        EYE_ASSEMBLY_POS,
        projected_trajectory_local=projected_trajectory_local
    )

    # Generate XML with only projected trajectory
    xml_projected = generate_xml_projected_only(
        lens_geometry,
        projected_trajectory_local,
        EYE_ASSEMBLY_POS
    )

    # Save both trajectories file
    output_file_both = '../scene/lens_visualization.xml'
    with open(output_file_both, 'w') as f:
        f.write(wrap_xml_for_mujoco(xml_both))

    # Save projected only file
    output_file_projected = '../scene/lens_visualization_projected.xml'
    with open(output_file_projected, 'w') as f:
        f.write(wrap_xml_for_mujoco(xml_projected))

    print(f"\n  ✓ Both trajectories saved to: {output_file_both}")
    print(f"    - Red markers: raw trajectory")
    print(f"    - Purple markers: projected onto surface")
    print(f"\n  ✓ Projected only saved to: {output_file_projected}")
    print(f"    - Purple markers: projected trajectory only")

    print(f"\n  To use with eye_scene_with_trajectory.xml:")
    print(f"    <include file=\"lens_visualization.xml\"/>  (both trajectories)")
    print(f"    <include file=\"lens_visualization_projected.xml\"/>  (projected only)")

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60 + "\n")

    return lens_geometry, trajectory_local, projected_trajectory_local


if __name__ == "__main__":
    lens_geometry, trajectory, projected = main()