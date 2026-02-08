#!/usr/bin/env python3
"""
Utils: Shared utilities for MuJoCo eye simulation
- Tool trajectory functions
- UV mapping and texture painting
- Texture reloading during simulation
"""

import random
import mujoco
import mujoco.viewer
import numpy as np
import subprocess
import shutil
import json
from pathlib import Path
from datetime import datetime
from PIL import Image


# =============================================================================
# TOOL OFFSET CALCULATIONS
# =============================================================================

def get_tool_offset():
    """Calculate tool tip offset in world coordinates"""
    # Tool tip offset in local coordinates (from XML: site pos="0 0.01 0.0025")
    tip_local = np.array([0, 0.01, 0])

    # Body rotation: euler="0 0 45" means 45 degrees around Z
    angle_z = np.radians(45)
    cos_z, sin_z = np.cos(angle_z), np.sin(angle_z)
    # Rotation matrix for Z-axis rotation
    R = np.array([
        [cos_z, -sin_z, 0],
        [sin_z,  cos_z, 0],
        [0,      0,     1]
    ])
    return R @ tip_local


# =============================================================================
# RCM (REMOTE CENTER OF MOTION) CONTROLLER
# =============================================================================

# Tool geometry constants (from eye_tool.xml)
# RCM is inside tip_frame body: tip_frame pos="0 0.01 0" euler="0 0 60", RCM pos="-0.008 0 0"
# Transform: RCM_body = tip_pos + R_z(60°) @ (-0.008, 0, 0) = (-0.004, 0.00307, 0.0025)
_cos60 = np.cos(np.radians(60))  # 0.5
_sin60 = np.sin(np.radians(60))  # 0.866
RCM_IN_TIP_FRAME = np.array([-0.008, 0, 0])      # RCM position in tip_frame local coords
TIP_LOCAL_OFFSET = np.array([0, 0.01, 0])   # Tip position in tool body frame
# RCM in tool body frame (tip_pos + rotated RCM_in_tip_frame)
RCM_LOCAL_OFFSET = TIP_LOCAL_OFFSET + np.array([
    _cos60 * RCM_IN_TIP_FRAME[0] - _sin60 * RCM_IN_TIP_FRAME[1],
    _sin60 * RCM_IN_TIP_FRAME[0] + _cos60 * RCM_IN_TIP_FRAME[1],
    RCM_IN_TIP_FRAME[2]
])  # ≈ (-0.004, 0.00307, 0.0025)


class RCMController:
    """
    Remote Center of Motion (RCM) Controller

    Calculates tool body position and orientation such that:
    - The tool's local RCM point coincides with a fixed world RCM position
    - The tool's tip follows the desired trajectory
    - The bent portion of the tool passes through the RCM point

    Tool geometry (local frame, looking from above):
        The RCM is on the bent portion, 8mm back from tip along tip_frame X axis.

                        Origin (0,0,0)
                           |
                           | bent 60° around Z
                           |
        RCM ←----8mm----→ Tip (0, 0.01, 0.0025)

        RCM in body frame ≈ (-0.004, 0.003, 0.0025)
        RCM-to-tip distance = 8mm
    """

    def __init__(self, rcm_world_pos):
        """
        Args:
            rcm_world_pos: [x,y,z] fixed RCM position in world coordinates
        """
        self.rcm_world = np.array(rcm_world_pos)
        self.rcm_local = RCM_LOCAL_OFFSET.copy()
        self.tip_local = TIP_LOCAL_OFFSET.copy()

        # Vector from RCM to tip in local frame (this is what we align)
        self.rcm_to_tip_local = self.tip_local - self.rcm_local

        # Distance from RCM to tip (fixed by tool geometry)
        self.rcm_to_tip_distance = np.linalg.norm(self.rcm_to_tip_local)

        # Debug output
        print(f"\n  [RCM DEBUG] RCM_LOCAL_OFFSET: {self.rcm_local}")
        print(f"  [RCM DEBUG] TIP_LOCAL_OFFSET: {self.tip_local}")
        print(f"  [RCM DEBUG] rcm_to_tip_local: {self.rcm_to_tip_local}")
        print(f"  [RCM DEBUG] rcm_to_tip_distance: {self.rcm_to_tip_distance*1000:.2f} mm")

    def calculate_tool_pose(self, tip_world_pos):
        """
        Calculate tool body position and orientation given desired tip position.

        The key insight: the vector from RCM to tip in local frame must map to
        the vector from rcm_world to tip_world. This determines the rotation.
        Then body_position = rcm_world - R @ rcm_local.

        Args:
            tip_world_pos: [x,y,z] desired tip position in world coordinates

        Returns:
            dict with:
                - body_position: [x,y,z] tool body position
                - quaternion: [w,x,y,z] tool orientation quaternion
                - euler_deg: [rx,ry,rz] euler angles in degrees
                - rotation_matrix: 3x3 rotation matrix
        """
        tip_world = np.array(tip_world_pos)

        # 1. Vector from RCM to tip in world frame
        rcm_to_tip_world = tip_world - self.rcm_world
        distance = np.linalg.norm(rcm_to_tip_world)

        if distance < 1e-6:
            raise ValueError("Tip position too close to RCM")

        # Normalize both vectors for rotation calculation
        rcm_to_tip_local_norm = self.rcm_to_tip_local / self.rcm_to_tip_distance
        rcm_to_tip_world_norm = rcm_to_tip_world / distance

        # 2. Calculate rotation matrix that aligns rcm_to_tip_local with rcm_to_tip_world
        rotation_matrix = self._rotation_matrix_from_vectors(
            rcm_to_tip_local_norm,   # Local direction from RCM to tip
            rcm_to_tip_world_norm    # World direction from RCM to tip
        )

        # 3. Calculate body position
        # Tool slides through RCM - tip must be at desired position
        # tip_world = body_position + R @ tip_local
        # So: body_position = tip_world - R @ tip_local
        body_position = tip_world - rotation_matrix @ self.tip_local

        # 4. Convert rotation matrix to quaternion
        quaternion = self._rotation_matrix_to_quaternion(rotation_matrix)

        # 5. Convert to euler angles (for logging)
        euler_deg = self._rotation_matrix_to_euler(rotation_matrix)

        # 6. Calculate shaft direction in world (for reference)
        shaft_dir = rotation_matrix @ np.array([1, 0, 0])

        return {
            'body_position': body_position,
            'quaternion': quaternion,
            'euler_deg': euler_deg,
            'rotation_matrix': rotation_matrix,
            'shaft_direction': shaft_dir
        }

    def _rotation_matrix_from_vectors(self, vec_from, vec_to):
        """
        Calculate rotation matrix that rotates vec_from to vec_to.
        Uses Rodrigues' rotation formula.
        """
        vec_from = vec_from / np.linalg.norm(vec_from)
        vec_to = vec_to / np.linalg.norm(vec_to)

        # Cross product gives rotation axis
        cross = np.cross(vec_from, vec_to)
        cross_norm = np.linalg.norm(cross)

        # Dot product gives cosine of angle
        dot = np.dot(vec_from, vec_to)

        # Handle parallel vectors
        if cross_norm < 1e-6:
            if dot > 0:
                # Same direction
                return np.eye(3)
            else:
                # Opposite direction - rotate 180° around any perpendicular axis
                perp = np.array([1, 0, 0]) if abs(vec_from[0]) < 0.9 else np.array([0, 1, 0])
                perp = perp - np.dot(perp, vec_from) * vec_from
                perp = perp / np.linalg.norm(perp)
                return 2 * np.outer(perp, perp) - np.eye(3)

        # Rodrigues' formula
        axis = cross / cross_norm
        angle = np.arctan2(cross_norm, dot)

        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])

        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        return R

    def _rotation_matrix_to_quaternion(self, R):
        """Convert 3x3 rotation matrix to quaternion [w, x, y, z]"""
        trace = np.trace(R)

        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

        return np.array([w, x, y, z])

    def _rotation_matrix_to_euler(self, R):
        """Convert rotation matrix to euler angles (XYZ intrinsic) in degrees"""
        sy = R[0, 2]

        if abs(sy) < 0.99999:
            y = np.arcsin(sy)
            x = np.arctan2(-R[1, 2], R[2, 2])
            z = np.arctan2(-R[0, 1], R[0, 0])
        else:
            # Gimbal lock
            y = np.pi / 2 * np.sign(sy)
            x = np.arctan2(R[2, 1], R[1, 1])
            z = 0

        return np.degrees([x, y, z])


def calculate_rcm_from_lens(lens_geometry, rotation_angle, eye_pos,
                            rcm_radius=0.005, rcm_height=0.006):
    """
    Calculate world RCM position based on lens geometry and trajectory start angle.

    The RCM is positioned:
    - At angle (rotation_angle + 180°) around lens center (opposite to trajectory start)
    - At rcm_radius from lens center (in lens plane)
    - At rcm_height TOWARD the cornea (opposite to lens normal, which points to back of eye)

    The tool enters from the front of the eye (cornea side), so RCM is placed
    in the NEGATIVE normal direction (toward cornea, away from vitreous).

    Args:
        lens_geometry: dict from analyze_lens_geometry()
        rotation_angle: trajectory rotation angle in radians
        eye_pos: [x,y,z] eye assembly position in world
        rcm_radius: distance from lens center in lens plane (default 5mm)
        rcm_height: height toward cornea from lens surface (default 6mm)

    Returns:
        rcm_world: [x,y,z] RCM position in world coordinates
    """
    # Get lens coordinate frame
    lens_center_local = lens_geometry['center_local']
    x_axis = lens_geometry['coord_frame']['x_axis']
    y_axis = lens_geometry['coord_frame']['y_axis']
    normal = lens_geometry['coord_frame']['z_axis']

    # RCM angle is opposite to trajectory start (180° offset)
    rcm_angle = rotation_angle + np.pi

    # Calculate RCM position in local coordinates
    # Note: NEGATIVE normal direction places RCM toward cornea (front of eye)
    # The lens normal points toward back of eye, so we subtract to go toward front
    rcm_local = (lens_center_local
                 + rcm_radius * np.cos(rcm_angle) * x_axis
                 + rcm_radius * np.sin(rcm_angle) * y_axis
                 - rcm_height * normal)  # NEGATIVE: toward cornea/front

    # Transform to world coordinates
    rcm_world = rcm_local + np.array(eye_pos)

    return rcm_world


# =============================================================================
# TRAJECTORY LOADING
# =============================================================================

def load_trajectory_file(filepath):
    """Load trajectory from JSON file (uses projected trajectory)"""
    with open(filepath, 'r') as f:
        data = json.load(f)

    metadata = data['metadata']

    print(f"✓ Loaded trajectory: {filepath}")
    print(f"  Points: {metadata['num_points']}")
    print(f"  Radius: {metadata['parameters']['radius_meters']*1000:.1f} mm")

    # Use projected trajectory if available, otherwise fall back to raw
    if 'projected_world' in data and data['projected_world']:
        trajectory_world = np.array(data['projected_world'])
        print(f"  ✓ Using PROJECTED trajectory")
    else:
        trajectory_world = np.array(data['trajectory_world'])
        print(f"  ⚠ No projected trajectory found, using RAW trajectory")

    return trajectory_world, metadata


# =============================================================================
# OBJ MESH LOADING WITH UV
# =============================================================================

def load_obj_with_uv(obj_path):
    """
    Load OBJ file with vertices, faces, and UV coordinates
    Triangulates faces with more than 3 vertices

    Returns:
        dict with:
            vertices: Nx3 array of vertex positions
            faces: Mx3 array of face vertex indices (triangulated)
            uvs: Kx2 array of UV coordinates
            face_uvs: Mx3 array of UV indices per face (triangulated)
    """
    vertices = []
    uvs = []
    faces = []
    face_uvs = []

    with open(obj_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            if parts[0] == 'v':
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'vt':
                uvs.append([float(parts[1]), float(parts[2])])
            elif parts[0] == 'f':
                face_v = []
                face_uv = []
                for p in parts[1:]:
                    indices = p.split('/')
                    face_v.append(int(indices[0]) - 1)  # vertex index (0-based)
                    if len(indices) > 1 and indices[1]:
                        face_uv.append(int(indices[1]) - 1)  # UV index (0-based)

                # Triangulate if more than 3 vertices (fan triangulation)
                for i in range(1, len(face_v) - 1):
                    faces.append([face_v[0], face_v[i], face_v[i + 1]])
                    if face_uv:
                        face_uvs.append([face_uv[0], face_uv[i], face_uv[i + 1]])

    return {
        'vertices': np.array(vertices),
        'faces': np.array(faces),
        'uvs': np.array(uvs) if uvs else None,
        'face_uvs': np.array(face_uvs) if face_uvs else None
    }


# =============================================================================
# UV MAPPING FUNCTIONS
# =============================================================================

def point_to_triangle_barycentric(point, v0, v1, v2):
    """
    Compute barycentric coordinates for closest point on triangle

    Returns:
        (u, v, w): barycentric coordinates where point ≈ u*v0 + v*v1 + w*v2
        closest_point: the closest point on the triangle
    """
    # Edge vectors
    e0 = v1 - v0
    e1 = v2 - v0
    d = point - v0

    # Dot products
    d00 = np.dot(e0, e0)
    d01 = np.dot(e0, e1)
    d11 = np.dot(e1, e1)
    d20 = np.dot(d, e0)
    d21 = np.dot(d, e1)

    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-12:
        return (1.0, 0.0, 0.0), v0.copy()

    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w

    # Clamp to triangle
    if u < 0:
        # Project to edge v1-v2
        t = np.clip(np.dot(point - v1, v2 - v1) / max(np.dot(v2 - v1, v2 - v1), 1e-12), 0, 1)
        closest = v1 + t * (v2 - v1)
        return (0.0, 1.0 - t, t), closest
    elif v < 0:
        # Project to edge v0-v2
        t = np.clip(np.dot(point - v0, v2 - v0) / max(np.dot(v2 - v0, v2 - v0), 1e-12), 0, 1)
        closest = v0 + t * (v2 - v0)
        return (1.0 - t, 0.0, t), closest
    elif w < 0:
        # Project to edge v0-v1
        t = np.clip(np.dot(point - v0, v1 - v0) / max(np.dot(v1 - v0, v1 - v0), 1e-12), 0, 1)
        closest = v0 + t * (v1 - v0)
        return (1.0 - t, t, 0.0), closest

    closest = u * v0 + v * v1 + w * v2
    return (u, v, w), closest


def world_to_uv(point_world, mesh_data, eye_assembly_pos):
    """
    Map a world coordinate point to UV coordinates on the mesh

    Args:
        point_world: 3D point in world coordinates
        mesh_data: dict from load_obj_with_uv()
        eye_assembly_pos: position of eye assembly in world coords

    Returns:
        (u, v): UV coordinates (0-1 range), or None if no UV data
        distance: distance to closest point on mesh
    """
    if mesh_data['uvs'] is None or mesh_data['face_uvs'] is None:
        return None, float('inf')

    # Convert to local coordinates
    point_local = np.array(point_world) - np.array(eye_assembly_pos)

    vertices = mesh_data['vertices']
    faces = mesh_data['faces']
    uvs = mesh_data['uvs']
    face_uvs = mesh_data['face_uvs']

    best_dist = float('inf')
    best_uv = None

    for face_idx, (face, face_uv) in enumerate(zip(faces, face_uvs)):
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        uv0, uv1, uv2 = uvs[face_uv[0]], uvs[face_uv[1]], uvs[face_uv[2]]

        (bu, bv, bw), closest = point_to_triangle_barycentric(point_local, v0, v1, v2)
        dist = np.linalg.norm(point_local - closest)

        if dist < best_dist:
            best_dist = dist
            # Interpolate UV using barycentric coordinates
            best_uv = bu * uv0 + bv * uv1 + bw * uv2

    return best_uv, best_dist


# =============================================================================
# TRAJECTORY GENERATION
# =============================================================================

def generate_ellipsoid_trajectory(lens_geometry, radius_x=0.003, radius_y=0.002,
                                   num_points=100, offset_meters=0.0,
                                   center_offset_x=0.0, center_offset_y=0.0,
                                   rotation_angle=0.0,
                                   noise_std=0.0001):
    """
    Generate an ellipsoid trajectory on the lens plane with optional randomization

    Args:
        lens_geometry: Output from analyze_lens_geometry()
        radius_x: Semi-major axis in meters
        radius_y: Semi-minor axis in meters
        num_points: Number of points around ellipse
        offset_meters: Offset along normal (positive = outward)
        center_offset_x: Offset of ellipse center along x-axis
        center_offset_y: Offset of ellipse center along y-axis
        rotation_angle: Rotation of ellipse in radians
        noise_std: Standard deviation of Gaussian noise added to points

    Returns:
        Nx3 array of points in lens local coordinates
    """
    center = lens_geometry['center_local']
    coord_frame = lens_geometry['coord_frame']

    # Offset center along the normal
    center = center + offset_meters * coord_frame['z_axis']

    # Apply center offset
    center = center + center_offset_x * coord_frame['x_axis']
    center = center + center_offset_y * coord_frame['y_axis']

    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    trajectory = []

    for angle in angles:
        # Ellipse in local coordinates
        x_local = radius_x * np.cos(angle)
        y_local = radius_y * np.sin(angle)

        # Apply rotation
        x_rot = x_local * np.cos(rotation_angle) - y_local * np.sin(rotation_angle)
        y_rot = x_local * np.sin(rotation_angle) + y_local * np.cos(rotation_angle)

        # Add Gaussian noise
        if noise_std > 0:
            x_rot += np.random.normal(0, noise_std)
            y_rot += np.random.normal(0, noise_std)

        # Transform to 3D lens coordinates
        point = (center +
                 x_rot * coord_frame['x_axis'] +
                 y_rot * coord_frame['y_axis'])

        trajectory.append(point)

    return np.array(trajectory)


def generate_random_trajectory_params(base_radius_x=0.003, base_radius_y=0.002,
                                       radius_std=0.0002,
                                       center_std=0.0003,
                                       rotation_range=(0, 2*np.pi)):
    """
    Generate randomized trajectory parameters

    Args:
        base_radius_x: Base semi-major axis
        base_radius_y: Base semi-minor axis
        radius_std: Std dev for radius variation (default 0.2mm)
        center_std: Std dev for center offset (default 0.3mm)
        rotation_range: (min, max) range for rotation angle

    Returns:
        dict with randomized parameters
    """
    params = {
        'radius_x': max(0.001, np.random.normal(base_radius_x, radius_std)),
        'radius_y': max(0.001, np.random.normal(base_radius_y, radius_std)),
        'center_offset_x': np.random.normal(0, center_std),
        'center_offset_y': np.random.normal(0, center_std),
        'rotation_angle': (3/2)*np.pi,
        'noise_std': np.random.uniform(0.00002, 0.0001)  # 0.02-0.1mm noise per point
    }
    return params


# =============================================================================
# TEXTURE PAINTING
# =============================================================================

class TexturePainter:
    """Handles painting on UV texture and reloading in MuJoCo"""

    # Fill mode constants
    FILL_SOLID = 'solid'
    FILL_RANDOM = 'random'
    FILL_BUBBLE = 'bubble'

    def __init__(self, texture_path, mesh_path, eye_assembly_pos, use_temp=True,
                 fill_mode='bubble', random_std=30):
        """
        Initialize texture painter

        Args:
            texture_path: Path to the lens UV texture PNG
            mesh_path: Path to lens OBJ file with UV coordinates
            eye_assembly_pos: [x, y, z] position of eye assembly
            use_temp: If True, create a temp texture file for runtime use
            fill_mode: 'solid' for uniform color, 'random' for normal distribution noise
            random_std: Standard deviation for random fill (default 30)
        """
        self.texture_path = Path(texture_path)
        self.original_texture = Image.open(texture_path).convert('RGB')
        self.texture = self.original_texture.copy()
        self.texture_array = np.array(self.texture)
        self.width, self.height = self.texture.size

        # Fill mode settings
        self.fill_mode = fill_mode
        self.random_std = random_std

        # Create temp texture file (used during runtime)
        self.use_temp = use_temp
        if use_temp:
            self.temp_texture_path = self.texture_path.parent / 'lens_uv_map_temp.png'
            self.original_texture.save(self.temp_texture_path)
        else:
            self.temp_texture_path = self.texture_path

        self.mesh_data = load_obj_with_uv(mesh_path)
        self.eye_assembly_pos = np.array(eye_assembly_pos)

        # Cache for MuJoCo texture info
        self._tex_id = None
        self._tex_adr = None
        self._tex_size = None

        print(f"✓ TexturePainter initialized")
        print(f"  Texture: {self.texture_path} ({self.width}x{self.height})")
        print(f"  Temp texture: {self.temp_texture_path}")
        print(f"  Fill mode: {fill_mode}" + (f" (std={random_std})" if fill_mode == 'random' else ""))
        print(f"  Mesh vertices: {len(self.mesh_data['vertices'])}")
        print(f"  Mesh UVs: {len(self.mesh_data['uvs']) if self.mesh_data['uvs'] is not None else 0}")

    def reset_texture(self):
        """Reset texture to original state"""
        self.texture = self.original_texture.copy()
        self.texture_array = np.array(self.texture)

    def _fill_solid(self, x, y, color):
        """Fill pixel with solid color"""
        self.texture_array[y, x] = color

    def _fill_random(self, x, y, color):
        """Fill pixel with color + normal distribution noise"""
        noise = np.random.normal(0, self.random_std, 3)
        noisy_color = np.clip(np.array(color) + noise, 0, 255).astype(np.uint8)
        self.texture_array[y, x] = noisy_color

    def _fill_bubble(self, x, y, color):
        """
        Fill pixel with noisy white color (for bubble mode)

        Args:
            x, y: Pixel coordinates
            color: Ignored - uses white range instead
        """
        # Noise in white range (200-255)
        noisy_color = np.array([
            random.randint(200, 255),
            random.randint(200, 255),
            random.randint(200, 255)
        ], dtype=np.uint8)
        self.texture_array[y, x] = noisy_color

    def set_fill_mode(self, mode, random_std=None):
        """
        Set the fill mode for painting

        Args:
            mode: 'solid', 'random', or 'bubble'
            random_std: Standard deviation for random mode (optional)
        """
        self.fill_mode = mode
        if random_std is not None:
            self.random_std = random_std

    def get_fill_function(self):
        """Get the current fill function based on fill_mode"""
        if self.fill_mode == self.FILL_RANDOM:
            return self._fill_random
        elif self.fill_mode == self.FILL_BUBBLE:
            return self._fill_bubble
        else:
            return self._fill_solid

    def paint_at_world_pos(self, point_world, radius_pixels=3, color=(255, 255, 255), fill_func=None):
        """
        Paint on texture at the UV location corresponding to world position

        Args:
            point_world: 3D point in world coordinates
            radius_pixels: radius of painted area in pixels
            color: RGB color tuple (default black)
            fill_func: Custom fill function(x, y, color), or None to use default

        Returns:
            (u, v) UV coordinates where painted, or None if outside mesh
        """
        uv, dist = world_to_uv(point_world, self.mesh_data, self.eye_assembly_pos)

        if uv is None:
            return None

        # Convert UV to pixel coordinates
        # UV origin is typically bottom-left, image origin is top-left
        px = int(uv[0] * self.width)
        py = int((1.0 - uv[1]) * self.height)  # Flip Y

        # Get fill function
        if fill_func is None:
            fill_func = self.get_fill_function()

        # For bubble mode, add random variation to radius
        if self.fill_mode == self.FILL_BUBBLE:
            radius_pixels = radius_pixels + random.randint(0, 9)

        # Paint circle of radius k pixels
        for dy in range(-radius_pixels, radius_pixels + 1):
            for dx in range(-radius_pixels, radius_pixels + 1):
                if dx * dx + dy * dy <= radius_pixels * radius_pixels:
                    x = px + dx
                    y = py + dy
                    current_color = tuple(min(255, max(0, int(c * random.uniform(0.6, 0.8)))) for c in color) if self.fill_mode == self.FILL_RANDOM else color
                    if 0 <= x < self.width and 0 <= y < self.height:
                        fill_func(x, y, current_color)

        return uv

    def get_texture_array(self):
        """Get current texture as numpy array (RGB, uint8)"""
        return self.texture_array

    def save_texture(self, path=None):
        """Save current texture to file"""
        if path is None:
            path = self.texture_path
        Image.fromarray(self.texture_array).save(path)
        return path

    def init_mujoco_texture(self, model, texture_name='lens_texture'):
        """
        Initialize texture info cache from MuJoCo model

        Args:
            model: MuJoCo model
            texture_name: name of texture in MuJoCo model
        """
        self._tex_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, texture_name)
        if self._tex_id >= 0:
            tex_height = model.tex_height[self._tex_id]
            tex_width = model.tex_width[self._tex_id]
            self._tex_adr = model.tex_adr[self._tex_id]
            self._tex_size = tex_height * tex_width * 3
            self._tex_width = tex_width
            self._tex_height = tex_height
            print(f"  MuJoCo texture: {texture_name} (id={self._tex_id}, {tex_width}x{tex_height})")

    def update_mujoco_texture(self, model, texture_name='lens_texture', renderer=None):
        """
        Update MuJoCo texture with current painted texture

        Args:
            model: MuJoCo model
            texture_name: name of texture in MuJoCo model
            renderer: MuJoCo Renderer instance (required for texture upload)

        Returns:
            True if texture was updated, False otherwise
        """
        # Initialize cache if needed
        if self._tex_id is None:
            self.init_mujoco_texture(model, texture_name)

        if self._tex_id is None or self._tex_id < 0:
            return False

        if renderer is None:
            return False

        # Resize if needed
        if self.width != self._tex_width or self.height != self._tex_height:
            resized = Image.fromarray(self.texture_array).resize(
                (self._tex_width, self._tex_height), Image.Resampling.NEAREST
            )
            tex_data = np.array(resized, dtype=np.uint8)
        else:
            tex_data = self.texture_array.astype(np.uint8)

        # Flip vertically (MuJoCo texture origin is bottom-left)
        tex_data = np.flipud(tex_data).copy()

        # Try to update texture through renderer context
        try:
            # Access renderer's internal context
            ctx = renderer._context

            # Flatten texture data
            flat_data = tex_data.flatten().astype(np.uint8)

            # Update the texture data in model
            model.tex_data[self._tex_adr:self._tex_adr + self._tex_size] = flat_data

            # Upload texture to GPU
            mujoco.mjr_uploadTexture(model, ctx, self._tex_id)
            return True
        except AttributeError:
            # Renderer might not have _context yet
            return False
        except Exception as e:
            print(f"  Warning: Texture upload failed: {e}")
            # Fallback: save to temp file
            self.save_texture(self.temp_texture_path)
            return False


# =============================================================================
# CAPTURE WITH TRAJECTORY PAINTING
# =============================================================================

def run_with_capture(model, data, trajectory_world, output_dir,
                     tool_body_name='diathermic_tip',
                     width=640, height=480,
                     painter=None, paint_radius=3,
                     scene_path=None,
                     rcm_controller=None,
                     vary_lights=False):
    """
    Run trajectory and capture images from both cameras
    Optionally paint trajectory on lens texture

    Args:
        model: MuJoCo model
        data: MuJoCo data
        trajectory_world: Nx3 array of world coordinates (tip positions)
        output_dir: Directory to save captured data
        tool_body_name: Name of tool body in XML
        width: Image width
        height: Image height
        painter: TexturePainter instance (optional)
        paint_radius: Radius in pixels for painting (default 3)
        scene_path: Path to scene XML (required if painter is used)
        rcm_controller: RCMController instance for tool orientation (optional)
                        If provided, tool orientation follows RCM constraint.
                        If None, tool uses fixed orientation (legacy behavior).
        vary_lights: If True, vary light intensity and position per frame

    Returns:
        Path to saved JSON log file
    """
    tip_world_offset = get_tool_offset()

    # Fake time increment per frame (no actual physics simulation)
    # We use model timestep for consistent time logging, but simulation is not stepped
    fake_timestep = model.opt.timestep

    # Create output directories
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / 'top_view').mkdir(exist_ok=True)
    (output_path / 'angle_view').mkdir(exist_ok=True)
    (output_path / 'tool_view').mkdir(exist_ok=True)

    # Data log
    position_log = []

    print(f"\n✓ Starting capture: {len(trajectory_world)} frames")
    print(f"  Output: {output_path}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Time mode: FAKE (no physics simulation, timestep={fake_timestep}s)")
    if rcm_controller:
        print(f"  RCM mode: ENABLED (tool orientation follows RCM constraint)")
        print(f"    RCM position: [{rcm_controller.rcm_world[0]:.4f}, {rcm_controller.rcm_world[1]:.4f}, {rcm_controller.rcm_world[2]:.4f}]")
    else:
        print(f"  RCM mode: DISABLED (fixed orientation)")
    if painter:
        print(f"  Painting trajectory on texture (radius={paint_radius}px)")
        print(f"  Reloading model each frame for texture updates")
        # Reset to clean texture at start
        painter.reset_texture()
        # Save initial clean texture to temp path
        painter.save_texture(painter.temp_texture_path)

    # Store base light values for variation (if enabled)
    base_top_diffuse = None
    base_side_diffuse = None
    base_top_pos = None
    base_side_pos = None
    if vary_lights:
        top_light_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, 'top_light')
        side_light_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, 'side_light')
        base_top_diffuse = model.light_diffuse[top_light_id].copy() if top_light_id >= 0 else None
        base_side_diffuse = model.light_diffuse[side_light_id].copy() if side_light_id >= 0 else None
        base_top_pos = model.light_pos[top_light_id].copy() if top_light_id >= 0 else None
        base_side_pos = model.light_pos[side_light_id].copy() if side_light_id >= 0 else None
        print(f"  Light variation: ENABLED")

    # Create renderer once if not using painter (no model reload needed)
    renderer = None
    if not painter:
        renderer = mujoco.Renderer(model, height, width)
        # Cache IDs
        tool_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, tool_body_name)
        mocap_id = model.body_mocapid[tool_body_id]
        top_view_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'top_view')
        angle_view_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'angle_view')
        tool_view_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'tool_view')

    for idx, tip_pos in enumerate(trajectory_world):
        # Paint on texture if painter provided
        if painter:
            painter.paint_at_world_pos(tip_pos, radius_pixels=paint_radius)
            # Save updated texture to temp file
            painter.save_texture(painter.temp_texture_path)
            # Reload model with updated texture
            model = mujoco.MjModel.from_xml_path(scene_path)
            data = mujoco.MjData(model)
            # Get IDs (need to get each time since model is reloaded)
            tool_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, tool_body_name)
            mocap_id = model.body_mocapid[tool_body_id]
            top_view_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'top_view')
            angle_view_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'angle_view')
            tool_view_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'tool_view')
            # Create renderer for this frame (model changed)
            renderer = mujoco.Renderer(model, height, width)

        # Update tool position and orientation
        if rcm_controller:
            # Use RCM controller for realistic tool orientation
            tool_pose = rcm_controller.calculate_tool_pose(tip_pos)
            body_pos = tool_pose['body_position']
            body_quat = tool_pose['quaternion']
            euler_deg = tool_pose['euler_deg']
            data.mocap_pos[mocap_id] = body_pos
            data.mocap_quat[mocap_id] = body_quat
        else:
            # Legacy behavior: fixed orientation, only translate
            body_pos = tip_pos - tip_world_offset
            data.mocap_pos[mocap_id] = body_pos
            euler_deg = np.array([0, 0, 45])  # Default euler angles

        # Vary lighting per frame (if enabled)
        if vary_lights:
            top_light_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, 'top_light')
            side_light_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, 'side_light')
            if top_light_id >= 0 and base_top_diffuse is not None:
                # Vary diffuse intensity (0.7 to 1.0 of base)
                intensity = random.uniform(0.7, 1.0)
                model.light_diffuse[top_light_id] = base_top_diffuse * intensity
                # Vary position slightly (±0.2m)
                model.light_pos[top_light_id] = base_top_pos + np.array([
                    random.uniform(-0.2, 0.2),
                    random.uniform(-0.2, 0.2),
                    random.uniform(-0.1, 0.1)
                ])
            if side_light_id >= 0 and base_side_diffuse is not None:
                # Vary diffuse intensity (0.5 to 1.0 of base)
                intensity = random.uniform(0.5, 1.0)
                model.light_diffuse[side_light_id] = base_side_diffuse * intensity
                # Vary position slightly (±0.3m)
                model.light_pos[side_light_id] = base_side_pos + np.array([
                    random.uniform(-0.3, 0.3),
                    random.uniform(-0.3, 0.3),
                    random.uniform(-0.1, 0.1)
                ])

        # Compute forward kinematics (no physics simulation)
        mujoco.mj_forward(model, data)

        # Increment fake time (no actual simulation stepping)
        # This provides consistent timestamps for logging without computing physics/contacts
        data.time = idx * fake_timestep

        # Render all three views
        renderer.update_scene(data, camera=top_view_id)
        img_top = renderer.render()
        Image.fromarray(img_top).save(output_path / 'top_view' / f'frame_{idx:05d}.png')

        renderer.update_scene(data, camera=angle_view_id)
        img_angle = renderer.render()
        Image.fromarray(img_angle).save(output_path / 'angle_view' / f'frame_{idx:05d}.png')

        renderer.update_scene(data, camera=tool_view_id)
        img_tool = renderer.render()
        Image.fromarray(img_tool).save(output_path / 'tool_view' / f'frame_{idx:05d}.png')

        # Close renderer if it will be recreated next iteration
        if painter:
            renderer.close()

        # Tool orientation as rotation vector (euler angles in radians)
        rot_vec = np.radians(euler_deg)

        # Log position
        position_log.append({
            'frame': idx,
            'tip_position': tip_pos.tolist(),  # [x, y, z]
            'tool_trv': np.concatenate([body_pos, rot_vec]).tolist(),  # [x, y, z, rx, ry, rz]
            'time': data.time  # Fake time: frame_idx * timestep (no physics simulation)
        })

        # Progress
        if (idx + 1) % 10 == 0 or idx == len(trajectory_world) - 1:
            progress = ((idx + 1) / len(trajectory_world)) * 100
            print(f"  Progress: {progress:5.1f}% ({idx + 1}/{len(trajectory_world)})")

    # Save painted texture
    if painter:
        painted_texture_path = output_path / 'painted_texture.png'
        painter.save_texture(painted_texture_path)
        print(f"\n✓ Painted texture saved: {painted_texture_path}")

    # Save position log
    log_file = output_path / 'positions.json'
    log_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'num_frames': len(trajectory_world),
            'resolution': [width, height],
            'cameras': ['top_view', 'angle_view', 'tool_view'],
            'trajectory_painted': painter is not None,
            'time_mode': 'fake',  # No physics simulation, time is frame_idx * timestep
            'timestep': fake_timestep
        },
        'frames': position_log
    }
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)

    # Close renderer if it was kept open (no painter case)
    if renderer is not None and not painter:
        renderer.close()

    print(f"\n✓ Capture complete!")
    print(f"  Images: {output_path}/top_view/, {output_path}/angle_view/, {output_path}/tool_view/")
    print(f"  Positions: {log_file}")

    # Generate side-by-side video
    create_side_by_side_video(output_path, fps=30)

    return str(log_file)


def move_tool_along_trajectory(model, data, trajectory_world,
                               tool_body_name='diathermic_tip',
                               speed=1.0,
                               painter=None, paint_radius=3,
                               rcm_controller=None):
    """
    Move tool along trajectory in interactive viewer
    Optionally paint trajectory on lens texture

    Args:
        model: MuJoCo model
        data: MuJoCo data
        trajectory_world: Nx3 array of world coordinates (tip positions)
        tool_body_name: Name of tool body in XML
        speed: Animation speed multiplier
        painter: TexturePainter instance (optional)
        paint_radius: Radius in pixels for painting (default 3)
        rcm_controller: RCMController instance for tool orientation (optional)
    """
    # Get tool body ID
    tool_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, tool_body_name)
    mocap_id = model.body_mocapid[tool_body_id]
    tip_world_offset = get_tool_offset()

    print(f"\n✓ Tool body ID: {tool_body_id}")
    print(f"✓ Mocap ID: {mocap_id}")
    print(f"✓ Tool body name: {tool_body_name}")
    print(f"✓ Number of mocap bodies: {model.nmocap}")
    print(f"✓ Tip offset (world): {tip_world_offset}")
    print(f"✓ Starting tip position: {trajectory_world[0]}")
    if rcm_controller:
        print(f"✓ RCM mode: ENABLED")
        print(f"  RCM position: [{rcm_controller.rcm_world[0]:.4f}, {rcm_controller.rcm_world[1]:.4f}, {rcm_controller.rcm_world[2]:.4f}]")
    else:
        print(f"✓ RCM mode: DISABLED (fixed orientation)")
    if painter:
        print(f"✓ Painting enabled (radius={paint_radius}px)")
        print(f"  Note: Live texture updates not visible in viewer.")
        print(f"        Use --capture mode for visible painting, or")
        print(f"        painted texture will be saved on exit.")
    print(f"✓ Starting animation...")
    print(f"\nControls:")
    print(f"  Tab - Switch camera views")
    print(f"  Esc - Exit")

    # Animation state
    current_idx = 0
    frame_counter = 0
    frames_per_point = max(1, int(30 / speed))

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Set initial view
        viewer.cam.fixedcamid = 0  # top_view
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED

        print(f"\nAnimation settings:")
        print(f"  Speed: {speed}x")
        print(f"  Frames per point: {frames_per_point}")
        print(f"  Total points: {len(trajectory_world)}")

        while viewer.is_running():
            # Update tool position and orientation
            tip_pos = trajectory_world[current_idx]
            if rcm_controller:
                # Use RCM controller for realistic tool orientation
                tool_pose = rcm_controller.calculate_tool_pose(tip_pos)
                data.mocap_pos[mocap_id] = tool_pose['body_position']
                data.mocap_quat[mocap_id] = tool_pose['quaternion']
            else:
                # Legacy behavior: fixed orientation, only translate
                data.mocap_pos[mocap_id] = tip_pos - tip_world_offset

            # Paint on texture if painter provided (only on new points)
            if painter and frame_counter == 0:
                painter.paint_at_world_pos(tip_pos, radius_pixels=paint_radius)

            # Advance to next point based on frame counter
            frame_counter += 1
            if frame_counter >= frames_per_point:
                frame_counter = 0
                current_idx = (current_idx + 1) % len(trajectory_world)

                # Print progress every 10 points
                if current_idx % 10 == 0:
                    progress = (current_idx / len(trajectory_world)) * 100
                    pos = trajectory_world[current_idx]
                    print(f"Progress: {progress:5.1f}% | Point: {current_idx:4d}/{len(trajectory_world)} | Pos: [{pos[0]:7.4f}, {pos[1]:7.4f}, {pos[2]:7.4f}]")

                # Loop complete notification
                if current_idx == 0:
                    print("\n✓ Completed one loop! Restarting...\n")
                    if painter:
                        painter.reset_texture()  # Reset for next loop

            # Step simulation
            mujoco.mj_step(model, data)
            viewer.sync()

        print("\n✓ Animation stopped")

        # Save painted texture if painter was used
        if painter:
            painted_path = Path('../textures/lens_uv_map_painted.png')
            painter.save_texture(painted_path)
            print(f"✓ Painted texture saved: {painted_path}")


# =============================================================================
# VIDEO GENERATION
# =============================================================================

def create_side_by_side_video(capture_dir, output_filename='video.mp4', fps=30):
    """
    Create a 2x2 grid video from top_view, angle_view, and tool_view frames.
    Fourth quadrant shows black.

    Args:
        capture_dir: Path to capture directory containing top_view/, angle_view/, tool_view/
        output_filename: Name of output video file
        fps: Frames per second for output video

    Returns:
        Path to created video file, or None if failed
    """
    capture_path = Path(capture_dir)
    top_view_dir = capture_path / 'top_view'
    angle_view_dir = capture_path / 'angle_view'
    tool_view_dir = capture_path / 'tool_view'
    output_path = capture_path / output_filename

    # Check if directories exist
    if not top_view_dir.exists() or not angle_view_dir.exists():
        print(f"  Warning: Camera directories not found in {capture_dir}")
        return None

    # Check if ffmpeg is available
    if shutil.which('ffmpeg') is None:
        print("  Warning: ffmpeg not found, skipping video generation")
        return None

    # Count frames
    top_frames = sorted(top_view_dir.glob('frame_*.png'))
    angle_frames = sorted(angle_view_dir.glob('frame_*.png'))
    tool_frames = sorted(tool_view_dir.glob('frame_*.png')) if tool_view_dir.exists() else []

    if len(top_frames) == 0 or len(angle_frames) == 0:
        print("  Warning: No frames found, skipping video generation")
        return None

    print(f"\n  Generating video: {output_filename}")
    print(f"    Frames: {len(top_frames)}")
    print(f"    FPS: {fps}")
    print(f"    Layout: 2x2 grid (top_view, angle_view, tool_view, black)")

    # Get frame dimensions from first frame
    first_frame = Image.open(top_frames[0])
    frame_width, frame_height = first_frame.size

    # Build ffmpeg command for 2x2 grid
    # Layout: [top_view | angle_view]
    #         [tool_view | black    ]
    # Add white text labels in top-left corner of each frame
    font_size = max(16, frame_height // 20)  # Scale font with frame size

    if len(tool_frames) > 0:
        # 2x2 grid with tool_view and text labels
        filter_complex = (
            f"color=black:{frame_width}x{frame_height}:d={len(top_frames)/fps}[black];"
            f"[0:v]drawtext=text='top_view':x=10:y=10:fontsize={font_size}:fontcolor=white[top_labeled];"
            f"[1:v]drawtext=text='angle_view':x=10:y=10:fontsize={font_size}:fontcolor=white[angle_labeled];"
            f"[2:v]drawtext=text='tool_view':x=10:y=10:fontsize={font_size}:fontcolor=white[tool_labeled];"
            f"[top_labeled][angle_labeled]hstack=inputs=2[top];"
            f"[tool_labeled][black]hstack=inputs=2[bottom];"
            f"[top][bottom]vstack=inputs=2[out]"
        )
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', str(top_view_dir / 'frame_%05d.png'),
            '-framerate', str(fps),
            '-i', str(angle_view_dir / 'frame_%05d.png'),
            '-framerate', str(fps),
            '-i', str(tool_view_dir / 'frame_%05d.png'),
            '-filter_complex', filter_complex,
            '-map', '[out]',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '23',
            str(output_path)
        ]
    else:
        # Fallback to side-by-side if no tool_view, with text labels
        filter_complex = (
            f"[0:v]drawtext=text='top_view':x=10:y=10:fontsize={font_size}:fontcolor=white[top_labeled];"
            f"[1:v]drawtext=text='angle_view':x=10:y=10:fontsize={font_size}:fontcolor=white[angle_labeled];"
            f"[top_labeled][angle_labeled]hstack=inputs=2[out]"
        )
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', str(top_view_dir / 'frame_%05d.png'),
            '-framerate', str(fps),
            '-i', str(angle_view_dir / 'frame_%05d.png'),
            '-filter_complex', filter_complex,
            '-map', '[out]',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '23',
            str(output_path)
        ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  ✓ Video saved: {output_path}")
            return str(output_path)
        else:
            print(f"  Warning: ffmpeg failed: {result.stderr[:200]}")
            return None
    except Exception as e:
        print(f"  Warning: Video generation failed: {e}")
        return None
