#!/usr/bin/env python3
"""
Utils: Shared utilities for MuJoCo eye simulation
- Tool trajectory functions
- UV mapping and texture painting
- Texture reloading during simulation
"""

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
    tip_local = np.array([0, 0.01, 0.0025])

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
        'rotation_angle': np.random.uniform(rotation_range[0], rotation_range[1]),
        'noise_std': np.random.uniform(0.00002, 0.0001)  # 0.02-0.1mm noise per point
    }
    return params


# =============================================================================
# TEXTURE PAINTING
# =============================================================================

class TexturePainter:
    """Handles painting on UV texture and reloading in MuJoCo"""

    def __init__(self, texture_path, mesh_path, eye_assembly_pos, use_temp=True):
        """
        Initialize texture painter

        Args:
            texture_path: Path to the lens UV texture PNG
            mesh_path: Path to lens OBJ file with UV coordinates
            eye_assembly_pos: [x, y, z] position of eye assembly
            use_temp: If True, create a temp texture file for runtime use
        """
        self.texture_path = Path(texture_path)
        self.original_texture = Image.open(texture_path).convert('RGB')
        self.texture = self.original_texture.copy()
        self.texture_array = np.array(self.texture)
        self.width, self.height = self.texture.size

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
        print(f"  Mesh vertices: {len(self.mesh_data['vertices'])}")
        print(f"  Mesh UVs: {len(self.mesh_data['uvs']) if self.mesh_data['uvs'] is not None else 0}")

    def reset_texture(self):
        """Reset texture to original state"""
        self.texture = self.original_texture.copy()
        self.texture_array = np.array(self.texture)

    def paint_at_world_pos(self, point_world, radius_pixels=3, color=(0, 0, 0)):
        """
        Paint on texture at the UV location corresponding to world position

        Args:
            point_world: 3D point in world coordinates
            radius_pixels: radius of painted area in pixels
            color: RGB color tuple (default white)

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

        # Paint circle of radius k pixels
        for dy in range(-radius_pixels, radius_pixels + 1):
            for dx in range(-radius_pixels, radius_pixels + 1):
                if dx * dx + dy * dy <= radius_pixels * radius_pixels:
                    x = px + dx
                    y = py + dy
                    if 0 <= x < self.width and 0 <= y < self.height:
                        self.texture_array[y, x] = color

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
                     scene_path=None):
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

    Returns:
        Path to saved JSON log file
    """
    tip_world_offset = get_tool_offset()

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
    if painter:
        print(f"  Painting trajectory on texture (radius={paint_radius}px)")
        print(f"  Reloading model each frame for texture updates")
        # Reset to clean texture at start
        painter.reset_texture()
        # Save initial clean texture to temp path
        painter.save_texture(painter.temp_texture_path)

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

        # Update tool position
        body_pos = tip_pos - tip_world_offset
        data.mocap_pos[mocap_id] = body_pos

        # Step simulation
        mujoco.mj_forward(model, data)

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

        # Tool orientation as rotation vector (axis * angle)
        rot_vec = np.array([0, 0, np.radians(45)])

        # Log position
        position_log.append({
            'frame': idx,
            'tip_position': tip_pos.tolist(),  # [x, y, z]
            'tool_trv': np.concatenate([body_pos, rot_vec]).tolist(),  # [x, y, z, rx, ry, rz]
            'simulation_time': data.time
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
            'trajectory_painted': painter is not None
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
                               painter=None, paint_radius=3):
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
            # Update tool position
            tip_pos = trajectory_world[current_idx]
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
    if len(tool_frames) > 0:
        # 2x2 grid with tool_view
        filter_complex = (
            f"color=black:{frame_width}x{frame_height}:d={len(top_frames)/fps}[black];"
            f"[0:v][1:v]hstack=inputs=2[top];"
            f"[2:v][black]hstack=inputs=2[bottom];"
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
        # Fallback to side-by-side if no tool_view
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', str(top_view_dir / 'frame_%05d.png'),
            '-framerate', str(fps),
            '-i', str(angle_view_dir / 'frame_%05d.png'),
            '-filter_complex', 'hstack=inputs=2',
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
