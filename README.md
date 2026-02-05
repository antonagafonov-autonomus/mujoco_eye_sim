# MuJoCo Eye Simulation

A MuJoCo-based simulation environment for visualizing and controlling surgical tool trajectories on an eye model. This project provides tools for lens geometry analysis, trajectory generation, real-time tool animation, and dataset generation with UV texture painting.

## Project Overview

This simulation models an anatomically detailed eye with support for:
- Surgical tool (diathermic tip) positioning and animation
- Randomized ellipsoid trajectory generation on the lens surface
- Mesh projection for accurate surface-following trajectories
- UV texture painting to visualize tool trajectory on the lens
- Multi-iteration dataset generation with capture
- Automatic side-by-side video generation
- Multiple camera viewpoints for visualization
- RCM (Remote Center of Motion) constraint for realistic tool orientation

### RCM Constraint

The RCM (Remote Center of Motion) feature simulates realistic surgical tool behavior where the tool shaft must pass through a fixed incision point (the RCM). This is common in minimally invasive surgery where the tool enters through a small incision.

When enabled (`--rcm`), the tool orientation is automatically calculated so that:
- The tool's local RCM point coincides with a fixed world RCM position
- The tool tip follows the desired trajectory on the lens surface
- The tool shaft always passes through the RCM point

The RCM position is calculated based on the trajectory's rotation angle, placed opposite to the trajectory start point at a configurable radius and height above the lens surface.

## Quick Start

```bash
cd scripts
./run_capture.sh
```

This will generate 10 iterations of randomized ellipsoid trajectories with image capture and UV painting.

## Scripts Reference

All scripts are located in the `scripts/` directory.

---

### run_capture.sh

**Main entry point for dataset generation.** Bash wrapper script with configurable parameters.

```bash
cd scripts
./run_capture.sh
```

**Configuration (edit the file to change):**
```bash
ITERATIONS=10        # Number of iterations
RADIUS_X=0.003       # 3mm semi-major axis
RADIUS_Y=0.002       # 2mm semi-minor axis
RADIUS_STD=0.0002    # Radius randomization std
CENTER_STD=0.0003    # Center offset std (keeps ellipse centered)
NUM_POINTS=100       # Points per trajectory
CAPTURE=true         # Enable/disable image capture
PAINT=true           # Enable/disable UV texture painting
WIDTH=640            # Capture width
HEIGHT=480           # Capture height
PAINT_RADIUS=3       # Paint brush size in pixels
RCM=true             # Enable RCM constraint for tool orientation
RCM_RADIUS=0.005     # 5mm distance from lens center to RCM
RCM_HEIGHT=0.006     # 6mm height above lens (tool reach is 8mm)
```

---

### tool_controller.py

**Main simulation controller.** Supports single trajectory mode and multi-iteration dataset generation.

```bash
# Multi-iteration mode (generates new trajectories)
python3 tool_controller.py --iterations 10 --capture --paint

# Single trajectory file mode
python3 tool_controller.py ../trajectories/trajectory.json --capture --paint

# Interactive viewer (no capture)
python3 tool_controller.py ../trajectories/trajectory.json

# Custom parameters
python3 tool_controller.py --iterations 5 --capture --paint \
    --radius-x 0.004 --radius-y 0.003 \
    --radius-std 0.0001 --center-std 0.0002 \
    --num-points 150 \
    --width 1920 --height 1080
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `trajectory_file` | - | Path to trajectory JSON (single mode) |
| `--iterations`, `-i` | 1 | Number of iterations |
| `--radius-x` | 0.003 | Base semi-major axis (m) |
| `--radius-y` | 0.002 | Base semi-minor axis (m) |
| `--radius-std` | 0.0002 | Std dev for radius randomization |
| `--center-std` | 0.0003 | Std dev for center offset |
| `--num-points`, `-n` | 100 | Points in trajectory |
| `--offset` | 0.0 | Offset along normal (m) |
| `--capture`, `-c` | flag | Enable image capture |
| `--paint`, `-p` | flag | Enable UV texture painting |
| `--paint-radius` | 3 | Paint radius in pixels |
| `--width` | 640 | Capture image width |
| `--height` | 480 | Capture image height |
| `--scene` | ../scene/eye_scene.xml | Scene file path |
| `--output`, `-o` | ../captures | Output directory |
| `--eye-pos` | 0 0 0.1 | Eye assembly position (X Y Z) |
| `--rcm` | flag | Enable RCM constraint for tool orientation |
| `--rcm-radius` | 0.005 | RCM distance from lens center (m) |
| `--rcm-height` | 0.006 | RCM height above lens surface (m) |

---

### generate_video.py

**Generate side-by-side videos from captured frames.** Videos are generated automatically during capture, but this script can regenerate them.

```bash
# Generate video for a specific iteration
python3 generate_video.py ../captures/run_20260204_163000/iter_0000

# Generate videos for all iterations in a run
python3 generate_video.py ../captures/run_20260204_163000

# Generate videos for all runs in captures folder
python3 generate_video.py ../captures

# Custom FPS
python3 generate_video.py ../captures/run_20260204_163000 --fps 60
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `directory` | - | Path to capture directory |
| `--fps` | 30 | Frames per second |

**Requirements:** ffmpeg (`sudo apt install ffmpeg`)

---

### trajectory_generator.py

**Standalone trajectory generation.** Generates circular trajectories and saves to JSON.

```bash
# Default parameters
python3 trajectory_generator.py

# Custom trajectory
python3 trajectory_generator.py \
    --radius 0.004 \
    --num-points 200 \
    --offset 0.0001

# View help
python3 trajectory_generator.py --help
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `-r`, `--radius` | 0.003 | Trajectory radius (m) |
| `-n`, `--num-points` | 600 | Number of points |
| `-o`, `--offset` | 0.0 | Offset along normal (m) |
| `--obj` | ../meshes/Lens_L_extracted.obj | Lens mesh path |
| `--output-dir` | ../trajectories | Output directory |
| `--scene-dir` | ../scene | XML output directory |
| `--eye-pos` | 0 0 0.1 | Eye position (X Y Z) |
| `--no-xml` | flag | Skip XML marker generation |

---

### reset_texture.py

**Reset lens UV texture to solid color.** Used to clear painted trajectories.

```bash
# Reset to default gray (128, 128, 128)
python3 reset_texture.py

# Reset to custom color
python3 reset_texture.py --color 100 100 100

# Reset to white
python3 reset_texture.py --color 255 255 255

# Custom texture path and size
python3 reset_texture.py --texture ../textures/lens_uv_map.png --size 1024 1024
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `-t`, `--texture` | ../textures/lens_uv_map.png | Texture file path |
| `-c`, `--color` | 128 128 128 | RGB color (0-255) |
| `-s`, `--size` | 512 512 | Image size (W H) |

---

### render_eye.py

**Render static images from scene cameras.**

```bash
# Render default scene
python3 render_eye.py ../scene/eye_scene.xml

# Render scene with trajectory markers
python3 render_eye.py ../scene/eye_scene_with_trajectory.xml
```

---

### analyze_lens.py

**Analyze lens mesh geometry.** Calculates center, normal, coordinate frame, and bounds.

```bash
python3 analyze_lens.py
```

Outputs lens parameters and can generate test trajectories with visualization markers.

---

### camera_control.py

**Interactive camera positioning tool.**

```bash
python3 camera_control.py
```

**Controls:**
| Key | Action |
|-----|--------|
| `1/2/3` | Select camera (front/angle/overview) |
| `W/A/S/D/Q/E` | Move camera |
| `I/K/J/L/U/O` | Rotate camera |
| `R` | Render images |
| `P` | Print camera pose |
| `ESC` | Exit |

---

### adjust_camera.py

**Adjust camera euler angles and FOV interactively.** Loads initial values from scene XML and prints final values for copy-paste.

```bash
# Adjust angle_view camera (default)
python3 adjust_camera.py

# Adjust top_view camera
python3 adjust_camera.py --camera top_view
python3 adjust_camera.py -c top_view

# Adjust tool_view camera (on-tool end effector camera)
python3 adjust_camera.py -c tool_view

# Override initial values
python3 adjust_camera.py --euler 90 0 70 --fovy 45
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `-c`, `--camera` | angle_view | Camera to adjust (angle_view, top_view, or tool_view) |
| `--euler` | from XML | Initial euler angles (X Y Z) |
| `--fovy` | from XML | Initial field of view |
| `--scene` | ../scene/eye_scene.xml | Scene file path |

**Controls:**
| Key | Action |
|-----|--------|
| `I/K` | Rotate X axis |
| `J/L` | Rotate Z axis |
| `U/O` | Rotate Y axis |
| `F/V` | Increase/decrease FOV |
| `+/-` | Change step size |
| `Q` | Quit and print final values |

---

### extract_lens.py

**Extract lens mesh from combined OBJ file.**

```bash
python3 extract_lens.py
```

---

### create_uv_test.py

**Generate UV test pattern textures.**

```bash
python3 create_uv_test.py
```

---

### draw_circle.py

**Draw circles on UV map for testing.**

```bash
python3 draw_circle.py
```

---

## Project Structure

```
mujoco_eye_sim/
├── scripts/                    # Python scripts and bash wrappers
│   ├── run_capture.sh          # Main entry point
│   ├── tool_controller.py      # Simulation controller
│   ├── generate_video.py       # Video generation
│   ├── utils.py                # Shared utilities
│   ├── analyze_lens.py         # Lens geometry analysis
│   ├── trajectory_generator.py # Standalone trajectory generation
│   ├── reset_texture.py        # Reset UV texture
│   ├── render_eye.py           # Static rendering
│   ├── camera_control.py       # Interactive camera positioning
│   ├── adjust_camera.py        # Camera euler/fov adjustment
│   ├── extract_lens.py         # Mesh extraction
│   ├── create_uv_test.py       # UV test patterns
│   └── draw_circle.py          # Circle drawing
│
├── scene/                      # MuJoCo XML scene definitions
│   ├── eye_scene.xml           # Main scene
│   ├── eye_scene_with_trajectory.xml
│   ├── eye_physics.xml         # Physics parameters (timestep, gravity)
│   ├── eye_axes.xml            # Coordinate axes visualization
│   ├── eye_assets.xml          # Meshes, textures, materials
│   ├── eye_body.xml            # Body hierarchy
│   ├── eye_anatomy.xml         # Eye components
│   ├── eye_cameras.xml         # Camera definitions
│   ├── eye_tool.xml            # Surgical tool + tool_view camera + local axes
│   └── lens_visualization*.xml # Trajectory markers
│
├── meshes/                     # 3D models (OBJ)
├── textures/                   # Texture images (PNG)
├── trajectories/               # Generated trajectory JSON files
├── captures/                   # Output data
│   └── run_YYYYMMDD_HHMMSS/
│       ├── config.json
│       └── iter_XXXX/
│           ├── trajectory.json
│           ├── positions.json
│           ├── painted_texture.png
│           ├── video.mp4            # 2x2 grid video
│           ├── top_view/
│           ├── angle_view/
│           └── tool_view/
├── utils/
└── parameters/
```

## Scene Architecture

```
eye_scene.xml
├── eye_assets.xml          # Meshes, textures, materials
├── eye_physics.xml         # timestep=0.5, gravity, contact
├── eye_axes.xml            # Coordinate axes (optional, can be commented out)
└── eye_body.xml
    ├── eye_anatomy.xml     # Lens, cornea, eye components
    ├── eye_cameras.xml     # top_view, angle_view (static cameras)
    └── eye_tool.xml        # diathermic_tip mocap body + tool_view camera + tool axes
```

## Output Data Format

### config.json (per run)
```json
{
  "timestamp": "20260204_163000",
  "iterations": 10,
  "base_radius_x": 0.003,
  "base_radius_y": 0.002,
  "radius_std": 0.0002,
  "center_std": 0.0003,
  "num_points": 100,
  "eye_pos": [0, 0, 0.1],
  "resolution": [640, 480],
  "paint": true,
  "paint_radius": 3,
  "cameras": {
    "top_view": {"pos": [x, y, z], "quat": [w, x, y, z], "fovy": 45},
    "angle_view": {"pos": [x, y, z], "quat": [w, x, y, z], "fovy": 45},
    "tool_view": {"pos": [x, y, z], "quat": [w, x, y, z], "fovy": 15}
  }
}
```

### trajectory.json (per iteration)
```json
{
  "metadata": {
    "timestamp": "2026-02-04T16:20:00",
    "num_points": 100,
    "parameters": {
      "radius_x": 0.003,
      "radius_y": 0.002,
      "center_offset_x": 0.0001,
      "center_offset_y": -0.0002,
      "rotation_angle": 1.57,
      "noise_std": 0.0001
    }
  },
  "trajectory_local": [[x, y, z], ...],
  "trajectory_world": [[x, y, z], ...],
  "projected_local": [[x, y, z], ...],
  "projected_world": [[x, y, z], ...]
}
```

### positions.json (per iteration)
```json
{
  "metadata": {
    "timestamp": "2026-02-04T16:20:00",
    "num_frames": 100,
    "resolution": [640, 480],
    "cameras": ["top_view", "angle_view", "tool_view"],
    "trajectory_painted": true
  },
  "frames": [
    {
      "frame": 0,
      "tip_position": [x, y, z],
      "tool_trv": [x, y, z, rx, ry, rz],
      "simulation_time": 0.5
    }
  ]
}
```

## Dependencies

- Python 3.8+
- MuJoCo 3.x
- NumPy
- Pillow (PIL)
- ffmpeg (for video generation)

```bash
# Python packages
pip install mujoco numpy pillow

# ffmpeg (Ubuntu/Debian)
sudo apt install ffmpeg
```

## Configuration Tips

### For more circular trajectories:
```bash
RADIUS_X=0.0025
RADIUS_Y=0.0025      # Equal radii = circle
RADIUS_STD=0.0001    # Less size variation
CENTER_STD=0.0001    # More centered
```

### For more variation:
```bash
RADIUS_STD=0.0005    # More size variation
CENTER_STD=0.001     # More position variation
```

## License

[Add license information]

## Contributing

[Add contribution guidelines]
