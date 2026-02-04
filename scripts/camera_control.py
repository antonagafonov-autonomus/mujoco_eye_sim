#!/usr/bin/env python3
import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time

class CameraController:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.active_camera = 'front'  # 'front' or 'angle'
        self.step_pos = 0.01  # position step
        self.step_rot = 5.0   # rotation step in degrees
        
        # Get mocap body IDs
        self.mocap_ids = {
            'front': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'camera_rig_front'),
            'angle': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'camera_rig_angle')
        }
        
        # Key state tracking
        self.last_key_time = {}
        self.key_repeat_delay = 0.1  # seconds
        
    def get_mocap_pos_quat(self, cam_name):
        """Get position and quaternion of mocap body"""
        body_id = self.mocap_ids[cam_name]
        pos = self.data.mocap_pos[body_id].copy()
        quat = self.data.mocap_quat[body_id].copy()
        return pos, quat
    
    def set_mocap_pos_quat(self, cam_name, pos, quat):
        """Set position and quaternion of mocap body"""
        body_id = self.mocap_ids[cam_name]
        self.data.mocap_pos[body_id] = pos
        self.data.mocap_quat[body_id] = quat
    
    def can_process_key(self, key):
        """Check if enough time has passed to process key again"""
        current_time = time.time()
        if key not in self.last_key_time:
            self.last_key_time[key] = current_time
            return True
        if current_time - self.last_key_time[key] > self.key_repeat_delay:
            self.last_key_time[key] = current_time
            return True
        return False
    
    def move_camera(self, axis, direction):
        """Move camera along specified axis"""
        pos, quat = self.get_mocap_pos_quat(self.active_camera)
        
        if axis == 'x':
            pos[0] += direction * self.step_pos
        elif axis == 'y':
            pos[1] += direction * self.step_pos
        elif axis == 'z':
            pos[2] += direction * self.step_pos
            
        self.set_mocap_pos_quat(self.active_camera, pos, quat)
        
    def rotate_camera(self, axis, direction):
        """Rotate camera around specified axis"""
        pos, quat = self.get_mocap_pos_quat(self.active_camera)
        
        # Create rotation quaternion
        angle = direction * np.radians(self.step_rot)
        
        if axis == 'rx':  # Pitch (around X)
            rot_quat = np.array([np.cos(angle/2), np.sin(angle/2), 0, 0])
        elif axis == 'ry':  # Roll (around Y)
            rot_quat = np.array([np.cos(angle/2), 0, np.sin(angle/2), 0])
        elif axis == 'rz':  # Yaw (around Z)
            rot_quat = np.array([np.cos(angle/2), 0, 0, np.sin(angle/2)])
        
        # Multiply quaternions
        new_quat = self.quat_multiply(quat, rot_quat)
        
        self.set_mocap_pos_quat(self.active_camera, pos, new_quat)
    
    def quat_multiply(self, q1, q2):
        """Multiply two quaternions"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    def quat_to_euler(self, quat):
        """Convert quaternion to euler angles (in degrees)"""
        w, x, y, z = quat
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(np.clip(sinp, -1, 1))
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.degrees([roll, pitch, yaw])
    
    def print_camera_pose(self):
        """Print current camera position and orientation"""
        pos, quat = self.get_mocap_pos_quat(self.active_camera)
        euler = self.quat_to_euler(quat)
        
        cam_display = "FRONT (Yellow)" if self.active_camera == 'front' else "ANGLE (Cyan)"
        print(f"\n=== Camera {cam_display} Pose ===")
        print(f"Position: x={pos[0]:.4f}, y={pos[1]:.4f}, z={pos[2]:.4f}")
        print(f"Rotation: roll={euler[0]:.2f}°, pitch={euler[1]:.2f}°, yaw={euler[2]:.2f}°")

def print_controls():
    print("\n" + "="*70)
    print("CAMERA CONTROL - EYE ABOVE FLOOR (STATIC CAMERAS)")
    print("="*70)
    print("\nScene Setup:")
    print("  - Yellow sphere at eye center (0, 0, 0.5)")
    print("  - FRONT camera (yellow) looks UP at eye from front/below")
    print("  - ANGLE camera (cyan) looks at eye from side at an angle")
    print("  - Red/Green/Blue axes show X/Y/Z at floor and eye")
    print("  - Cameras are now STATIC (mocap) - won't drift!")
    print("\nCamera Selection:")
    print("  1          - Select FRONT camera (yellow marker)")
    print("  2          - Select ANGLE camera (cyan marker)")
    print("  3          - Switch to overview camera")
    print("\nPosition Control (selected camera):")
    print("  W/S        - Move forward/backward (Y axis)")
    print("  A/D        - Move left/right (X axis)")
    print("  Q/E        - Move up/down (Z axis)")
    print("\nRotation Control (selected camera):")
    print("  I/K        - Pitch up/down (rotate around X)")
    print("  J/L        - Yaw left/right (rotate around Z)")
    print("  U/O        - Roll (rotate around Y)")
    print("\nOther:")
    print("  P          - Print current camera pose")
    print("  R          - Render and save images from both cameras")
    print("  H          - Show this help")
    print("  +/-        - Increase/decrease movement speed")
    print("  Esc        - Exit")
    print("="*70 + "\n")

def render_from_cameras(model, data):
    """Render and save images from both cameras"""
    print("\nRendering images...")
    renderer = mujoco.Renderer(model, height=1080, width=1920)
    
    # Render from front camera
    renderer.update_scene(data, camera="front_view")
    img_front = renderer.render()
    Image.fromarray(img_front).save('eye_front_camera.png')
    print("✓ Saved: eye_front_camera.png")
    
    # Render from angle camera
    renderer.update_scene(data, camera="angle_view")
    img_angle = renderer.render()
    Image.fromarray(img_angle).save('eye_angle_camera.png')
    print("✓ Saved: eye_angle_camera.png")
    
    # Render from overview
    renderer.update_scene(data, camera="overview")
    img_overview = renderer.render()
    Image.fromarray(img_overview).save('eye_overview.png')
    print("✓ Saved: eye_overview.png")
    
    # Display all three views
    fig = plt.figure(figsize=(20, 6))
    
    ax1 = plt.subplot(131)
    ax1.imshow(img_front)
    ax1.set_title('Front Camera View\n(Looking up at eye)', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    ax2 = plt.subplot(132)
    ax2.imshow(img_angle)
    ax2.set_title('Angle Camera View\n(Side perspective)', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    ax3 = plt.subplot(133)
    ax3.imshow(img_overview)
    ax3.set_title('Overview\n(Scene layout)', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    plt.tight_layout()
    plt.savefig('eye_all_cameras.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: eye_all_cameras.png")
    plt.close()
    print("Rendering complete!\n")

def key_callback(keycode, controller, viewer, model, data):
    """Handle keyboard input"""
    key = chr(keycode) if 0 <= keycode < 128 else None
    
    if not key or not controller.can_process_key(key):
        return
    
    # Camera selection
    if key == '1':
        controller.active_camera = 'front'
        print("✓ Selected FRONT camera (yellow marker)")
    elif key == '2':
        controller.active_camera = 'angle'
        print("✓ Selected ANGLE camera (cyan marker)")
    elif key == '3':
        viewer.cam.fixedcamid = 2  # overview
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        print("✓ Switched to OVERVIEW camera")
    
    # Position control
    elif key == 'w' or key == 'W':
        controller.move_camera('y', 1)
    elif key == 's' or key == 'S':
        controller.move_camera('y', -1)
    elif key == 'a' or key == 'A':
        controller.move_camera('x', -1)
    elif key == 'd' or key == 'D':
        controller.move_camera('x', 1)
    elif key == 'q' or key == 'Q':
        controller.move_camera('z', 1)
    elif key == 'e' or key == 'E':
        controller.move_camera('z', -1)
    
    # Rotation control
    elif key == 'i' or key == 'I':
        controller.rotate_camera('rx', 1)
    elif key == 'k' or key == 'K':
        controller.rotate_camera('rx', -1)
    elif key == 'j' or key == 'J':
        controller.rotate_camera('rz', 1)
    elif key == 'l' or key == 'L':
        controller.rotate_camera('rz', -1)
    elif key == 'u' or key == 'U':
        controller.rotate_camera('ry', 1)
    elif key == 'o' or key == 'O':
        controller.rotate_camera('ry', -1)
    
    # Speed control
    elif key == '=' or key == '+':
        controller.step_pos *= 1.5
        controller.step_rot *= 1.5
        print(f"Speed increased: pos={controller.step_pos:.4f}, rot={controller.step_rot:.1f}°")
    elif key == '-' or key == '_':
        controller.step_pos /= 1.5
        controller.step_rot /= 1.5
        print(f"Speed decreased: pos={controller.step_pos:.4f}, rot={controller.step_rot:.1f}°")
    
    # Other commands
    elif key == 'p' or key == 'P':
        controller.print_camera_pose()
    elif key == 'r' or key == 'R':
        render_from_cameras(model, data)
    elif key == 'h' or key == 'H':
        print_controls()

def main():
    # Load model
    print("Loading MuJoCo model...")
    model = mujoco.MjModel.from_xml_path('../scene/eye_scene.xml')
    data = mujoco.MjData(model)
    
    # Initialize
    mujoco.mj_forward(model, data)
    
    # Create controller
    controller = CameraController(model, data)
    
    print_controls()
    print("✓ Cameras are now STATIC using mocap - they won't move on their own!")
    print("✓ Starting interactive viewer...\n")
    
    # Launch viewer with keyboard callback
    with mujoco.viewer.launch_passive(model, data, key_callback=lambda keycode: key_callback(keycode, controller, None, model, data)) as viewer:
        # Update key callback to include viewer reference
        viewer._key_callback = lambda keycode: key_callback(keycode, controller, viewer, model, data)
        
        # Start with overview camera
        viewer.cam.fixedcamid = 2  # overview camera
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        
        while viewer.is_running():
            # Step simulation
            mujoco.mj_step(model, data)
            viewer.sync()

if __name__ == "__main__":
    main()