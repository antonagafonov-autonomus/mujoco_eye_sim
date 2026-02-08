#!/usr/bin/env python3
"""
Simple camera angle adjustment tool.
Rotate the angle_view camera with keyboard and print final euler angles.
"""
import mujoco
import mujoco.viewer
import numpy as np
import time
import xml.etree.ElementTree as ET
import os


def find_camera_params(scene_path, camera_name='angle_view'):
    """Parse scene XML and included files to find camera pos, euler and fovy"""
    scene_dir = os.path.dirname(os.path.abspath(scene_path))

    def search_xml(xml_path):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Search for camera in this file
            for cam in root.iter('camera'):
                if cam.get('name') == camera_name:
                    pos_str = cam.get('pos', '0 0 0')
                    pos = [float(x) for x in pos_str.split()]
                    euler_str = cam.get('euler', '0 0 0')
                    euler = [float(x) for x in euler_str.split()]
                    fovy = float(cam.get('fovy', '45'))
                    return pos, euler, fovy

            # Search in included files
            for inc in root.iter('include'):
                inc_file = inc.get('file')
                if inc_file:
                    inc_path = os.path.join(scene_dir, inc_file)
                    result = search_xml(inc_path)
                    if result:
                        return result
        except Exception as e:
            pass
        return None

    result = search_xml(scene_path)
    return result if result else ([0, 0, 0], [0, 0, 0], 45.0)

class CameraAdjuster:
    def __init__(self, model, camera_name, initial_pos, initial_euler, initial_fovy):
        self.model = model
        self.camera_name = camera_name
        self.cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)

        # Track position
        self.pos = np.array(initial_pos, dtype=float)

        # Track euler angles (for display only - actual rotation uses quaternion deltas)
        self.euler = np.array(initial_euler, dtype=float)
        self.fovy = float(initial_fovy)

        # Store the initial quaternion from MuJoCo (already correctly computed from XML)
        self.base_quat = self.model.cam_quat[self.cam_id].copy()

        self.step = 5.0  # degrees per keypress
        self.pos_step = 0.01  # meters per keypress
        self.fovy_step = 5.0  # fovy step
        self.last_key_time = {}
        self.key_delay = 0.1

        print(f"Camera: {camera_name}")
        print(f"Initial pos: {self.pos[0]:.3f} {self.pos[1]:.3f} {self.pos[2]:.3f}")
        print(f"Initial euler: {self.euler[0]:.1f} {self.euler[1]:.1f} {self.euler[2]:.1f}, fovy: {self.fovy:.1f}")

    def euler_to_quat_mujoco(self, euler_deg):
        """Convert euler angles (degrees) to quaternion using MuJoCo's XYZ intrinsic convention"""
        # MuJoCo uses intrinsic XYZ: rotate around X, then new Y, then new Z
        ax, ay, az = np.radians(euler_deg)

        # Quaternion for each axis rotation
        qx = np.array([np.cos(ax/2), np.sin(ax/2), 0, 0])
        qy = np.array([np.cos(ay/2), 0, np.sin(ay/2), 0])
        qz = np.array([np.cos(az/2), 0, 0, np.sin(az/2)])

        # Intrinsic XYZ = extrinsic ZYX, so multiply: qz * qy * qx
        q = self.quat_mul(qz, self.quat_mul(qy, qx))
        return q

    def quat_mul(self, q1, q2):
        """Multiply two quaternions [w,x,y,z]"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    def quat_to_euler_mujoco(self, quat):
        """Convert quaternion [w,x,y,z] to euler XYZ intrinsic (MuJoCo convention)"""
        w, x, y, z = quat

        # Rotation matrix from quaternion
        r00 = 1 - 2*(y*y + z*z)
        r01 = 2*(x*y - w*z)
        r02 = 2*(x*z + w*y)
        r10 = 2*(x*y + w*z)
        r11 = 1 - 2*(x*x + z*z)
        r12 = 2*(y*z - w*x)
        r20 = 2*(x*z - w*y)
        r21 = 2*(y*z + w*x)
        r22 = 1 - 2*(x*x + y*y)

        # Extract euler XYZ intrinsic from rotation matrix
        # For intrinsic XYZ: R = Rx * Ry * Rz
        sy = r02
        if abs(sy) < 0.99999:
            y_angle = np.arcsin(sy)
            x_angle = np.arctan2(-r12, r22)
            z_angle = np.arctan2(-r01, r00)
        else:
            # Gimbal lock
            y_angle = np.pi/2 * np.sign(sy)
            x_angle = np.arctan2(r21, r11)
            z_angle = 0

        return np.degrees([x_angle, y_angle, z_angle])

    def can_process(self, key):
        now = time.time()
        if key not in self.last_key_time or now - self.last_key_time[key] > self.key_delay:
            self.last_key_time[key] = now
            return True
        return False

    def rotate(self, axis, direction):
        """Rotate euler angle on axis (0=x, 1=y, 2=z)"""
        self.euler[axis] += direction * self.step

        # Apply incremental rotation to current quaternion
        angle = np.radians(direction * self.step)
        if axis == 0:  # X
            delta_quat = np.array([np.cos(angle/2), np.sin(angle/2), 0, 0])
        elif axis == 1:  # Y
            delta_quat = np.array([np.cos(angle/2), 0, np.sin(angle/2), 0])
        else:  # Z
            delta_quat = np.array([np.cos(angle/2), 0, 0, np.sin(angle/2)])

        # Apply rotation: new_quat = current_quat * delta_quat (local rotation)
        current_quat = self.model.cam_quat[self.cam_id].copy()
        new_quat = self.quat_mul(current_quat, delta_quat)
        self.model.cam_quat[self.cam_id] = new_quat

        self._print_current()

    def adjust_fovy(self, direction):
        """Adjust field of view"""
        self.fovy += direction * self.fovy_step
        self.fovy = np.clip(self.fovy, 5, 120)  # Keep fovy in reasonable range

        self.model.cam_fovy[self.cam_id] = self.fovy
        self._print_current()

    def translate(self, axis, direction):
        """Translate position on axis (0=x, 1=y, 2=z)"""
        self.pos[axis] += direction * self.pos_step
        self.model.cam_pos[self.cam_id] = self.pos
        self._print_current()

    def _print_current(self):
        # Get actual euler from current quaternion
        euler = self.quat_to_euler_mujoco(self.model.cam_quat[self.cam_id])
        pos = self.model.cam_pos[self.cam_id]
        print(f"pos=\"{pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f}\" euler=\"{euler[0]:.0f} {euler[1]:.0f} {euler[2]:.0f}\" fovy=\"{self.fovy:.0f}\"")

    def print_final(self):
        # Get actual euler from current quaternion
        euler = self.quat_to_euler_mujoco(self.model.cam_quat[self.cam_id])
        pos = self.model.cam_pos[self.cam_id]
        print("\n" + "="*50)
        print(f"FINAL CAMERA PARAMETERS ({self.camera_name})")
        print("="*50)
        print(f"pos=\"{pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f}\" euler=\"{euler[0]:.0f} {euler[1]:.0f} {euler[2]:.0f}\" fovy=\"{self.fovy:.0f}\"")
        print("="*50)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Adjust camera rotation, position and FOV')
    parser.add_argument('--camera', '-c', type=str, default='top_view',
                        choices=['angle_view', 'top_view', 'tool_view'],
                        help='Camera to adjust (default: top_view)')
    parser.add_argument('--pos', type=float, nargs=3, default=None,
                        metavar=('X', 'Y', 'Z'),
                        help='Initial position (default: read from scene XML)')
    parser.add_argument('--euler', type=float, nargs=3, default=None,
                        metavar=('X', 'Y', 'Z'),
                        help='Initial euler angles (default: read from scene XML)')
    parser.add_argument('--fovy', type=float, default=None,
                        help='Initial field of view (default: read from scene XML)')
    parser.add_argument('--scene', type=str, default='../scene/eye_scene.xml',
                        help='Scene file path')
    args = parser.parse_args()

    # Get initial params from XML if not specified
    pos_xml, euler_xml, fovy_xml = find_camera_params(args.scene, args.camera)
    if args.pos is None:
        args.pos = pos_xml
    if args.euler is None:
        args.euler = euler_xml
    if args.fovy is None:
        args.fovy = fovy_xml
    print(f"Loaded from XML ({args.camera}): pos={args.pos}, euler={args.euler}, fovy={args.fovy}")

    print("\n" + "="*50)
    print("CAMERA ADJUSTER")
    print("="*50)
    print("\nRotation Controls:")
    print("  I/K  - Rotate X axis")
    print("  J/L  - Rotate Z axis")
    print("  U/O  - Rotate Y axis")
    print("\nPosition Controls:")
    print("  A/D  - Move X axis")
    print("  W/S  - Move Y axis")
    print("  R/E  - Move Z axis")
    print("\nOther:")
    print("  F/V  - Increase/decrease FOV")
    print("  +/-  - Change step size")
    print("  Q    - Quit and print final values")
    print("="*50 + "\n")

    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    adjuster = CameraAdjuster(model, args.camera, args.pos, args.euler, args.fovy)
    running = True

    def on_key(keycode):
        nonlocal running
        key = chr(keycode) if 0 <= keycode < 128 else None
        if not key or not adjuster.can_process(key):
            return

        key = key.lower()

        if key == 'q':
            running = False
        elif key == 'i':
            adjuster.rotate(0, 1)
        elif key == 'k':
            adjuster.rotate(0, -1)
        elif key == 'j':
            adjuster.rotate(2, 1)
        elif key == 'l':
            adjuster.rotate(2, -1)
        elif key == 'u':
            adjuster.rotate(1, 1)
        elif key == 'o':
            adjuster.rotate(1, -1)
        elif key == 'a':
            adjuster.translate(0, -1)  # X-
        elif key == 'd':
            adjuster.translate(0, 1)   # X+
        elif key == 'w':
            adjuster.translate(1, 1)   # Y+
        elif key == 's':
            adjuster.translate(1, -1)  # Y-
        elif key == 'r':
            adjuster.translate(2, 1)   # Z+
        elif key == 'e':
            adjuster.translate(2, -1)  # Z-
        elif key == 'f':
            adjuster.adjust_fovy(1)
        elif key == 'v':
            adjuster.adjust_fovy(-1)
        elif key == '=' or key == '+':
            adjuster.step = min(45, adjuster.step * 1.5)
            adjuster.pos_step = min(0.1, adjuster.pos_step * 1.5)
            adjuster.fovy_step = min(20, adjuster.fovy_step * 1.5)
            print(f"Rotation step: {adjuster.step:.1f} deg, pos step: {adjuster.pos_step:.4f}m, fovy step: {adjuster.fovy_step:.1f}")
        elif key == '-':
            adjuster.step = max(1, adjuster.step / 1.5)
            adjuster.pos_step = max(0.001, adjuster.pos_step / 1.5)
            adjuster.fovy_step = max(1, adjuster.fovy_step / 1.5)
            print(f"Rotation step: {adjuster.step:.1f} deg, pos step: {adjuster.pos_step:.4f}m, fovy step: {adjuster.fovy_step:.1f}")

    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        # Set to angle_view camera
        viewer.cam.fixedcamid = adjuster.cam_id
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED

        while viewer.is_running() and running:
            mujoco.mj_step(model, data)
            viewer.sync()

    adjuster.print_final()


if __name__ == "__main__":
    main()
