#!/usr/bin/env python3
import mujoco
import mujoco.viewer
import numpy as np

def test_tool():
    print("Loading model with diathermic tip...")
    model = mujoco.MjModel.from_xml_path('../scene/eye_scene.xml')
    data = mujoco.MjData(model)
    
    mujoco.mj_forward(model, data)
    
    # Get tool body ID
    tool_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'diathermic_tip')
    
    print("\n=== Tool Test ===")
    print(f"Tool body ID: {tool_body_id}")
    print(f"Initial tool position: {data.mocap_pos[tool_body_id]}")
    print("\nControls:")
    print("  Arrow keys: Move tool X/Y")
    print("  Page Up/Down: Move tool Z (up/down)")
    print("  Tab: Switch camera views")
    print("  ESC: Exit")
    
    # Animation variables
    t = 0
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Start with angle view to see both eye and tool
        viewer.cam.fixedcamid = 1
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        
        while viewer.is_running():
            # Simple circular motion demo
            t += 0.01
            radius = 0.04
            center_x = 0
            center_y = -0.05
            center_z = 0.12
            
            # Move tool in a circle
            data.mocap_pos[tool_body_id][0] = center_x + radius * np.cos(t)
            data.mocap_pos[tool_body_id][1] = center_y + radius * np.sin(t)
            data.mocap_pos[tool_body_id][2] = center_z
            
            mujoco.mj_step(model, data)
            viewer.sync()
            
            # Print position every 100 steps
            if int(t * 100) % 100 == 0:
                print(f"Tool position: x={data.mocap_pos[tool_body_id][0]:.4f}, "
                      f"y={data.mocap_pos[tool_body_id][1]:.4f}, "
                      f"z={data.mocap_pos[tool_body_id][2]:.4f}")

if __name__ == "__main__":
    test_tool()