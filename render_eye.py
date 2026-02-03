#!/usr/bin/env python3
import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def render_eye_scene(show_viewer=True, save_image=True):
    """Render the eye scene with camera on top"""
    
    # Load model
    print("Loading MuJoCo model...")
    model = mujoco.MjModel.from_xml_path('eye_scene.xml')
    data = mujoco.MjData(model)
    
    # Initialize simulation
    mujoco.mj_forward(model, data)
    
    if show_viewer:
        # Interactive viewer
        print("Launching interactive viewer...")
        print("Use mouse to rotate view")
        print("Press Tab to switch cameras")
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # Set initial camera to top view
            viewer.cam.fixedcamid = 0  # top_view camera
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            
            while viewer.is_running():
                mujoco.mj_step(model, data)
                viewer.sync()
    
    if save_image:
        # Render and save images from both cameras
        print("\nRendering images...")
        
        # Create renderer
        renderer = mujoco.Renderer(model, height=1080, width=1920)
        
        # Render from top camera
        renderer.update_scene(data, camera="top_view")
        top_img = renderer.render()
        
        # Save top view
        Image.fromarray(top_img).save('eye_top_view.png')
        print("Saved: eye_top_view.png")
        
        # Render from angle camera
        renderer.update_scene(data, camera="angle_view")
        angle_img = renderer.render()
        
        # Save angle view
        Image.fromarray(angle_img).save('eye_angle_view.png')
        print("Saved: eye_angle_view.png")
        
        # Display both views
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        ax1.imshow(top_img)
        ax1.set_title('Top View (Camera Above Eye)')
        ax1.axis('off')
        
        ax2.imshow(angle_img)
        ax2.set_title('Angle View')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig('eye_both_views.png', dpi=150, bbox_inches='tight')
        print("Saved: eye_both_views.png")
        plt.show()

if __name__ == "__main__":
    render_eye_scene(show_viewer=True, save_image=True)