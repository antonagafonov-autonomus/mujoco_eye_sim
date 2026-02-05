#!/usr/bin/env python3
import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse

def render_eye_scene(scene_path='../scene/eye_scene.xml', show_viewer=True, save_image=True):
    """Render the eye scene with camera on top"""

    # Load model
    print(f"Loading MuJoCo model: {scene_path}")
    model = mujoco.MjModel.from_xml_path(scene_path)
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
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            
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
    parser = argparse.ArgumentParser(description='Render eye scene with MuJoCo')
    parser.add_argument('scene_path', type=str, nargs='?', default='../scene/eye_scene.xml',
                        help='Path to the scene XML file')
    parser.add_argument('--no-viewer', action='store_true',
                        help='Disable interactive viewer')
    parser.add_argument('--no-save', action='store_true',
                        help='Disable saving images')
    args = parser.parse_args()

    render_eye_scene(scene_path=args.scene_path,
                     show_viewer=not args.no_viewer,
                     save_image=not args.no_save)