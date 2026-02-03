#!/usr/bin/env python3
"""Interactive UV map viewer - shows pixel coordinates on hover/click."""

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def main():
    img = Image.open('textures/lens_uv_map_example.png')
    img_array = np.array(img)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img_array, origin='upper')
    ax.set_title('Lens UV Map - Move mouse to see coordinates')
    ax.set_xlabel('U (pixels)')
    ax.set_ylabel('V (pixels)')

    # Add coordinate display
    coord_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                         fontsize=12, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    def on_move(event):
        if event.inaxes == ax:
            x, y = int(event.xdata), int(event.ydata)
            if 0 <= x < img_array.shape[1] and 0 <= y < img_array.shape[0]:
                # UV coordinates (normalized 0-1)
                u = x / img_array.shape[1]
                v = y / img_array.shape[0]
                pixel = img_array[y, x]
                if len(pixel) >= 3:
                    rgb = f'RGB({pixel[0]}, {pixel[1]}, {pixel[2]})'
                else:
                    rgb = f'Gray({pixel})'
                coord_text.set_text(f'Pixel: ({x}, {y})\nUV: ({u:.3f}, {v:.3f})\n{rgb}')
                fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes == ax:
            x, y = int(event.xdata), int(event.ydata)
            u = x / img_array.shape[1]
            v = y / img_array.shape[0]
            print(f'Clicked: pixel=({x}, {y}), UV=({u:.4f}, {v:.4f})')

    fig.canvas.mpl_connect('motion_notify_event', on_move)
    fig.canvas.mpl_connect('button_press_event', on_click)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
