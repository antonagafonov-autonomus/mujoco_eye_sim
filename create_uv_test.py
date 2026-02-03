#!/usr/bin/env python3
from PIL import Image, ImageDraw
import numpy as np

# Create a test UV map with checker pattern
width, height = 1024, 1024
img = Image.new('RGB', (width, height), color='white')
draw = ImageDraw.Draw(img)

# Draw checker pattern
checker_size = 64
for y in range(0, height, checker_size):
    for x in range(0, width, checker_size):
        if ((x // checker_size) + (y // checker_size)) % 2 == 0:
            draw.rectangle([x, y, x + checker_size, y + checker_size], fill='black')

# Add colored corners to identify orientation
# Red corner (top-left)
draw.rectangle([0, 0, 100, 100], fill='red')
# Green corner (top-right)
draw.rectangle([width-100, 0, width, 100], fill='green')
# Blue corner (bottom-left)
draw.rectangle([0, height-100, 100, height], fill='blue')
# Yellow corner (bottom-right)
draw.rectangle([width-100, height-100, width, height], fill='yellow')

# Save
img.save('textures/lens_uv_map.png')
print("Created test UV map: textures/lens_uv_map.png")
print("- Checkerboard pattern to show UV mapping")
print("- Red (top-left), Green (top-right), Blue (bottom-left), Yellow (bottom-right)")