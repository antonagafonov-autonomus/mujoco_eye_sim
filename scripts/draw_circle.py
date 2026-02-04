from PIL import Image, ImageDraw

# Create gray image large enough for the circle (center 770,770 with radius 100)
size = 1024
img = Image.new('RGB', (size, size), (128, 128, 128))

draw = ImageDraw.Draw(img)

# Draw circle with center (770, 770) and radius 100
cx, cy = 770, 770
radius = 120
width = 10
draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], outline=(255, 0, 0), width=width)

img.save('../textures/lens_uv_map.png')
print(f"Saved lens_uv_map.png with circle at ({cx}, {cy}) radius {radius}")
