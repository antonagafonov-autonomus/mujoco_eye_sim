#!/usr/bin/env python3
"""
Reset lens UV texture to a solid color
"""

import argparse
from PIL import Image
from pathlib import Path


def reset_texture(texture_path, color=(128, 128, 128), size=(512, 512)):
    """
    Reset texture to a solid color

    Args:
        texture_path: Path to texture file
        color: RGB tuple (default gray 128,128,128)
        size: Image size (width, height)
    """
    img = Image.new('RGB', size, color)
    img.save(texture_path)
    print(f"✓ Texture reset to RGB{color}: {texture_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Reset lens UV texture to solid color',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--texture', '-t', type=str,
                        default='../textures/lens_uv_map.png',
                        help='Path to texture file')
    parser.add_argument('--color', '-c', type=int, nargs=3,
                        default=[128, 128, 128],
                        metavar=('R', 'G', 'B'),
                        help='RGB color values (0-255)')
    parser.add_argument('--size', '-s', type=int, nargs=2,
                        default=[512, 512],
                        metavar=('W', 'H'),
                        help='Image size in pixels')

    args = parser.parse_args()

    reset_texture(
        texture_path=args.texture,
        color=tuple(args.color),
        size=tuple(args.size)
    )


if __name__ == "__main__":
    main()
