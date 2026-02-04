#!/usr/bin/env python3
"""
Generate side-by-side video from captured frames.
Works with individual iteration directories or entire run directories.
"""

import argparse
from pathlib import Path
from utils import create_side_by_side_video


def process_directory(dir_path, fps=30):
    """
    Process a directory to generate video.
    Handles both iteration directories and run directories.
    """
    dir_path = Path(dir_path)

    # Check if this is an iteration directory (has top_view/ and angle_view/)
    if (dir_path / 'top_view').exists() and (dir_path / 'angle_view').exists():
        print(f"\nProcessing: {dir_path}")
        create_side_by_side_video(dir_path, fps=fps)
        return 1

    # Check if this is a run directory (has iter_XXXX subdirectories)
    iter_dirs = sorted(dir_path.glob('iter_*'))
    if iter_dirs:
        print(f"\nFound {len(iter_dirs)} iteration directories in {dir_path}")
        count = 0
        for iter_dir in iter_dirs:
            if (iter_dir / 'top_view').exists() and (iter_dir / 'angle_view').exists():
                print(f"\nProcessing: {iter_dir.name}")
                result = create_side_by_side_video(iter_dir, fps=fps)
                if result:
                    count += 1
        print(f"\n✓ Generated {count} videos")
        return count

    # Check for capture_* directories
    capture_dirs = sorted(dir_path.glob('capture_*'))
    if capture_dirs:
        print(f"\nFound {len(capture_dirs)} capture directories in {dir_path}")
        count = 0
        for capture_dir in capture_dirs:
            if (capture_dir / 'top_view').exists():
                print(f"\nProcessing: {capture_dir.name}")
                result = create_side_by_side_video(capture_dir, fps=fps)
                if result:
                    count += 1
        print(f"\n✓ Generated {count} videos")
        return count

    # Check for run_* directories
    run_dirs = sorted(dir_path.glob('run_*'))
    if run_dirs:
        print(f"\nFound {len(run_dirs)} run directories in {dir_path}")
        total_count = 0
        for run_dir in run_dirs:
            count = process_directory(run_dir, fps=fps)
            total_count += count
        print(f"\n✓ Total: {total_count} videos generated")
        return total_count

    print(f"No valid capture directories found in {dir_path}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Generate side-by-side videos from captured frames',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('directory', type=str,
                        help='Path to capture directory (iter_*, capture_*, run_*, or captures/)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frames per second for output video')

    args = parser.parse_args()

    dir_path = Path(args.directory)
    if not dir_path.exists():
        print(f"Error: Directory not found: {dir_path}")
        return

    print("="*60)
    print("VIDEO GENERATOR")
    print("="*60)

    process_directory(dir_path, fps=args.fps)

    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
