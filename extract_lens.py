#!/usr/bin/env python3
"""Extract Lens_L.001 object from merged OBJ file and create a standalone lens OBJ."""

import re

def extract_lens_object(input_file, output_file):
    """Extract the Lens_L.001 object and renumber indices."""

    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Find where Lens_L.001 starts
    lens_start = None
    for i, line in enumerate(lines):
        if line.strip() == 'o Lens_L.001':
            lens_start = i
            break

    if lens_start is None:
        print("Could not find 'o Lens_L.001' in the file")
        return False

    print(f"Found Lens_L.001 at line {lens_start + 1}")

    # Count v, vt, vn before the lens section to calculate offset
    v_offset = 0
    vt_offset = 0
    vn_offset = 0

    for i in range(lens_start):
        line = lines[i]
        if line.startswith('v '):
            v_offset += 1
        elif line.startswith('vt '):
            vt_offset += 1
        elif line.startswith('vn '):
            vn_offset += 1

    print(f"Offsets - v: {v_offset}, vt: {vt_offset}, vn: {vn_offset}")

    # Extract lens section
    lens_lines = []
    lens_lines.append("# Extracted Lens_L object\n")
    lens_lines.append("o Lens_L\n")

    v_count = 0
    vt_count = 0
    vn_count = 0

    for i in range(lens_start + 1, len(lines)):
        line = lines[i]

        # Stop if we hit another object
        if line.startswith('o '):
            break

        if line.startswith('v '):
            lens_lines.append(line)
            v_count += 1
        elif line.startswith('vt '):
            lens_lines.append(line)
            vt_count += 1
        elif line.startswith('vn '):
            lens_lines.append(line)
            vn_count += 1
        elif line.startswith('f '):
            # Renumber face indices
            new_line = renumber_face(line, v_offset, vt_offset, vn_offset)
            lens_lines.append(new_line)
        elif line.startswith('s ') or line.startswith('usemtl '):
            lens_lines.append(line)

    print(f"Extracted - v: {v_count}, vt: {vt_count}, vn: {vn_count}")

    with open(output_file, 'w') as f:
        f.writelines(lens_lines)

    print(f"Written to {output_file}")
    return True

def renumber_face(line, v_off, vt_off, vn_off):
    """Renumber face indices by subtracting offsets."""
    parts = line.split()
    new_parts = ['f']

    for part in parts[1:]:
        # Face indices can be: v, v/vt, v/vt/vn, or v//vn
        if '//' in part:
            # v//vn format
            indices = part.split('//')
            v_idx = int(indices[0]) - v_off
            vn_idx = int(indices[1]) - vn_off
            new_parts.append(f"{v_idx}//{vn_idx}")
        elif '/' in part:
            indices = part.split('/')
            if len(indices) == 2:
                # v/vt format
                v_idx = int(indices[0]) - v_off
                vt_idx = int(indices[1]) - vt_off
                new_parts.append(f"{v_idx}/{vt_idx}")
            elif len(indices) == 3:
                # v/vt/vn format
                v_idx = int(indices[0]) - v_off
                vt_idx = int(indices[1]) - vt_off if indices[1] else ''
                vn_idx = int(indices[2]) - vn_off
                if vt_idx:
                    new_parts.append(f"{v_idx}/{vt_idx}/{vn_idx}")
                else:
                    new_parts.append(f"{v_idx}//{vn_idx}")
        else:
            # Just vertex index
            v_idx = int(part) - v_off
            new_parts.append(str(v_idx))

    return ' '.join(new_parts) + '\n'

if __name__ == '__main__':
    extract_lens_object(
        'meshes/Lens_L.obj',
        'meshes/Lens_L_extracted.obj'
    )
