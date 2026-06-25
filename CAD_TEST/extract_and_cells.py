#!/usr/bin/env python3
"""
Extract AND cells (AN2*, AN3*, AN4*) from .tlib files
"""

import os
import re
from pathlib import Path


def extract_and_cells(input_file, output_file):
    """
    Extract AND cells from a .tlib file.
    AND cells start with AN2, AN3, or AN4.
    """
    with open(input_file, 'r') as f:
        content = f.read()

    # Extract header (everything before first cell)
    header_match = re.search(r'^(.*?)(\s*cell\s*\()', content, re.DOTALL)
    if not header_match:
        print(f"  No cells found in {input_file}")
        return 0

    header = header_match.group(1)

    # Find all cell blocks
    # Pattern: cell (NAME) { ... } - need to match balanced braces
    cell_pattern = r'(\s*cell\s*\(([^)]+)\)\s*\{)'

    cells = []
    pos = 0
    and_count = 0

    for match in re.finditer(cell_pattern, content):
        cell_name = match.group(2)
        cell_start = match.start()

        # Check if it's an AND cell (AN2*, AN3*, AN4*)
        if re.match(r'^AN[234]', cell_name):
            # Find the matching closing brace
            brace_count = 0
            in_cell = False
            cell_end = cell_start

            for i in range(match.end() - 1, len(content)):
                if content[i] == '{':
                    brace_count += 1
                    in_cell = True
                elif content[i] == '}':
                    brace_count -= 1
                    if in_cell and brace_count == 0:
                        cell_end = i + 1
                        break

            cell_content = content[cell_start:cell_end]
            cells.append(cell_content)
            and_count += 1

    if and_count == 0:
        print(f"  No AND cells found in {input_file}")
        return 0

    # Write output file
    with open(output_file, 'w') as f:
        f.write(header)
        for cell in cells:
            f.write(cell)
            f.write('\n')
        f.write('}\n')  # Close library block

    print(f"  Extracted {and_count} AND cells -> {output_file}")
    return and_count


def process_directory(input_dir, output_dir):
    """Process all .tlib files in a directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tlib_files = sorted(input_path.glob('*.tlib'))

    total_cells = 0
    for tlib_file in tlib_files:
        output_file = output_path / f"AND_{tlib_file.name}"
        count = extract_and_cells(str(tlib_file), str(output_file))
        total_cells += count

    return total_cells


def main():
    base_dirs = [
        '/home/tkdgn2907/Deepsets_test/MAML/Projects/CAD_TEST/20200414_base_nom_0p8v/base_nom_0p8v',
        '/home/tkdgn2907/Deepsets_test/MAML/Projects/CAD_TEST/20200414_base_nom_0p9v/base_nom_0p9v',
        '/home/tkdgn2907/Deepsets_test/MAML/Projects/CAD_TEST/20200414_base_nom_1p0v/base_nom_1p0v',
    ]

    output_base = '/home/tkdgn2907/Deepsets_test/MAML/Projects/CAD_TEST/AND_cells_extracted'

    for input_dir in base_dirs:
        # Extract voltage from path
        if '0p8v' in input_dir:
            voltage = '0p8v'
        elif '0p9v' in input_dir:
            voltage = '0p9v'
        elif '1p0v' in input_dir:
            voltage = '1p0v'
        else:
            voltage = 'unknown'

        output_dir = os.path.join(output_base, voltage)

        print(f"\nProcessing {voltage}:")
        print(f"  Input: {input_dir}")
        print(f"  Output: {output_dir}")

        if not os.path.exists(input_dir):
            print(f"  ERROR: Input directory not found!")
            continue

        total = process_directory(input_dir, output_dir)
        print(f"  Total AND cells extracted: {total}")


if __name__ == '__main__':
    main()
