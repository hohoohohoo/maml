#!/usr/bin/env python3
"""
Filter Liberty (.lib) files to keep only cells that are common across all voltage variations.

For each directory in voltage_variation, this script:
1. Extracts cell names from all 61 lib files
2. Finds the intersection of cell names (cells present in ALL files)
3. Creates filtered versions of lib files containing only common cells
"""

import os
import re
from pathlib import Path
from collections import defaultdict
import argparse


def extract_cell_names(lib_file_path):
    """
    Extract all cell names from a .lib file

    Args:
        lib_file_path: Path to .lib file

    Returns:
        set: Set of cell names found in the file
    """
    cell_names = set()

    with open(lib_file_path, 'r') as f:
        for line in f:
            # Match lines like: cell(AO221x1_ASAP7_75t_L) {
            match = re.match(r'^\s*cell\(([^)]+)\)\s*\{', line)
            if match:
                cell_name = match.group(1)
                cell_names.add(cell_name)

    return cell_names


def extract_cell_block(lib_file_path, cell_name):
    """
    Extract the complete cell block for a given cell name

    Args:
        lib_file_path: Path to .lib file
        cell_name: Name of the cell to extract

    Returns:
        str: Complete cell block including cell(...) { ... }
    """
    cell_block = []
    in_target_cell = False
    brace_count = 0

    with open(lib_file_path, 'r') as f:
        for line in f:
            # Check if we're entering the target cell
            if not in_target_cell:
                match = re.match(r'^\s*cell\(([^)]+)\)\s*\{', line)
                if match and match.group(1) == cell_name:
                    in_target_cell = True
                    brace_count = 1
                    cell_block.append(line)
            else:
                cell_block.append(line)
                # Count braces to find the end of the cell block
                brace_count += line.count('{') - line.count('}')

                if brace_count == 0:
                    # End of cell block
                    break

    return ''.join(cell_block)


def get_lib_header_and_footer(lib_file_path):
    """
    Extract the header (before first cell) and footer (after last cell) from lib file

    Args:
        lib_file_path: Path to .lib file

    Returns:
        tuple: (header_lines, footer_lines)
    """
    header = []
    footer = []
    in_header = True
    past_all_cells = False

    with open(lib_file_path, 'r') as f:
        lines = f.readlines()

    brace_depth = 0

    for i, line in enumerate(lines):
        # Count library braces
        if 'library(' in line:
            brace_depth += 1

        # Check for cell definition
        if re.match(r'^\s*cell\(', line):
            in_header = False

        if in_header:
            header.append(line)
        else:
            # Track brace depth to find end of library
            brace_depth += line.count('{') - line.count('}')

            # If we're at the closing brace of the library
            if brace_depth == 0 and '}' in line:
                footer.append(line)
                past_all_cells = True
            elif past_all_cells:
                footer.append(line)

    return header, footer


def filter_lib_file(lib_file_path, common_cells, output_path):
    """
    Create a filtered version of lib file containing only common cells

    Args:
        lib_file_path: Path to input .lib file
        common_cells: Set of cell names to keep
        output_path: Path to output filtered .lib file
    """
    # Get header and footer
    header, footer = get_lib_header_and_footer(lib_file_path)

    # Extract all common cell blocks
    cell_blocks = []
    for cell_name in sorted(common_cells):  # Sort for consistent ordering
        cell_block = extract_cell_block(lib_file_path, cell_name)
        if cell_block:
            cell_blocks.append(cell_block)

    # Write filtered lib file
    with open(output_path, 'w') as f:
        # Write header
        f.writelines(header)

        # Write cell blocks
        for cell_block in cell_blocks:
            f.write(cell_block)
            f.write('\n')  # Add spacing between cells

        # Write footer
        f.writelines(footer)


def process_directory(dir_path, output_base_dir=None, dry_run=False):
    """
    Process a single directory: find common cells and filter lib files

    Args:
        dir_path: Path to directory containing .lib files
        output_base_dir: Base directory for filtered output (default: dir_path + "_filtered")
        dry_run: If True, only analyze without creating filtered files

    Returns:
        dict: Statistics about the processing
    """
    dir_path = Path(dir_path)
    dir_name = dir_path.name

    print(f"\n{'='*80}")
    print(f"Processing directory: {dir_name}")
    print(f"{'='*80}")

    # Find all .lib files
    lib_files = sorted(dir_path.glob("*.lib"))

    if len(lib_files) == 0:
        print(f"  ⚠️  No .lib files found in {dir_name}")
        return None

    print(f"  📚 Found {len(lib_files)} .lib files")

    # Extract cell names from each file
    print(f"  🔍 Extracting cell names from all files...")

    file_cell_names = {}
    for lib_file in lib_files:
        cell_names = extract_cell_names(lib_file)
        file_cell_names[lib_file.name] = cell_names
        print(f"     {lib_file.name}: {len(cell_names)} cells")

    # Find common cells (intersection)
    all_cell_sets = list(file_cell_names.values())
    common_cells = set.intersection(*all_cell_sets) if all_cell_sets else set()

    print(f"\n  📊 Cell statistics:")
    print(f"     Total files: {len(lib_files)}")
    print(f"     Common cells (in ALL files): {len(common_cells)}")

    # Find cells that are NOT in all files
    all_unique_cells = set.union(*all_cell_sets) if all_cell_sets else set()
    uncommon_cells = all_unique_cells - common_cells

    if uncommon_cells:
        print(f"     Cells NOT in all files: {len(uncommon_cells)}")
        print(f"\n  ⚠️  Cells that will be removed:")
        for cell in sorted(uncommon_cells):
            # Find which files contain this cell
            file_count = sum(1 for cells in file_cell_names.values() if cell in cells)
            print(f"       {cell} (in {file_count}/{len(lib_files)} files)")

    if dry_run:
        print(f"\n  ℹ️  DRY RUN: No files will be created")
        return {
            'dir_name': dir_name,
            'num_files': len(lib_files),
            'common_cells': len(common_cells),
            'total_unique_cells': len(all_unique_cells),
            'uncommon_cells': len(uncommon_cells)
        }

    # Create output directory
    if output_base_dir is None:
        output_dir = dir_path.parent / f"{dir_name}_filtered"
    else:
        output_dir = Path(output_base_dir) / dir_name

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  📝 Creating filtered lib files...")
    print(f"     Output directory: {output_dir}")

    # Filter each lib file
    for lib_file in lib_files:
        output_path = output_dir / lib_file.name
        filter_lib_file(lib_file, common_cells, output_path)
        print(f"     ✓ Created {lib_file.name}")

    print(f"\n  ✅ Successfully processed {dir_name}")

    return {
        'dir_name': dir_name,
        'num_files': len(lib_files),
        'common_cells': len(common_cells),
        'total_unique_cells': len(all_unique_cells),
        'uncommon_cells': len(uncommon_cells),
        'output_dir': str(output_dir)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Filter Liberty files to keep only cells common across all voltage variations"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/ASAP7_lib_files/voltage_variation",
        help="Base directory containing subdirectories with .lib files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output base directory for filtered files (default: base_dir + '_filtered')"
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        default=None,
        help="Process only this specific subdirectory (e.g., 'AO_LVT_FF')"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Analyze only without creating filtered files"
    )

    args = parser.parse_args()

    base_dir = Path(args.base_dir)

    if not base_dir.exists():
        print(f"❌ Base directory does not exist: {base_dir}")
        return

    print(f"🚀 Starting Liberty File Filtering")
    print(f"{'='*80}")
    print(f"Base directory: {base_dir}")
    if args.dry_run:
        print(f"Mode: DRY RUN (analysis only)")
    else:
        print(f"Output directory: {args.output_dir or str(base_dir) + '_filtered'}")
    print(f"{'='*80}")

    # Get list of directories to process
    if args.target_dir:
        target_path = base_dir / args.target_dir
        if not target_path.exists():
            print(f"❌ Target directory does not exist: {target_path}")
            return
        directories = [target_path]
    else:
        # Process all subdirectories
        directories = [d for d in base_dir.iterdir() if d.is_dir()]
        directories.sort()

    print(f"\n📁 Found {len(directories)} directories to process")

    # Process each directory
    results = []
    for dir_path in directories:
        result = process_directory(
            dir_path,
            output_base_dir=args.output_dir,
            dry_run=args.dry_run
        )
        if result:
            results.append(result)

    # Print summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Directories processed: {len(results)}")

    if results:
        print(f"\n{'Directory':<30} {'Files':<8} {'Common':<10} {'Total':<10} {'Removed':<10}")
        print(f"{'-'*30} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")

        for result in results:
            print(f"{result['dir_name']:<30} "
                  f"{result['num_files']:<8} "
                  f"{result['common_cells']:<10} "
                  f"{result['total_unique_cells']:<10} "
                  f"{result['uncommon_cells']:<10}")

    if not args.dry_run:
        print(f"\n✅ Filtering complete!")
    else:
        print(f"\n✅ Analysis complete! (No files created)")


if __name__ == "__main__":
    main()
