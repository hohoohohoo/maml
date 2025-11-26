#!/usr/bin/env python
# coding: utf-8

"""
Voltage Variation Data Preprocessing Wrapper

Wrapper script for voltage variation data preprocessing (TSMC and ASAP7).
Supports both interactive and command-line modes.
"""

import os
import sys
import subprocess
import argparse
import re
from pathlib import Path

# Dataset type configurations
DATASET_CONFIGS = {
    'tsmc': {
        'name': 'TSMC Voltage Variation',
        'data_dir': '/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/TSMC_lib_files',
        'folder_pattern': r'TSMC_([A-Z]+)_(\d+)',
        'start': 60,
        'end': 121,
        'output_base_dir': '/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/temp_dataset_TSMC_dim5'
    },
    'asap7': {
        'name': 'ASAP7 Voltage Variation',
        'data_dir': '/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/ASAP7_lib_files/voltage_variation',
        'folder_pattern': r'([A-Z]+)_([A-Z]+)_(FF|TT|SS)',  # e.g., AO_LVT_FF
        'start': 40,
        'end': 101,
        'output_base_dir': '/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/temp_dataset_ASAP7_dim5',
        'corner_mapping': {'FF': 1, 'TT': 2, 'SS': 3}
    }
}

def print_banner():
    """Print welcome banner"""
    print("\n" + "="*80)
    print(" "*20 + "Voltage Variation Data Preprocessing Wrapper")
    print("="*80 + "\n")

def select_dataset_type():
    """Let user select dataset type"""
    print("Available dataset types:")
    print("-" * 80)
    print("  [0] TSMC Voltage Variation (Default)")
    print("      • Path: TSMC_lib_files/TSMC_<CORNER>_<TEMP>")
    print("      • Voltage range: 60-120")
    print("      • Output: dataset_TSMC_dim5/processed/cell_TSMC_<CORNER>_<TEMP>_dataset_*.pth")
    print()
    print("  [1] ASAP7 Voltage Variation")
    print("      • Path: ASAP7_lib_files/voltage_variation/<CELL>_<VT>_<CORNER>")
    print("      • Voltage range: 40-101")
    print("      • Output: dataset_ASAP7_dim5/dataset_test5(dim5)_<CORNER>/processed/")
    print("               cell_<CELL>_<VT>_<NUM>_25_dataset_*.pth")
    print("      • Corner mapping: FF->1, TT->2, SS->3")
    print()

    while True:
        choice = input("Select dataset type [0/1] (default: 0): ").strip()
        if choice == '' or choice == '0':
            return 'tsmc'
        elif choice == '1':
            return 'asap7'
        print("Invalid choice. Please select 0 or 1.\n")

def select_delay_type():
    """Let user select delay type"""
    print("\nAvailable delay types:")
    print("-" * 80)
    print("  [0] Cell Delay (Default)")
    print("      • Uses libdata_extract_MAML_cell module")
    print("      • Cell-level timing analysis")
    print()
    print("  [1] Transition Delay")
    print("      • Uses libdata_extract_MAML_transition module")
    print("      • Transition-level timing analysis")
    print()

    while True:
        choice = input("Select delay type [0/1] (default: 0): ").strip()
        if choice == '' or choice == '0':
            return 'cell'
        elif choice == '1':
            return 'transition'
        print("Invalid choice. Please select 0 or 1.\n")

def select_asap7_vt_and_corner():
    """Let user select VT type and Corner for ASAP7"""
    vt_types = ['LVT', 'RVT', 'SLVT', 'SRAM']
    corners = ['FF', 'TT', 'SS']

    print("\nSelect VT Type:")
    print("-" * 80)
    for i, vt in enumerate(vt_types):
        print(f"  [{i}] {vt}")
    print()

    while True:
        choice = input(f"Select VT type [0-{len(vt_types)-1}] (default: 0): ").strip()
        if choice == '':
            choice = '0'
        try:
            idx = int(choice)
            if 0 <= idx < len(vt_types):
                vt_type = vt_types[idx]
                break
        except ValueError:
            pass
        print("Invalid choice. Please enter a valid number.\n")

    print(f"\nSelected VT Type: {vt_type}")
    print("\nSelect Corner:")
    print("-" * 80)
    for i, corner in enumerate(corners):
        print(f"  [{i}] {corner}")
    print()

    while True:
        choice = input(f"Select corner [0-{len(corners)-1}] (default: 0): ").strip()
        if choice == '':
            choice = '0'
        try:
            idx = int(choice)
            if 0 <= idx < len(corners):
                corner = corners[idx]
                break
        except ValueError:
            pass
        print("Invalid choice. Please enter a valid number.\n")

    return vt_type, corner

def find_asap7_folders_by_vt_corner(vt_type, corner):
    """Find all ASAP7 folders matching the given VT type and corner"""
    config = DATASET_CONFIGS['asap7']
    data_path = Path(config['data_dir'])

    if not data_path.exists():
        return []

    # Pattern: <CELL>_<VT>_<CORNER>
    pattern = re.compile(config['folder_pattern'])
    matching_folders = []

    for folder in data_path.iterdir():
        if folder.is_dir():
            match = pattern.match(folder.name)
            if match:
                folder_vt = match.group(2)
                folder_corner = match.group(3)
                if folder_vt == vt_type and folder_corner == corner:
                    matching_folders.append(folder)

    return sorted(matching_folders, key=lambda x: x.name)

def count_matching_folders(dataset_type):
    """Count folders matching the dataset type pattern"""
    config = DATASET_CONFIGS[dataset_type]
    data_path = Path(config['data_dir'])

    if not data_path.exists():
        return 0, []

    # Both TSMC and ASAP7 now use folder patterns
    pattern = re.compile(config['folder_pattern'])
    matching_folders = []
    for folder in data_path.iterdir():
        if folder.is_dir():
            match = pattern.match(folder.name)
            if match:
                matching_folders.append(folder)
    return len(matching_folders), matching_folders

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Voltage Variation Data Preprocessing Wrapper',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (recommended)
  python run_voltage_variation_preprocessing.py

  # TSMC - single folder
  python run_voltage_variation_preprocessing.py -t tsmc --folder TSMC_FF_0
  python run_voltage_variation_preprocessing.py -t tsmc --folder TSMC_TT_25 --delay-type transition --yes

  # ASAP7 - single folder
  python run_voltage_variation_preprocessing.py -t asap7 --folder AO_LVT_FF

  # ASAP7 - all folders matching VT and Corner (NEW!)
  python run_voltage_variation_preprocessing.py -t asap7 --vt-type LVT --corner FF
  python run_voltage_variation_preprocessing.py -t asap7 --vt-type RVT --corner TT --delay-type transition --yes
  # This will process all matching folders: AO_LVT_FF, OA_LVT_FF, simple_LVT_FF, INVBUF_LVT_FF, etc.

Dataset Types:
  tsmc   - TSMC voltage variation dataset
           Voltage range: 60-120
           Folder pattern: TSMC_<CORNER>_<TEMP> (e.g., TSMC_FF_0, TSMC_TT_25)
           Output: dataset_TSMC_dim5/processed/cell_TSMC_<CORNER>_<TEMP>_dataset_input.pth

  asap7  - ASAP7 voltage variation dataset
           Voltage range: 40-101
           Folder pattern: <CELL>_<VT>_<CORNER> (e.g., AO_LVT_FF, OA_RVT_TT)
           Output: dataset_ASAP7_dim5/dataset_test5(dim5)_<CORNER>/processed/
                   cell_<CELL>_<VT>_<NUM>_25_dataset_input.pth
           Corner mapping: FF->1, TT->2, SS->3

           ASAP7 Modes:
           1. Single folder: --folder AO_LVT_FF
           2. All folders matching VT/Corner: --vt-type LVT --corner FF
""")

    parser.add_argument('-t', '--dataset-type', type=str,
                       choices=['tsmc', 'asap7'],
                       help='Dataset type: tsmc or asap7')
    parser.add_argument('--delay-type', type=str,
                       choices=['cell', 'transition'],
                       default='cell',
                       help='Delay type: cell or transition (default: cell)')
    parser.add_argument('--folder', type=str,
                       help='Folder to process. Examples: TSMC_FF_0, AO_LVT_FF')
    parser.add_argument('--vt-type', type=str,
                       choices=['LVT', 'RVT', 'SLVT', 'SRAM'],
                       help='ASAP7 only: VT type to process all matching folders')
    parser.add_argument('--corner', type=str,
                       choices=['FF', 'TT', 'SS'],
                       help='ASAP7 only: Corner to process all matching folders')
    parser.add_argument('--yes', '-y', action='store_true',
                       help='Skip confirmation prompt')

    return parser.parse_args()

def build_command(dataset_type, folder_name=None, delay_type='cell'):
    """Build the command to execute"""
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'build_and_split_dataset_test5dim.py')

    config = DATASET_CONFIGS[dataset_type]

    # Determine data directory, prefix, and output directory
    if dataset_type == 'tsmc':
        if folder_name:
            # Specific TSMC folder (e.g., TSMC_FF_0)
            data_dir = os.path.join(config['data_dir'], folder_name)
            pattern = re.compile(config['folder_pattern'])
            match = pattern.match(folder_name)
            if match:
                corner = match.group(1)
                temperature = match.group(2)
                prefix = f"TSMC_{corner}_{temperature}_"
                # Output to processed directory
                output_dir = os.path.join(config['output_base_dir'], 'processed')
            else:
                raise ValueError(f"Folder name '{folder_name}' doesn't match TSMC pattern")
        else:
            # Should not reach here - TSMC requires specific folder
            raise ValueError("TSMC requires specific folder selection")
    else:
        # ASAP7 voltage variation (e.g., AO_LVT_FF)
        if folder_name:
            data_dir = os.path.join(config['data_dir'], folder_name)
            pattern = re.compile(config['folder_pattern'])
            match = pattern.match(folder_name)
            if match:
                cell_type = match.group(1)  # e.g., AO
                vt_type = match.group(2)     # e.g., LVT
                corner_name = match.group(3) # e.g., FF
                corner_num = config['corner_mapping'][corner_name]  # FF->1, TT->2, SS->3
                prefix = f"{cell_type}_{vt_type}_{corner_num}_25_"
                # Output to dataset_test5(dim5)_{CORNER}/processed
                output_dir = os.path.join(config['output_base_dir'],
                                         f"dataset_test5(dim5)_{corner_name}",
                                         'processed')
            else:
                raise ValueError(f"Folder name '{folder_name}' doesn't match ASAP7 pattern")
        else:
            # Should not reach here - ASAP7 requires specific folder
            raise ValueError("ASAP7 requires specific folder selection")

    start = config['start']
    end = config['end']

    # Build Python execution command
    cmd = [
        sys.executable, script_path,
        '--data-dir', data_dir,
        '--prefix', prefix,
        '--start', str(start),
        '--end', str(end),
        '--output-dir', output_dir,
        '--delay-type', delay_type
    ]

    return cmd, output_dir

def confirm_execution(dataset_type, folder_name, output_dir, delay_type, auto_confirm=False):
    """Show summary and confirm execution"""
    config = DATASET_CONFIGS[dataset_type]

    print("\n" + "="*80)
    print(" "*25 + "EXECUTION SUMMARY")
    print("="*80)

    print(f"\nDataset Type: {config['name']}")
    print(f"  Data directory: {config['data_dir']}")
    print(f"  Processing folder: {folder_name}")
    print(f"  Output directory: {output_dir}")

    print(f"\nDelay Type: {delay_type}")
    print(f"Voltage Range: {config['start']} - {config['end']}")

    # Show parsed information for better clarity
    if dataset_type == 'tsmc':
        pattern = re.compile(config['folder_pattern'])
        match = pattern.match(folder_name)
        if match:
            corner = match.group(1)
            temperature = match.group(2)
            print(f"\nParsed Information:")
            print(f"  Corner: {corner}")
            print(f"  Temperature: {temperature}°C")
            print(f"  Prefix: TSMC_{corner}_{temperature}_")
    else:  # ASAP7
        pattern = re.compile(config['folder_pattern'])
        match = pattern.match(folder_name)
        if match:
            cell_type = match.group(1)
            vt_type = match.group(2)
            corner_name = match.group(3)
            corner_num = config['corner_mapping'][corner_name]
            print(f"\nParsed Information:")
            print(f"  Cell Type: {cell_type}")
            print(f"  VT Type: {vt_type}")
            print(f"  Corner: {corner_name} (mapped to {corner_num})")
            print(f"  Prefix: {cell_type}_{vt_type}_{corner_num}_25_")

    print("="*80)

    if auto_confirm:
        return True

    confirm = input("\nProceed with execution? [Y/n]: ").strip().lower()
    return confirm not in ['n', 'no']

def execute_command(cmd):
    """Execute the preprocessing script"""
    print("\n" + "="*80)
    print(" "*25 + "STARTING PREPROCESSING")
    print("="*80 + "\n")

    try:
        result = subprocess.run(cmd, check=True)

        print("\n" + "="*80)
        print(" "*25 + "PREPROCESSING COMPLETED")
        print("="*80)

        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\n\nError: Command failed with exit code {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n\nPreprocessing interrupted by user.")
        return 130

def main_interactive():
    """Run interactive mode"""
    print_banner()

    # Step 1: Select dataset type
    dataset_type = select_dataset_type()
    print(f"\nSelected dataset type: {DATASET_CONFIGS[dataset_type]['name']}")

    # Step 2: Select delay type
    delay_type = select_delay_type()
    print(f"\nSelected delay type: {delay_type}")

    # Step 3: For ASAP7, select VT and Corner to find all matching folders
    if dataset_type == 'asap7':
        vt_type, corner = select_asap7_vt_and_corner()
        print(f"\nSelected: {vt_type} - {corner}")

        # Find all folders matching this VT and corner
        folders = find_asap7_folders_by_vt_corner(vt_type, corner)

        if len(folders) == 0:
            print(f"\n❌ No folders found for {vt_type} - {corner}")
            print(f"   Please check data directory: {DATASET_CONFIGS[dataset_type]['data_dir']}")
            return 1

        print(f"\nFound {len(folders)} matching folders:")
        for folder in folders:
            print(f"  • {folder.name}")

        # Confirm processing all folders
        print("\n" + "="*80)
        print(f"Will process all {len(folders)} folders sequentially")
        print("="*80)
        confirm = input("\nProceed with processing all folders? [Y/n]: ").strip().lower()
        if confirm in ['n', 'no']:
            print("\nExecution cancelled by user.")
            return 0

        # Process each folder
        success_count = 0
        fail_count = 0

        for i, folder in enumerate(folders, 1):
            folder_name = folder.name
            print(f"\n{'='*80}")
            print(f"Processing folder {i}/{len(folders)}: {folder_name}")
            print(f"{'='*80}")

            try:
                cmd, output_dir = build_command(dataset_type, folder_name, delay_type)
                result = execute_command(cmd)
                if result == 0:
                    success_count += 1
                    print(f"✅ Successfully processed: {folder_name}")
                else:
                    fail_count += 1
                    print(f"❌ Failed to process: {folder_name}")
            except Exception as e:
                fail_count += 1
                print(f"❌ Error processing {folder_name}: {e}")

        # Summary
        print(f"\n{'='*80}")
        print(f"PROCESSING SUMMARY")
        print(f"{'='*80}")
        print(f"Total folders: {len(folders)}")
        print(f"✅ Successful: {success_count}")
        print(f"❌ Failed: {fail_count}")
        print(f"{'='*80}")

        return 0 if fail_count == 0 else 1

    else:
        # TSMC: Select specific folder
        folder_count, folders = count_matching_folders(dataset_type)
        if folder_count == 0:
            print(f"\n❌ No folders found for dataset type '{dataset_type}'")
            print(f"   Please check data directory: {DATASET_CONFIGS[dataset_type]['data_dir']}")
            return 1

        print(f"Found {folder_count} matching folders")

        # Step 4: Select specific folder
        print("\nAvailable folders:")
        print("-" * 80)
        for i, folder in enumerate(folders):
            print(f"  [{i}] {folder.name}")

        while True:
            choice = input(f"\nSelect folder [0-{len(folders)-1}]: ").strip()
            try:
                idx = int(choice)
                if 0 <= idx < len(folders):
                    folder_name = folders[idx].name
                    break
            except ValueError:
                pass
            print("Invalid choice. Please enter a valid number.")

        # Step 5: Build command
        cmd, output_dir = build_command(dataset_type, folder_name, delay_type)

        # Step 6: Confirm and execute
        if confirm_execution(dataset_type, folder_name, output_dir, delay_type, auto_confirm=False):
            return execute_command(cmd)
        else:
            print("\nExecution cancelled by user.")
            return 0

def main_commandline(args):
    """Run command-line mode"""
    print_banner()

    # Use defaults if not provided
    dataset_type = args.dataset_type if args.dataset_type else 'tsmc'
    delay_type = args.delay_type if args.delay_type else 'cell'
    folder_name = args.folder if args.folder else None

    print(f"Dataset Type: {DATASET_CONFIGS[dataset_type]['name']}")
    print(f"Delay Type: {delay_type}")

    # For ASAP7, support VT and Corner specification
    if dataset_type == 'asap7' and not folder_name:
        # Check if VT and Corner are provided
        vt_type = getattr(args, 'vt_type', None)
        corner = getattr(args, 'corner', None)

        if vt_type and corner:
            # Find all folders matching VT and Corner
            folders = find_asap7_folders_by_vt_corner(vt_type, corner)

            if len(folders) == 0:
                print(f"\n❌ No folders found for {vt_type} - {corner}")
                print(f"   Please check data directory: {DATASET_CONFIGS[dataset_type]['data_dir']}")
                return 1

            print(f"\nFound {len(folders)} matching folders for {vt_type} - {corner}:")
            for folder in folders:
                print(f"  • {folder.name}")

            auto_confirm = hasattr(args, 'yes') and args.yes

            # Process each folder
            success_count = 0
            fail_count = 0

            for i, folder in enumerate(folders, 1):
                folder_name = folder.name
                print(f"\n{'='*80}")
                print(f"Processing folder {i}/{len(folders)}: {folder_name}")
                print(f"{'='*80}")

                try:
                    cmd, output_dir = build_command(dataset_type, folder_name, delay_type)
                    result = execute_command(cmd)
                    if result == 0:
                        success_count += 1
                        print(f"✅ Successfully processed: {folder_name}")
                    else:
                        fail_count += 1
                        print(f"❌ Failed to process: {folder_name}")
                except Exception as e:
                    fail_count += 1
                    print(f"❌ Error processing {folder_name}: {e}")

            # Summary
            print(f"\n{'='*80}")
            print(f"PROCESSING SUMMARY")
            print(f"{'='*80}")
            print(f"Total folders: {len(folders)}")
            print(f"✅ Successful: {success_count}")
            print(f"❌ Failed: {fail_count}")
            print(f"{'='*80}")

            return 0 if fail_count == 0 else 1
        else:
            print(f"\n❌ Error: For ASAP7, either specify --folder or both --vt-type and --corner")
            print(f"\n   Example: python run_voltage_variation_preprocessing.py -t asap7 --vt-type LVT --corner FF")
            print(f"            python run_voltage_variation_preprocessing.py -t asap7 --folder AO_LVT_FF")
            return 1

    # For TSMC or specific folder mode
    if not folder_name:
        print(f"\n❌ Error: --folder argument is required")
        print(f"   Please specify a folder to process")
        print(f"\n   Example: python run_voltage_variation_preprocessing.py -t tsmc --folder TSMC_FF_0")
        return 1

    print(f"Processing folder: {folder_name}")

    # Count matching folders
    folder_count, folders = count_matching_folders(dataset_type)
    if folder_count == 0:
        print(f"\n❌ No folders found for dataset type '{dataset_type}'")
        print(f"   Please check data directory: {DATASET_CONFIGS[dataset_type]['data_dir']}")
        return 1

    print(f"Found {folder_count} matching folders")

    # Verify folder exists
    folder_names = [f.name for f in folders]
    if folder_name not in folder_names:
        print(f"\n❌ Error: Folder '{folder_name}' not found in available folders")
        print(f"   Available folders:")
        for f in folders[:10]:
            print(f"     • {f.name}")
        if len(folders) > 10:
            print(f"     ... and {len(folders) - 10} more")
        return 1

    # Build command
    cmd, output_dir = build_command(dataset_type, folder_name, delay_type)

    # Confirm and execute
    auto_confirm = hasattr(args, 'yes') and args.yes
    if confirm_execution(dataset_type, folder_name, output_dir, delay_type, auto_confirm=auto_confirm):
        return execute_command(cmd)
    else:
        print("\nExecution cancelled.")
        return 0

def main():
    """Main function - determines interactive or command-line mode"""
    # Check if any command-line arguments were provided
    if len(sys.argv) > 1:
        args = parse_args()
        return main_commandline(args)
    else:
        return main_interactive()

if __name__ == "__main__":
    sys.exit(main())
