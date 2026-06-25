#!/usr/bin/env python
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# coding: utf-8

"""
TSMC Data Preprocessing Wrapper

Wrapper script for TSMC data preprocessing.
Supports both interactive and command-line modes.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# Data directory
TSMC_DATA_DIR = "/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/TSMC_lib_files"

# Dataset type configurations
DATASET_CONFIGS = {
    'original_agnostic': {
        'name': 'Original - Technology Agnostic (Standard TSMC)',
        'folder_pattern': 'TSMC_??_*',
        'exclude_patterns': [],
        'output_suffix': 'topology_agnostic_data_reduced_patched',
        'topology_type': 'agnostic',
        'test_cell_types': ["HA1D0BWP30P140", "FA1D0BWP30P140", "IOA21D0BWP30P140", "IOA21D1BWP30P140",
                           "OA21D0BWP30P140", "OA21D1BWP30P140", "OA211D0BWP30P140", "OA211D1BWP30P140",
                           "IAO21D0BWP30P140", "IAO21D1BWP30P140", "AO21D0BWP30P140", "AO21D1BWP30P140",
                           "AO211D0BWP30P140", "AO211D1BWP30P140"]
    },
    'original_intra': {
        'name': 'Original - Intra Topology (Standard TSMC)',
        'folder_pattern': 'TSMC_??_*',
        'exclude_patterns': [],
        'output_suffix': 'intratopology_data_patched',
        'topology_type': 'intra',
        'test_cell_types': ["AN4D0BWP30P140","ND3D0BWP30P140","NR3D1BWP30P140","OR4D0BWP30P140","XNR3D1BWP30P140","XOR3D1BWP30P140"]
    },
    'nor_nand': {
        'name': 'Adding NOR/NAND (TSMC_*2_*)',
        'folder_pattern': 'TSMC_*2_*',
        'exclude_patterns': [],
        'output_suffix': 'topology_agnostic_data2_patched',
        'topology_type': 'agnostic',
        'test_cell_types': []  # No test cells, all for training
    },
    'seq': {
        'name': 'Sequential Cells (TSMC_*seq_*)',
        'folder_pattern': 'TSMC_*seq_*',
        'exclude_patterns': [],
        'data_dir': f'{TSMC_DATA_DIR}/TSMC_seq_cell',
        'output_suffix': 'seq_data',
        'topology_type': 'agnostic',
        'test_cell_types': ["DFCNQD1BWP30P140", "SDFSNQD0BWP30P140", "SDFCSNQD1BWP30P140"]
    },
    'combined': {
        # GNN convention (matches 1-D and 2-D GNN_dataset_TSMC train pools):
        # one train pool whose excluded cells are the union of the historical
        # intra (6) and agnostic (14) test_cell_types. The downstream consumer
        # can filter the test set by either subset to recover the intra-only
        # or agnostic-only evaluation metric.
        'name': 'Combined intra+agnostic (1-D/2-D GNN convention)',
        'folder_pattern': 'TSMC_??_*',
        'exclude_patterns': [],
        'output_suffix': 'combined_data',
        'topology_type': 'agnostic',
        'test_cell_types': [
            # intra topology test cells (6)
            'AN4D0BWP30P140', 'ND3D0BWP30P140', 'NR3D1BWP30P140',
            'OR4D0BWP30P140', 'XNR3D1BWP30P140', 'XOR3D1BWP30P140',
            # agnostic topology test cells (14)
            'HA1D0BWP30P140',  'FA1D0BWP30P140',
            'IOA21D0BWP30P140', 'IOA21D1BWP30P140',
            'OA21D0BWP30P140',  'OA21D1BWP30P140',
            'OA211D0BWP30P140', 'OA211D1BWP30P140',
            'IAO21D0BWP30P140', 'IAO21D1BWP30P140',
            'AO21D0BWP30P140',  'AO21D1BWP30P140',
            'AO211D0BWP30P140', 'AO211D1BWP30P140',
        ],
    },
}

# Parameter mappings (same for all datasets)
PARAM_A = "1.427,1.457,1.430,1.470,1.443,1.483,1.43,1.47,1.43,1.47"
PARAM_B = "0.026,0.045,0,0,-0.026,-0.05,0.0208,-0.04,0.036,-0.0208"
PARAM_C = "0.024,2.000,0.024,2.000,0.024,2.000,0.024,2.000,0.024,2.000"

def print_banner():
    """Print welcome banner"""
    print("\n" + "="*80)
    print(" "*20 + "TSMC Data Preprocessing Wrapper")
    print("="*80 + "\n")

def select_dataset_type():
    """Let user select dataset type"""
    print("Available dataset types:")
    print("-" * 80)
    print("  [0] Original - Technology Agnostic (Default)")
    print("      • All TSMC_* folders (excluding *2_* and *Seq*)")
    print("      • 14 test cell types for topology agnostic validation")
    print("      • Output: topology_agnostic_data_reduced")
    print()
    print("  [1] Original - Intra Topology")
    print("      • All TSMC_* folders (excluding *2_* and *Seq*)")
    print("      • 14 test cell types for intra topology validation")
    print("      • Output: intratopology_data")
    print()
    print("  [2] Adding NOR/NAND")
    print("      • TSMC_*2_* folders (with additional NOR/NAND cells)")
    print("      • All cells for training (no separate test cells)")
    print("      • Output: topology_agnostic_data2")
    print()
    print("  [3] Sequential Cells")
    print("      • TSMC_Seq* and TSMC_*Seq* folders (sequential cells only)")
    print("      • 2 sequential test cell types (DFCNQD1BWP30P140, SDFSNQD0BWP30P140)")
    print("      • Output: topology_agnostic_data_Seq")
    print()

    while True:
        choice = input("Select dataset type [0/1/2/3] (default: 0): ").strip()
        if choice == '' or choice == '0':
            return 'original_agnostic'
        elif choice == '1':
            return 'original_intra'
        elif choice == '2':
            return 'nor_nand'
        elif choice == '3':
            return 'seq'
        print("Invalid choice. Please select 0, 1, 2, or 3.\n")

def select_delay_type():
    """Let user select delay type"""
    print("\nAvailable delay types:")
    print("-" * 80)
    print("  [0] Transition Delay (Default)")
    print("      • Uses libdata_extract_MAML_transition module")
    print("      • Transition-level timing analysis")
    print()
    print("  [1] Cell Delay")
    print("      • Uses libdata_extract_MAML_cell module")
    print("      • Cell-level timing analysis")
    print()

    while True:
        choice = input("Select delay type [0/1] (default: 0): ").strip()
        if choice == '' or choice == '0':
            return 'transition'
        elif choice == '1':
            return 'cell'
        print("Invalid choice. Please select 0 or 1.\n")

def get_dataset_name(dataset_type):
    """Get dataset name from type"""
    return DATASET_CONFIGS[dataset_type]['name']

def count_matching_folders(dataset_type):
    """Count folders matching the dataset type pattern"""
    config = DATASET_CONFIGS[dataset_type]
    data_path = Path(config.get('data_dir', TSMC_DATA_DIR))

    if not data_path.exists():
        return 0, []

    # Get folder patterns
    patterns = config['folder_pattern']
    if isinstance(patterns, str):
        patterns = [patterns]

    exclude_patterns = config['exclude_patterns']

    # Find matching folders
    matching_folders = []
    for pattern in patterns:
        folders = list(data_path.glob(pattern))
        for folder in folders:
            # Check if should be excluded
            should_exclude = False
            for exclude_pattern in exclude_patterns:
                if folder.match(exclude_pattern):
                    should_exclude = True
                    break

            if not should_exclude and folder.is_dir():
                matching_folders.append(folder)

    return len(matching_folders), matching_folders

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='TSMC Data Preprocessing Wrapper',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python run_tsmc_preprocessing.py

  # Command-line mode
  python run_tsmc_preprocessing.py --dataset-type original --delay-type transition
  python run_tsmc_preprocessing.py -t nor_nand -d cell
  python run_tsmc_preprocessing.py -t seq -d transition --train-only

Dataset Types:
  original_agnostic - Standard TSMC dataset, technology agnostic (all TSMC_* folders)
  original_intra    - Standard TSMC dataset, intra topology (all TSMC_* folders)
  nor_nand          - TSMC dataset with additional NOR/NAND cells (TSMC_*2_* folders)
  seq               - TSMC Sequential cells dataset (TSMC_Seq*, TSMC_*Seq* folders)
""")

    parser.add_argument('-t', '--dataset-type', type=str,
                       choices=['original_agnostic', 'original_intra', 'nor_nand', 'seq', 'combined'],
                       help='Dataset type: original_agnostic, original_intra, nor_nand, seq, or combined')
    parser.add_argument('-d', '--delay-type', type=str, choices=['cell', 'transition'],
                       help='Delay type: cell or transition')
    parser.add_argument('--train-only', action='store_true',
                       help='Only create shared train dataset')
    parser.add_argument('--test-only', action='store_true',
                       help='Only create test datasets (skip train)')
    parser.add_argument('--yes', '-y', action='store_true',
                       help='Skip confirmation prompt')

    return parser.parse_args()

def build_command(dataset_type, delay_type, train_only=False, test_only=False):
    """Build the command to execute"""
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'build_and_split_dataset_tsmc.py')

    config = DATASET_CONFIGS[dataset_type]

    # Output directory
    output_dir = f"../../dataset_all/MLP_dataset_TSMC/{config['output_suffix']}"

    # Test cell types
    test_cell_types = config['test_cell_types']
    if not test_cell_types:
        test_cell_types = ["NONE"]  # No test cells for nor_nand

    cmd = [sys.executable, script_path]
    cmd.extend(['--data-dirs', config.get('data_dir', TSMC_DATA_DIR)])
    cmd.extend(['--output-dir', output_dir])
    cmd.extend(['--test-cell-types'] + test_cell_types)
    cmd.extend(['--param-a', PARAM_A])
    cmd.extend(['--param-b', PARAM_B])
    cmd.extend(['--param-c', PARAM_C])
    cmd.extend(['--delay-type', delay_type])
    cmd.extend(['--topology-type', config['topology_type']])

    if train_only:
        cmd.append('--train-only')
    if test_only:
        cmd.append('--test-only')

    return cmd, output_dir

def confirm_execution(dataset_type, delay_type, output_dir, folder_count, folders, train_only=False, test_only=False, auto_confirm=False):
    """Show summary and confirm execution"""
    config = DATASET_CONFIGS[dataset_type]

    print("\n" + "="*80)
    print(" "*25 + "EXECUTION SUMMARY")
    print("="*80)

    print(f"\nDataset Type: {config['name']}")
    print(f"  Data directory: {TSMC_DATA_DIR}")
    print(f"  Output directory: {output_dir}")
    print(f"  Output suffix: {config['output_suffix']}")

    print(f"\nDelay Type: {delay_type}")
    print(f"Topology Type: {config['topology_type']}")
    mode = 'Test only' if test_only else ('Train only' if train_only else 'Train + Test')
    print(f"Mode: {mode}")

    print(f"\nMatching Folders: {folder_count} folders")
    if folder_count > 0 and folder_count <= 10:
        for folder in folders[:10]:
            print(f"  • {folder.name}")
    elif folder_count > 10:
        for folder in folders[:5]:
            print(f"  • {folder.name}")
        print(f"  ... and {folder_count - 5} more folders")

    print(f"\nParameters:")
    print(f"  A: {PARAM_A}")
    print(f"  B: {PARAM_B}")
    print(f"  C: {PARAM_C}")

    test_cell_types = config['test_cell_types']
    if test_cell_types:
        print(f"\nTest Cell Types ({len(test_cell_types)} types):")
        for cell_type in test_cell_types:
            print(f"  • {cell_type}")
    else:
        print(f"\nTest Cell Types: None (all cells for training)")

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
    print(f"\nSelected dataset type: {get_dataset_name(dataset_type)}")

    # Step 2: Count matching folders
    folder_count, folders = count_matching_folders(dataset_type)
    if folder_count == 0:
        print(f"\n❌ No folders found matching dataset type '{dataset_type}'")
        print(f"   Please check data directory: {TSMC_DATA_DIR}")
        return 1

    print(f"Found {folder_count} matching folders")

    # Step 3: Select delay type
    delay_type = select_delay_type()
    print(f"\nSelected delay type: {delay_type}")

    # Step 4: Ask for processing mode
    print("\nProcessing mode:")
    print("-" * 80)
    print("  [0] Train + Test (default)")
    print("  [1] Train only")
    print("  [2] Test only")
    mode_input = input("Select mode [0-2]: ").strip()
    train_only = mode_input == '1'
    test_only = mode_input == '2'

    # Step 5: Build command
    cmd, output_dir = build_command(dataset_type, delay_type, train_only, test_only)

    # Step 6: Confirm and execute
    if confirm_execution(dataset_type, delay_type, output_dir, folder_count, folders, train_only, test_only, auto_confirm=False):
        return execute_command(cmd)
    else:
        print("\nExecution cancelled by user.")
        return 0

def main_commandline(args):
    """Run command-line mode"""
    print_banner()

    # Use defaults if not provided
    dataset_type = args.dataset_type if args.dataset_type else 'original_agnostic'
    delay_type = args.delay_type if args.delay_type else 'transition'
    train_only = args.train_only
    test_only = args.test_only

    print(f"Dataset Type: {get_dataset_name(dataset_type)}")
    print(f"Delay Type: {delay_type}")
    mode = 'Test only' if test_only else ('Train only' if train_only else 'Train + Test')
    print(f"Mode: {mode}")

    # Count matching folders
    folder_count, folders = count_matching_folders(dataset_type)
    if folder_count == 0:
        print(f"\n❌ No folders found matching dataset type '{dataset_type}'")
        print(f"   Please check data directory: {TSMC_DATA_DIR}")
        return 1

    print(f"Found {folder_count} matching folders")

    # Build command
    cmd, output_dir = build_command(dataset_type, delay_type, train_only, test_only)

    # Confirm and execute
    auto_confirm = hasattr(args, 'yes') and args.yes
    if confirm_execution(dataset_type, delay_type, output_dir, folder_count, folders, train_only, test_only, auto_confirm=auto_confirm):
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
