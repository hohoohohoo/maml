#!/usr/bin/env python
# coding: utf-8

"""
Separate Cell Types Data Preprocessing Wrapper

Wrapper script for separate cell types data preprocessing.
Supports both interactive and command-line modes.
"""

import os
import sys
import subprocess
import argparse

# Data directory options
DATA_DIR_OPTIONS = [
    "/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/ASAP7_lib_files/processed",
    "/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/ASAP7_lib_files/processed_simple",
    "/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/ASAP7_lib_files/test_processed",
    "/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/ASAP7_lib_files/test_processed_simple"
]

def print_banner():
    """Print welcome banner"""
    print("\n" + "="*80)
    print(" "*15 + "Separate Cell Types Data Preprocessing Wrapper")
    print("="*80 + "\n")

def select_data_directory():
    """Let user select data directory"""
    print("Available data directories:")
    print("-" * 80)
    print("  [0] processed")
    print("      • Path: ASAP7_lib_files/processed")
    print("      • Parameters: A=0.625,0.875,1.125,1.375 (no 'test' in name)")
    print()
    print("  [1] processed_simple (Default)")
    print("      • Path: ASAP7_lib_files/processed_simple")
    print("      • Parameters: A=0.625,0.875,1.125,1.375 (no 'test' in name)")
    print()
    print("  [2] test_processed")
    print("      • Path: ASAP7_lib_files/test_processed")
    print("      • Parameters: A=0.75,1.0,1.25 ('test' in name)")
    print()
    print("  [3] test_processed_simple")
    print("      • Path: ASAP7_lib_files/test_processed_simple")
    print("      • Parameters: A=0.75,1.0,1.25 ('test' in name)")
    print()

    while True:
        choice = input("Select data directory [0/1/2/3] (default: 1): ").strip()
        if choice == '' or choice == '1':
            return 1
        elif choice in ['0', '2', '3']:
            return int(choice)
        print("Invalid choice. Please select 0, 1, 2, or 3.\n")

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

def select_topology_type():
    """Let user select topology type"""
    print("\nAvailable topology types:")
    print("-" * 80)
    print("  [0] Intra-Topology (Default)")
    print("      • Test cell types: AND2x6, NAND3x2, NOR2xp67, OR2x6")
    print("      • 4 test cell types")
    print()
    print("  [1] Technology-Agnostic")
    print("      • Test cell types: MAJIxp5, MAJx2, MAJx3, HAxp5, FAx1,")
    print("        XOR2xp5, XOR2x2, XOR2x1, XNOR2xp5, XNOR2x2, XNOR2x1")
    print("      • 11 test cell types")
    print()

    while True:
        choice = input("Select topology type [0/1] (default: 0): ").strip()
        if choice == '' or choice == '0':
            return 'intra'
        elif choice == '1':
            return 'agnostic'
        print("Invalid choice. Please select 0 or 1.\n")

def get_dir_name(dir_index):
    """Get directory name from index"""
    dir_names = {
        0: 'processed',
        1: 'processed_simple',
        2: 'test_processed',
        3: 'test_processed_simple'
    }
    return dir_names.get(dir_index, f'Directory {dir_index}')

def get_parameters(dir_index, topology_type):
    """Get parameters based on directory and topology type"""
    data_dir = DATA_DIR_OPTIONS[dir_index]
    dir_name = os.path.basename(data_dir)

    # Determine parameters based on directory name (contains 'test' or not)
    if 'test' in dir_name:
        param_a = "0.75,1.0,1.25"
        param_b = "0.09,0.062,0.092,0.066,0.094,0.07"
        param_c = "0.36,0.47,0.38,0.475,0.40,0.48"
        param_type = "test"
    else:
        param_a = "0.625,0.875,1.125,1.375"
        param_b = "0.089,0.06,0.091,0.064,0.093,0.068,0.095,0.072"
        param_c = "0.35,0.465,0.37,0.473,0.39,0.478,0.41,0.485"
        param_type = "non-test"

    # Determine test cell types based on topology type
    if topology_type == 'intra':
        test_cell_types = ["AND2x6", "NAND3x2", "NOR2xp67", "OR2x6"]
    else:
        test_cell_types = [
                        #     "MAJIxp5", "MAJx2", "MAJx3", "HAxp5", "FAx1",
                        #   "XOR2xp5", "XOR2x2", "XOR2x1", "XNOR2xp5", "XNOR2x2", "XNOR2x1",
                          "A2O1A1O1Ixp25", "AO21x1", "AO32x1",
                          "O2A1O1Ixp5", "OAI22x1"]

    return {
        'data_dir': data_dir,
        'dir_name': dir_name,
        'param_a': param_a,
        'param_b': param_b,
        'param_c': param_c,
        'param_type': param_type,
        'test_cell_types': test_cell_types
    }

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Separate Cell Types Data Preprocessing Wrapper',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python run_separate_cell_types_preprocessing.py

  # Command-line mode
  python run_separate_cell_types_preprocessing.py --data-dir-index 0 --delay-type cell --topology-type intra
  python run_separate_cell_types_preprocessing.py -i 2 -d transition -t agnostic
  python run_separate_cell_types_preprocessing.py -i 1 -d cell -t intra --train-only

Data Directories (index):
  [0] ASAP7_lib_files/processed
  [1] ASAP7_lib_files/processed_simple
  [2] ASAP7_lib_files/test_processed
  [3] ASAP7_lib_files/test_processed_simple
""")

    parser.add_argument('-i', '--data-dir-index', type=int, choices=[0, 1, 2, 3],
                       help='Data directory index: 0-3')
    parser.add_argument('-d', '--delay-type', type=str, choices=['cell', 'transition'],
                       help='Delay type: cell or transition')
    parser.add_argument('-t', '--topology-type', type=str, choices=['intra', 'agnostic'],
                       help='Topology type: intra or agnostic')
    parser.add_argument('--train-only', action='store_true',
                       help='Only create shared train dataset')
    parser.add_argument('--yes', '-y', action='store_true',
                       help='Skip confirmation prompt')

    return parser.parse_args()

def build_command(dir_index, delay_type, topology_type, params, train_only=False):
    """Build the command to execute"""
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'build_and_split_dataset_asap7.py')

    # Output directory
    if topology_type == 'intra':
      output_suffix = 'intra_topology_data'
    else:  # agnostic
        output_suffix = 'topology_agnostic_data'

    output_dir = f"../../dataset_all/temp_dataset_ASAP7/{output_suffix}"

    # Create temporary directory with symbolic links
    temp_data_dir = f"{output_dir}/temp_combined_data_{delay_type}_{topology_type}_{dir_index}"

    cmd = [sys.executable, script_path]
    cmd.extend(['--data-dirs', params['data_dir']])
    cmd.extend(['--output-dir', output_dir])
    cmd.extend(['--test-cell-types'] + params['test_cell_types'])
    cmd.extend(['--param-a', params['param_a']])
    cmd.extend(['--param-b', params['param_b']])
    cmd.extend(['--param-c', params['param_c']])
    cmd.extend(['--delay-type', delay_type])
    cmd.extend(['--topology-type', topology_type])

    if train_only:
        cmd.append('--train-only')

    return cmd

def confirm_execution(dir_index, delay_type, topology_type, params, cmd, train_only=False, auto_confirm=False):
    """Show summary and confirm execution"""
    print("\n" + "="*80)
    print(" "*25 + "EXECUTION SUMMARY")
    print("="*80)

    print(f"\nData Directory: {get_dir_name(dir_index)}")
    print(f"  Path: {params['data_dir']}")
    print(f"  Parameter type: {params['param_type']}")

    print(f"\nDelay Type: {delay_type}")
    print(f"Topology Type: {topology_type}")
    print(f"Mode: {'Train only' if train_only else 'Train + Test'}")

    print(f"\nParameters:")
    print(f"  A: {params['param_a']}")
    print(f"  B: {params['param_b']}")
    print(f"  C: {params['param_c']}")

    print(f"\nTest Cell Types ({len(params['test_cell_types'])} types):")
    for cell_type in params['test_cell_types']:
        print(f"  • {cell_type}")

    print(f"\nCommand to execute:")
    print(f"  {' '.join(cmd)}")
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

    # Step 1: Select data directory
    dir_index = select_data_directory()
    print(f"\nSelected data directory: [{dir_index}] {get_dir_name(dir_index)}")

    # Step 2: Select delay type
    delay_type = select_delay_type()
    print(f"\nSelected delay type: {delay_type}")

    # Step 3: Select topology type
    topology_type = select_topology_type()
    print(f"\nSelected topology type: {topology_type}")

    # Step 4: Ask for train-only mode
    print("\nProcessing mode:")
    print("-" * 80)
    train_only_input = input("Train only mode? [y/N]: ").strip().lower()
    train_only = train_only_input in ['y', 'yes']

    # Step 5: Get parameters
    params = get_parameters(dir_index, topology_type)

    # Step 6: Build command
    cmd = build_command(dir_index, delay_type, topology_type, params, train_only)

    # Step 7: Confirm and execute
    if confirm_execution(dir_index, delay_type, topology_type, params, cmd, train_only, auto_confirm=False):
        return execute_command(cmd)
    else:
        print("\nExecution cancelled by user.")
        return 0

def main_commandline(args):
    """Run command-line mode"""
    print_banner()

    # Use defaults if not provided
    dir_index = args.data_dir_index if args.data_dir_index is not None else 1
    delay_type = args.delay_type if args.delay_type else 'cell'
    topology_type = args.topology_type if args.topology_type else 'intra'
    train_only = args.train_only

    print(f"Data Directory: [{dir_index}] {get_dir_name(dir_index)}")
    print(f"Delay Type: {delay_type}")
    print(f"Topology Type: {topology_type}")
    print(f"Mode: {'Train only' if train_only else 'Train + Test'}")

    # Get parameters
    params = get_parameters(dir_index, topology_type)

    # Build command
    cmd = build_command(dir_index, delay_type, topology_type, params, train_only)

    # Confirm and execute
    auto_confirm = hasattr(args, 'yes') and args.yes
    if confirm_execution(dir_index, delay_type, topology_type, params, cmd, train_only, auto_confirm=auto_confirm):
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
