#!/usr/bin/env python
# coding: utf-8

"""
Dataset Merge and Train/Test Split Wrapper

Wrapper script for creating merged train/test datasets from processed data.
Supports both ASAP7 and TSMC datasets.
"""

import os
import sys
import subprocess
import argparse

def print_banner():
    """Print welcome banner"""
    print("\n" + "="*80)
    print(" "*20 + "Dataset Merge and Train/Test Split Wrapper")
    print("="*80 + "\n")

def select_dataset_type():
    """Let user select dataset type"""
    print("Available dataset types:")
    print("-" * 80)
    print("  [0] ASAP7 (Default)")
    print("      • VT types: LVT, RVT, SLVT, SRAM")
    print("      • Corners: FF, TT, SS")
    print("      • Total: 12 combinations")
    print()
    print("  [1] TSMC")
    print("      • Corners: FF, SS, TT")
    print("      • Temperatures: 0, 25, 50, 75, 100")
    print("      • Total: 15 combinations")
    print()

    while True:
        choice = input("Select dataset type [0/1] (default: 0): ").strip()
        if choice == '' or choice == '0':
            return 'asap7'
        elif choice == '1':
            return 'tsmc'
        print("Invalid choice. Please select 0 or 1.\n")

def confirm_execution(dataset_type, auto_confirm=False):
    """Show summary and confirm execution"""
    print("\n" + "="*80)
    print(" "*25 + "EXECUTION SUMMARY")
    print("="*80)

    if dataset_type == 'asap7':
        print(f"\nDataset Type: ASAP7")
        print(f"  Base path: /home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_ASAP7_dim5")
        print(f"  VT types: LVT, RVT, SLVT, SRAM")
        print(f"  Corners: FF, TT, SS")
        print(f"  Total combinations: 12")
        print(f"\nOperation:")
        print(f"  • Merge cell types (AO, OA, simple, INVBUF) for each VT/corner combination")
        print(f"  • Split into 80% train / 20% test")
        print(f"  • Save to taskdivide_<vt>_<corner> directories")
    else:
        print(f"\nDataset Type: TSMC")
        print(f"  Base path: /home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_TSMC_dim5")
        print(f"  Corners: FF, SS, TT")
        print(f"  Temperatures: 0, 25, 50, 75, 100")
        print(f"  Total combinations: 15")
        print(f"\nOperation:")
        print(f"  • Split each corner/temperature combination into 80% train / 20% test")
        print(f"  • Save to taskdivide_<corner>_<temp> directories")

    print("="*80)

    if auto_confirm:
        return True

    confirm = input("\nProceed with execution? [Y/n]: ").strip().lower()
    return confirm not in ['n', 'no']

def execute_script(dataset_type):
    """Execute the appropriate dataset processing script"""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if dataset_type == 'asap7':
        script_path = os.path.join(script_dir, 'create_asap7_merged_datasets.py')
        script_name = 'ASAP7 Dataset Merger'
    else:
        script_path = os.path.join(script_dir, 'create_tsmc_merged_datasets.py')
        script_name = 'TSMC Dataset Splitter'

    print("\n" + "="*80)
    print(" "*25 + f"STARTING {script_name.upper()}")
    print("="*80 + "\n")

    try:
        result = subprocess.run([sys.executable, script_path], check=True)

        print("\n" + "="*80)
        print(" "*25 + "PROCESSING COMPLETED")
        print("="*80)

        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\n\nError: Script failed with exit code {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user.")
        return 130

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Dataset Merge and Train/Test Split Wrapper',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python run_dataset_merge_split.py

  # Command-line mode
  python run_dataset_merge_split.py --dataset-type asap7
  python run_dataset_merge_split.py -t tsmc
  python run_dataset_merge_split.py -t asap7 --yes

Dataset Types:
  asap7  - ASAP7 dataset (12 combinations: 4 VT types x 3 corners)
  tsmc   - TSMC dataset (15 combinations: 3 corners x 5 temperatures)
""")

    parser.add_argument('-t', '--dataset-type', type=str,
                       choices=['asap7', 'tsmc'],
                       help='Dataset type: asap7 or tsmc')
    parser.add_argument('--yes', '-y', action='store_true',
                       help='Skip confirmation prompt')

    return parser.parse_args()

def main_interactive():
    """Run interactive mode"""
    print_banner()

    # Step 1: Select dataset type
    dataset_type = select_dataset_type()

    # Step 2: Confirm and execute
    if confirm_execution(dataset_type, auto_confirm=False):
        return execute_script(dataset_type)
    else:
        print("\nExecution cancelled by user.")
        return 0

def main_commandline(args):
    """Run command-line mode"""
    print_banner()

    # Use defaults if not provided
    dataset_type = args.dataset_type if args.dataset_type else 'asap7'

    # Confirm and execute
    auto_confirm = hasattr(args, 'yes') and args.yes
    if confirm_execution(dataset_type, auto_confirm=auto_confirm):
        return execute_script(dataset_type)
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
