#!/usr/bin/env python
# coding: utf-8

"""
Unified Testing Wrapper Script

Simplified wrapper that uses MAML_topology_validation.py or MLP_topology_validation.py
with test_dataset_config.py. Supports both interactive and command-line modes.
"""

import os
import sys
import subprocess
import argparse

# Import configuration to get default values
from utils.test_dataset_config import get_test_config, print_available_configs

def print_banner():
    """Print welcome banner"""
    print("\n" + "="*80)
    print(" "*20 + "MAML/MLP Testing Wrapper")
    print("="*80 + "\n")

def select_model_type():
    """Let user select model type"""
    print("Available model types:")
    print("-" * 80)
    print("  [0] MAML - Model-Agnostic Meta-Learning")
    print("  [1] MLP  - Multi-Layer Perceptron")
    print()

    while True:
        choice = input("Select model type [0/1]: ").strip()
        if choice == '0':
            return 'maml'
        elif choice == '1':
            return 'mlp'
        print("Invalid choice. Please select 0 or 1.\n")

def select_config():
    """Let user select configuration"""
    print_available_configs()

    while True:
        choice = input("\nSelect configuration [0/1/2/3]: ").strip()
        if choice in ['0', '1', '2', '3']:
            return int(choice)
        print("Invalid choice. Please select 0, 1, 2, or 3.\n")

def select_cells(default_cells):
    """Let user select which cells to test"""
    print("\n" + "-"*80)
    print("Default cells for this configuration:")
    for i, cell in enumerate(default_cells):
        print(f"  {i+1}. {cell}")
    print("-"*80)

    print("\nOptions:")
    print("  [Enter]     - Use all default cells")
    print("  [cell names] - Space-separated cell names (e.g., 'NAND3x2 OR2x6')")
    print("  [indices]    - Cell indices (e.g., '1 3' for first and third cell)")

    user_input = input("\nYour choice: ").strip()

    if not user_input:
        return default_cells

    # Check if input is indices or cell names
    if all(part.isdigit() for part in user_input.split()):
        indices = [int(x) - 1 for x in user_input.split()]
        selected = []
        for idx in indices:
            if 0 <= idx < len(default_cells):
                selected.append(default_cells[idx])
            else:
                print(f"Warning: Index {idx+1} out of range, skipping.")
        return selected if selected else default_cells
    else:
        return user_input.split()

def get_maml_parameters(config):
    """Get MAML-specific parameters from user"""
    print("\n" + "-"*80)
    print("MAML parameters (press Enter for defaults):")
    print("-"*80)

    mode_input = input(f"Mode [extrapolation/interpolation] (default: extrapolation): ").strip().lower()
    mode = mode_input if mode_input in ['extrapolation', 'interpolation'] else 'extrapolation'

    data_type_input = input(f"Data type [cell/transition] (default: {config['default_data_type']}): ").strip().lower()
    data_type = data_type_input if data_type_input in ['cell', 'transition'] else config['default_data_type']

    gpu_input = input(f"GPU ID (default: {config['default_gpu']}): ").strip()
    gpu_id = gpu_input if gpu_input else config['default_gpu']

    inner_input = input(f"Inner loop steps (default: 1): ").strip()
    inner = inner_input if inner_input else '1'

    innerdiv_input = input(f"Inner learning rate divisor (default: 100): ").strip()
    innerdiv = innerdiv_input if innerdiv_input else '100'

    meta_input = input(f"Meta batch size (default: {config['default_meta']}): ").strip()
    meta = meta_input if meta_input else str(config['default_meta'])

    print(f"\nSupport set indices:")
    print(f"  - Extrapolation default: [5, 30, 55]")
    print(f"  - Interpolation default: [15, 30, 45]")
    indices_input = input(f"Indices (space-separated, e.g., '5 30 55') (default: mode-dependent): ").strip()
    indices = indices_input.split() if indices_input else None

    save_input = input(f"Save results to .npy files? [y/N]: ").strip().lower()
    save_results = save_input in ['y', 'yes']

    return {
        'mode': mode,
        'data_type': data_type,
        'gpu_id': gpu_id,
        'inner': inner,
        'innerdiv': innerdiv,
        'meta': meta,
        'indices': indices,
        'save_results': save_results
    }

def get_mlp_parameters(config):
    """Get MLP-specific parameters from user"""
    print("\n" + "-"*80)
    print("MLP parameters (press Enter for defaults):")
    print("-"*80)

    mode_input = input(f"Mode [extrapolation/interpolation] (default: extrapolation): ").strip().lower()
    mode = mode_input if mode_input in ['extrapolation', 'interpolation'] else 'extrapolation'

    data_type_input = input(f"Data type [cell/transition] (default: {config['default_data_type']}): ").strip().lower()
    data_type = data_type_input if data_type_input in ['cell', 'transition'] else config['default_data_type']

    model_type_input = input(f"Model type [aadam/mlp] (default: aadam): ").strip().lower()
    model_type = model_type_input if model_type_input in ['aadam', 'mlp'] else 'aadam'

    gpu_input = input(f"GPU ID (default: {config['default_gpu']}): ").strip()
    gpu_id = gpu_input if gpu_input else config['default_gpu']

    iterations_input = input(f"Number of iterations (default: 300000): ").strip()
    num_iterations = iterations_input if iterations_input else '300000'

    print(f"\nSupport set indices:")
    print(f"  - Extrapolation default: [5, 30, 55]")
    print(f"  - Interpolation default: [15, 30, 45]")
    indices_input = input(f"Indices (space-separated, e.g., '5 30 55') (default: mode-dependent): ").strip()
    indices = indices_input.split() if indices_input else None

    save_input = input(f"Save results to .npy files? [y/N]: ").strip().lower()
    save_results = save_input in ['y', 'yes']

    return {
        'mode': mode,
        'data_type': data_type,
        'model_type': model_type,
        'gpu_id': gpu_id,
        'num_iterations': num_iterations,
        'indices': indices,
        'save_results': save_results
    }

def build_maml_command(config_id, cells, params):
    """Build the MAML command to execute"""
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MAML_topology_validation.py')

    cmd = [sys.executable, script_path]
    cmd.extend(['--config', str(config_id)])
    cmd.extend(['--mode', params['mode']])
    cmd.extend(['--cells'] + cells)
    cmd.extend(['--data_type', params['data_type']])
    cmd.extend(['--gpu_id', params['gpu_id']])
    cmd.extend(['--inner', params['inner']])
    cmd.extend(['--innerdiv', params['innerdiv']])
    cmd.extend(['--meta', params['meta']])

    if params['indices']:
        cmd.extend(['--indices'] + params['indices'])

    if params['save_results']:
        cmd.append('--save_results')

    return cmd

def build_mlp_command(config_id, cells, params):
    """Build the MLP command to execute"""
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MLP_topology_validation.py')

    cmd = [sys.executable, script_path]
    cmd.extend(['--config', str(config_id)])
    cmd.extend(['--mode', params['mode']])
    cmd.extend(['--cells'] + cells)
    cmd.extend(['--data_type', params['data_type']])
    cmd.extend(['--model_type', params['model_type']])
    cmd.extend(['--gpu_id', params['gpu_id']])
    cmd.extend(['--num_iterations', params['num_iterations']])

    if params['indices']:
        cmd.extend(['--indices'] + params['indices'])

    if params['save_results']:
        cmd.append('--save_results')

    return cmd

def confirm_execution(model_type, config, cells, params, cmd, auto_confirm=False):
    """Show summary and confirm execution"""
    print("\n" + "="*80)
    print(" "*30 + "EXECUTION SUMMARY")
    print("="*80)
    print(f"\nModel Type: {model_type.upper()}")
    print(f"Configuration: {config['name']}")
    print(f"Script: {model_type.upper()}_topology_validation.py")
    print(f"\nParameters:")
    print(f"  Mode: {params['mode']}")
    print(f"  Data type: {params['data_type']}")

    if model_type == 'maml':
        print(f"  Cells: {', '.join(cells)}")
        print(f"  GPU ID: {params['gpu_id']}")
        print(f"  Inner steps: {params['inner']}")
        print(f"  Inner LR divisor: {params['innerdiv']}")
        print(f"  Meta batch size: {params['meta']}")
    else:  # mlp
        print(f"  Model type: {params['model_type']}")
        print(f"  Cells: {', '.join(cells)}")
        print(f"  GPU ID: {params['gpu_id']}")
        print(f"  Iterations: {params['num_iterations']}")

    if params['indices']:
        print(f"  Indices: {', '.join(params['indices'])}")
    else:
        print(f"  Indices: mode-dependent (auto)")
    print(f"  Save results: {'Yes' if params['save_results'] else 'No'}")

    print(f"\nCommand to execute:")
    print(f"  {' '.join(cmd)}")
    print("="*80)

    if auto_confirm:
        return True

    confirm = input("\nProceed with execution? [Y/n]: ").strip().lower()
    return confirm not in ['n', 'no']

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Unified Testing Wrapper - Interactive or Command-line Mode',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Interactive mode:
    python run_test.py

  Command-line mode (MAML):
    python run_test.py --model maml --config 0 --cells NAND3x2 OR2x6
    python run_test.py --model maml --config 2 --mode interpolation --inner 2

  Command-line mode (MLP):
    python run_test.py --model mlp --config 0 --cells NAND3x2 OR2x6
    python run_test.py --model mlp --config 2 --model_type mlp --num_iterations 100000
        """
    )

    parser.add_argument('--model', type=str, choices=['maml', 'mlp'],
                        help='Model type: maml or mlp')
    parser.add_argument('--config', type=int, choices=[0, 1, 2, 3],
                        help='Configuration: 0=ASAP7 Intra, 1=ASAP7 Agnostic, 2=TSMC Intra, 3=TSMC Agnostic')
    parser.add_argument('--cells', type=str, nargs='+',
                        help='Cell types to test (space-separated)')
    parser.add_argument('--mode', type=str, choices=['extrapolation', 'interpolation'],
                        help='Testing mode (default: extrapolation)')
    parser.add_argument('--data_type', type=str, choices=['cell', 'transition'],
                        help='Data type (default: config-dependent)')
    parser.add_argument('--gpu_id', type=str,
                        help='GPU device ID')
    parser.add_argument('--indices', type=str, nargs='+',
                        help='Support set indices (space-separated)')
    parser.add_argument('--save_results', action='store_true',
                        help='Save prediction results to .npy files')

    # MAML-specific arguments
    parser.add_argument('--inner', type=str,
                        help='[MAML] Inner loop steps (default: 1)')
    parser.add_argument('--innerdiv', type=str,
                        help='[MAML] Inner learning rate divisor (default: 100)')
    parser.add_argument('--meta', type=str,
                        help='[MAML] Meta batch size')

    # MLP-specific arguments
    parser.add_argument('--model_type', type=str, choices=['aadam', 'mlp'],
                        help='[MLP] Model type: aadam (hidden=256) or mlp (hidden=40)')
    parser.add_argument('--num_iterations', type=str,
                        help='[MLP] Number of iterations (default: 300000)')

    parser.add_argument('--yes', '-y', action='store_true',
                        help='Skip confirmation prompt')

    return parser.parse_args()

def main_interactive():
    """Run interactive mode"""
    print_banner()

    # Step 1: Select model type
    model_type = select_model_type()
    print(f"\nSelected model type: {model_type.upper()}")

    # Step 2: Select configuration
    config_id = select_config()
    config = get_test_config(config_id)
    print(f"\nSelected configuration: {config['name']}")

    # Step 3: Select cells
    cells = select_cells(config['default_cells'])

    if not cells:
        print("\nError: No cells selected. Exiting.")
        return 1

    print(f"\nSelected cells: {', '.join(cells)}")

    # Step 4: Get parameters
    if model_type == 'maml':
        params = get_maml_parameters(config)
        cmd = build_maml_command(config_id, cells, params)
    else:  # mlp
        params = get_mlp_parameters(config)
        cmd = build_mlp_command(config_id, cells, params)

    # Step 5: Confirm and execute
    if confirm_execution(model_type, config, cells, params, cmd, auto_confirm=False):
        return execute_command(cmd)
    else:
        print("\nExecution cancelled by user.")
        return 0

def main_commandline(args):
    """Run command-line mode"""
    print_banner()

    # Validate required arguments
    if args.model is None:
        print("Error: --model is required in command-line mode")
        print("Use -h for help or run without arguments for interactive mode")
        return 1

    if args.config is None:
        print("Error: --config is required in command-line mode")
        print("Use -h for help or run without arguments for interactive mode")
        return 1

    model_type = args.model
    config_id = args.config
    config = get_test_config(config_id)

    print(f"Model Type: {model_type.upper()}")
    print(f"Configuration: {config['name']}")

    # Determine cells
    if args.cells:
        cells = args.cells
    else:
        cells = config['default_cells']
        print(f"Using default cells: {', '.join(cells)}")

    # Build parameters based on model type
    if model_type == 'maml':
        params = {
            'mode': args.mode or 'extrapolation',
            'data_type': args.data_type or config['default_data_type'],
            'gpu_id': args.gpu_id or config['default_gpu'],
            'inner': args.inner or '1',
            'innerdiv': args.innerdiv or '100',
            'meta': args.meta or str(config['default_meta']),
            'indices': args.indices,
            'save_results': args.save_results
        }
        cmd = build_maml_command(config_id, cells, params)
    else:  # mlp
        params = {
            'mode': args.mode or 'extrapolation',
            'data_type': args.data_type or config['default_data_type'],
            'model_type': args.model_type or 'aadam',
            'gpu_id': args.gpu_id or config['default_gpu'],
            'num_iterations': args.num_iterations or '300000',
            'indices': args.indices,
            'save_results': args.save_results
        }
        cmd = build_mlp_command(config_id, cells, params)

    # Confirm and execute
    if confirm_execution(model_type, config, cells, params, cmd, auto_confirm=args.yes):
        return execute_command(cmd)
    else:
        print("\nExecution cancelled.")
        return 0

def execute_command(cmd):
    """Execute the command"""
    print("\n" + "="*80)
    print(" "*30 + "STARTING EXECUTION")
    print("="*80 + "\n")

    try:
        result = subprocess.run(cmd, check=True)

        print("\n" + "="*80)
        print(" "*30 + "EXECUTION COMPLETED")
        print("="*80)

        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\n\nError: Command failed with exit code {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user.")
        return 130

def main():
    """Main function - determines interactive or command-line mode"""
    args = parse_args()

    # Check if any command-line arguments were provided
    if args.model is not None or args.config is not None:
        return main_commandline(args)
    else:
        return main_interactive()

if __name__ == "__main__":
    sys.exit(main())
