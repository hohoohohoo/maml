#!/usr/bin/env python
# coding: utf-8

"""
Voltage Variation Testing Wrapper Script

Wrapper script for voltage variation validation supporting both ASAP7 and TSMC PDKs.
Supports both interactive and command-line modes.
"""

import os
import sys
import subprocess
import argparse

def print_banner():
    """Print welcome banner"""
    print("\n" + "="*80)
    print(" "*20 + "Voltage Variation Testing Wrapper")
    print("="*80 + "\n")

def select_pdk():
    """Let user select PDK (ASAP7 or TSMC)"""
    print("Available PDKs:")
    print("-" * 80)
    print("  [0] ASAP7 - 7nm PDK")
    print("  [1] TSMC  - TSMC PDK")
    print()

    while True:
        choice = input("Select PDK [0/1]: ").strip()
        if choice == '0':
            return 'asap7'
        elif choice == '1':
            return 'tsmc'
        print("Invalid choice. Please select 0 or 1.\n")

def select_model_framework():
    """Let user select model framework"""
    print("\nAvailable model frameworks:")
    print("-" * 80)
    print("  [0] MLP  - Multi-Layer Perceptron")
    print("  [1] MAML - Model-Agnostic Meta-Learning")
    print()

    while True:
        choice = input("Select model framework [0/1]: ").strip()
        if choice == '0':
            return 'mlp'
        elif choice == '1':
            return 'maml'
        print("Invalid choice. Please select 0 or 1.\n")

def get_common_parameters():
    """Get common parameters for both PDKs"""
    print("\n" + "-"*80)
    print("Common parameters (press Enter for defaults):")
    print("-"*80)

    mode_input = input("Mode [extrapolation/interpolation] (default: extrapolation): ").strip().lower()
    mode = mode_input if mode_input in ['extrapolation', 'interpolation'] else 'extrapolation'

    data_type_input = input("Data type [cell/transition] (default: cell): ").strip().lower()
    data_type = data_type_input if data_type_input in ['cell', 'transition'] else 'cell'

    gpu_input = input("GPU ID (default: 4): ").strip()
    gpu_id = gpu_input if gpu_input else '4'

    num_samples_input = input("Number of test samples (default: 100000): ").strip()
    num_test_samples = num_samples_input if num_samples_input else '100000'

    print(f"\nSupport set indices:")
    print(f"  - Extrapolation default: [5, 30, 55] (K=3)")
    print(f"  - Interpolation default: [0, 13, 30, 45, 60] (K=5) ")
    indices_input = input(f"Indices (space-separated, e.g., '0 13 30 45 60') (default: mode-dependent): ").strip()
    indices = indices_input.split() if indices_input else None

    save_input = input(f"Save results to .npy files? [y/N]: ").strip().lower()
    save_results = save_input in ['y', 'yes']

    return {
        'mode': mode,
        'data_type': data_type,
        'gpu_id': gpu_id,
        'num_test_samples': num_test_samples,
        'indices': indices,
        'save_results': save_results
    }

def get_asap7_parameters():
    """Get ASAP7-specific parameters"""
    print("\n" + "-"*80)
    print("ASAP7-specific parameters:")
    print("-"*80)

    corner_input = input("Corner [SS/FF/TT] (default: FF): ").strip().upper()
    corner = corner_input if corner_input in ['SS', 'FF', 'TT'] else 'FF'

    print("\nCell types:")
    print("  [0] lvt  - Low Voltage Threshold")
    print("  [1] rvt  - Regular Voltage Threshold")
    print("  [2] slvt - Super Low Voltage Threshold")
    print("  [3] sram - SRAM")
    cell_choice = input("Select cell type [0/1/2/3] (default: 0): ").strip()
    cell_types = ['lvt', 'rvt', 'slvt', 'sram']
    cell_type = cell_types[int(cell_choice)] if cell_choice in ['0','1','2','3'] else 'lvt'

    return {
        'corner': corner,
        'cell_type': cell_type
    }

def get_tsmc_parameters():
    """Get TSMC-specific parameters"""
    print("\n" + "-"*80)
    print("TSMC-specific parameters:")
    print("-"*80)

    corner_input = input("Corner [ff/ss/tt] (default: ff): ").strip().lower()
    corner = corner_input if corner_input in ['ff', 'ss', 'tt'] else 'ff'

    print("\nTemperatures: 0, 25, 50, 75, 100°C")
    temp_input = input("Temperatures (space-separated, e.g., '0 25 50') (default: all): ").strip()
    temperatures = temp_input.split() if temp_input else ['0', '25', '50', '75', '100']

    return {
        'corner': corner,
        'temperatures': temperatures
    }

def get_mlp_parameters():
    """Get MLP-specific parameters"""
    print("\n" + "-"*80)
    print("MLP parameters:")
    print("-"*80)

    model_type_input = input("Model type [aadam/mlp] (default: aadam): ").strip().lower()
    model_type = model_type_input if model_type_input in ['aadam', 'mlp'] else 'aadam'

    iterations_input = input("Number of iterations (default: 100000): ").strip()
    num_iterations = iterations_input if iterations_input else '100000'

    return {
        'model_type': model_type,
        'num_iterations': num_iterations
    }

def get_maml_parameters():
    """Get MAML-specific parameters"""
    print("\n" + "-"*80)
    print("MAML parameters:")
    print("-"*80)

    layer_input = input("Layer length/hidden size (default: 40): ").strip()
    layer_length = layer_input if layer_input else '40'

    inner_input = input("Inner loop steps (default: 3 for ASAP7, 1 for TSMC): ").strip()
    inner_step = inner_input if inner_input else None  # Will be set based on PDK

    innerdiv_input = input("Inner learning rate divisor (default: 10): ").strip()
    innerdiv = innerdiv_input if innerdiv_input else '10'

    meta_input = input("Meta batch size (default: 16 for ASAP7, 32 for TSMC): ").strip()
    meta = meta_input if meta_input else None  # Will be set based on PDK

    iterations_input = input("Number of iterations (default: 100000): ").strip()
    num_iterations = iterations_input if iterations_input else '100000'

    return {
        'layer_length': layer_length,
        'inner_step': inner_step,
        'innerdiv': innerdiv,
        'meta': meta,
        'num_iterations': num_iterations
    }

def build_asap7_command(model_framework, common_params, asap7_params, model_params):
    """Build the ASAP7 voltage variation command"""
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'ASAP7_voltage_variation_validation.py')

    cmd = [sys.executable, script_path]
    cmd.extend(['--mode', common_params['mode']])
    cmd.extend(['--corner', asap7_params['corner']])
    cmd.extend(['--cell_type', asap7_params['cell_type']])
    cmd.extend(['--data_type', common_params['data_type']])
    cmd.extend(['--gpu_id', common_params['gpu_id']])
    cmd.extend(['--model_framework', model_framework])
    cmd.extend(['--num_test_samples', common_params['num_test_samples']])

    if model_framework == 'mlp':
        cmd.extend(['--model_type', model_params['model_type']])
        cmd.extend(['--num_iterations', model_params['num_iterations']])
    else:  # maml
        cmd.extend(['--layer_length', model_params['layer_length']])
        # Set default inner_step if not provided
        inner_step = model_params['inner_step'] if model_params['inner_step'] else '3'
        cmd.extend(['--inner_step', inner_step])
        cmd.extend(['--innerdiv', model_params['innerdiv']])
        # Set default meta if not provided
        meta = model_params['meta'] if model_params['meta'] else '16'
        cmd.extend(['--meta', meta])
        cmd.extend(['--num_iterations', model_params['num_iterations']])

    if common_params['indices']:
        cmd.extend(['--indices'] + common_params['indices'])

    if common_params['save_results']:
        cmd.append('--save_results')

    return cmd

def build_tsmc_command(model_framework, common_params, tsmc_params, model_params):
    """Build the TSMC voltage variation command"""
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'TSMC_voltage_variation_validation.py')

    cmd = [sys.executable, script_path]
    cmd.extend(['--mode', common_params['mode']])
    cmd.extend(['--corner', tsmc_params['corner']])
    cmd.extend(['--temperatures'] + tsmc_params['temperatures'])
    cmd.extend(['--data_type', common_params['data_type']])
    cmd.extend(['--gpu_id', common_params['gpu_id']])
    cmd.extend(['--model_framework', model_framework])
    cmd.extend(['--num_test_samples', common_params['num_test_samples']])

    if model_framework == 'mlp':
        cmd.extend(['--model_type', model_params['model_type']])
        cmd.extend(['--num_iterations', model_params['num_iterations']])
    else:  # maml
        cmd.extend(['--layer_length', model_params['layer_length']])
        # Set default inner_step if not provided
        inner_step = model_params['inner_step'] if model_params['inner_step'] else '1'
        cmd.extend(['--inner_step', inner_step])
        cmd.extend(['--innerdiv', model_params['innerdiv']])
        # Set default meta if not provided
        meta = model_params['meta'] if model_params['meta'] else '32'
        cmd.extend(['--meta', meta])
        cmd.extend(['--num_iterations', model_params['num_iterations']])

    if common_params['indices']:
        cmd.extend(['--indices'] + common_params['indices'])

    if common_params['save_results']:
        cmd.append('--save_results')

    return cmd

def confirm_execution(pdk, model_framework, common_params, pdk_params, model_params, cmd, auto_confirm=False):
    """Show summary and confirm execution"""
    print("\n" + "="*80)
    print(" "*30 + "EXECUTION SUMMARY")
    print("="*80)
    print(f"\nPDK: {pdk.upper()}")
    print(f"Model Framework: {model_framework.upper()}")
    print(f"Script: {pdk.upper()}_voltage_variation_validation.py")

    print(f"\nCommon Parameters:")
    print(f"  Mode: {common_params['mode']}")
    print(f"  Data type: {common_params['data_type']}")
    print(f"  GPU ID: {common_params['gpu_id']}")
    print(f"  Test samples: {common_params['num_test_samples']}")

    if pdk == 'asap7':
        print(f"\nASAP7 Parameters:")
        print(f"  Corner: {pdk_params['corner']}")
        print(f"  Cell type: {pdk_params['cell_type']}")
    else:  # tsmc
        print(f"\nTSMC Parameters:")
        print(f"  Corner: {pdk_params['corner']}")
        print(f"  Temperatures: {', '.join(pdk_params['temperatures'])}°C")

    if model_framework == 'mlp':
        print(f"\nMLP Parameters:")
        print(f"  Model type: {model_params['model_type']}")
        print(f"  Iterations: {model_params['num_iterations']}")
    else:  # maml
        print(f"\nMAML Parameters:")
        print(f"  Layer length: {model_params['layer_length']}")
        print(f"  Inner steps: {model_params['inner_step'] or 'default'}")
        print(f"  Inner div: {model_params['innerdiv']}")
        print(f"  Meta batch: {model_params['meta'] or 'default'}")
        print(f"  Iterations: {model_params['num_iterations']}")

    if common_params['indices']:
        print(f"\n  Indices: {', '.join(common_params['indices'])}")
    else:
        print(f"\n  Indices: mode-dependent (auto)")
    print(f"  Save results: {'Yes' if common_params['save_results'] else 'No'}")

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
        description='Voltage Variation Testing Wrapper - Interactive or Command-line Mode',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Interactive mode:
    python run_voltage_variation_validation.py

  Command-line mode (ASAP7):
    python run_voltage_variation_validation.py --pdk asap7 --model mlp --corner FF --cell_type lvt
    python run_voltage_variation_validation.py --pdk asap7 --model maml --corner SS --cell_type rvt --inner_step 3

  Command-line mode (TSMC):
    python run_voltage_variation_validation.py --pdk tsmc --model mlp --corner ff --temperatures 0 25 50
    python run_voltage_variation_validation.py --pdk tsmc --model maml --corner tt --temperatures 25 --inner_step 1
        """
    )

    parser.add_argument('--pdk', type=str, choices=['asap7', 'tsmc'],
                        help='PDK: asap7 or tsmc')
    parser.add_argument('--model', type=str, choices=['mlp', 'maml'],
                        help='Model framework: mlp or maml')

    # Common parameters
    parser.add_argument('--mode', type=str, choices=['extrapolation', 'interpolation'],
                        help='Testing mode (default: extrapolation)')
    parser.add_argument('--data_type', type=str, choices=['cell', 'transition'],
                        help='Data type (default: cell)')
    parser.add_argument('--gpu_id', type=str,
                        help='GPU device ID (default: 4)')
    parser.add_argument('--num_test_samples', type=str,
                        help='Number of test samples (default: 100000)')
    parser.add_argument('--indices', type=str, nargs='+',
                        help='Support set indices (space-separated)')
    parser.add_argument('--save_results', action='store_true',
                        help='Save prediction results to .npy files')

    # ASAP7-specific parameters
    parser.add_argument('--corner', type=str,
                        help='Corner condition: SS/FF/TT (ASAP7) or ff/ss/tt (TSMC)')
    parser.add_argument('--cell_type', type=str, choices=['lvt', 'rvt', 'slvt', 'sram'],
                        help='[ASAP7] Cell type')

    # TSMC-specific parameters
    parser.add_argument('--temperatures', type=str, nargs='+',
                        help='[TSMC] Temperatures (0/25/50/75/100)')

    # MLP-specific parameters
    parser.add_argument('--model_type', type=str, choices=['aadam', 'mlp'],
                        help='[MLP] Model type: aadam (hidden=256) or mlp (hidden=40)')
    parser.add_argument('--num_iterations', type=str,
                        help='Number of iterations (default: 100000)')

    # MAML-specific parameters
    parser.add_argument('--layer_length', type=str,
                        help='[MAML] Hidden layer size (default: 40)')
    parser.add_argument('--inner_step', type=str,
                        help='[MAML] Inner loop steps (default: 3 for ASAP7, 1 for TSMC)')
    parser.add_argument('--innerdiv', type=str,
                        help='[MAML] Inner learning rate divisor (default: 10)')
    parser.add_argument('--meta', type=str,
                        help='[MAML] Meta batch size (default: 16 for ASAP7, 32 for TSMC)')

    parser.add_argument('--yes', '-y', action='store_true',
                        help='Skip confirmation prompt')

    return parser.parse_args()

def main_interactive():
    """Run interactive mode"""
    print_banner()

    # Step 1: Select PDK
    pdk = select_pdk()
    print(f"\nSelected PDK: {pdk.upper()}")

    # Step 2: Select model framework
    model_framework = select_model_framework()
    print(f"\nSelected model framework: {model_framework.upper()}")

    # Step 3: Get common parameters
    common_params = get_common_parameters()

    # Step 4: Get PDK-specific parameters
    if pdk == 'asap7':
        pdk_params = get_asap7_parameters()
    else:  # tsmc
        pdk_params = get_tsmc_parameters()

    # Step 5: Get model-specific parameters
    if model_framework == 'mlp':
        model_params = get_mlp_parameters()
    else:  # maml
        model_params = get_maml_parameters()

    # Step 6: Build command
    if pdk == 'asap7':
        cmd = build_asap7_command(model_framework, common_params, pdk_params, model_params)
    else:  # tsmc
        cmd = build_tsmc_command(model_framework, common_params, pdk_params, model_params)

    # Step 7: Confirm and execute
    if confirm_execution(pdk, model_framework, common_params, pdk_params, model_params, cmd, auto_confirm=False):
        return execute_command(cmd)
    else:
        print("\nExecution cancelled by user.")
        return 0

def main_commandline(args):
    """Run command-line mode"""
    print_banner()

    # Validate required arguments
    if args.pdk is None:
        print("Error: --pdk is required in command-line mode")
        print("Use -h for help or run without arguments for interactive mode")
        return 1

    if args.model is None:
        print("Error: --model is required in command-line mode")
        print("Use -h for help or run without arguments for interactive mode")
        return 1

    pdk = args.pdk
    model_framework = args.model

    print(f"PDK: {pdk.upper()}")
    print(f"Model Framework: {model_framework.upper()}")

    # Build common parameters
    common_params = {
        'mode': args.mode or 'extrapolation',
        'data_type': args.data_type or 'cell',
        'gpu_id': args.gpu_id or '4',
        'num_test_samples': args.num_test_samples or '100000',
        'indices': args.indices,
        'save_results': args.save_results
    }

    # Build PDK-specific parameters
    if pdk == 'asap7':
        if args.corner is None:
            print("Error: --corner is required for ASAP7")
            return 1
        if args.cell_type is None:
            print("Error: --cell_type is required for ASAP7")
            return 1
        pdk_params = {
            'corner': args.corner,
            'cell_type': args.cell_type
        }
    else:  # tsmc
        if args.corner is None:
            print("Error: --corner is required for TSMC")
            return 1
        pdk_params = {
            'corner': args.corner,
            'temperatures': args.temperatures or ['0', '25', '50', '75', '100']
        }

    # Build model-specific parameters
    if model_framework == 'mlp':
        model_params = {
            'model_type': args.model_type or 'aadam',
            'num_iterations': args.num_iterations or '100000'
        }
    else:  # maml
        model_params = {
            'layer_length': args.layer_length or '40',
            'inner_step': args.inner_step,
            'innerdiv': args.innerdiv or '10',
            'meta': args.meta,
            'num_iterations': args.num_iterations or '100000'
        }

    # Build command
    if pdk == 'asap7':
        cmd = build_asap7_command(model_framework, common_params, pdk_params, model_params)
    else:  # tsmc
        cmd = build_tsmc_command(model_framework, common_params, pdk_params, model_params)

    # Confirm and execute
    if confirm_execution(pdk, model_framework, common_params, pdk_params, model_params, cmd, auto_confirm=args.yes):
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
    if args.pdk is not None or args.model is not None:
        return main_commandline(args)
    else:
        return main_interactive()

if __name__ == "__main__":
    sys.exit(main())
