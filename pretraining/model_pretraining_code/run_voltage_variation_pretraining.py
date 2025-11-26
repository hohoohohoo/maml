#!/usr/bin/env python
# coding: utf-8

"""
Voltage Variation Pretraining Wrapper Script

Top-level wrapper that routes to PDK-specific pretraining scripts.
Supports both interactive and command-line modes with PDK-specific arguments.
"""

import os
import sys
import subprocess
import argparse

def print_banner():
    """Print welcome banner"""
    print("\n" + "="*80)
    print(" "*20 + "Voltage Variation Pretraining Wrapper")
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

def get_common_parameters(pdk='asap7'):
    """Get common parameters for both PDKs"""
    print("\n" + "-"*80)
    print("Common parameters (press Enter for defaults):")
    print("-"*80)

    data_type_input = input("Data type [cell/transition] (default: cell): ").strip().lower()
    data_type = data_type_input if data_type_input in ['cell', 'transition'] else 'cell'

    gpu_input = input("GPU ID (default: 4): ").strip()
    gpu_id = gpu_input if gpu_input else '4'

    # PDK-specific default for num_iterations
    default_iterations = '100000' if pdk == 'asap7' else '30000'
    iterations_input = input(f"Number of training iterations (default: {default_iterations}): ").strip()
    num_iterations = iterations_input if iterations_input else default_iterations

    return {
        'data_type': data_type,
        'gpu_id': gpu_id,
        'num_iterations': num_iterations
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

    # Corner selection
    print("\nCorner conditions:")
    print("  [0] FF (Fast-Fast)")
    print("  [1] TT (Typical-Typical)")
    print("  [2] SS (Slow-Slow)")
    corner_input = input("Select corner [0-2] (default: 0 for FF): ").strip()
    corner_map = {'0': 'ff', '1': 'tt', '2': 'ss', '': 'ff'}
    corner = corner_map.get(corner_input, 'ff')

    # Temperature selection
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

    return {
        'model_type': model_type
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

    return {
        'layer_length': layer_length,
        'inner_step': inner_step,
        'innerdiv': innerdiv,
        'meta': meta
    }

def parse_args():
    """Parse command-line arguments with PDK-specific options"""
    parser = argparse.ArgumentParser(
        description='Voltage Variation Pretraining Wrapper',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Create subparsers for each PDK
    subparsers = parser.add_subparsers(dest='pdk', help='PDK selection')

    # ASAP7 subparser
    asap7_parser = subparsers.add_parser('asap7', help='ASAP7 PDK pretraining',
                                         formatter_class=argparse.RawDescriptionHelpFormatter,
                                         epilog="""
Examples:
  python run_voltage_variation_pretraining.py asap7 --model mlp --corner FF --cell_type lvt
  python run_voltage_variation_pretraining.py asap7 --model maml --corner SS --cell_type rvt --inner_step 5
""")

    # ASAP7 required arguments
    asap7_parser.add_argument('--model', type=str, required=True, choices=['mlp', 'maml'],
                             help='Model framework: mlp or maml')

    # ASAP7 common parameters
    asap7_parser.add_argument('--corner', type=str, default='FF', choices=['SS', 'FF', 'TT'],
                             help='Corner condition (default: FF)')
    asap7_parser.add_argument('--cell_type', type=str, default='lvt', choices=['lvt', 'rvt', 'slvt', 'sram'],
                             help='Cell type (default: lvt)')
    asap7_parser.add_argument('--data_type', type=str, default='cell', choices=['cell', 'transition'],
                             help='Data type (default: cell)')
    asap7_parser.add_argument('--gpu_id', type=str, default='4',
                             help='GPU device ID (default: 4)')
    asap7_parser.add_argument('--num_iterations', type=int, default=100000,
                             help='Number of training iterations (default: 100000)')

    # ASAP7 MLP-specific parameters
    asap7_parser.add_argument('--model_type', type=str, default='aadam', choices=['aadam', 'mlp'],
                             help='[MLP] Model type: aadam (hidden=256) or mlp (hidden=40) (default: aadam)')

    # ASAP7 MAML-specific parameters
    asap7_parser.add_argument('--layer_length', type=int, default=40,
                             help='[MAML] Hidden layer size (default: 40)')
    asap7_parser.add_argument('--inner_step', type=int, default=3,
                             help='[MAML] Inner loop steps (default: 3)')
    asap7_parser.add_argument('--innerdiv', type=int, default=10,
                             help='[MAML] Inner learning rate divisor (default: 10)')
    asap7_parser.add_argument('--meta', type=int, default=16,
                             help='[MAML] Meta batch size (default: 16)')

    # TSMC subparser
    tsmc_parser = subparsers.add_parser('tsmc', help='TSMC PDK pretraining',
                                        formatter_class=argparse.RawDescriptionHelpFormatter,
                                        epilog="""
Examples:
  python run_voltage_variation_pretraining.py tsmc --model mlp --temperatures 0 25 50 75 100
  python run_voltage_variation_pretraining.py tsmc --model maml --temperatures 25 50 --inner_step 2
""")

    # TSMC required arguments
    tsmc_parser.add_argument('--model', type=str, required=True, choices=['mlp', 'maml'],
                            help='Model framework: mlp or maml')

    # TSMC common parameters
    tsmc_parser.add_argument('--temperatures', type=int, nargs='+', default=[0, 25, 50, 75, 100],
                            help='List of temperatures to train (default: [0,25,50,75,100])')
    tsmc_parser.add_argument('--corner', type=str, default='ff', choices=['ss', 'ff', 'tt'],
                            help='Corner condition (default: ff)')
    tsmc_parser.add_argument('--data_type', type=str, default='transition', choices=['cell', 'transition'],
                            help='Data type (default: transition)')
    tsmc_parser.add_argument('--gpu_id', type=str, default='1',
                            help='GPU device ID (default: 1)')
    tsmc_parser.add_argument('--num_iterations', type=int, default=30000,
                            help='Number of training iterations (default: 30000)')

    # TSMC MLP-specific parameters
    tsmc_parser.add_argument('--model_type', type=str, default='aadam', choices=['aadam', 'mlp'],
                            help='[MLP] Model type: aadam (hidden=256) or mlp (hidden=40) (default: aadam)')

    # TSMC MAML-specific parameters
    tsmc_parser.add_argument('--layer_length', type=int, default=40,
                            help='[MAML] Hidden layer size (default: 40)')
    tsmc_parser.add_argument('--inner_step', type=int, default=1,
                            help='[MAML] Inner loop steps (default: 1)')
    tsmc_parser.add_argument('--innerdiv', type=int, default=10,
                            help='[MAML] Inner learning rate divisor (default: 10)')
    tsmc_parser.add_argument('--meta', type=int, default=32,
                            help='[MAML] Meta batch size (default: 32)')

    # Common arguments for all subparsers
    for subparser in [asap7_parser, tsmc_parser]:
        subparser.add_argument('--yes', '-y', action='store_true',
                              help='Skip confirmation prompt')

    return parser.parse_args()

def build_asap7_command(model_framework, common_params, asap7_params, model_params):
    """Build the ASAP7 voltage variation pretraining command"""
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'ASAP7_voltage_variation_pretraining.py')

    cmd = [sys.executable, script_path]
    cmd.extend(['--model', model_framework])
    cmd.extend(['--corner', asap7_params['corner']])
    cmd.extend(['--cell_type', asap7_params['cell_type']])
    cmd.extend(['--data_type', common_params['data_type']])
    cmd.extend(['--gpu_id', common_params['gpu_id']])
    cmd.extend(['--num_iterations', common_params['num_iterations']])

    if model_framework == 'mlp':
        cmd.extend(['--model_type', model_params['model_type']])
    else:  # maml
        cmd.extend(['--layer_length', model_params['layer_length']])
        # Set default inner_step if not provided
        inner_step = model_params['inner_step'] if model_params['inner_step'] else '3'
        cmd.extend(['--inner_step', inner_step])
        cmd.extend(['--innerdiv', model_params['innerdiv']])
        # Set default meta if not provided
        meta = model_params['meta'] if model_params['meta'] else '16'
        cmd.extend(['--meta', meta])

    return cmd

def build_tsmc_command(model_framework, common_params, tsmc_params, model_params):
    """Build the TSMC voltage variation pretraining command"""
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'TSMC_voltage_variation_pretraining.py')

    cmd = [sys.executable, script_path]
    cmd.extend(['--model', model_framework])
    cmd.extend(['--temperatures'] + tsmc_params['temperatures'])
    cmd.extend(['--corner', tsmc_params['corner']])  # Corner is now a common parameter
    cmd.extend(['--data_type', common_params['data_type']])
    cmd.extend(['--gpu_id', common_params['gpu_id']])
    cmd.extend(['--num_iterations', common_params['num_iterations']])

    if model_framework == 'mlp':
        cmd.extend(['--model_type', model_params['model_type']])
    else:  # maml
        cmd.extend(['--layer_length', model_params['layer_length']])
        # Set default inner_step if not provided
        inner_step = model_params['inner_step'] if model_params['inner_step'] else '1'
        cmd.extend(['--inner_step', inner_step])
        cmd.extend(['--innerdiv', model_params['innerdiv']])
        # Set default meta if not provided
        meta = model_params['meta'] if model_params['meta'] else '32'
        cmd.extend(['--meta', meta])

    return cmd

def confirm_execution(pdk, model_framework, common_params, pdk_params, model_params, cmd, auto_confirm=False):
    """Show summary and confirm execution"""
    print("\n" + "="*80)
    print(" "*30 + "EXECUTION SUMMARY")
    print("="*80)
    print(f"\nPDK: {pdk.upper()}")
    print(f"Model Framework: {model_framework.upper()}")
    print(f"Script: {pdk.upper()}_voltage_variation_pretraining.py")

    print(f"\nCommon Parameters:")
    print(f"  Data type: {common_params['data_type']}")
    print(f"  GPU ID: {common_params['gpu_id']}")
    print(f"  Training iterations: {common_params['num_iterations']}")

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
    else:  # maml
        print(f"\nMAML Parameters:")
        print(f"  Layer length: {model_params['layer_length']}")
        print(f"  Inner steps: {model_params['inner_step'] or 'default'}")
        print(f"  Inner div: {model_params['innerdiv']}")
        print(f"  Meta batch: {model_params['meta'] or 'default'}")

    print(f"\nCommand to execute:")
    print(f"  {' '.join(cmd)}")
    print("="*80)

    if auto_confirm:
        return True

    confirm = input("\nProceed with execution? [Y/n]: ").strip().lower()
    return confirm not in ['n', 'no']

def execute_command(cmd):
    """Execute the command"""
    print("\n" + "="*80)
    print(" "*30 + "STARTING TRAINING")
    print("="*80 + "\n")

    try:
        result = subprocess.run(cmd, check=True)

        print("\n" + "="*80)
        print(" "*30 + "TRAINING COMPLETED")
        print("="*80)

        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\n\nError: Command failed with exit code {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        return 130

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
    common_params = get_common_parameters(pdk)

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

    pdk = args.pdk
    model_framework = args.model

    print(f"PDK: {pdk.upper()}")
    print(f"Model Framework: {model_framework.upper()}")

    # Build common parameters
    common_params = {
        'data_type': args.data_type,
        'gpu_id': args.gpu_id,
        'num_iterations': str(args.num_iterations)
    }

    # Build PDK-specific parameters
    if pdk == 'asap7':
        pdk_params = {
            'corner': args.corner,
            'cell_type': args.cell_type
        }
    else:  # tsmc
        pdk_params = {
            'corner': args.corner,
            'temperatures': [str(t) for t in args.temperatures]
        }

    # Build model-specific parameters
    if model_framework == 'mlp':
        model_params = {
            'model_type': args.model_type
        }
    else:  # maml
        model_params = {
            'layer_length': str(args.layer_length),
            'inner_step': str(args.inner_step),
            'innerdiv': str(args.innerdiv),
            'meta': str(args.meta)
        }

    # Build command
    if pdk == 'asap7':
        cmd = build_asap7_command(model_framework, common_params, pdk_params, model_params)
    else:  # tsmc
        cmd = build_tsmc_command(model_framework, common_params, pdk_params, model_params)

    # Confirm and execute
    auto_confirm = hasattr(args, 'yes') and args.yes
    if confirm_execution(pdk, model_framework, common_params, pdk_params, model_params, cmd, auto_confirm=auto_confirm):
        return execute_command(cmd)
    else:
        print("\nExecution cancelled.")
        return 0

def main():
    """Main function - determines interactive or command-line mode"""
    # Check if any command-line arguments were provided
    if len(sys.argv) > 1:
        args = parse_args()
        if args.pdk is None:
            print("Error: PDK not specified. Use 'asap7' or 'tsmc' as the first argument.")
            print("\nUsage:")
            print("  python run_voltage_variation_pretraining.py asap7 --model mlp --corner FF --cell_type lvt")
            print("  python run_voltage_variation_pretraining.py tsmc --model maml --temperatures 0 25 50")
            print("\nFor help:")
            print("  python run_voltage_variation_pretraining.py asap7 -h")
            print("  python run_voltage_variation_pretraining.py tsmc -h")
            return 1
        if args.model is None:
            print("Error: Model not specified. Use --model mlp or --model maml")
            return 1
        return main_commandline(args)
    else:
        return main_interactive()

if __name__ == "__main__":
    sys.exit(main())
