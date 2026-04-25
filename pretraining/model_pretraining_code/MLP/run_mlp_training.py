#!/usr/bin/env python
# coding: utf-8

"""
MLP Training Wrapper Script

Wrapper script for MLP pretraining supporting both Baseline MLP and MAML.
Supports both interactive and command-line modes with model-specific arguments.
"""

import os
import sys
import subprocess
import argparse

# Import configuration (utils is in parent directory)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.dataset_config import get_dataset_config

def print_banner():
    """Print welcome banner"""
    print("\n" + "="*80)
    print(" "*20 + "MLP Training Wrapper")
    print("="*80 + "\n")

def select_model_framework():
    """Let user select model framework"""
    print("Available model frameworks:")
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

def select_dataset_config():
    """Let user select dataset configuration"""
    print("\nAvailable dataset configurations:")
    print("-" * 80)
    print("  [0] ASAP7 Intra Topology")
    print("  [1] ASAP7 Technology Agnostic")
    print("  [2] TSMC Intra Topology")
    print("  [3] TSMC Technology Agnostic")
    print()

    while True:
        choice = input("Select dataset configuration [0/1/2/3]: ").strip()
        if choice in ['0', '1', '2', '3']:
            return int(choice)
        print("Invalid choice. Please select 0, 1, 2, or 3.\n")

def get_config_name(config_id):
    """Get configuration name from ID"""
    config_names = {
        0: 'ASAP7 Intra Topology',
        1: 'ASAP7 Technology Agnostic',
        2: 'TSMC Intra Topology',
        3: 'TSMC Technology Agnostic'
    }
    return config_names.get(config_id, f'Config {config_id}')

def get_maml_parameters(config):
    """Get MAML-specific parameters from user"""
    print("\n" + "-"*80)
    print("MAML parameters (press Enter for defaults):")
    print("-"*80)

    data_type_input = input(f"Data type [cell/transition] (default: cell): ").strip().lower()
    data_type = data_type_input if data_type_input in ['cell', 'transition'] else 'cell'

    gpu_input = input(f"GPU ID (default: 0): ").strip()
    gpu_id = gpu_input if gpu_input else '0'

    inner_input = input(f"Inner loop steps (default: 1): ").strip()
    inner = inner_input if inner_input else '1'

    innerdiv_input = input(f"Inner learning rate divisor (default: 100): ").strip()
    innerdiv = innerdiv_input if innerdiv_input else '100'

    meta_input = input(f"Meta batch size (default: 32): ").strip()
    meta = meta_input if meta_input else '32'

    iterations_input = input(f"Number of iterations (default: 300000): ").strip()
    num_iterations = iterations_input if iterations_input else '300000'

    resume_input = input(f"Resume from model? [y/N]: ").strip().lower()
    resume = None
    auto_resume = False

    if resume_input in ['y', 'yes']:
        resume_mode = input(f"  [1] Auto-resume (find latest model)\n  [2] Manual path\nSelect [1/2]: ").strip()
        if resume_mode == '1':
            auto_resume = True
        elif resume_mode == '2':
            manual_path = input(f"  Enter model path: ").strip()
            resume = manual_path if manual_path else None

    return {
        'data_type': data_type,
        'gpu_id': gpu_id,
        'inner': inner,
        'innerdiv': innerdiv,
        'meta': meta,
        'num_iterations': num_iterations,
        'resume': resume,
        'auto_resume': auto_resume
    }

def get_mlp_parameters(config):
    """Get MLP-specific parameters from user"""
    print("\n" + "-"*80)
    print("MLP parameters (press Enter for defaults):")
    print("-"*80)

    data_type_input = input(f"Data type [cell/transition] (default: cell): ").strip().lower()
    data_type = data_type_input if data_type_input in ['cell', 'transition'] else 'cell'

    model_type_input = input(f"Model type [aadam/mlp] (default: aadam): ").strip().lower()
    model_type = model_type_input if model_type_input in ['aadam', 'mlp'] else 'aadam'

    gpu_input = input(f"GPU ID (default: 0): ").strip()
    gpu_id = gpu_input if gpu_input else '0'

    iterations_input = input(f"Number of iterations (default: 300000): ").strip()
    num_iterations = iterations_input if iterations_input else '300000'

    lr_input = input(f"Learning rate (default: 1e-4): ").strip()
    learning_rate = lr_input if lr_input else '1e-4'

    return {
        'data_type': data_type,
        'model_type': model_type,
        'gpu_id': gpu_id,
        'num_iterations': num_iterations,
        'learning_rate': learning_rate
    }

def parse_args():
    """Parse command-line arguments with model-specific options"""
    parser = argparse.ArgumentParser(
        description='MLP Training Wrapper',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Create subparsers for each model
    subparsers = parser.add_subparsers(dest='model', help='Model framework selection')

    # MLP subparser
    mlp_parser = subparsers.add_parser('mlp', help='MLP pretraining',
                                       formatter_class=argparse.RawDescriptionHelpFormatter,
                                       epilog="""
Examples:
  python run_mlp_training.py mlp --config 0 --data_type cell
  python run_mlp_training.py mlp --config 1 --model_type mlp --num_iterations 100000
""")

    # MLP required arguments
    mlp_parser.add_argument('--config', type=int, required=True, choices=[0, 1, 2, 3],
                           help='Dataset config: 0=ASAP7 intra, 1=ASAP7 agnostic, 2=TSMC intra, 3=TSMC agnostic')

    # MLP common parameters
    mlp_parser.add_argument('--data_type', type=str, default='cell', choices=['cell', 'transition'],
                           help='Data type (default: cell)')
    mlp_parser.add_argument('--gpu_id', type=str, default='0',
                           help='GPU device ID (default: 0)')

    # MLP-specific parameters
    mlp_parser.add_argument('--model_type', type=str, default='aadam', choices=['aadam', 'mlp'],
                           help='Model type: aadam (hidden=256) or mlp (hidden=40) (default: aadam)')
    mlp_parser.add_argument('--num_iterations', type=int, default=300000,
                           help='Number of iterations (default: 300000)')
    mlp_parser.add_argument('--learning_rate', type=str, default='1e-4',
                           help='Learning rate (default: 1e-4)')

    # MAML subparser
    maml_parser = subparsers.add_parser('maml', help='MAML pretraining',
                                        formatter_class=argparse.RawDescriptionHelpFormatter,
                                        epilog="""
Examples:
  python run_mlp_training.py maml --config 0 --data_type transition
  python run_mlp_training.py maml --config 2 --inner 2 --meta 64 --auto_resume
""")

    # MAML required arguments
    maml_parser.add_argument('--config', type=int, required=True, choices=[0, 1, 2, 3],
                            help='Dataset config: 0=ASAP7 intra, 1=ASAP7 agnostic, 2=TSMC intra, 3=TSMC agnostic')

    # MAML common parameters
    maml_parser.add_argument('--data_type', type=str, default='cell', choices=['cell', 'transition'],
                            help='Data type (default: cell)')
    maml_parser.add_argument('--gpu', type=str, default='0',
                            help='GPU device ID (default: 0)')

    # MAML-specific parameters
    maml_parser.add_argument('--inner', type=int, default=1,
                            help='Inner loop steps (default: 1)')
    maml_parser.add_argument('--innerdiv', type=int, default=100,
                            help='Inner learning rate divisor (default: 100)')
    maml_parser.add_argument('--meta', type=int, default=32,
                            help='Meta batch size (default: 32)')
    maml_parser.add_argument('--num_iterations', type=int, default=300000,
                            help='Number of training iterations (default: 300000)')
    maml_parser.add_argument('--auto_resume', action='store_true',
                            help='Auto-find and resume from latest model')
    maml_parser.add_argument('--resume', type=str,
                            help='Resume from specific model path')

    # Common arguments for all subparsers
    for subparser in [mlp_parser, maml_parser]:
        subparser.add_argument('--yes', '-y', action='store_true',
                              help='Skip confirmation prompt')

    return parser.parse_args()

def build_maml_command(config_id, params):
    """Build the MAML command to execute"""
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'maml_mlp_training.py')

    cmd = [sys.executable, script_path]
    cmd.extend(['--dataset_config', str(config_id)])
    cmd.extend(['--data_type', params['data_type']])
    cmd.extend(['--gpu', params['gpu_id']])
    cmd.extend(['--inner', params['inner']])
    cmd.extend(['--innerdiv', params['innerdiv']])
    cmd.extend(['--meta', params['meta']])
    cmd.extend(['--num_iterations', params['num_iterations']])

    if params['auto_resume']:
        cmd.append('--auto_resume')
    elif params['resume']:
        cmd.extend(['--resume', params['resume']])

    return cmd

def build_mlp_command(config_id, params):
    """Build the MLP command to execute"""
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'baseline_mlp_training.py')

    cmd = [sys.executable, script_path]
    cmd.extend(['--dataset_config', str(config_id)])
    cmd.extend(['--data_type', params['data_type']])
    cmd.extend(['--gpu_id', params['gpu_id']])
    cmd.extend(['--model_type', params['model_type']])
    cmd.extend(['--num_iterations', params['num_iterations']])
    cmd.extend(['--learning_rate', params['learning_rate']])

    return cmd

def confirm_execution(model_type, config_id, params, cmd, auto_confirm=False):
    """Show summary and confirm execution"""
    print("\n" + "="*80)
    print(" "*30 + "EXECUTION SUMMARY")
    print("="*80)
    print(f"\nModel Type: {model_type.upper()}")
    print(f"Configuration: {get_config_name(config_id)}")
    print(f"Script: {'maml_mlp_training.py' if model_type == 'maml' else 'baseline_mlp_training.py'}")

    print(f"\nParameters:")
    print(f"  Data type: {params['data_type']}")
    print(f"  GPU ID: {params['gpu_id']}")

    if model_type == 'maml':
        print(f"  Inner steps: {params['inner']}")
        print(f"  Inner LR divisor: {params['innerdiv']}")
        print(f"  Meta batch size: {params['meta']}")
        print(f"  Iterations: {params['num_iterations']}")
        if params.get('auto_resume'):
            print(f"  Resume: Auto-resume (find latest model)")
        elif params.get('resume'):
            print(f"  Resume: {params['resume']}")
        else:
            print(f"  Resume: No")
    else:  # mlp
        print(f"  Model type: {params['model_type']}")
        print(f"  Iterations: {params['num_iterations']}")
        print(f"  Learning rate: {params['learning_rate']}")

    print(f"\nCommand to execute:")
    print(f"  {' '.join(cmd)}")
    print("="*80)

    if auto_confirm:
        return True

    confirm = input("\nProceed with execution? [Y/n]: ").strip().lower()
    return confirm not in ['n', 'no']

def execute_command(cmd):
    """Execute the model-specific script"""
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

    # Step 1: Select model framework
    model_framework = select_model_framework()
    print(f"\nSelected model framework: {model_framework.upper()}")

    # Step 2: Select dataset configuration
    config_id = select_dataset_config()
    config = get_dataset_config(config_id)
    print(f"\nSelected configuration: [{config_id}] {get_config_name(config_id)}")

    # Step 3: Get parameters
    if model_framework == 'maml':
        params = get_maml_parameters(config)
        cmd = build_maml_command(config_id, params)
    else:  # mlp
        params = get_mlp_parameters(config)
        cmd = build_mlp_command(config_id, params)

    # Step 4: Confirm and execute
    if confirm_execution(model_framework, config_id, params, cmd, auto_confirm=False):
        return execute_command(cmd)
    else:
        print("\nExecution cancelled by user.")
        return 0

def main_commandline(args):
    """Run command-line mode"""
    print_banner()

    model = args.model
    config_id = args.config

    print(f"Model: {model.upper()}")
    print(f"Dataset Configuration: [{config_id}] {get_config_name(config_id)}")

    # Build parameters based on model type
    if model == 'maml':
        params = {
            'data_type': args.data_type,
            'gpu_id': args.gpu,
            'inner': str(args.inner),
            'innerdiv': str(args.innerdiv),
            'meta': str(args.meta),
            'num_iterations': str(args.num_iterations),
            'resume': args.resume,
            'auto_resume': args.auto_resume
        }
        cmd = build_maml_command(config_id, params)
    else:  # mlp
        params = {
            'data_type': args.data_type,
            'gpu_id': args.gpu_id,
            'model_type': args.model_type,
            'num_iterations': str(args.num_iterations),
            'learning_rate': args.learning_rate
        }
        cmd = build_mlp_command(config_id, params)

    # Confirm and execute
    auto_confirm = hasattr(args, 'yes') and args.yes
    if confirm_execution(model, config_id, params, cmd, auto_confirm=auto_confirm):
        return execute_command(cmd)
    else:
        print("\nExecution cancelled.")
        return 0

def main():
    """Main function - determines interactive or command-line mode"""
    # Check if any command-line arguments were provided
    if len(sys.argv) > 1:
        args = parse_args()
        if args.model is None:
            print("Error: Model not specified. Use 'mlp' or 'maml' as the first argument.")
            print("\nUsage:")
            print("  python run_mlp_training.py mlp --config 0 --data_type cell")
            print("  python run_mlp_training.py maml --config 2 --inner 2 --meta 64")
            print("\nFor help:")
            print("  python run_mlp_training.py mlp -h")
            print("  python run_mlp_training.py maml -h")
            return 1
        return main_commandline(args)
    else:
        return main_interactive()

if __name__ == "__main__":
    sys.exit(main())
