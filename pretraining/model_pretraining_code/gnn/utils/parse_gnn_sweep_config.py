#!/usr/bin/env python3
"""
Parse GNN architecture sweep configuration JSON and generate training commands
"""

import json
import argparse
import sys
from pathlib import Path


def parse_sweep_config(config_file):
    """
    Parse sweep configuration JSON file

    Returns:
        dict: Parsed configuration with common_params and sweep_configs
    """
    with open(config_file, 'r') as f:
        config = json.load(f)

    return config


def generate_training_command(common_params, sweep_params, script_name='maml_gnn_training_cached.py'):
    """
    Generate training command from parameters

    Args:
        common_params: Common parameters dict
        sweep_params: Sweep-specific parameters dict
        script_name: Training script name

    Returns:
        str: Complete command string
    """
    cmd_parts = [f'python {script_name}']

    # Merge common and sweep parameters
    all_params = {**common_params, **sweep_params}

    # Build command arguments
    for key, value in all_params.items():
        if isinstance(value, bool):
            if value:
                cmd_parts.append(f'--{key}')
        elif isinstance(value, (int, float, str)):
            cmd_parts.append(f'--{key} {value}')

    return ' '.join(cmd_parts)


def main():
    parser = argparse.ArgumentParser(
        description='Parse GNN sweep config and generate training commands',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--config', type=str, required=True,
                       help='Path to sweep configuration JSON file')
    parser.add_argument('--script', type=str, default='maml_gnn_training_cached.py',
                       help='Training script name (default: maml_gnn_training_cached.py)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output shell script file (default: stdout)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Print commands without executing')

    args = parser.parse_args()

    # Parse configuration
    config = parse_sweep_config(args.config)

    print(f"=" * 80)
    print(f"GNN ARCHITECTURE SWEEP CONFIGURATION")
    print(f"=" * 80)
    print(f"Description: {config.get('sweep_description', 'N/A')}")
    print(f"Sweep type: {config.get('sweep_type', 'N/A')}")
    print(f"Varying parameter: {config.get('vary_parameter', 'N/A')}")
    print(f"Number of configs: {len(config['sweep_configs'])}")
    print(f"=" * 80)
    print()

    # Generate commands
    commands = []
    for sweep_config in config['sweep_configs']:
        config_id = sweep_config['id']
        description = sweep_config['description']
        sweep_params = sweep_config['parameters']

        cmd = generate_training_command(
            config['common_parameters'],
            sweep_params,
            args.script
        )

        commands.append({
            'id': config_id,
            'description': description,
            'command': cmd,
            'params': sweep_params
        })

    # Output commands
    if args.output:
        # Write to shell script
        with open(args.output, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(f"# GNN Architecture Sweep: {config.get('sweep_description', '')}\n")
            f.write(f"# Generated from: {args.config}\n")
            f.write(f"# Total configurations: {len(commands)}\n\n")

            for cmd_info in commands:
                f.write(f"# Config {cmd_info['id']}: {cmd_info['description']}\n")
                f.write(f"echo 'Starting config {cmd_info['id']}: {cmd_info['description']}'\n")
                f.write(f"{cmd_info['command']}\n\n")

        print(f"✅ Shell script written to: {args.output}")
        print(f"   Run with: bash {args.output}")

    else:
        # Print to stdout
        print("Generated commands:")
        print()
        for cmd_info in commands:
            print(f"# Config {cmd_info['id']}: {cmd_info['description']}")
            print(f"{cmd_info['command']}")
            print()

    # Print summary
    print(f"\n{'=' * 80}")
    print(f"SWEEP SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total configurations: {len(commands)}\n")

    # Group by varying parameter
    vary_param = config.get('vary_parameter', 'unknown')
    param_values = {}
    for cmd_info in commands:
        if vary_param in cmd_info['params']:
            val = cmd_info['params'][vary_param]
            if val not in param_values:
                param_values[val] = []
            param_values[val].append(cmd_info['id'])

    if param_values:
        print(f"Varying parameter: {vary_param}")
        for val, ids in sorted(param_values.items()):
            print(f"  {vary_param}={val}: configs {ids}")

    print(f"{'=' * 80}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
