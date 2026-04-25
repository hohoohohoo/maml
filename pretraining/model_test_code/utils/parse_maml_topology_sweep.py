#!/usr/bin/env python3
"""
Parse MAML topology validation sweep JSON configuration and generate experiment commands.

This script reads a JSON configuration file with base config and sweep parameters,
generates all parameter combinations, and outputs shell commands.
"""

import sys
import json
import itertools
from typing import Dict, List, Any


def parse_sweep_config(config_file: str) -> List[Dict[str, Any]]:
    """
    Parse sweep configuration file and generate all experiment combinations.

    Args:
        config_file: Path to JSON sweep configuration file

    Returns:
        List of experiment configurations
    """
    with open(config_file, 'r') as f:
        config = json.load(f)

    base_config = config.get('base_config', {})
    sweep_params = config.get('sweep_params', {})

    if not sweep_params:
        print("Error: No sweep_params defined in configuration", file=sys.stderr)
        sys.exit(1)

    # Get all parameter names and their value lists
    param_names = list(sweep_params.keys())
    param_values = [sweep_params[name] for name in param_names]

    # Generate all combinations using itertools.product
    combinations = list(itertools.product(*param_values))

    # Build experiment configs
    experiments = []
    for combo in combinations:
        exp_config = base_config.copy()

        # Add sweep parameters
        for param_name, param_value in zip(param_names, combo):
            exp_config[param_name] = param_value

        experiments.append(exp_config)

    return experiments


def config_to_args(config: Dict[str, Any]) -> str:
    """
    Convert configuration dictionary to command-line arguments.

    Args:
        config: Configuration dictionary

    Returns:
        Command-line argument string
    """
    args = []

    for key, value in config.items():
        if value is None:
            continue

        if isinstance(value, bool):
            if value:
                args.append(f"--{key}")
        elif isinstance(value, list):
            args.append(f"--{key}")
            args.extend([str(v) for v in value])
        else:
            args.append(f"--{key} {value}")

    return ' '.join(args)


def main():
    if len(sys.argv) != 2:
        print("Usage: parse_maml_topology_sweep.py <sweep_config.json>", file=sys.stderr)
        sys.exit(1)

    config_file = sys.argv[1]

    try:
        experiments = parse_sweep_config(config_file)
    except Exception as e:
        print(f"Error parsing configuration: {e}", file=sys.stderr)
        sys.exit(1)

    # Output experiments
    for i, exp_config in enumerate(experiments, 1):
        args_str = config_to_args(exp_config)

        # Create compact summary for display
        config_id = exp_config.get('config', 'N/A')
        mode = exp_config.get('mode', 'N/A')
        data_type = exp_config.get('data_type', 'N/A')

        print(f"EXPERIMENT {i}: config={config_id} mode={mode} data_type={data_type}: {args_str}")


if __name__ == "__main__":
    main()
