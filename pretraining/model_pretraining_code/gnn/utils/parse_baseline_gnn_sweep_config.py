#!/usr/bin/env python3
"""
Parse Baseline GNN sweep configuration JSON
Supports multi-parameter sweeps with cartesian product
Similar to parse_gnn_maml_sweep_config.py but for baseline training
"""

import json
import sys
import itertools


def parse_config(config_file):
    """Parse JSON config file"""
    with open(config_file, 'r') as f:
        return json.load(f)


def generate_sweep_combinations(sweep_params):
    """
    Generate all combinations of sweep parameters (cartesian product)

    Args:
        sweep_params: Dict of parameter names to list of values

    Returns:
        List of dicts, each representing one experiment configuration
    """
    param_names = list(sweep_params.keys())
    param_values = [sweep_params[name] for name in param_names]

    combinations = []
    for values in itertools.product(*param_values):
        combo = dict(zip(param_names, values))
        combinations.append(combo)

    return combinations


def format_value(value):
    """Format value for shell output"""
    if isinstance(value, bool):
        return "True" if value else "False"
    elif value is None:
        return ""
    else:
        return str(value)


def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_baseline_gnn_sweep_config.py <config.json>", file=sys.stderr)
        sys.exit(1)

    config_file = sys.argv[1]
    config = parse_config(config_file)

    experiment_name = config.get('experiment_name', 'baseline_gnn_sweep')
    base_config = config.get('base_config', {})
    sweep_params = config.get('sweep_params', {})

    # Output metadata
    print(f"EXPERIMENT_NAME={experiment_name}")

    # Output base config
    for key, value in base_config.items():
        upper_key = f"BASE_{key.upper()}"
        print(f"{upper_key}={format_value(value)}")

    # Generate sweep combinations
    combinations = generate_sweep_combinations(sweep_params)

    print(f"TOTAL_EXPERIMENTS={len(combinations)}", file=sys.stderr)

    # Output each experiment
    for idx, combo in enumerate(combinations, 1):
        exp_line_parts = [f"EXP_{idx}"]

        # Add sweep parameters
        for key, value in combo.items():
            exp_line_parts.append(f"{key}={format_value(value)}")

        print(" ".join(exp_line_parts))

    return 0


if __name__ == "__main__":
    sys.exit(main())
