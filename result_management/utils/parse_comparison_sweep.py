#!/usr/bin/env python3
"""
Parse comparison sweep configuration JSON and generate experiment combinations
"""

import json
import sys
import itertools


def expand_param(spec):
    """
    Expand parameter specification to list of values

    Args:
        spec: Either a list of values or a dict with min/max/step

    Returns:
        list of values
    """
    if isinstance(spec, list):
        return spec
    elif isinstance(spec, dict) and 'min' in spec and 'max' in spec:
        min_val = spec['min']
        max_val = spec['max']
        step = spec.get('step', 1)
        values = []
        current = min_val
        while current <= max_val:
            values.append(current)
            current += step
        return values
    else:
        return [spec]


def parse_sweep_config(config_file):
    """
    Parse sweep configuration and output shell variables

    Outputs:
        Shell variables that can be eval'd:
        - EXPERIMENT_NAME
        - BASE_* variables for base config
        - MODE (sweep or fixed)
        - EXP_* lines for each experiment
    """
    with open(config_file, 'r') as f:
        config = json.load(f)

    # Base config
    base_config = config.get('base_config', {})
    experiment_name = config.get('experiment_name', 'comparison_sweep')

    print(f"EXPERIMENT_NAME={experiment_name}")
    for key, value in base_config.items():
        if value is not None:
            # Convert boolean to lowercase string for shell
            if isinstance(value, bool):
                value = str(value).lower()
            # Convert list to space-separated string for shell
            elif isinstance(value, list):
                value = ' '.join(map(str, value))
            print(f"BASE_{key.upper()}={value}")

    # Check mode
    if 'sweep_params' in config:
        print("MODE=sweep")
        sweep_params = config['sweep_params']

        # Expand parameters
        param_names = list(sweep_params.keys())
        param_values = []
        for name in param_names:
            expanded = expand_param(sweep_params[name])
            param_values.append(expanded)
            print(f"PARAM_{name.upper()}=({' '.join(map(str, expanded))})")

        # Generate combinations
        print("COMBINATIONS_START")
        for i, combo in enumerate(itertools.product(*param_values), 1):
            cmd_parts = []
            for param_name, param_value in zip(param_names, combo):
                # Handle list parameters
                if isinstance(param_value, list):
                    # Convert list to comma-separated string
                    value_str = ",".join(str(v) for v in param_value)
                    cmd_parts.append(f"{param_name}={value_str}")
                else:
                    cmd_parts.append(f"{param_name}={param_value}")
            print(f"EXP_{i}:" + " ".join(cmd_parts))
        print("COMBINATIONS_END")

    elif 'fixed_experiments' in config:
        print("MODE=fixed")
        fixed_exps = config['fixed_experiments']
        print(f"NUM_FIXED={len(fixed_exps)}")

        for i, exp_config in enumerate(fixed_exps, 1):
            cmd_parts = []
            for key, value in exp_config.items():
                cmd_parts.append(f"{key}={value}")
            print(f"EXP_{i}:" + " ".join(cmd_parts))
    else:
        print("ERROR: No sweep_params or fixed_experiments found")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: parse_comparison_sweep.py <config.json>", file=sys.stderr)
        sys.exit(1)

    config_file = sys.argv[1]
    parse_sweep_config(config_file)
