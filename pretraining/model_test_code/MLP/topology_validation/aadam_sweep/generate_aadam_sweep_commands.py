#!/usr/bin/env python3
"""
Generate AADAM iteration sweep commands from JSON config
"""
import json
import sys

def generate_commands(config_file):
    """Generate AADAM sweep commands from JSON config"""
    with open(config_file, 'r') as f:
        config = json.load(f)

    experiment_name = config['experiment_name']
    base_config = config['base_config']
    sweep_params = config['sweep_params']

    # Output experiment info
    print("MODE=aadam_sweep")
    print(f"EXPERIMENT_NAME={experiment_name}")
    print()

    # Base configuration
    for key, value in base_config.items():
        env_key = f"BASE_{key.upper()}"
        if isinstance(value, bool):
            print(f"{env_key}={str(value)}")
        else:
            print(f"{env_key}={value}")
    print()

    # Generate experiments
    data_types = sweep_params.get('data_type', ['transition'])
    modes = sweep_params.get('mode', ['interpolation'])
    iterations_list = sweep_params.get('num_iterations', [300000])

    exp_num = 1

    for data_type in data_types:
        for mode in modes:
            for iterations in iterations_list:
                # Build experiment command
                cmd_parts = []
                cmd_parts.append(f"--data_type {data_type}")
                cmd_parts.append(f"--mode {mode}")
                cmd_parts.append(f"--num_iterations {iterations}")

                # Create experiment label
                label = f"{data_type}_{mode}_iter{iterations}"

                print(f"EXP_{exp_num}={' '.join(cmd_parts)}  # {label}")
                exp_num += 1

    print()
    print(f"TOTAL_EXPERIMENTS={exp_num - 1}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generate_aadam_sweep_commands.py <config.json>")
        sys.exit(1)

    generate_commands(sys.argv[1])
