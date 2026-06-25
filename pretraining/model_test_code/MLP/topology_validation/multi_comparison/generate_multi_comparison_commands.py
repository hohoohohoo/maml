#!/usr/bin/env python3
"""
Generate MAML multi-comparison commands from JSON config
"""
import json
import sys
from itertools import product

def generate_commands(config_file):
    """Generate comparison commands from JSON config"""
    with open(config_file, 'r') as f:
        config = json.load(f)

    experiment_name = config['experiment_name']
    base_config = config['base_config']
    sweep_params = config['sweep_params']
    mlp_config = config.get('mlp_comparison', {})

    # Output mode
    print("MODE=multi_comparison")
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

    # MLP configuration
    if mlp_config.get('enabled', False):
        print(f"MLP_ENABLED=true")
        print(f"MLP_MODEL_TYPE={mlp_config.get('mlp_model_type', 'aadam')}")
        print(f"MLP_ITERATIONS={mlp_config.get('mlp_iterations', 300000)}")
    else:
        print(f"MLP_ENABLED=false")
    print()

    # Generate experiments
    data_types = sweep_params.get('data_type', ['transition'])
    vary_params = sweep_params.get('vary', ['innerdiv'])
    modes = sweep_params.get('mode', ['interpolation'])  # Add mode support
    comparison_configs = sweep_params.get('comparison_configs', [])

    exp_num = 1

    for data_type in data_types:
        for mode in modes:  # Add mode loop
            for vary in vary_params:
                for comp_config in comparison_configs:
                    # Extract parameter lists
                    innerdiv_list = comp_config.get('innerdiv', [100])
                    meta_list = comp_config.get('meta', [32])
                    num_iterations_list = comp_config.get('num_iterations', [300000])
                    layer_length_list = comp_config.get('layer_length', [40])

                    # Validate based on vary parameter
                    if vary == 'innerdiv':
                        if len(innerdiv_list) < 2 or len(innerdiv_list) > 5:
                            print(f"# Skipping: innerdiv list must have 2-5 values", file=sys.stderr)
                            continue
                        if len(meta_list) != 1 or len(num_iterations_list) != 1 or len(layer_length_list) != 1:
                            print(f"# Skipping: when varying innerdiv, meta, num_iterations, and layer_length must be single values", file=sys.stderr)
                            continue
                    elif vary == 'meta':
                        if len(meta_list) < 2 or len(meta_list) > 5:
                            print(f"# Skipping: meta list must have 2-5 values", file=sys.stderr)
                            continue
                        if len(innerdiv_list) != 1 or len(num_iterations_list) != 1 or len(layer_length_list) != 1:
                            print(f"# Skipping: when varying meta, innerdiv, num_iterations, and layer_length must be single values", file=sys.stderr)
                            continue
                    elif vary == 'num_iterations':
                        if len(num_iterations_list) < 2 or len(num_iterations_list) > 5:
                            print(f"# Skipping: num_iterations list must have 2-5 values", file=sys.stderr)
                            continue
                        if len(innerdiv_list) != 1 or len(meta_list) != 1 or len(layer_length_list) != 1:
                            print(f"# Skipping: when varying num_iterations, innerdiv, meta, and layer_length must be single values", file=sys.stderr)
                            continue
                    elif vary == 'layer_length':
                        if len(layer_length_list) < 2 or len(layer_length_list) > 5:
                            print(f"# Skipping: layer_length list must have 2-5 values", file=sys.stderr)
                            continue
                        if len(innerdiv_list) != 1 or len(meta_list) != 1 or len(num_iterations_list) != 1:
                            print(f"# Skipping: when varying layer_length, innerdiv, meta, and num_iterations must be single values", file=sys.stderr)
                            continue

                    # Build experiment command
                    cmd_parts = []
                    cmd_parts.append(f"--data_type {data_type}")
                    cmd_parts.append(f"--mode {mode}")  # Add mode to command
                    cmd_parts.append(f"--vary {vary}")

                    # Add parameter lists
                    innerdiv_str = ' '.join(map(str, innerdiv_list))
                    meta_str = ' '.join(map(str, meta_list))
                    iterations_str = ' '.join(map(str, num_iterations_list))
                    layer_str = ' '.join(map(str, layer_length_list))

                    cmd_parts.append(f"--innerdiv {innerdiv_str}")
                    cmd_parts.append(f"--meta {meta_str}")
                    cmd_parts.append(f"--num_iterations {iterations_str}")
                    cmd_parts.append(f"--layer_length {layer_str}")

                    # Create experiment label
                    if vary == 'innerdiv':
                        vary_values = '_'.join(map(str, innerdiv_list))
                        label = f"{data_type}_{mode}_innerdiv[{vary_values}]_meta{meta_list[0]}_iter{num_iterations_list[0]}_layer{layer_length_list[0]}"
                    elif vary == 'meta':
                        vary_values = '_'.join(map(str, meta_list))
                        label = f"{data_type}_{mode}_innerdiv{innerdiv_list[0]}_meta[{vary_values}]_iter{num_iterations_list[0]}_layer{layer_length_list[0]}"
                    elif vary == 'num_iterations':
                        vary_values = '_'.join(map(str, num_iterations_list))
                        label = f"{data_type}_{mode}_innerdiv{innerdiv_list[0]}_meta{meta_list[0]}_iter[{vary_values}]_layer{layer_length_list[0]}"
                    else:  # layer_length
                        vary_values = '_'.join(map(str, layer_length_list))
                        label = f"{data_type}_{mode}_innerdiv{innerdiv_list[0]}_meta{meta_list[0]}_iter{num_iterations_list[0]}_layer[{vary_values}]"

                    print(f"EXP_{exp_num}={' '.join(cmd_parts)}  # {label}")
                    exp_num += 1

    print()
    print(f"TOTAL_EXPERIMENTS={exp_num - 1}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generate_multi_comparison_commands.py <config.json>")
        sys.exit(1)

    generate_commands(sys.argv[1])
