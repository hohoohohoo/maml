#!/usr/bin/env python3
"""
Parse voltage variation sweep configuration JSON and generate experiment combinations
"""

import json
import sys
import itertools


def parse_voltage_sweep_config(config_file):
    """
    Parse voltage variation sweep configuration and output shell variables

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
    experiment_name = config.get('experiment_name', 'voltage_variation_sweep')

    print(f"EXPERIMENT_NAME={experiment_name}")
    for key, value in base_config.items():
        if value is not None:
            # Convert boolean to string for shell
            if isinstance(value, bool):
                value = str(value)
            print(f"BASE_{key.upper()}={value}")

    # Check mode
    if 'sweep_params' in config:
        print("MODE=sweep")
        sweep_params = config['sweep_params']
        pdk_specific = config.get('pdk_specific', {})
        model_params = config.get('model_params', {})

        # Get PDK list and data_type list from sweep_params
        pdks = sweep_params.get('pdk', [])
        data_types = sweep_params.get('data_type', [base_config.get('data_type', 'cell')])

        # If data_type is a string, convert to list
        if isinstance(data_types, str):
            data_types = [data_types]

        # Get model framework from base config
        model_framework = base_config.get('model_framework', 'maml')

        experiments = []

        for pdk in pdks:
            for data_type in data_types:
                # Get PDK-specific parameters
                pdk_config = pdk_specific.get(pdk, {})

                if pdk == 'asap7':
                    corners = pdk_config.get('corner', ['FF'])
                    cell_types = pdk_config.get('cell_type', ['lvt'])

                    # Get model-specific parameters for ASAP7
                    if model_framework == 'maml':
                        model_config = model_params.get('maml', {})
                        asap7_model = model_config.get('asap7', {})

                        layer_length = model_config.get('layer_length', 40)
                        innerdiv = model_config.get('innerdiv', 100)
                        # Check for ASAP7-specific num_iterations first, then fall back to global
                        num_iterations = asap7_model.get('num_iterations', model_config.get('num_iterations', 300000))
                        inner_step = asap7_model.get('inner_step', 3)
                        meta = asap7_model.get('meta', 16)

                        # Generate combinations for ASAP7
                        for corner in corners:
                            for cell_type in cell_types:
                                exp_params = {
                                    'pdk': pdk,
                                    'data_type': data_type,
                                    'corner': corner,
                                    'cell_type': cell_type,
                                    'layer_length': layer_length,
                                    'inner_step': inner_step,
                                    'innerdiv': innerdiv,
                                    'meta': meta,
                                    'num_iterations': num_iterations
                                }
                                experiments.append(exp_params)

                    elif model_framework == 'mlp':
                        model_config = model_params.get('mlp', {})
                        asap7_model = model_config.get('asap7', {})

                        model_type = model_config.get('model_type', 'aadam')
                        # Check for ASAP7-specific num_iterations first, then fall back to global
                        num_iterations = asap7_model.get('num_iterations', model_config.get('num_iterations', 300000))

                        # Generate combinations for ASAP7 MLP
                        for corner in corners:
                            for cell_type in cell_types:
                                exp_params = {
                                    'pdk': pdk,
                                    'data_type': data_type,
                                    'corner': corner,
                                    'cell_type': cell_type,
                                    'model_type': model_type,
                                    'num_iterations': num_iterations
                                }
                                experiments.append(exp_params)

                elif pdk == 'tsmc':
                    corners = pdk_config.get('corner', ['ff'])
                    temperatures_list = pdk_config.get('temperatures', [['0', '25', '50', '75', '100']])

                    # Flatten temperatures if nested list (for individual temperature sweep)
                    # If temperatures_list is [["0", "25", "50"]], flatten to ["0", "25", "50"]
                    if temperatures_list and isinstance(temperatures_list[0], list):
                        temperatures_flat = temperatures_list[0]
                    else:
                        temperatures_flat = temperatures_list

                    # Get model-specific parameters for TSMC
                    if model_framework == 'maml':
                        model_config = model_params.get('maml', {})
                        tsmc_model = model_config.get('tsmc', {})

                        layer_length = model_config.get('layer_length', 40)
                        innerdiv = model_config.get('innerdiv', 100)
                        # Check for TSMC-specific num_iterations first, then fall back to global
                        num_iterations = tsmc_model.get('num_iterations', model_config.get('num_iterations', 300000))
                        inner_step = tsmc_model.get('inner_step', 1)
                        meta = tsmc_model.get('meta', 32)

                        # Generate combinations for TSMC (sweep each temperature individually)
                        for corner in corners:
                            for temp in temperatures_flat:
                                exp_params = {
                                    'pdk': pdk,
                                    'data_type': data_type,
                                    'corner': corner,
                                    'temperatures': temp,
                                    'layer_length': layer_length,
                                    'inner_step': inner_step,
                                    'innerdiv': innerdiv,
                                    'meta': meta,
                                    'num_iterations': num_iterations
                                }
                                experiments.append(exp_params)

                    elif model_framework == 'mlp':
                        model_config = model_params.get('mlp', {})
                        tsmc_model = model_config.get('tsmc', {})

                        model_type = model_config.get('model_type', 'aadam')
                        # Check for TSMC-specific num_iterations first, then fall back to global
                        num_iterations = tsmc_model.get('num_iterations', model_config.get('num_iterations', 300000))

                        # Generate combinations for TSMC MLP (sweep each temperature individually)
                        for corner in corners:
                            for temp in temperatures_flat:
                                exp_params = {
                                    'pdk': pdk,
                                    'data_type': data_type,
                                    'corner': corner,
                                    'temperatures': temp,
                                    'model_type': model_type,
                                    'num_iterations': num_iterations
                                }
                                experiments.append(exp_params)

        # Output experiments
        print("COMBINATIONS_START")
        for i, exp_params in enumerate(experiments, 1):
            cmd_parts = []
            for param_name, param_value in exp_params.items():
                cmd_parts.append(f"{param_name}={param_value}")
            print(f"EXP_{i}:" + " ".join(cmd_parts))
        print("COMBINATIONS_END")

    else:
        print("ERROR: No sweep_params found")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: parse_voltage_variation_sweep.py <config.json>", file=sys.stderr)
        sys.exit(1)

    config_file = sys.argv[1]
    parse_voltage_sweep_config(config_file)
