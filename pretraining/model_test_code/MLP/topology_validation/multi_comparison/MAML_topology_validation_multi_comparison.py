#!/usr/bin/env python
# coding: utf-8

"""
MAML Multi-Model Comparison Script

Compare up to 3 MAML models varying one parameter (innerdiv, meta, or num_iterations)
on the same validation tasks in real-time.
"""

import os
import sys
import torch
import numpy as np
import random
import argparse

# Add parent directory to path for utils access
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import itertools

# Import configuration
from utils.test_dataset_config import (
    get_test_config,
    get_train_data_paths,
    get_test_data_paths,
    get_maml_model_path,
    get_mlp_model_path,
    print_available_configs
)

# Import utility functions
from utils.data_management_utils import (
    analyze_continuity,
    load_and_normalize_data,
    apply_normalization
)
from utils.maml_functions import evaluate_model_performance_maml
from utils.mlp_functions import evaluate_model_performance_mlp

# MAML import
sys.path.append('../../../../../model_code/')
from mlp_maml import OptimizedMAML, MAMLModel_3hidden
from baseline_mlp import MLP_Aadam, MLP


def load_maml_model(model_path, device, input_features=9, layer_length=40):
    """Load MAML model from path"""
    maml_model = OptimizedMAML(
        model=MAMLModel_3hidden(in_features=input_features, layer_length=layer_length),
        dataset_in=None,
        dataset_out=None,
        inner_lr=0.001,
        meta_lr=0.0001
    )

    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        maml_model.model.load_state_dict(state_dict)
        print(f"✅ Loaded model: {os.path.basename(model_path)}")
        return maml_model
    else:
        print(f"❌ Model file not found: {model_path}")
        return None


def load_mlp_model(model_path, model_type, device, input_features=9):
    """Load MLP model from path"""
    if model_type == 'aadam':
        mlp_model = MLP_Aadam(input_size=input_features, output_size=1).to(device)
    else:  # mlp
        mlp_model = MLP(input_size=input_features, output_size=1).to(device)

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        mlp_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded MLP model: {os.path.basename(model_path)}")
        return mlp_model
    else:
        print(f"❌ MLP model file not found: {model_path}")
        return None


def evaluate_single_task(model, test_data_input, test_data_output, randomtask,
                         indices, middle_idx, left_bound, right_bound, total_points,
                         mode, device, model_type='maml', layer_length=40,
                         adaptation_method='selective_adam'):
    """Evaluate a single task for a model (MAML or MLP)"""
    try:
        X = test_data_input[randomtask][indices]
        y = test_data_output[randomtask][indices]

        # Define regions based on mode
        testdata_inter_output = test_data_output[randomtask][left_bound:right_bound]
        y_inter_range = testdata_inter_output.max() - testdata_inter_output.min()
        y1_range = test_data_output[randomtask].max() - test_data_output[randomtask].min()

        # Only calculate left/right regions for extrapolation mode
        if mode == 'extrapolation':
            testdata_rightex_output = test_data_output[randomtask][right_bound:]
            testdata_leftex_output = test_data_output[randomtask][:left_bound]
            y_leftex_range = testdata_leftex_output.max() - testdata_leftex_output.min()
            y_rightex_range = testdata_rightex_output.max() - testdata_rightex_output.min()

        y_mean = y.mean()
        y_std = y.std()

        if y_std > 0:  # Only process if there's variation in the data
            y_norm = (y - y_mean) / y_std

            # Create center input
            center_input = torch.zeros((1, X.shape[1])).to(device)
            center_input[0, 4] = 0.0  # voltage = 0 (normalized)
            center_input[0, :4] = X[0, :4]  # copy first 4 features
            center_input[0, 5:] = X[0, 5:]  # copy remaining features

            # Get center and predictions based on model type
            if model_type == 'maml':
                center = model.model.model(center_input).item()
                predictions_scaling = model.model.model(test_data_input[randomtask][left_bound:right_bound])
            else:  # mlp
                center = model(center_input).item()
                predictions_scaling = model(test_data_input[randomtask][left_bound:right_bound])

            y_max = y_norm[:,0].max()
            y_min = y_norm[:,0].min()

            min_val = predictions_scaling.min().item()
            max_val = predictions_scaling.max().item()

            if abs(max_val - min_val) > 0:  # Avoid division by zero
                grad = (y_max - y_min) / (max_val - min_val)
                move = center - y_norm[middle_idx,0] / grad

                # Call appropriate evaluation function based on model type
                if model_type == 'maml':
                    (total_loss1, inter_loss1, leftex_loss1, rightex_loss1,
                    mape_loss1, leftex_mape1, inter_mape1, rightex_mape1,
                    predictions, actual_values, _, _, _, _, _, _, adam_used,
                    mae_loss, mae_l_loss, mae_r_loss, mae_in_loss) = evaluate_model_performance_maml(
                        model.model.model, 'MAML', X, y,
                        test_data_input[randomtask], test_data_output[randomtask], grad, move,
                        left_bound=left_bound, right_bound=right_bound, total_points=total_points,
                        mode=mode, adaptation_method=adaptation_method, layer_length=layer_length
                    )
                else:  # mlp
                    (total_loss1, inter_loss1, leftex_loss1, rightex_loss1,
                    mape_loss1, leftex_mape1, inter_mape1, rightex_mape1,
                    predictions, actual_values, _, _, _, _, _, _, adam_used,
                    mae_loss, mae_l_loss, mae_r_loss, mae_in_loss) = evaluate_model_performance_mlp(
                        model, 'MLP', X, y,
                        test_data_input[randomtask], test_data_output[randomtask], grad, move,
                        left_bound=left_bound, right_bound=right_bound, total_points=total_points,
                        mode=mode, adaptation_method=adaptation_method
                    )

                # Calculate RMSE values
                rmse_total = total_loss1 ** 0.5
                rmse_inter = inter_loss1 ** 0.5

                # Calculate region-specific metrics based on mode
                if mode == 'extrapolation':
                    rmse_leftex = leftex_loss1 ** 0.5
                    rmse_rightex = rightex_loss1 ** 0.5
                    nrmse_leftex = rmse_leftex / (abs(y_leftex_range) + 1e-4) * 100
                    nrmse_rightex = rmse_rightex / (abs(y_rightex_range) + 1e-4) * 100
                else:  # interpolation mode
                    rmse_leftex = 0
                    rmse_rightex = 0
                    nrmse_leftex = 0
                    nrmse_rightex = 0

                # Calculate common metrics (NRMSE using max-min range)
                nrmse1 = rmse_total / (abs(y1_range) + 1e-4) * 100
                nrmse_inter = rmse_inter / (abs(y_inter_range) + 1e-4) * 100

                return {
                    'rmse_total': rmse_total.item() if torch.is_tensor(rmse_total) else rmse_total,
                    'rmse_left': rmse_leftex.item() if torch.is_tensor(rmse_leftex) else rmse_leftex,
                    'rmse_right': rmse_rightex.item() if torch.is_tensor(rmse_rightex) else rmse_rightex,
                    'rmse_inter': rmse_inter.item() if torch.is_tensor(rmse_inter) else rmse_inter,
                    'nrmse_total': nrmse1.item(),
                    'nrmse_left': nrmse_leftex.item() if torch.is_tensor(nrmse_leftex) else nrmse_leftex,
                    'nrmse_right': nrmse_rightex.item() if torch.is_tensor(nrmse_rightex) else nrmse_rightex,
                    'nrmse_inter': nrmse_inter.item(),
                    'adam_used': adam_used,
                    'predictions': predictions,
                    'actuals': actual_values
                }
    except Exception as e:
        return None

    return None


def _build_argparser():
    """Build the argparse parser with all arguments."""
    parser = argparse.ArgumentParser(
        description='MAML Multi-Model Comparison (Compare up to 5 models)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare 3 different innerdiv values
  python MAML_topology_validation_multi_comparison.py --config 0 --vary innerdiv --innerdiv 50 100 200 --meta 32 --num_iterations 300000 --layer_length 40

  # Compare 5 different meta values
  python MAML_topology_validation_multi_comparison.py --config 0 --vary meta --meta 16 32 64 128 256 --innerdiv 100 --num_iterations 300000 --layer_length 40

  # Compare 2 different num_iterations
  python MAML_topology_validation_multi_comparison.py --config 1 --vary num_iterations --num_iterations 100000 300000 --innerdiv 100 --meta 32 --layer_length 40

  # Compare 3 different layer_length values
  python MAML_topology_validation_multi_comparison.py --config 0 --vary layer_length --layer_length 40 80 120 --innerdiv 100 --meta 32 --num_iterations 300000
        """
    )

    parser.add_argument('--config', type=int, required=True, choices=[0, 1, 2, 3, 6, 7],
                        help='Configuration ID. 0..3 = legacy datasets; '
                             '6 = TSMC combined patched intra eval; '
                             '7 = TSMC combined patched agnostic eval')
    parser.add_argument('--mode', type=str, choices=['extrapolation', 'interpolation'], default='extrapolation',
                        help='Testing mode (default: extrapolation)')
    parser.add_argument('--cells', type=str, nargs='+', default=None,
                        help='Cell types to test (default: config-dependent)')

    # Parameter variation selection
    parser.add_argument('--vary', type=str, required=True,
                        choices=['innerdiv', 'meta', 'num_iterations', 'layer_length'],
                        help='Which parameter to vary (innerdiv, meta, num_iterations, or layer_length)')

    # Parameter values (use the one specified in --vary)
    parser.add_argument('--innerdiv', type=int, nargs='+', required=True,
                        help='Inner learning rate divisor(s) - single value if not varying, or 2-5 values if varying')
    parser.add_argument('--meta', type=int, nargs='+', required=True,
                        help='Meta batch size(s) - single value if not varying, or 2-5 values if varying')
    parser.add_argument('--num_iterations', type=int, nargs='+', required=True,
                        help='Training iterations - single value if not varying, or 2-5 values if varying')
    parser.add_argument('--layer_length', type=int, nargs='+', default=[40],
                        help='Hidden layer size(s) - single value if not varying, or 2-5 values if varying (default: 40)')

    parser.add_argument('--inner', type=int, default=1,
                        help='Inner loop steps (default: 1)')
    parser.add_argument('--data_type', type=str, default=None,
                        help='Data type: cell/transition (default: config-dependent)')
    parser.add_argument('--gpu_id', type=str, default=None,
                        help='GPU device ID (default: config-dependent)')
    parser.add_argument('--num_test_samples', type=int, default=1000000,
                        help='Number of test samples (default: 1000000)')
    parser.add_argument('--save_results', action='store_true',
                        help='Save prediction results to .npy files')
    parser.add_argument('--output_prefix', type=str, default=None,
                        help='Prefix for output files')
    parser.add_argument('--indices', type=int, nargs='+', default=None,
                        help='Sampling indices for support set')
    parser.add_argument('--total_points', type=int, default=61,
                        help='Total number of data points (default: 61)')

    # MLP comparison options
    parser.add_argument('--compare_with_mlp', action='store_true',
                        help='Also compare with MLP (Aadam) model')
    parser.add_argument('--mlp_iterations', type=int, default=None,
                        help='Number of iterations for MLP model (default: 300000)')
    parser.add_argument('--mlp_model_type', type=str, default='aadam', choices=['aadam', 'mlp'],
                        help='MLP model type: aadam (hidden=256) or mlp (hidden=40) (default: aadam)')
    parser.add_argument('--mlp_model_path', type=str, default=None,
                        help='Path to pretrained MLP model (optional)')

    # Adaptation method
    parser.add_argument('--adaptation_method', type=str, default='selective_adam',
                        choices=['selective_adam', 'adam'],
                        help='Adaptation method: selective_adam (grad/move + conditional Adam) or adam (direct Adam, no grad/move) (default: selective_adam)')

    return parser


def _validate_vary_counts(args, parser):
    """Validate that --vary parameter has 2-5 values and others are single-valued."""
    varying_param = args.vary
    if varying_param == 'innerdiv':
        if len(args.innerdiv) < 2 or len(args.innerdiv) > 5:
            parser.error(f"When varying innerdiv, provide 2-5 values. Got {len(args.innerdiv)}")
        if len(args.meta) != 1 or len(args.num_iterations) != 1 or len(args.layer_length) != 1:
            parser.error("When varying innerdiv, meta, num_iterations, and layer_length must be single values")
    elif varying_param == 'meta':
        if len(args.meta) < 2 or len(args.meta) > 5:
            parser.error(f"When varying meta, provide 2-5 values. Got {len(args.meta)}")
        if len(args.innerdiv) != 1 or len(args.num_iterations) != 1 or len(args.layer_length) != 1:
            parser.error("When varying meta, innerdiv, num_iterations, and layer_length must be single values")
    elif varying_param == 'num_iterations':
        if len(args.num_iterations) < 2 or len(args.num_iterations) > 5:
            parser.error(f"When varying num_iterations, provide 2-5 values. Got {len(args.num_iterations)}")
        if len(args.innerdiv) != 1 or len(args.meta) != 1 or len(args.layer_length) != 1:
            parser.error("When varying num_iterations, innerdiv, meta, and layer_length must be single values")
    elif varying_param == 'layer_length':
        if len(args.layer_length) < 2 or len(args.layer_length) > 5:
            parser.error(f"When varying layer_length, provide 2-5 values. Got {len(args.layer_length)}")
        if len(args.innerdiv) != 1 or len(args.meta) != 1 or len(args.num_iterations) != 1:
            parser.error("When varying layer_length, innerdiv, meta, and num_iterations must be single values")


def _resolve_config_defaults(args, config):
    """Fill args defaults from the loaded test config."""
    if args.cells is None:
        args.cells = config['default_cells']
    if args.data_type is None:
        args.data_type = config['default_data_type']
    if args.gpu_id is None:
        args.gpu_id = config['default_gpu']
    if args.output_prefix is None:
        args.output_prefix = config['tech'].upper()

    # Set MLP iterations default if not specified
    if args.compare_with_mlp and args.mlp_iterations is None:
        args.mlp_iterations = 300000


def _setup_indices_and_bounds(args):
    """Populate mode-dependent default indices and derive k/left/right bounds."""
    # Set mode-dependent default indices
    if args.indices is None:
        if args.mode == 'extrapolation':
            args.indices = [5, 30, 55]
        else:  # interpolation
            args.indices = [13, 30, 45]

    # For interpolation mode: add endpoints
    if args.mode == 'interpolation':
        middle_indices = sorted(set(args.indices))
        if 0 not in middle_indices:
            middle_indices = [0] + middle_indices
        if args.total_points - 1 not in middle_indices:
            middle_indices = middle_indices + [args.total_points - 1]
        args.indices = middle_indices

    # Calculate bounds
    args.k = len(args.indices)
    args.left_bound = min(args.indices)
    args.right_bound = max(args.indices) + 1


def _build_model_configs(args, varying_param):
    """Build the list of per-model config dicts based on the varying parameter."""
    model_configs = []
    if varying_param == 'innerdiv':
        for innerdiv_val in args.innerdiv:
            model_configs.append({
                'innerdiv': innerdiv_val,
                'meta': args.meta[0],
                'num_iterations': args.num_iterations[0],
                'layer_length': args.layer_length[0],
                'label': f'innerdiv{innerdiv_val}'
            })
    elif varying_param == 'meta':
        for meta_val in args.meta:
            model_configs.append({
                'innerdiv': args.innerdiv[0],
                'meta': meta_val,
                'num_iterations': args.num_iterations[0],
                'layer_length': args.layer_length[0],
                'label': f'meta{meta_val}'
            })
    elif varying_param == 'num_iterations':
        for iter_val in args.num_iterations:
            model_configs.append({
                'innerdiv': args.innerdiv[0],
                'meta': args.meta[0],
                'num_iterations': iter_val,
                'layer_length': args.layer_length[0],
                'label': f'iter{iter_val}'
            })
    else:  # layer_length
        for layer_val in args.layer_length:
            model_configs.append({
                'innerdiv': args.innerdiv[0],
                'meta': args.meta[0],
                'num_iterations': args.num_iterations[0],
                'layer_length': layer_val,
                'label': f'layer{layer_val}'
            })
    return model_configs


def _print_config_banner(args, config, varying_param, model_configs):
    """Print the multi-model comparison configuration banner."""
    num_models = len(model_configs)
    print(f"\n{'='*80}")
    print("MULTI-MODEL COMPARISON CONFIGURATION")
    print(f"{'='*80}")
    print(f"   Config ID: {args.config}")
    print(f"   Config name: {config['name']}")
    print(f"   Mode: {args.mode}")
    print(f"   Data type: {args.data_type}")
    print(f"   Cells: {args.cells}")
    print(f"   Inner steps: {args.inner}")
    print(f"   Adaptation method: {args.adaptation_method}")
    print(f"   Varying parameter: {varying_param}")
    print(f"   Number of models: {num_models}")
    print(f"\n🔄 COMPARING {num_models} MODELS:")
    for i, cfg in enumerate(model_configs, 1):
        print(f"   Model {i}: innerdiv={cfg['innerdiv']}, meta={cfg['meta']}, iterations={cfg['num_iterations']}, layer_length={cfg['layer_length']}")


def _load_all_maml_models(args, model_configs, device):
    """Load all MAML models; returns the list or None on failure."""
    num_models = len(model_configs)
    print(f"\n🤖 Loading {num_models} MAML models...")
    models = []
    for i, cfg in enumerate(model_configs, 1):
        model_path = get_maml_model_path(
            args.config,
            data_type=args.data_type,
            innerdiv=cfg['innerdiv'],
            meta=cfg['meta'],
            inner=args.inner,
            num_iterations=cfg['num_iterations'],
            layer_length=cfg['layer_length']
        )
        print(f"\n📦 Model {i} ({cfg['label']}):")
        model = load_maml_model(model_path, device, layer_length=cfg['layer_length'])
        if model is None:
            print(f"❌ Failed to load model {i}, exiting")
            return None
        models.append(model)
    return models


def _load_mlp_model_optional(args, config, device):
    """Load MLP model if --compare_with_mlp; returns model, None (unused), or None on failure."""
    if not args.compare_with_mlp:
        return None
    print(f"\n🤖 Loading MLP model ({args.mlp_model_type})...")
    mlp_model_path = get_mlp_model_path(
        args.config,
        data_type=args.data_type,
        model_type=args.mlp_model_type,
        num_iterations=args.mlp_iterations,
        custom_path=args.mlp_model_path
    )
    mlp_model = load_mlp_model(mlp_model_path, args.mlp_model_type, device)
    if mlp_model is None:
        print(f"❌ Failed to load MLP model, exiting")
        return None
    return mlp_model


def _new_model_data_slot():
    """Factory: empty per-label accumulator dict."""
    return {
        'predictions': [],
        'actuals': [],
        'rmse': [], 'rmse_l': [], 'rmse_r': [], 'rmse_in': [],
        'nrmse': [], 'extra_l': [], 'extra_r': [], 'inter': [],
        'adam_count': 0
    }


def _load_and_prepare_test_data(args, cell, norm_stats, device):
    """Load, reshape, normalize, and analyze continuity of test data.

    Returns (test_data_input, test_data_output, discontinuous_task_ids) or
    (None, None, None) if data missing.
    """
    print("\n📊 Loading TEST dataset...")
    test_input_path, test_output_path = get_test_data_paths(args.config, cell, args.data_type)

    try:
        test_data_input = torch.load(test_input_path)
        test_data_output = torch.load(test_output_path)
    except FileNotFoundError as e:
        print(f"⚠️ Test data not found for cell {cell}: {e}")
        return None, None, None

    if len(test_data_output.shape) == 2:
        test_data_output = test_data_output.unsqueeze(-1)

    print(f"Test input shape: {test_data_input.shape}")
    print(f"Number of test samples: {len(test_data_input)}")

    # Apply normalization
    apply_normalization(test_data_input, norm_stats)

    # Analyze continuity
    print("\n🔍 Analyzing data continuity...")
    continuous_task_ids, discontinuous_task_ids, continuity_analysis = analyze_continuity(
        test_data_input, test_data_output, threshold_ratio=0.18
    )

    # Move to GPU
    test_data_input = test_data_input.to(device)
    test_data_output = test_data_output.to(device)

    return test_data_input, test_data_output, discontinuous_task_ids


def _evaluate_task_all_models(models, model_configs, mlp_model, args,
                              test_data_input, test_data_output, randomtask,
                              indices, middle_idx, device):
    """Run evaluate_single_task for every MAML model + optional MLP model.

    Returns (results_list, mlp_result_or_None).
    """
    results = []
    for model, cfg in zip(models, model_configs):
        result = evaluate_single_task(
            model, test_data_input, test_data_output, randomtask,
            indices, middle_idx, args.left_bound, args.right_bound,
            args.total_points, args.mode, device, model_type='maml', layer_length=cfg['layer_length'],
            adaptation_method=args.adaptation_method
        )
        results.append(result)

    mlp_result = None
    if args.compare_with_mlp:
        mlp_result = evaluate_single_task(
            mlp_model, test_data_input, test_data_output, randomtask,
            indices, middle_idx, args.left_bound, args.right_bound,
            args.total_points, args.mode, device, model_type='mlp',
            adaptation_method=args.adaptation_method
        )
    return results, mlp_result


def _check_all_results_valid(results, mlp_result, compare_with_mlp):
    """Return True iff all results are non-None with finite nrmse_total."""
    all_models_succeeded = all(r is not None for r in results)
    if compare_with_mlp:
        all_models_succeeded = all_models_succeeded and (mlp_result is not None)
    if not all_models_succeeded:
        return False

    all_valid = all(
        not (torch.isinf(torch.tensor(r['nrmse_total'])) or
             torch.isnan(torch.tensor(r['nrmse_total'])))
        for r in results
    )
    if compare_with_mlp and all_valid:
        all_valid = all_valid and not (
            torch.isinf(torch.tensor(mlp_result['nrmse_total'])) or
            torch.isnan(torch.tensor(mlp_result['nrmse_total']))
        )
    return all_valid


def _store_task_result(data, result):
    """Append a single evaluate_single_task result into an accumulator slot."""
    data['rmse'].append(result['rmse_total'])
    data['rmse_l'].append(result['rmse_left'])
    data['rmse_r'].append(result['rmse_right'])
    data['rmse_in'].append(result['rmse_inter'])
    data['nrmse'].append(result['nrmse_total'])
    data['extra_l'].append(result['nrmse_left'])
    data['extra_r'].append(result['nrmse_right'])
    data['inter'].append(result['nrmse_inter'])
    if result['adam_used']:
        data['adam_count'] += 1
    data['predictions'].extend(result['predictions'])
    data['actuals'].extend(result['actuals'])


def _print_progress_comparison(model_data, model_configs, args, task_idx, num_test_samples):
    """Print the every-100-tasks comparison table (RMSE + NRMSE)."""
    num_models = len(model_configs)
    first_label = model_configs[0]['label']
    if len(model_data[first_label]['nrmse']) == 0:
        return
    print(f"\n📊 Progress: {task_idx}/{num_test_samples} | Valid: {len(model_data[first_label]['nrmse'])}")

    # Dynamic column widths
    col_width = 18
    total_models = num_models + (1 if args.compare_with_mlp else 0)
    line_width = 20 + col_width * total_models + 12 * (total_models - 1)
    print(f"{'─'*line_width}")

    # Header
    header = f"{'Metric':<20}"
    for cfg in model_configs:
        header += f"{cfg['label']:>{col_width}}"
    if args.compare_with_mlp:
        header += f"{'MLP':>{col_width}}"
    for i in range(total_models - 1):
        header += f"{'Diff(' + str(i+1) + '-' + str(i) + ')':>12}"
    print(header)
    print(f"{'─'*line_width}")

    # RMSE
    rmse_line = f"{'RMSE (ns)':<20}"
    rmse_vals = []
    for cfg in model_configs:
        data = model_data[cfg['label']]
        avg = sum(data['rmse']) / len(data['rmse'])
        rmse_vals.append(avg)
        rmse_line += f"{avg*1000:>{col_width}.4f}"
    if args.compare_with_mlp:
        mlp_data = model_data['MLP']
        if len(mlp_data['rmse']) > 0:
            avg = sum(mlp_data['rmse']) / len(mlp_data['rmse'])
            rmse_vals.append(avg)
            rmse_line += f"{avg*1000:>{col_width}.4f}"
    for i in range(len(rmse_vals) - 1):
        diff = (rmse_vals[i+1] - rmse_vals[i]) * 1000
        symbol = "✓" if diff < 0 else "✗"
        rmse_line += f"{diff:>11.4f}{symbol}"
    print(rmse_line)

    # NRMSE (max-min based)
    nrmse_line = f"{'NRMSE (%)':<20}"
    nrmse_vals = []
    for cfg in model_configs:
        data = model_data[cfg['label']]
        avg = sum(data['nrmse']) / len(data['nrmse'])
        nrmse_vals.append(avg)
        nrmse_line += f"{avg:>{col_width}.2f}"
    if args.compare_with_mlp:
        mlp_data = model_data['MLP']
        if len(mlp_data['nrmse']) > 0:
            avg = sum(mlp_data['nrmse']) / len(mlp_data['nrmse'])
            nrmse_vals.append(avg)
            nrmse_line += f"{avg:>{col_width}.2f}"
    for i in range(len(nrmse_vals) - 1):
        diff = nrmse_vals[i+1] - nrmse_vals[i]
        symbol = "✓" if diff < 0 else "✗"
        nrmse_line += f"{diff:>11.2f}{symbol}"
    print(nrmse_line)

    print(f"{'─'*line_width}")


def _run_task_loop(models, model_configs, mlp_model, args, test_data_input,
                   test_data_output, discontinuous_task_ids, indices, middle_idx,
                   device, model_data):
    """Iterate over random test tasks, evaluating & accumulating into model_data."""
    num_test_samples = min(args.num_test_samples, len(test_data_input))
    test_indices = random.sample(range(len(test_data_input)), num_test_samples)

    print(f"\n🎲 Testing {num_test_samples} random tasks")
    print(f"{'='*80}")

    valid_count = 0
    for task_idx, randomtask in enumerate(test_indices):
        # Simple progress indicator (always prints)
        if task_idx % 100 == 0:
            print(f"  Task {task_idx}/{num_test_samples} | Valid: {valid_count}", flush=True)

        # Skip discontinuous tasks
        if randomtask in discontinuous_task_ids:
            continue

        results, mlp_result = _evaluate_task_all_models(
            models, model_configs, mlp_model, args,
            test_data_input, test_data_output, randomtask,
            indices, middle_idx, device
        )

        if _check_all_results_valid(results, mlp_result, args.compare_with_mlp):
            valid_count += 1
            # Store results for all models
            for cfg, result in zip(model_configs, results):
                _store_task_result(model_data[cfg['label']], result)
            # Store MLP results if enabled
            if args.compare_with_mlp:
                _store_task_result(model_data['MLP'], mlp_result)

        # Real-time comparison every 100 samples
        if task_idx % 100 == 0 and task_idx > 0:
            _print_progress_comparison(model_data, model_configs, args, task_idx, num_test_samples)

    return num_test_samples


def _aggregate_cell_results(model_data, all_results, cell, model_configs, args):
    """Fold per-cell accumulators into the all_results dict of avg RMSE/NRMSE rows."""
    for cfg in model_configs:
        label = cfg['label']
        data = model_data[label]

        if len(data['nrmse']) > 0:
            all_results[label][f"RMSE_{cell}"] = [[
                sum(data['rmse'])/len(data['rmse']),
                sum(data['rmse_l'])/len(data['rmse_l']),
                sum(data['rmse_r'])/len(data['rmse_r']),
                sum(data['rmse_in'])/len(data['rmse_in'])
            ]]
            all_results[label][f"NRMSE_{cell}"] = [[
                sum(data['nrmse'])/len(data['nrmse']),
                sum(data['extra_l'])/len(data['extra_l']),
                sum(data['extra_r'])/len(data['extra_r']),
                sum(data['inter'])/len(data['inter'])
            ]]

    if args.compare_with_mlp:
        mlp_data = model_data['MLP']
        if len(mlp_data['nrmse']) > 0:
            all_results['MLP'][f"RMSE_{cell}"] = [[
                sum(mlp_data['rmse'])/len(mlp_data['rmse']),
                sum(mlp_data['rmse_l'])/len(mlp_data['rmse_l']),
                sum(mlp_data['rmse_r'])/len(mlp_data['rmse_r']),
                sum(mlp_data['rmse_in'])/len(mlp_data['rmse_in'])
            ]]
            all_results['MLP'][f"NRMSE_{cell}"] = [[
                sum(mlp_data['nrmse'])/len(mlp_data['nrmse']),
                sum(mlp_data['extra_l'])/len(mlp_data['extra_l']),
                sum(mlp_data['extra_r'])/len(mlp_data['extra_r']),
                sum(mlp_data['inter'])/len(mlp_data['inter'])
            ]]


def _save_cell_npy_results(model_data, model_configs, args, config, cell):
    """Save per-model prediction/actual arrays to .npy files."""
    os.makedirs("data_result_npy_directory", exist_ok=True)

    # Adaptation method suffix (only for adam, selective_adam has no suffix for backward compatibility)
    adapt_suffix = "_adam" if args.adaptation_method == 'adam' else ""

    for cfg in model_configs:
        label = cfg['label']
        data = model_data[label]

        # Build full parameter string for filename
        innerdiv = cfg['innerdiv']
        meta = cfg['meta']
        iterations = cfg['num_iterations']
        layer_length = cfg['layer_length']

        pred_filename = f"data_result_npy_directory/{args.output_prefix}_{config['topology_type']}_{cell}_{args.data_type}_{args.mode}_MAML_innerdiv{innerdiv}_meta{meta}_layer{layer_length}_{iterations}{adapt_suffix}_pred.npy"
        act_filename = f"data_result_npy_directory/{args.output_prefix}_{config['topology_type']}_{cell}_{args.data_type}_{args.mode}_MAML_innerdiv{innerdiv}_meta{meta}_layer{layer_length}_{iterations}{adapt_suffix}_act.npy"

        np.save(pred_filename, data['predictions'])
        np.save(act_filename, data['actuals'])
        print(f"💾 Saved {label}: {os.path.basename(pred_filename)}")

    # Save MLP results if enabled
    if args.compare_with_mlp:
        mlp_data = model_data['MLP']
        mlp_pred_filename = f"data_result_npy_directory/{args.output_prefix}_{config['topology_type']}_{cell}_{args.data_type}_{args.mode}_{args.mlp_model_type}_{args.mlp_iterations}{adapt_suffix}_pred.npy"
        mlp_act_filename = f"data_result_npy_directory/{args.output_prefix}_{config['topology_type']}_{cell}_{args.data_type}_{args.mode}_{args.mlp_model_type}_{args.mlp_iterations}{adapt_suffix}_act.npy"

        np.save(mlp_pred_filename, mlp_data['predictions'])
        np.save(mlp_act_filename, mlp_data['actuals'])
        print(f"💾 Saved MLP: {os.path.basename(mlp_pred_filename)}")


def _print_cell_final_results(all_results, model_configs, cell, args):
    """Print detailed per-model FINAL RESULTS block for a cell."""
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS FOR {cell}")
    print(f"{'='*80}")

    for cfg in model_configs:
        label = cfg['label']
        print(f"\n{label}:")
        print(f"{'─'*80}")
        for key, value in all_results[label].items():
            if cell in key:
                print(f"  {key}: {np.array(value[0])}")

    # Print MLP results if enabled
    if args.compare_with_mlp:
        print(f"\nMLP ({args.mlp_model_type}, iter={args.mlp_iterations}):")
        print(f"{'─'*80}")
        for key, value in all_results['MLP'].items():
            if cell in key:
                print(f"  {key}: {np.array(value[0])}")


def _process_cell(cell, args, config, models, model_configs, mlp_model,
                  all_results, norm_stats, indices, middle_idx, device):
    """Run the full evaluation pipeline for a single cell type."""
    print(f"\n{'='*80}")
    print(f"Processing cell: {cell}")
    print(f"{'='*80}")

    # Initialize collections for all models
    model_data = {cfg['label']: _new_model_data_slot() for cfg in model_configs}
    if args.compare_with_mlp:
        model_data['MLP'] = _new_model_data_slot()

    test_data_input, test_data_output, discontinuous_task_ids = _load_and_prepare_test_data(
        args, cell, norm_stats, device
    )
    if test_data_input is None:
        return

    num_test_samples = _run_task_loop(
        models, model_configs, mlp_model, args,
        test_data_input, test_data_output, discontinuous_task_ids,
        indices, middle_idx, device, model_data
    )

    # Final results for this cell
    first_label = model_configs[0]['label']
    num_valid = len(model_data[first_label]['nrmse'])
    print(f"\n✅ Completed {num_valid} valid tasks for {cell}")

    _aggregate_cell_results(model_data, all_results, cell, model_configs, args)

    if args.save_results:
        _save_cell_npy_results(model_data, model_configs, args, config, cell)

    _print_cell_final_results(all_results, model_configs, cell, args)


def _print_overall_summary(all_results, model_configs):
    """Print the OVERALL COMPARISON SUMMARY block."""
    print(f"\n{'='*80}")
    print("OVERALL COMPARISON SUMMARY")
    print(f"{'='*80}")

    for cfg in model_configs:
        label = cfg['label']
        print(f"\n{label} (innerdiv={cfg['innerdiv']}, meta={cfg['meta']}, iter={cfg['num_iterations']}, layer={cfg['layer_length']}):")
        print(f"{'─'*80}")
        for key, value in all_results[label].items():
            print(f"{key}: {np.array(value[0])}")


def main():
    parser = _build_argparser()
    args = parser.parse_args()

    _validate_vary_counts(args, parser)

    # Get configuration
    try:
        config = get_test_config(args.config)
    except ValueError as e:
        print(f"Error: {e}")
        print_available_configs()
        return 1

    _resolve_config_defaults(args, config)
    _setup_indices_and_bounds(args)

    # GPU settings
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:', device)

    varying_param = args.vary
    model_configs = _build_model_configs(args, varying_param)
    _print_config_banner(args, config, varying_param, model_configs)

    # Load training data for normalization
    print("\n📊 Loading TRAINING dataset for normalization...")
    train_data_paths = get_train_data_paths(args.config, args.data_type)
    norm_stats = load_and_normalize_data(train_data_paths)

    models = _load_all_maml_models(args, model_configs, device)
    if models is None:
        return 1

    mlp_model = _load_mlp_model_optional(args, config, device)
    if args.compare_with_mlp and mlp_model is None:
        return 1

    # Initialize results storage for all models
    all_results = {cfg['label']: {} for cfg in model_configs}
    if args.compare_with_mlp:
        all_results['MLP'] = {}

    # Random sampling setup
    indices = args.indices
    middle_idx = len(indices) // 2

    for cell in args.cells:
        _process_cell(cell, args, config, models, model_configs, mlp_model,
                      all_results, norm_stats, indices, middle_idx, device)

    _print_overall_summary(all_results, model_configs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
