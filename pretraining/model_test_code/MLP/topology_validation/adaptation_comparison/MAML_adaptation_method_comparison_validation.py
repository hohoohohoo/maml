#!/usr/bin/env python
# coding: utf-8

"""
MAML Optimization Comparison Validation Script

This script compares 5 optimization methods for MAML fine-tuning:
- Grad+Move Only: No optimization, just scaling
- SGD 40 steps: Direct SGD optimization (no grad/move)
- Adam 40 steps: Direct Adam optimization (no grad/move)
- Selective Adam: Grad+Move + Adam if loss > threshold
- Full Adam 40: Full Adam optimization with 40 steps

Configuration options:
- Config 0: ASAP7 Intra Topology
- Config 1: ASAP7 Topology Agnostic
- Config 2: TSMC Intra Topology
- Config 3: TSMC Topology Agnostic

Usage:
  Single run:
    python MAML_optim_comparison_validation.py --config 0 --data_type cell --save_results

  Sweep mode (JSON config):
    python MAML_optim_comparison_validation.py json_configs/maml_optim_comparison_sweep.json
"""

import os
import sys
import torch
import numpy as np
import random
import argparse

# Add parent directory to path for utils access
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import json
from datetime import datetime
from itertools import product

# Import configuration
from utils.test_dataset_config import (
    get_test_config,
    get_train_data_paths,
    get_test_data_paths,
    get_maml_model_path,
    print_available_configs
)

# Import utility functions
from utils.data_management_utils import (
    analyze_continuity,
    load_and_normalize_data,
    apply_normalization
)
from utils.maml_functions import compare_optimization_methods_maml

# MAML import
sys.path.append('../../../../../model_code/')
from mlp_maml import OptimizedMAML, MAMLModel_3hidden


# Optimization methods to compare (shared across helpers)
_METHODS = ['none', 'sgd', 'adam', 'selective_adam', 'full_adam']


def _method_names(num_optim_steps):
    """Return the display names for each optimization method."""
    return {
        'none': 'Grad+Move Only',
        'sgd': f'SGD {num_optim_steps} steps',
        'adam': f'Adam {num_optim_steps} steps',
        'selective_adam': 'Selective Adam',
        'full_adam': f'Full Adam {num_optim_steps}'
    }


def _prepare_run_config(args):
    """Resolve CLI defaults from the test config and compute derived bounds.

    Returns the loaded config dict, or None if the config ID is invalid.
    """
    try:
        config = get_test_config(args.config)
    except ValueError as e:
        print(f"Error: {e}")
        print_available_configs()
        return None

    # Set defaults from config
    if args.cells is None:
        args.cells = config['default_cells']
    if args.meta is None:
        args.meta = config['default_meta']
    if args.data_type is None:
        args.data_type = config['default_data_type']
    if args.gpu_id is None:
        args.gpu_id = config['default_gpu']
    if args.output_prefix is None:
        args.output_prefix = config['tech'].upper()

    # Set mode-dependent default indices
    if args.indices is None:
        if args.mode == 'extrapolation':
            args.indices = [5, 30, 55]
        else:
            args.indices = [13, 30, 45]

    # For interpolation mode: add endpoints
    if args.mode == 'interpolation':
        middle_indices = sorted(set(args.indices))
        if 0 not in middle_indices:
            middle_indices = [0] + middle_indices
        if args.total_points - 1 not in middle_indices:
            middle_indices = middle_indices + [args.total_points - 1]
        args.indices = middle_indices
        print(f"\nInterpolation mode: Added endpoints -> {args.indices}")

    # Calculate bounds
    args.k = len(args.indices)
    args.left_bound = min(args.indices)
    args.right_bound = max(args.indices) + 1

    return config


def _setup_device(args):
    """Apply GPU environment variables and return the torch device."""
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:', device)
    if torch.cuda.is_available():
        print('Current cuda device:', torch.cuda.current_device())
    return device


def _print_run_header(args, config, data_type):
    """Print the top-level run banner."""
    print(f"\n{'='*80}")
    print("MAML OPTIMIZATION COMPARISON VALIDATION")
    print(f"{'='*80}")
    print(f"Config: {args.config} ({config['name']})")
    print(f"Mode: {args.mode}")
    print(f"Data type: {data_type}")
    print(f"Cells: {args.cells}")
    print(f"Optimization steps: {args.num_optim_steps}")
    print(f"Indices: {args.indices}")
    print(f"Left bound: {args.left_bound}, Right bound: {args.right_bound}")
    if args.measure_time:
        print(f"Timing mode: CPU (for consistent measurement)")
    print(f"{'='*80}\n")


def _load_model_for_run(args, config, device, data_type):
    """Load the MAML model checkpoint. Returns (maml_model, model_path) or (None, path) on failure."""
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = get_maml_model_path(
            args.config, data_type, args.innerdiv, args.meta, args.inner,
            args.num_iterations if args.num_iterations else config['default_num_iterations']
        )

    print(f"Using model: {model_path}")

    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return None, model_path

    # Load MAML model (input_features=9 for cell/transition delay models)
    input_features = 9
    maml_model = OptimizedMAML(
        model=MAMLModel_3hidden(input_features, args.layer_length),
        inner_lr=0.01 / args.innerdiv,
        meta_lr=0.0001
    )
    maml_model.model.load_state_dict(torch.load(model_path, map_location=device))
    maml_model.model.to(device)
    maml_model.model.model.eval()
    return maml_model, model_path


def _load_cell_test_data(args, cell, data_type, norm_stats, device):
    """Load test data for one cell and apply normalization. Returns (input, output) or None."""
    test_input_path, test_output_path = get_test_data_paths(args.config, cell, data_type)

    if not os.path.exists(test_input_path) or not os.path.exists(test_output_path):
        print(f"Warning: Test data not found for {cell}, skipping...")
        return None

    test_data_input = torch.load(test_input_path)
    test_data_output = torch.load(test_output_path)

    # Add dimension to output if needed
    if len(test_data_output.shape) == 2:
        test_data_output = test_data_output.unsqueeze(-1)

    # Apply normalization using training statistics
    apply_normalization(test_data_input, norm_stats)

    test_data_input = test_data_input.to(device)
    test_data_output = test_data_output.to(device)
    return test_data_input, test_data_output


def _init_method_metrics():
    """Create the per-method metric accumulator dict."""
    return {m: {
        'total_rmse': [], 'inter_rmse': [], 'leftex_rmse': [], 'rightex_rmse': [],
        'total_mape': [], 'inter_mape': [], 'leftex_mape': [], 'rightex_mape': [],
        'total_nrmse': [], 'inter_nrmse': [], 'leftex_nrmse': [], 'rightex_nrmse': [],
        'time_ms': [],
        'adam_triggered_count': 0
    } for m in _METHODS}


def _compute_y_ranges(test_data_output, randomtask, args):
    """Compute (y1_range, y_inter_range, y_leftex_range, y_rightex_range) for NRMSE."""
    y_total = test_data_output[randomtask]
    y1_range = (y_total.max() - y_total.min()).item()

    testdata_inter_output = test_data_output[randomtask][args.left_bound:args.right_bound]
    y_inter_range = (testdata_inter_output.max() - testdata_inter_output.min()).item()

    if args.mode == 'extrapolation':
        y_leftex = test_data_output[randomtask][:args.left_bound]
        y_rightex = test_data_output[randomtask][args.right_bound:]
        y_leftex_range = (y_leftex.max() - y_leftex.min()).item() if len(y_leftex) > 0 else y1_range
        y_rightex_range = (y_rightex.max() - y_rightex.min()).item() if len(y_rightex) > 0 else y1_range
    else:
        y_leftex_range = y1_range
        y_rightex_range = y1_range
    return y1_range, y_inter_range, y_leftex_range, y_rightex_range


def _compute_grad_move(maml_model, X, y_norm, test_data_input, randomtask, args, device, middle_idx):
    """Compute (grad, move, center) for MAML scaling, or None if degenerate."""
    center_input = torch.zeros((1, X.shape[1])).to(device)
    center_input[0, 4] = 0.0
    center_input[0, :4] = X[0, :4]
    center_input[0, 5:] = X[0, 5:]
    center = maml_model.model.model(center_input).item()

    y_max = y_norm[:, 0].max()
    y_min = y_norm[:, 0].min()

    predictions = maml_model.model.model(test_data_input[randomtask][args.left_bound:args.right_bound])
    min_val = predictions.min().item()
    max_val = predictions.max().item()

    if abs(max_val - min_val) <= 0:
        return None

    grad = (y_max - y_min) / (max_val - min_val)
    move = center - y_norm[middle_idx, 0] / grad
    return grad, move, center


def _record_method_metrics(method_metrics, results, y_ranges):
    """Append per-method results from a single task to the accumulators."""
    y1_range, y_inter_range, y_leftex_range, y_rightex_range = y_ranges

    for method in _METHODS:
        r = results[method]

        method_metrics[method]['total_rmse'].append(r['total_rmse'])
        method_metrics[method]['inter_rmse'].append(r['inter_rmse'])
        method_metrics[method]['leftex_rmse'].append(r['leftex_rmse'])
        method_metrics[method]['rightex_rmse'].append(r['rightex_rmse'])

        method_metrics[method]['total_mape'].append(r['total_mape'])
        method_metrics[method]['inter_mape'].append(r['inter_mape'])
        method_metrics[method]['leftex_mape'].append(r['leftex_mape'])
        method_metrics[method]['rightex_mape'].append(r['rightex_mape'])

        # Collect timing data
        method_metrics[method]['time_ms'].append(r.get('time_ms', 0))

        # Calculate NRMSE (using max-min as denominator)
        total_nrmse = r['total_rmse'] / (y1_range + 1e-8) * 100
        inter_nrmse = r['inter_rmse'] / (y_inter_range + 1e-8) * 100
        leftex_nrmse = r['leftex_rmse'] / (y_leftex_range + 1e-8) * 100
        rightex_nrmse = r['rightex_rmse'] / (y_rightex_range + 1e-8) * 100

        method_metrics[method]['total_nrmse'].append(total_nrmse)
        method_metrics[method]['inter_nrmse'].append(inter_nrmse)
        method_metrics[method]['leftex_nrmse'].append(leftex_nrmse)
        method_metrics[method]['rightex_nrmse'].append(rightex_nrmse)

        # Track adam_triggered for selective_adam
        if method == 'selective_adam' and r.get('adam_triggered', False):
            method_metrics[method]['adam_triggered_count'] += 1


def _print_progress(method_metrics, method_names, valid_tasks, i, num_test_samples):
    """Print intermediate progress summary."""
    sel_adam_ratio = method_metrics['selective_adam']['adam_triggered_count'] / valid_tasks * 100
    print(f"\n  Progress: {valid_tasks} valid tasks (iter {i+1}/{num_test_samples}) | Selective Adam Ratio: {sel_adam_ratio:.1f}%")
    print(f"  {'Method':<20} | {'NRMSE':<10} | {'RMSE':<10}")
    print(f"  {'-'*45}")
    for m in _METHODS:
        n = len(method_metrics[m]['total_nrmse'])
        if n > 0:
            avg_nrmse = sum(method_metrics[m]['total_nrmse']) / n
            avg_rmse = sum(method_metrics[m]['total_rmse']) / n
            print(f"  {method_names[m]:<20} | {avg_nrmse:<10.3f} | {avg_rmse:<10.6f}")


def _evaluate_tasks_with_adaptation(args, maml_model, test_data_input, test_data_output,
                                    num_test_samples, method_metrics, method_names, device):
    """Run the per-task inner loop with adaptation method dispatch. Returns valid_tasks count."""
    indices = args.indices
    middle_idx = len(indices) // 2
    valid_tasks = 0

    for i in range(num_test_samples):
        randomtask = random.randint(0, len(test_data_input) - 1)

        try:
            X = test_data_input[randomtask][indices]
            y = test_data_output[randomtask][indices]

            y_ranges = _compute_y_ranges(test_data_output, randomtask, args)

            y_mean = y.mean()
            y_std = y.std()

            if y_std <= 0:
                continue

            y_norm = (y - y_mean) / y_std

            gm = _compute_grad_move(maml_model, X, y_norm, test_data_input,
                                    randomtask, args, device, middle_idx)
            if gm is None:
                continue
            grad, move, _center = gm

            # Run comparison
            results = compare_optimization_methods_maml(
                initial_model=maml_model.model.model,
                X=X,
                y=y,
                true_x=test_data_input[randomtask],
                true_function=test_data_output[randomtask],
                grad=grad,
                move=move,
                num_steps=args.num_optim_steps,
                left_bound=args.left_bound,
                right_bound=args.right_bound,
                total_points=args.total_points,
                mode=args.mode,
                layer_length=args.layer_length,
                use_cpu_for_timing=args.measure_time
            )

            valid_tasks += 1
            _record_method_metrics(method_metrics, results, y_ranges)

            # Print intermediate results every 100 valid tasks
            if valid_tasks % 100 == 0:
                _print_progress(method_metrics, method_names, valid_tasks, i, num_test_samples)

        except Exception as e:
            if i < 5:
                print(f"Error processing task {randomtask}: {e}")
            continue

    return valid_tasks


def _aggregate_cell_metrics(method_metrics, method_names):
    """Compute average metrics per method for a single cell."""
    cell_results = {}
    for method in _METHODS:
        m = method_metrics[method]
        n = len(m['total_rmse'])
        if n == 0:
            continue

        cell_results[method] = {
            'name': method_names[method],
            'num_tasks': n,
            'avg_total_rmse': sum(m['total_rmse']) / n,
            'avg_inter_rmse': sum(m['inter_rmse']) / n,
            'avg_leftex_rmse': sum(m['leftex_rmse']) / n,
            'avg_rightex_rmse': sum(m['rightex_rmse']) / n,
            'avg_total_mape': sum(m['total_mape']) / n * 100,
            'avg_inter_mape': sum(m['inter_mape']) / n * 100,
            'avg_leftex_mape': sum(m['leftex_mape']) / n * 100,
            'avg_rightex_mape': sum(m['rightex_mape']) / n * 100,
            'avg_total_nrmse': sum(m['total_nrmse']) / n,
            'avg_inter_nrmse': sum(m['inter_nrmse']) / n,
            'avg_leftex_nrmse': sum(m['leftex_nrmse']) / n,
            'avg_rightex_nrmse': sum(m['rightex_nrmse']) / n,
            'avg_time_ms': sum(m['time_ms']) / n,
        }
        # Add adam_triggered_ratio for selective_adam
        if method == 'selective_adam':
            cell_results[method]['adam_triggered_ratio'] = m['adam_triggered_count'] / n * 100
    return cell_results


def _print_cell_results(cell, cell_results, method_metrics, valid_tasks, args):
    """Print the per-cell results table."""
    sel_adam_ratio = method_metrics['selective_adam']['adam_triggered_count'] / valid_tasks * 100 if valid_tasks > 0 else 0

    print(f"\n{'='*100}")
    print(f"RESULTS FOR {cell} ({valid_tasks} tasks) | Selective Adam Triggered: {sel_adam_ratio:.1f}%")
    print(f"{'='*100}")
    print(f"{'Method':<20} | {'NRMSE Total':<12} | {'NRMSE Inter':<12} | {'RMSE Total':<12} | {'RMSE Inter':<12}")
    print("-" * 100)

    for method in _METHODS:
        if method in cell_results:
            r = cell_results[method]
            print(f"{r['name']:<20} | {r['avg_total_nrmse']:<12.3f} | {r['avg_inter_nrmse']:<12.3f} | "
                  f"{r['avg_total_rmse']:<12.6f} | {r['avg_inter_rmse']:<12.6f}")

    if args.mode == 'extrapolation':
        print("-" * 100)
        print(f"{'Method':<20} | {'NRMSE Left':<12} | {'NRMSE Right':<12} | {'RMSE Left':<12} | {'RMSE Right':<12}")
        print("-" * 100)
        for method in _METHODS:
            if method in cell_results:
                r = cell_results[method]
                print(f"{r['name']:<20} | {r['avg_leftex_nrmse']:<12.3f} | {r['avg_rightex_nrmse']:<12.3f} | "
                      f"{r['avg_leftex_rmse']:<12.6f} | {r['avg_rightex_rmse']:<12.6f}")

    print("=" * 100)


def _process_cell(cell, args, maml_model, norm_stats, data_type, method_names, device):
    """Run evaluation for a single cell. Returns cell_results dict or None if skipped."""
    print(f"\n{'='*80}")
    print(f"Processing cell: {cell}")
    print(f"{'='*80}")

    loaded = _load_cell_test_data(args, cell, data_type, norm_stats, device)
    if loaded is None:
        return None
    test_data_input, test_data_output = loaded

    num_test_samples = min(args.num_test_samples, len(test_data_input))
    print(f"Number of test samples: {num_test_samples}")

    method_metrics = _init_method_metrics()

    valid_tasks = _evaluate_tasks_with_adaptation(
        args, maml_model, test_data_input, test_data_output,
        num_test_samples, method_metrics, method_names, device
    )

    print(f"\nCompleted {valid_tasks} valid tasks for {cell}")

    cell_results = _aggregate_cell_metrics(method_metrics, method_names)
    _print_cell_results(cell, cell_results, method_metrics, valid_tasks, args)
    return cell_results


def _save_run_json_results(args, config, data_type, all_results):
    """Write the JSON results file when --save_results is set."""
    if not args.save_results:
        return

    output_dir = "adaptation_method_comparison_results"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{output_dir}/{args.output_prefix}_{config['topology_type']}_{data_type}_{args.mode}_optim_comparison_{timestamp}.json"

    save_data = {
        'config': {
            'config_id': args.config,
            'config_name': config['name'],
            'mode': args.mode,
            'data_type': data_type,
            'cells': args.cells,
            'indices': args.indices,
            'num_optim_steps': args.num_optim_steps,
            'layer_length': args.layer_length,
            'timestamp': timestamp,
            'measure_time_on_cpu': args.measure_time
        },
        'results': {}
    }

    # Convert results to serializable format
    for cell, cell_results in all_results.items():
        save_data['results'][cell] = {}
        for method, metrics in cell_results.items():
            save_data['results'][cell][method] = {
                k: float(v) if isinstance(v, (int, float)) else v
                for k, v in metrics.items()
            }

    with open(output_file, 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")


def _print_run_summary(all_results):
    """Print the final overall summary across all cells."""
    print(f"\n{'='*100}")
    print("OVERALL SUMMARY")
    print(f"{'='*100}")

    for cell in all_results:
        # Get selective_adam ratio if available
        sel_ratio_str = ""
        if 'selective_adam' in all_results[cell] and 'adam_triggered_ratio' in all_results[cell]['selective_adam']:
            sel_ratio_str = f" | Selective Adam Ratio: {all_results[cell]['selective_adam']['adam_triggered_ratio']:.1f}%"
        print(f"\n{cell}:{sel_ratio_str}")
        for method in _METHODS:
            if method in all_results[cell]:
                r = all_results[cell][method]
                print(f"  {r['name']:<20}: NRMSE={r['avg_total_nrmse']:.3f}%, RMSE={r['avg_total_rmse']:.6f}")


def run_single_validation(args):
    """Run validation for a single configuration."""
    # Resolve config defaults, indices, bounds
    config = _prepare_run_config(args)
    if config is None:
        return 1

    device = _setup_device(args)

    data_type = args.data_type.lower()
    _print_run_header(args, config, data_type)

    # Load training data for normalization statistics
    train_data_paths = get_train_data_paths(args.config, data_type)
    norm_stats = load_and_normalize_data(train_data_paths)

    # Load MAML model
    maml_model, _model_path = _load_model_for_run(args, config, device, data_type)
    if maml_model is None:
        return 1

    method_names = _method_names(args.num_optim_steps)
    all_results = {}

    # Process each cell
    for cell in args.cells:
        cell_results = _process_cell(cell, args, maml_model, norm_stats,
                                     data_type, method_names, device)
        if cell_results is not None:
            all_results[cell] = cell_results

    # Save + summarize
    _save_run_json_results(args, config, data_type, all_results)
    _print_run_summary(all_results)

    return 0


def run_sweep(sweep_config_path):
    """Run sweep over multiple configurations from JSON config file."""
    print(f"\n{'='*80}")
    print("MAML OPTIMIZATION COMPARISON SWEEP")
    print(f"{'='*80}")
    print(f"Loading sweep config from: {sweep_config_path}")

    with open(sweep_config_path, 'r') as f:
        sweep_config = json.load(f)

    print(f"Experiment: {sweep_config.get('experiment_name', 'unnamed')}")
    print(f"Description: {sweep_config.get('description', 'N/A')}")

    base_config = sweep_config.get('base_config', {})
    sweep_params = sweep_config.get('sweep_params', {})

    # Get all parameter combinations
    param_names = list(sweep_params.keys())
    param_values = [sweep_params[name] for name in param_names]

    combinations = list(product(*param_values))
    total_runs = len(combinations)

    print(f"\nSweep parameters: {param_names}")
    print(f"Total combinations: {total_runs}")
    print(f"{'='*80}\n")

    for run_idx, combo in enumerate(combinations, 1):
        # Create args namespace
        args = argparse.Namespace()

        # Set base config values
        args.inner = base_config.get('inner', 1)
        args.innerdiv = base_config.get('innerdiv', 100)
        args.meta = base_config.get('meta', 32)
        args.num_iterations = base_config.get('num_iterations', 300000)
        args.gpu_id = base_config.get('gpu_id', '0')
        args.num_test_samples = base_config.get('num_test_samples', 1000)
        args.save_results = base_config.get('save_results', True)
        args.indices = base_config.get('indices', None)
        args.total_points = base_config.get('total_points', 61)
        args.num_optim_steps = base_config.get('num_optim_steps', 40)
        args.layer_length = base_config.get('layer_length', 40)
        args.mode = base_config.get('mode', 'extrapolation')
        args.model_path = base_config.get('model_path', None)
        args.output_prefix = base_config.get('output_prefix', None)
        args.cells = base_config.get('cells', None)
        args.measure_time = base_config.get('measure_time', False)

        # Override with sweep parameters
        for param_name, param_value in zip(param_names, combo):
            setattr(args, param_name, param_value)

        print(f"\n{'#'*80}")
        print(f"RUN {run_idx}/{total_runs}")
        print(f"{'#'*80}")
        print(f"Parameters: {dict(zip(param_names, combo))}")

        try:
            run_single_validation(args)
        except Exception as e:
            print(f"Error in run {run_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*80}")
    print(f"SWEEP COMPLETED: {total_runs} runs")
    print(f"{'='*80}")

    return 0


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='MAML Optimization Comparison Validation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Optimization methods compared:
  1. Grad+Move Only: No optimization, just scaling
  2. SGD 40 steps: Direct SGD (no grad/move)
  3. Adam 40 steps: Direct Adam (no grad/move)
  4. Selective Adam: Grad+Move + Adam if loss > threshold
  5. Full Adam 40: Full Adam optimization

Examples:
  Single run:
    python MAML_optim_comparison_validation.py --config 0 --cells NAND3x2
    python MAML_optim_comparison_validation.py --config 2 --mode extrapolation --data_type cell
    python MAML_optim_comparison_validation.py --config 1 --save_results

  Sweep mode:
    python MAML_optim_comparison_validation.py json_configs/maml_optim_comparison_sweep.json
        """
    )

    # Positional argument for sweep config (optional)
    parser.add_argument('sweep_config', type=str, nargs='?', default=None,
                        help='Path to JSON sweep config file (optional)')

    parser.add_argument('--config', type=int, choices=[0, 1, 2, 3, 6, 7],
                        help='Configuration ID. 0..3 = legacy datasets; '
                             '6 = TSMC combined patched intra eval; '
                             '7 = TSMC combined patched agnostic eval '
                             '(configs 6 and 7 share the same trained model)')
    parser.add_argument('--mode', type=str, choices=['extrapolation', 'interpolation'], default='extrapolation',
                        help='Testing mode (default: extrapolation)')
    parser.add_argument('--cells', type=str, nargs='+', default=None,
                        help='Cell types to test (default: config-dependent)')
    parser.add_argument('--inner', type=int, default=1,
                        help='Inner loop steps (default: 1)')
    parser.add_argument('--innerdiv', type=int, default=100,
                        help='Inner learning rate divisor (default: 100)')
    parser.add_argument('--meta', type=int, default=32,
                        help='Tasks per meta batch (default: config-dependent)')
    parser.add_argument('--num_iterations', type=int, default=300000,
                        help='Number of training iterations (default: config-dependent)')
    parser.add_argument('--data_type', type=str, default='cell',
                        help='Data type: cell/transition (default: config-dependent)')
    parser.add_argument('--gpu_id', type=str, default=None,
                        help='GPU device ID (default: config-dependent)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to pretrained model (optional)')
    parser.add_argument('--num_test_samples', type=int, default=2000,
                        help='Number of test samples (default: 100000)')
    parser.add_argument('--num_optim_steps', type=int, default=40,
                        help='Number of optimization steps for SGD/Adam (default: 40)')
    parser.add_argument('--save_results', action='store_true',
                        help='Save results to JSON file')
    parser.add_argument('--output_prefix', type=str, default=None,
                        help='Prefix for output files (default: config-dependent)')
    parser.add_argument('--indices', type=int, nargs='+', default=None,
                        help='Sampling indices for support set')
    parser.add_argument('--total_points', type=int, default=61,
                        help='Total number of data points (default: 61)')
    parser.add_argument('--layer_length', type=int, default=40,
                        help='Hidden layer size for MAML models (default: 40)')
    parser.add_argument('--measure_time', action='store_true',
                        help='Measure timing on CPU for consistent timing measurement')

    args = parser.parse_args()

    # Check if sweep config is provided
    if args.sweep_config:
        # Sweep mode
        if not os.path.exists(args.sweep_config):
            print(f"Error: Sweep config file not found: {args.sweep_config}")
            return 1
        return run_sweep(args.sweep_config)
    else:
        # Single run mode - require --config
        if args.config is None:
            print("Error: --config is required for single run mode")
            print("Usage:")
            print("  Single run: python MAML_optim_comparison_validation.py --config 0 --data_type cell")
            print("  Sweep mode: python MAML_optim_comparison_validation.py json_configs/maml_optim_comparison_sweep.json")
            return 1
        return run_single_validation(args)


if __name__ == "__main__":
    sys.exit(main())
