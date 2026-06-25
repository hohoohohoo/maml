#!/usr/bin/env python
# coding: utf-8

"""
Unified MLP Testing Script

This script handles all MLP testing configurations:
- Config 0: ASAP7 Intra Topology
- Config 1: ASAP7 Topology Agnostic
- Config 2: TSMC Intra Topology
- Config 3: TSMC Topology Agnostic

All dataset configurations are managed through test_dataset_config.py
"""

import os
import sys
import torch
import numpy as np
import random
import argparse

# Add parent directory to path for utils access
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import configuration
from utils.test_dataset_config import (
    get_test_config,
    get_train_data_paths,
    get_test_data_paths,
    get_mlp_model_path,
    print_available_configs
)

# Import utility functions
from utils.data_management_utils import (
    analyze_continuity,
    load_and_normalize_data,
    apply_normalization
)
from utils.mlp_functions import evaluate_model_performance_mlp

# MLP import
sys.path.append('../../../../model_code/')
from baseline_mlp import MLP_Aadam, MLP


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Unified MLP Testing (Extrapolation/Interpolation)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available configurations:
  0: ASAP7 Intra Topology
  1: ASAP7 Topology Agnostic
  2: TSMC Intra Topology
  3: TSMC Topology Agnostic

Examples:
  python unified_mlp_test.py --config 0 --cells NAND3x2 OR2x6
  python unified_mlp_test.py --config 2 --mode interpolation --model_type mlp
  python unified_mlp_test.py --config 1 --save_results
        """
    )

    parser.add_argument('--config', type=int, required=True, choices=[0, 1, 2, 3, 6, 7],
                        help='Configuration ID. 0..3 = legacy datasets; '
                             '6 = TSMC combined patched intra eval; '
                             '7 = TSMC combined patched agnostic eval')
    parser.add_argument('--mode', type=str, choices=['extrapolation', 'interpolation'], default='extrapolation',
                        help='Testing mode: extrapolation or interpolation (default: extrapolation)')
    parser.add_argument('--cells', type=str, nargs='+', default=None,
                        help='Cell types to test (default: config-dependent)')
    parser.add_argument('--data_type', type=str, default=None,
                        help='Data type: cell/transition (default: config-dependent)')
    parser.add_argument('--model_type', type=str, default='aadam', choices=['aadam', 'mlp'],
                        help='Model type: aadam (hidden=256) or mlp (hidden=40) (default: aadam)')
    parser.add_argument('--num_iterations', type=int, default=300000,
                        help='Number of iterations used during pretraining (default: 300000)')
    parser.add_argument('--gpu_id', type=str, default=None,
                        help='GPU device ID (default: config-dependent)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to pretrained model (optional)')
    parser.add_argument('--num_test_samples', type=int, default=1000000,
                        help='Number of test samples to process (default: 1000000)')
    parser.add_argument('--save_results', action='store_true',
                        help='Save prediction results to .npy files')
    parser.add_argument('--output_prefix', type=str, default=None,
                        help='Prefix for output files (default: config-dependent)')
    parser.add_argument('--indices', type=int, nargs='+', default=None,
                        help='Sampling indices for support set (default: mode-dependent)')
    parser.add_argument('--total_points', type=int, default=61,
                        help='Total number of data points (default: 61)')
    parser.add_argument('--adaptation_method', type=str, default='selective_adam',
                        choices=['selective_adam', 'adam', 'sgd'],
                        help='Adaptation method: selective_adam (grad/move + conditional Adam), '
                             'adam (direct Adam, no grad/move), or sgd (direct vanilla SGD, '
                             'no grad/move — mirrors MAML inner-loop). Default: selective_adam.')

    args = parser.parse_args()

    # Get configuration
    try:
        config = get_test_config(args.config)
    except ValueError as e:
        print(f"Error: {e}")
        print("\n")
        print_available_configs()
        return 1

    # Set defaults from config
    if args.cells is None:
        args.cells = config['default_cells']
    if args.data_type is None:
        args.data_type = config['default_data_type']
    if args.gpu_id is None:
        args.gpu_id = config['default_gpu']
    if args.output_prefix is None:
        args.output_prefix = config['tech'].upper()

    # Set mode-dependent default indices if not provided
    if args.indices is None:
        if args.mode == 'extrapolation':
            args.indices = [5, 30, 55]
        else:  # interpolation
            args.indices = [13, 30, 45]

    # For interpolation mode: automatically add endpoints if not present
    if args.mode == 'interpolation':
        middle_indices = sorted(set(args.indices))
        if 0 not in middle_indices:
            middle_indices = [0] + middle_indices
        if args.total_points - 1 not in middle_indices:
            middle_indices = middle_indices + [args.total_points - 1]
        args.indices = middle_indices
        print(f"\n🔧 Interpolation mode: Added endpoints to indices → {args.indices}")

    # Calculate K, left_bound, right_bound from indices
    args.k = len(args.indices)
    args.left_bound = min(args.indices)
    args.right_bound = max(args.indices) + 1

    # Validate indices
    if args.k == 0:
        raise ValueError("At least one index must be provided")

    if args.left_bound < 0 or args.right_bound > args.total_points:
        raise ValueError(f"Indices must be within [0, {args.total_points - 1}]. "
                        f"Got indices range [{args.left_bound}, {max(args.indices)}]")

    # GPU settings
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:', device)
    if torch.cuda.is_available():
        print('Current cuda device:', torch.cuda.current_device())
        print('Count of using GPUs:', torch.cuda.device_count())

    # Configuration
    data_type = args.data_type.lower()
    model_type = args.model_type.lower()

    print(f"\n⚙️ Configuration:")
    print(f"   Config ID: {args.config}")
    print(f"   Config name: {config['name']}")
    print(f"   Mode: {args.mode}")
    print(f"   Data type: {data_type}")
    print(f"   Model type: {model_type}")
    print(f"   Adaptation method: {args.adaptation_method}")
    print(f"   Cells: {args.cells}")
    print(f"   GPU ID: {args.gpu_id}")
    print(f"   Num iterations: {args.num_iterations}")
    print(f"   Indices: {args.indices}")
    print(f"   → Calculated K (support samples): {args.k}")
    print(f"   → Calculated left_bound: {args.left_bound}")
    print(f"   → Calculated right_bound: {args.right_bound}")
    print(f"   → Total points: {args.total_points}")

    # Load training data and calculate normalization statistics
    print("\n📊 Loading TRAINING dataset for normalization statistics...")
    train_data_paths = get_train_data_paths(args.config, data_type)
    norm_stats = load_and_normalize_data(train_data_paths)

    # Load MLP model
    print(f"\n🤖 Loading {model_type.upper()} model...")

    # Get model path from config
    model_path = get_mlp_model_path(
        args.config,
        data_type=data_type,
        model_type=model_type,
        num_iterations=args.num_iterations,
        custom_path=args.model_path
    )

    input_features = 9  # Both ASAP7 and TSMC have 9 features

    # Select model based on model_type
    if model_type == 'aadam':
        mlp_model = MLP_Aadam(input_size=input_features, output_size=1).to(device)
    else:  # mlp
        mlp_model = MLP(input_size=input_features, output_size=1).to(device)

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        mlp_model.load_state_dict(checkpoint['model_state_dict'])
        mlp_model.eval()
        print(f"✅ Loaded model: {model_path}")
    else:
        print(f"⚠️ Model file not found: {model_path}")
        print("Please update the model path or ensure the model has been trained")
        return 1

    # Random sampling setup
    K = args.k
    indices = args.indices
    middle_idx = len(indices) // 2  # Calculate middle index for move parameter

    results_dict = {}

    # Process each cell type
    for cell in args.cells:
        print(f"\n{'='*80}")
        print(f"Processing cell: {cell}")
        print(f"{'='*80}")

        # Reset per-cell collections
        all_predictions_global = []
        all_actuals_global = []

        # Load test dataset from config
        print("\n📊 Loading TEST dataset...")
        test_input_path, test_output_path = get_test_data_paths(args.config, cell, data_type)

        try:
            test_data_input = torch.load(test_input_path)
            test_data_output = torch.load(test_output_path)
        except FileNotFoundError as e:
            print(f"⚠️ Test data not found for cell {cell}: {e}")
            print(f"   Input:  {test_input_path}")
            print(f"   Output: {test_output_path}")
            continue

        # Add dimension to output if needed
        if len(test_data_output.shape) == 2:
            test_data_output = test_data_output.unsqueeze(-1)

        print(f"Test input shape: {test_data_input.shape}")
        print(f"Test output shape: {test_data_output.shape}")
        print(f"Number of test samples: {len(test_data_input)}")

        # Apply normalization
        apply_normalization(test_data_input, norm_stats)

        # Analyze continuity
        print("\n🔍 Analyzing data continuity...")
        continuous_task_ids, discontinuous_task_ids, continuity_analysis = analyze_continuity(
            test_data_input, test_data_output, threshold_ratio=0.18
        )

        # Move data to GPU
        test_data_input = test_data_input.to(device)
        test_data_output = test_data_output.to(device)

        # Test sampling
        num_test_samples = min(args.num_test_samples, len(test_data_input))
        test_indices = random.sample(range(len(test_data_input)), num_test_samples)

        print(f"\n🎲 Selected {num_test_samples} random test tasks from {len(test_data_input)} total tasks")

        # Process test tasks
        print(f"\n🔄 Processing {num_test_samples} test tasks (excluding discontinuous ones)...")
        adam_condition_count = 0
        total_nrmse = []
        total_mape = []
        total_mae = []
        print("parameter check")
        print(f"K:{K},middle:{middle_idx},left_value:{args.left_bound},right_value:{args.right_bound}")
        for i, randomtask in enumerate(test_indices):
            # Skip discontinuous tasks
            if randomtask in discontinuous_task_ids:
                continue

            if i % 100 == 0:
                print(f"Processing task {i+1}/{num_test_samples} (index: {randomtask})")

            try:
                X = test_data_input[randomtask][indices]
                y = test_data_output[randomtask][indices]

                # Calculate mean for NRMSE normalization
                y1_mean = test_data_output[randomtask].mean()

                y_mean = y.mean()
                y_std = y.std()

                if y_std > 0:  # Only process if there's variation in the data
                    y_norm = (y - y_mean) / y_std

                    # Create center input
                    center_input = torch.zeros((1, X.shape[1])).to(device)
                    center_input[0, 4] = 0.0  # voltage = 0 (normalized)
                    center_input[0, :4] = X[0, :4]  # copy first 4 features
                    center_input[0, 5:] = X[0, 5:]  # copy remaining features
                    center = mlp_model(center_input).item()

                    y_max = y_norm[:,0].max()
                    y_min = y_norm[:,0].min()

                    # Get model predictions for scaling
                    predictions = mlp_model(test_data_input[randomtask][args.left_bound:args.right_bound])

                    min_val = predictions.min().item()
                    max_val = predictions.max().item()

                    if abs(max_val - min_val) > 0:  # Avoid division by zero
                        grad = (y_max - y_min) / (max_val - min_val)

                        # Calculate move parameter
                        move = center - y_norm[middle_idx,0] / grad

                        (total_loss1, mape_loss1,
                        predictions, actual_values, _, adam_used,
                        mae_loss) = evaluate_model_performance_mlp(
                            mlp_model, 'MLP', X, y,
                            test_data_input[randomtask], test_data_output[randomtask], grad, move,
                            left_bound=args.left_bound, right_bound=args.right_bound, total_points=args.total_points,
                            mode=args.mode, adaptation_method=args.adaptation_method
                        )

                        if adam_used:
                            adam_condition_count += 1

                        # Add to global collections
                        all_predictions_global.extend(predictions)
                        all_actuals_global.extend(actual_values)

                        # Calculate metrics
                        nrmse1 = (total_loss1 ** 0.5) / (abs(y1_mean) + 1e-4) * 100
                        mape_percent = mape_loss1 * 100

                        if not(torch.isinf(nrmse1) or torch.isnan(nrmse1)):
                            total_nrmse.append(nrmse1.item())
                            total_mape.append(mape_percent)
                            total_mae.append(mae_loss)

                        if i % 100 == 0 and i > 0:
                            print(f"  Adam usage: {(adam_condition_count/len(total_nrmse)):.2f}")
                            print(f"  Current avg NRMSE: {sum(total_nrmse)/len(total_nrmse):.2f}% - Tasks: {len(total_nrmse)}")
                            print(f"  Current avg MAPE: {sum(total_mape)/len(total_mape):.2f}% - Tasks: {len(total_mape)}")
                            print(f"  Current avg MAE: {1000*sum(total_mae)/len(total_mae):.3f}ps - Tasks: {len(total_mae)}")
            except Exception as e:
                if i < 10:
                    print(f"Error processing task {randomtask}: {e}")
                continue

        print(f"\n✅ Completed processing {len(total_nrmse)} valid tasks for {cell}")

        # Store results
        key_nrmse = f"NRMSE_{cell}"
        key_mape = f"MAPE_{cell}"
        key_mae = f"MAE_{cell}"

        results_dict[key_nrmse] = sum(total_nrmse)/len(total_nrmse)
        results_dict[key_mape] = sum(total_mape)/len(total_mape)
        results_dict[key_mae] = sum(total_mae)/len(total_mae)

        # Save results for this cell if requested
        if args.save_results:
            # Adaptation method suffix: 'selective_adam' is default → no suffix,
            # 'adam' → _adam, 'sgd' → _sgd.
            adapt_suffix = {
                'adam': '_adam',
                'sgd':  '_sgd',
            }.get(args.adaptation_method, '')

            pred_filename = f"data_result_npy_directory_baseline/{args.output_prefix}_{config['topology_type']}_{cell}_{data_type}_{args.mode}_{model_type}_{args.num_iterations}{adapt_suffix}_pred.npy"
            act_filename = f"data_result_npy_directory_baseline/{args.output_prefix}_{config['topology_type']}_{cell}_{data_type}_{args.mode}_{model_type}_{args.num_iterations}{adapt_suffix}_act.npy"

            os.makedirs("data_result_npy_directory_baseline", exist_ok=True)

            np.save(pred_filename, all_predictions_global)
            np.save(act_filename, all_actuals_global)

            print(f"\n💾 Saved predictions to: {pred_filename}")
            print(f"💾 Saved actuals to: {act_filename}")

        print(f"\n📊 Results for {cell}:")
        print(f"   NRMSE: {results_dict[key_nrmse]:.4f}%")
        print(f"   MAPE: {results_dict[key_mape]:.4f}%")
        print(f"   MAE: {results_dict[key_mae]*1000:.3f}ps")

    # Print overall results
    print(f"\n{'='*80}")
    print("Overall Results")
    print(f"{'='*80}")

    for key, value in results_dict.items():
        print(f"{key}: {value:.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
