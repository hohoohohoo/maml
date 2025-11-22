#!/usr/bin/env python
# coding: utf-8

"""
ASAP7 Intra Topology Testing with MLP (Aadam)

This script evaluates MLP model performance on ASAP7 test datasets.
Supports both extrapolation and interpolation testing modes.
"""

import os
import sys
import torch
import numpy as np
import random
import argparse

# Import utility functions
from data_management_utils import (
    analyze_continuity,
    load_and_normalize_data,
    apply_normalization
)
from mlp_functions import evaluate_model_performance_mlp

# MLP import
sys.path.append('../../model_code/')
from networks import MLP_Aadam, MLP


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='ASAP7 Intra Topology Testing with MLP (Aadam)')
    parser.add_argument('--mode', type=str, default='extrapolation', choices=['extrapolation', 'interpolation'],
                        help='Testing mode: extrapolation or interpolation (default: extrapolation)')
    parser.add_argument('--cells', type=str, nargs='+', default=['NAND3x2', 'OR2x6', 'NOR2xp67', 'AND2x6'],
                        help='Cell types to test (default: NAND3x2 OR2x6 NOR2xp67 AND2x6)')
    parser.add_argument('--data_type', type=str, default='cell',
                        help='Data type: cell/transition (default: cell)')
    parser.add_argument('--gpu_id', type=str, default='2',
                        help='GPU device ID (default: 2)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to pretrained model (optional)')
    parser.add_argument('--num_test_samples', type=int, default=1000000,
                        help='Number of test samples to process (default: 1000000)')
    parser.add_argument('--save_results', action='store_true',
                        help='Save prediction results to .npy files')
    parser.add_argument('--output_prefix', type=str, default='ASAP7',
                        help='Prefix for output files (default: ASAP7)')
    parser.add_argument('--indices', type=int, nargs='+', default=None,
                        help='Sampling indices for support set (extrapolation: all indices, interpolation: middle indices only)')
    parser.add_argument('--total_points', type=int, default=61,
                        help='Total number of data points (default: 61)')
    parser.add_argument('--model_type', type=str, default='aadam', choices=['aadam', 'mlp'],
                        help='Model type: aadam (hidden=256) or mlp (hidden=40) (default: aadam)')
    parser.add_argument('--num_iterations', type=int, default=300000,
                        help='Number of iterations used during pretraining (default: 300000)')

    args = parser.parse_args()

    # Set default indices based on mode if not provided
    if args.indices is None:
        if args.mode == 'extrapolation':
            args.indices = [5, 30, 55]
        else:  # interpolation
            args.indices = [15, 30, 45]

    # Process indices based on mode
    if args.mode == 'interpolation':
        # For interpolation: automatically add endpoints (0 and total_points-1)
        middle_indices = sorted(set(args.indices))

        # Add endpoints
        if 0 not in middle_indices:
            middle_indices = [0] + middle_indices
        if args.total_points - 1 not in middle_indices:
            middle_indices = middle_indices + [args.total_points - 1]

        args.indices = middle_indices

    # Calculate K, left_bound, right_bound from indices
    args.k = len(args.indices)
    args.left_bound = min(args.indices)
    args.right_bound = max(args.indices) + 1

    # Validate indices
    if args.k == 0:
        raise ValueError("At least one index must be provided")

    if args.mode == 'interpolation' and args.k < 2:
        raise ValueError("At least 2 indices (including endpoints) are required for interpolation")

    for idx in args.indices:
        if idx < 0 or idx >= args.total_points:
            raise ValueError(f"All indices must be within [0, {args.total_points - 1}]. Got: {args.indices}")

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
    print(f"   Mode: {args.mode}")
    print(f"   Data type: {data_type}")
    print(f"   Model type: {model_type}")
    print(f"   Cells: {args.cells}")
    print(f"   GPU ID: {args.gpu_id}")
    print(f"   Indices: {args.indices}")
    print(f"   → Calculated K (support samples): {args.k}")
    print(f"   → Calculated left_bound: {args.left_bound}")
    print(f"   → Calculated right_bound: {args.right_bound}")
    print(f"   → Total points: {args.total_points}")
    print(f"   → Num iterations: {args.num_iterations}")

    # Load training data and calculate normalization statistics
    train_data_paths = [
        (f"/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_ASAP7/unified_invbuf/merged_invbuf_input_{data_type}.pth",
         f"/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_ASAP7/unified_invbuf/merged_invbuf_output_{data_type}.pth"),
        (f"/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_ASAP7/intra_topology_data_upgraded/{data_type}_intratopology_train_input.pth",
         f"/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_ASAP7/intra_topology_data_upgraded/{data_type}_intratopology_train_output.pth")
    ]

    norm_stats = load_and_normalize_data(train_data_paths)

    # Load MLP model
    print(f"\n🤖 Loading {model_type.upper()} model...")

    # Auto-detect model path if not provided
    if args.model_path is None:
        model_path = f"/home/tkdgn2907/Deepsets_test/MAML/Projects/pretraining/model_pretraining_code/MLP_pretrained_model/pretrained_mlp1_intratopology_{data_type}_{model_type}_{args.num_iterations}.pth"
    else:
        model_path = args.model_path

    input_features = 9  # ASAP7 has 9 features

    # Select model based on model_type
    if model_type == 'aadam':
        mlp_model = MLP_Aadam(input_size=input_features, output_size=1).to(device)
    else:  # mlp
        mlp_model = MLP(input_size=input_features, output_size=1).to(device)

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        mlp_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded model: {model_path}")
    else:
        print(f"⚠️ Model file not found: {model_path}")
        print("Please update the model path or ensure the model has been trained")
        return

    # Random sampling setup (using command line arguments)
    K = args.k
    indices = args.indices
    middle_idx = len(indices) // 2  # Calculate middle index for move parameter

    results_dict = {}
    all_predictions_global = []
    all_actuals_global = []

    # Process each cell type
    for cell in args.cells:
        print(f"\n{'='*80}")
        print(f"Processing cell: {cell}")
        print(f"{'='*80}")

        # Load test dataset
        print("\n📊 Loading TEST dataset...")
        data_dir = "/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_ASAP7/intra_topology_data_upgraded"
        test_input_path = f"{data_dir}/{cell}/{data_type}_{cell}_test_input.pth"
        test_output_path = f"{data_dir}/{cell}/{data_type}_{cell}_test_output.pth"

        test_data_input = torch.load(test_input_path)
        test_data_output = torch.load(test_output_path)

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
        total_extra_l = []
        total_extra_r = []
        total_inter = []
        total_l_mape = []
        total_r_mape = []
        total_in_mape = []
        total_mape = []
        total_mae = []
        total_l_mae = []
        total_r_mae = []
        total_in_mae = []

        for i, randomtask in enumerate(test_indices):
            # Skip discontinuous tasks
            if randomtask in discontinuous_task_ids:
                continue

            if i % 100 == 0:
                print(f"Processing task {i+1}/{num_test_samples} (index: {randomtask})")

            try:
                X = test_data_input[randomtask][indices]
                y = test_data_output[randomtask][indices]

                # Define regions based on mode
                testdata_inter_output = test_data_output[randomtask][args.left_bound:args.right_bound]
                y_inter_mean = testdata_inter_output.mean()
                y1_mean = test_data_output[randomtask].mean()

                # Only calculate left/right regions for extrapolation mode
                if args.mode == 'extrapolation':
                    testdata_rightex_output = test_data_output[randomtask][args.right_bound:]
                    testdata_leftex_output = test_data_output[randomtask][:args.left_bound]
                    y_leftex_mean = testdata_leftex_output.mean()
                    y_rightex_mean = testdata_rightex_output.mean()

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

                    # Get model predictions for scaling (using parameterized boundaries)
                    predictions = mlp_model(test_data_input[randomtask][args.left_bound:args.right_bound])

                    min_val = predictions.min().item()
                    max_val = predictions.max().item()

                    if abs(max_val - min_val) > 0:  # Avoid division by zero
                        grad = (y_max - y_min) / (max_val - min_val)

                        # Calculate move parameter (using middle index of support set)
                        move = center - y_norm[middle_idx,0] / grad

                        (total_loss1, inter_loss1, leftex_loss1, rightex_loss1,
                        mape_loss1, leftex_mape1, inter_mape1, rightex_mape1,
                        predictions, actual_values, _, _, _, _, _, _, adam_used) = evaluate_model_performance_mlp(
                            mlp_model, 'Aadam', X, y,
                            test_data_input[randomtask], test_data_output[randomtask], grad, move,
                            left_bound=args.left_bound, right_bound=args.right_bound, total_points=args.total_points,
                            mode=args.mode
                        )

                        if adam_used:
                            adam_condition_count += 1

                        # Add to global collections
                        all_predictions_global.extend(predictions)
                        all_actuals_global.extend(actual_values)

                        # Calculate MAE values for each region (using parameterized boundaries)
                        mae_total = 0
                        mae_leftex = 0
                        mae_inter = 0
                        mae_rightex = 0

                        for idx in range(args.total_points):
                            abs_error = abs(predictions[idx] - actual_values[idx])
                            mae_total += abs_error
                            if idx < args.left_bound:  # Left extrapolation
                                mae_leftex += abs_error
                            elif idx < args.right_bound:  # Interpolation
                                mae_inter += abs_error
                            else:  # Right extrapolation
                                mae_rightex += abs_error

                        # Average MAE for each region
                        mae_total = mae_total / args.total_points
                        mae_inter = mae_inter / (args.right_bound - args.left_bound) if (args.right_bound - args.left_bound) > 0 else 0

                        # Calculate region-specific metrics based on mode
                        if args.mode == 'extrapolation':
                            mae_leftex = mae_leftex / args.left_bound if args.left_bound > 0 else 0
                            mae_rightex = mae_rightex / (args.total_points - args.right_bound) if (args.total_points - args.right_bound) > 0 else 0
                            nrmse_leftex = (leftex_loss1 ** 0.5) / (abs(y_leftex_mean) + 1e-4) * 100
                            nrmse_rightex = (rightex_loss1 ** 0.5) / (abs(y_rightex_mean) + 1e-4) * 100
                            mape_l_percent = leftex_mape1 * 100
                            mape_r_percent = rightex_mape1 * 100
                        else:  # interpolation mode
                            mae_leftex = 0
                            mae_rightex = 0
                            nrmse_leftex = 0
                            nrmse_rightex = 0
                            mape_l_percent = 0
                            mape_r_percent = 0

                        # Calculate common metrics
                        nrmse1 = (total_loss1 ** 0.5) / (abs(y1_mean) + 1e-4) * 100
                        nrmse_inter = (inter_loss1 ** 0.5) / (abs(y_inter_mean) + 1e-4) * 100
                        mape_percent = mape_loss1 * 100
                        mape_in_percent = inter_mape1 * 100

                        if not(torch.isinf(nrmse1) or nrmse1 > 1000 or torch.isnan(nrmse1)):
                            total_nrmse.append(nrmse1.item())
                            # Handle scalar values in interpolation mode
                            total_extra_l.append(nrmse_leftex.item() if torch.is_tensor(nrmse_leftex) else nrmse_leftex)
                            total_extra_r.append(nrmse_rightex.item() if torch.is_tensor(nrmse_rightex) else nrmse_rightex)
                            total_inter.append(nrmse_inter.item())
                            total_mape.append(mape_percent)
                            total_r_mape.append(mape_r_percent)
                            total_l_mape.append(mape_l_percent)
                            total_in_mape.append(mape_in_percent)
                            total_mae.append(mae_total)
                            total_l_mae.append(mae_leftex)
                            total_r_mae.append(mae_rightex)
                            total_in_mae.append(mae_inter)

                        if i % 100 == 0 and i > 0:  # Print progress every 100 samples
                            print(f"  Adam usage: {(adam_condition_count/len(total_nrmse)):.2f}")
                            print(f"  Current avg NRMSE: {sum(total_nrmse)/len(total_nrmse):.2f}% - Tasks: {len(total_nrmse)}")
                            print(f"  Current avg MAPE: {sum(total_mape)/len(total_mape):.2f}% - Tasks: {len(total_mape)}")
                            print(f"  Current avg MAE: {sum(total_mae)/len(total_mae):.6f} - Tasks: {len(total_mae)}")
            except Exception as e:
                if i < 10:  # Print first few errors
                    print(f"Error processing task {randomtask}: {e}")
                continue

        print(f"\n✅ Completed processing {len(total_nrmse)} valid tasks for {cell}")

        # Store results
        key_nrmse = f"NRMSE_{cell}"
        key_mape = f"MAPE_{cell}"
        key_mae = f"MAE_{cell}"

        results_dict[key_nrmse] = [[sum(total_nrmse)/len(total_nrmse),
                                    sum(total_extra_l)/len(total_extra_l),
                                    sum(total_extra_r)/len(total_extra_r),
                                    sum(total_inter)/len(total_inter)]]
        results_dict[key_mape] = [[sum(total_mape)/len(total_mape),
                                   sum(total_l_mape)/len(total_l_mape),
                                   sum(total_r_mape)/len(total_r_mape),
                                   sum(total_in_mape)/len(total_in_mape)]]
        results_dict[key_mae] = [[sum(total_mae)/len(total_mae),
                                  sum(total_l_mae)/len(total_l_mae),
                                  sum(total_r_mae)/len(total_r_mae),
                                  sum(total_in_mae)/len(total_in_mae)]]
        if args.save_results:
            pred_filename = f"data_result_npy_directory/{args.output_prefix}_intra_topology_{cell}_{data_type}_{args.mode}_Aadam_pred.npy"
            act_filename = f"data_result_npy_directory/{args.output_prefix}_intra_topology_{cell}_{data_type}_{args.mode}_Aadam_act.npy"

            np.save(pred_filename, all_predictions_global)
            np.save(act_filename, all_actuals_global)

            print(f"\n💾 Saved predictions to: {pred_filename}")
            print(f"💾 Saved actuals to: {act_filename}")

        print(f"\n📊 Results for {cell}:")
        print(f"   NRMSE (Total/Left/Right/Inter): {results_dict[key_nrmse]}")
        print(f"   MAPE (Total/Left/Right/Inter): {results_dict[key_mape]}")
        print(f"   MAE (Total/Left/Right/Inter): {results_dict[key_mae]}")

    # Print overall results
    print(f"\n{'='*80}")
    print("Overall Results")
    print(f"{'='*80}")

    for key, value in results_dict.items():
        print(f"{key}:")
        for arr in value:
            print(f"  {np.array(arr)}")

    # Save results if requested


if __name__ == "__main__":
    main()
