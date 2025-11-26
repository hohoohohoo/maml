#!/usr/bin/env python
# coding: utf-8

"""
TSMC Voltage Variation Validation with MLP/MAML

This script validates MLP or MAML model performance on TSMC voltage variation test datasets.
Supports multiple corner conditions (ff/ss/tt) and temperatures (0/25/50/75/100).
Supports both extrapolation and interpolation testing modes.
"""

import os
import sys
import torch
import numpy as np
import random
import argparse

# Import utility functions
from utils.data_management_utils import (
    analyze_continuity
)
from utils.mlp_functions import evaluate_model_performance_mlp
from utils.maml_functions import evaluate_model_performance_maml

# Model imports
sys.path.append('../../model_code/')
from networks import MLP_Aadam, MLP
from maml import MAMLModel_3hidden


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='TSMC Voltage Variation Validation with MLP/MAML')
    parser.add_argument('--mode', type=str, choices=['extrapolation', 'interpolation'], default='extrapolation',
                        help='Testing mode: extrapolation or interpolation (default: extrapolation)')
    parser.add_argument('--corner', type=str, default='ff', choices=['ff', 'ss', 'tt'],
                        help='Corner condition: ff/ss/tt (default: ff)')
    parser.add_argument('--temperatures', type=str, nargs='+', default=['0', '25', '50', '75', '100'],
                        help='Temperatures to process: 0/25/50/75/100 (default: all 5 temperatures)')
    parser.add_argument('--data_type', type=str, default='cell',
                        help='Data type: cell/transition (default: cell)')
    parser.add_argument('--gpu_id', type=str, default='6',
                        help='GPU device ID (default: 6)')
    parser.add_argument('--model_framework', type=str, default='mlp', choices=['mlp', 'maml'],
                        help='Model framework: mlp or maml (default: mlp)')
    parser.add_argument('--model_type', type=str, default='aadam', choices=['aadam', 'mlp'],
                        help='For MLP: aadam (hidden=256) or mlp (hidden=40) (default: aadam)')
    parser.add_argument('--num_iterations', type=int, default=100000,
                        help='Number of iterations used during pretraining (default: 100000)')

    # MAML-specific parameters
    parser.add_argument('--layer_length', type=int, default=40,
                        help='MAML hidden layer size (default: 40)')
    parser.add_argument('--inner_step', type=int, default=1,
                        help='MAML inner loop steps (default: 1)')
    parser.add_argument('--innerdiv', type=int, default=10,
                        help='MAML inner learning rate divisor (default: 10)')
    parser.add_argument('--meta', type=int, default=32,
                        help='MAML tasks per meta batch (default: 32)')

    parser.add_argument('--num_test_samples', type=int, default=100000,
                        help='Number of test samples to process (default: 100000)')
    parser.add_argument('--save_results', action='store_true',
                        help='Save prediction and actual results to .npy files')
    parser.add_argument('--indices', type=int, nargs='+', default=None,
                        help='Sampling indices for support set (default: mode-dependent)')
    parser.add_argument('--total_points', type=int, default=61,
                        help='Total number of data points (default: 61)')

    args = parser.parse_args()

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

    # GPU settings
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:', device)
    if torch.cuda.is_available():
        print('Current cuda device:', torch.cuda.current_device())
        print('Count of using GPUs:', torch.cuda.device_count())

    # Configuration
    corner = args.corner.lower()
    temperatures = args.temperatures
    data_type = args.data_type.lower()
    model_framework = args.model_framework.lower()
    model_type = args.model_type.lower()

    print(f"\n⚙️ Configuration:")
    print(f"   Mode: {args.mode}")
    print(f"   Corner: {corner.upper()}")
    print(f"   Temperatures: {', '.join(temperatures)}°C")
    print(f"   Data type: {data_type}")
    print(f"   Model framework: {model_framework.upper()}")
    if model_framework == 'mlp':
        print(f"   → Model type: {model_type} (hidden={'256' if model_type=='aadam' else '40'})")
        print(f"   → Num iterations: {args.num_iterations}")
    else:  # maml
        print(f"   → Layer length: {args.layer_length}")
        print(f"   → Inner steps: {args.inner_step}")
        print(f"   → Inner div: {args.innerdiv}")
        print(f"   → Meta batch: {args.meta}")
        print(f"   → Num iterations: {args.num_iterations}")
    print(f"   GPU ID: {args.gpu_id}")
    print(f"   Support set indices: {args.indices} (K={args.k})")
    print(f"   Left bound: {args.left_bound}, Right bound: {args.right_bound}")

    # Store results for all temperatures
    results_dict = {}

    # Loop through all temperatures
    for temp in temperatures:
        print(f"\n{'='*80}")
        print(f"Processing Temperature: {temp}°C")
        print(f"{'='*80}")

        # Load test data
        print("\n📊 Loading test data...")
        test_data_input = torch.load(f"/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_test5(dim5)_TSMC/taskdivide_{corner}_{temp}/testdatainput/{data_type}_test_input.pth")
        test_data_output = torch.load(f"/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_test5(dim5)_TSMC/taskdivide_{corner}_{temp}/testdataoutput/{data_type}_test_output.pth")

        # Determine model path based on framework
        if model_framework == 'mlp':
            model_path = f"../../pretrained_models/MLP_pretrained_model/pretrained_tsmc_{corner.upper()}_{temp}_test5_{data_type}_{model_type}_{args.num_iterations}.pth"
        else:  # maml
            model_path = f"../../pretrained_models/taskdivide_all/{data_type}_innerdiv{args.innerdiv}_meta{args.meta}_full1DMAML_weights_3hidden_({args.layer_length})_{args.num_iterations}_TSMC_{corner.upper()}_{temp}_test5(dim5)_inner{args.inner_step}.pth"

        print(f"\n🤖 Loading {model_framework.upper()} model...")
        print(f"Model path: {model_path}")

        if not os.path.exists(model_path):
            print(f"⚠️ Model file not found: {model_path}")
            print(f"   Skipping temperature {temp}°C")
            continue

        # Load model and apply normalization if MLP
        if model_framework == 'mlp':
            checkpoint = torch.load(model_path, map_location=device)
            feature_means = checkpoint['feature_means']
            feature_stds = checkpoint['feature_stds']

            # Apply same normalization as during MLP training (features 0,3,4 only)
            normalize_indices = [0, 3, 4]
            for feature_idx in normalize_indices:
                if feature_means[feature_idx] is not None and feature_stds[feature_idx] is not None:
                    mean_val = feature_means[feature_idx].item() if torch.is_tensor(feature_means[feature_idx]) else feature_means[feature_idx]
                    std_val = feature_stds[feature_idx].item() if torch.is_tensor(feature_stds[feature_idx]) else feature_stds[feature_idx]
                    test_data_input[:,:,feature_idx] = ((test_data_input[:,:,feature_idx] - mean_val) / std_val)
                    print(f"Applied normalization to feature {feature_idx}: mean={mean_val:.4f}, std={std_val:.4f}")

            print(f"Input shape: {test_data_input.shape}")
            print(f"Output shape: {test_data_output.shape}")
            print(f"Features normalized: {normalize_indices}")
        else:  # maml - no normalization
            print(f"Input shape: {test_data_input.shape}")
            print(f"Output shape: {test_data_output.shape}")
            print(f"No normalization applied for MAML")

        # Analyze continuity
        continuous_task_ids, discontinuous_task_ids, continuity_analysis = analyze_continuity(
            test_data_input, test_data_output, threshold_ratio=0.18, max_check_samples=100000
        )

        # Create and load model
        input_features = test_data_input.shape[2]  # 5D input

        if model_framework == 'mlp':
            # Select MLP model based on model_type
            if model_type == 'aadam':
                model = MLP_Aadam(input_size=input_features, output_size=1).to(device)
            else:  # mlp
                model = MLP(input_size=input_features, output_size=1).to(device)

            # Load trained MLP model weights
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded MLP model: {model_path}")
        else:  # maml
            # Create MAML model
            model = MAMLModel_3hidden(in_features=input_features, layer_length=args.layer_length).to(device)

            # Load trained MAML model weights
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"✅ Loaded MAML model: {model_path}")

        # Test on a subset of data with random sampling
        num_test_samples = min(args.num_test_samples, len(test_data_input))
        test_indices = random.sample(range(len(test_data_input)), num_test_samples)

        print(f"\n🎲 Selected {num_test_samples} random test tasks from {len(test_data_input)} total tasks")

        # Global collections for analysis
        total_nrmse = []
        total_extra_l = []
        total_extra_r = []
        total_inter = []
        total_mape = []
        total_mape_l = []
        total_mape_r = []
        total_mape_inter = []
        total_mae = []
        total_mae_l = []
        total_mae_r = []
        total_mae_inter = []

        all_predictions_global = []
        all_actuals_global = []

        testdata_output = test_data_output.to(device)
        testdata_input = test_data_input.to(device)

        print(f"\n🔄 Processing {num_test_samples} random tasks (excluding discontinuous ones)...")

        # Get middle index for move calculation
        middle_idx = len(args.indices) // 2

        for i, randomtask in enumerate(test_indices):
            if randomtask not in discontinuous_task_ids:
                if i % 50 == 0:
                    print(f"Processing task {i+1}/{num_test_samples} (index: {randomtask})")

                try:
                    X = testdata_input[randomtask][args.indices]
                    y = testdata_output[randomtask][args.indices]

                    # Define regions
                    testdata_rightex_output = testdata_output[randomtask][args.right_bound:]
                    testdata_leftex_output = testdata_output[randomtask][:args.left_bound]
                    testdata_inter_output = testdata_output[randomtask][args.left_bound:args.right_bound]

                    y_leftex_mean = testdata_leftex_output.mean()
                    y_rightex_mean = testdata_rightex_output.mean()
                    y_inter_mean = testdata_inter_output.mean()
                    y1_mean = testdata_output[randomtask].mean()

                    y_mean = y.mean()
                    y_std = y.std()

                    if y_std > 1e-8:
                        y_norm = (y - y_mean) / y_std

                        # Create center input for 5D
                        center_input = torch.zeros((1, X.shape[1])).to(device)
                        center_input[0, 0] = 0.0
                        center_input[0, 1:] = X[0, 1:]
                        center = model(center_input).item()

                        y_max = y_norm[:,0].max()
                        y_min = y_norm[:,0].min()

                        # Get model predictions for scaling
                        predictions = model(testdata_input[randomtask])
                        min_val = predictions.min().item()
                        max_val = predictions.max().item()

                        if abs(max_val - min_val) > 1e-8:
                            grad = (y_max - y_min) / (max_val - min_val)
                            move = center - y_norm[middle_idx,0] / grad

                            # Call appropriate evaluate function
                            if model_framework == 'mlp':
                                (total_loss1, inter_loss1, leftex_loss1, rightex_loss1,
                                 mape_loss1, leftex_mape1, inter_mape1, rightex_mape1,
                                 predictions, actual_values, _, _, _, _, _, _, adam_used) = evaluate_model_performance_mlp(
                                    model, model_framework.upper(), X, y,
                                    testdata_input[randomtask], testdata_output[randomtask], grad, move,
                                    left_bound=args.left_bound, right_bound=args.right_bound,
                                    total_points=args.total_points, mode=args.mode
                                )
                            else:  # maml
                                (total_loss1, inter_loss1, leftex_loss1, rightex_loss1,
                                 mape_loss1, leftex_mape1, inter_mape1, rightex_mape1,
                                 predictions, actual_values, _, _, _, _, _, _, adam_used,
                                 mae_loss, mae_l_loss, mae_r_loss, mae_in_loss) = evaluate_model_performance_maml(
                                    model, model_framework.upper(), X, y,
                                    testdata_input[randomtask], testdata_output[randomtask], grad, move,
                                    left_bound=args.left_bound, right_bound=args.right_bound,
                                    total_points=args.total_points, mode=args.mode
                                )

                            # Add to global collections
                            all_predictions_global.extend(predictions)
                            all_actuals_global.extend(actual_values)

                            # Calculate NRMSE values
                            nrmse1 = (total_loss1 ** 0.5) / (abs(y1_mean) + 1e-4) * 100
                            nrmse_inter = (inter_loss1 ** 0.5) / (abs(y_inter_mean) + 1e-4) * 100
                            nrmse_leftex = (leftex_loss1 ** 0.5) / (abs(y_leftex_mean) + 1e-4) * 100
                            nrmse_rightex = (rightex_loss1 ** 0.5) / (abs(y_rightex_mean) + 1e-4) * 100
                            mape_percent = mape_loss1 * 100
                            mape_l_percent = leftex_mape1 * 100
                            mape_r_percent = rightex_mape1 * 100
                            mape_inter_percent = inter_mape1 * 100

                            if not(torch.isinf(nrmse1) or nrmse1 > 1000 or torch.isnan(nrmse1)):
                                total_nrmse.append(nrmse1.item())
                                total_extra_l.append(nrmse_leftex.item())
                                total_extra_r.append(nrmse_rightex.item())
                                total_inter.append(nrmse_inter.item())
                                total_mape.append(mape_percent)
                                total_mape_l.append(mape_l_percent)
                                total_mape_r.append(mape_r_percent)
                                total_mape_inter.append(mape_inter_percent)

                                if model_framework == 'maml':
                                    total_mae.append(mae_loss)
                                    total_mae_l.append(mae_l_loss)
                                    total_mae_r.append(mae_r_loss)
                                    total_mae_inter.append(mae_in_loss)

                            if i % 100 == 0 and i > 0:
                                print(f"  Current avg NRMSE: {sum(total_nrmse)/len(total_nrmse):.3f}%, Tasks completed: {len(total_nrmse)}")
                                print(f"  Current avg MAPE: {sum(total_mape)/len(total_mape):.3f}%, Tasks completed: {len(total_mape)}")
                except Exception as e:
                    if i < 10:
                        print(f"Error processing task {randomtask}: {e}")
                    continue

        print(f"\n✅ Completed processing {len(total_nrmse)} valid tasks for {temp}°C")
        print(f"📈 Total global predictions collected: {len(all_predictions_global)}")

        # Print final results for this temperature
        print(f"\n{'='*80}")
        print(f"Final Results for {temp}°C on {len(total_nrmse)} valid tasks")
        print(f"{'='*80}")

        if len(total_nrmse) > 0:
            print(f"\n📊 NRMSE Results:")
            print(f"   Average Total NRMSE: {sum(total_nrmse)/len(total_nrmse):.3f}%")
            print(f"   Average Left Extrapolation NRMSE: {sum(total_extra_l)/len(total_extra_l):.3f}%")
            print(f"   Average Right Extrapolation NRMSE: {sum(total_extra_r)/len(total_extra_r):.3f}%")
            print(f"   Average Interpolation NRMSE: {sum(total_inter)/len(total_inter):.3f}%")

            print(f"\n📊 MAPE Results:")
            print(f"   Average Total MAPE: {sum(total_mape)/len(total_mape):.3f}%")
            print(f"   Average Left Extrapolation MAPE: {sum(total_mape_l)/len(total_mape_l):.3f}%")
            print(f"   Average Right Extrapolation MAPE: {sum(total_mape_r)/len(total_mape_r):.3f}%")
            print(f"   Average Interpolation MAPE: {sum(total_mape_inter)/len(total_mape_inter):.3f}%")

            if model_framework == 'maml' and len(total_mae) > 0:
                print(f"\n📊 MAE Results:")
                print(f"   Average Total MAE: {sum(total_mae)/len(total_mae):.6f}")
                print(f"   Average Left Extrapolation MAE: {sum(total_mae_l)/len(total_mae_l):.6f}")
                print(f"   Average Right Extrapolation MAE: {sum(total_mae_r)/len(total_mae_r):.6f}")
                print(f"   Average Interpolation MAE: {sum(total_mae_inter)/len(total_mae_inter):.6f}")

            # Calculate R² score
            if len(all_predictions_global) > 0 and len(all_actuals_global) > 0:
                actual_np = np.array(all_actuals_global)
                pred_np = np.array(all_predictions_global)

                valid_mask = ~(np.isnan(actual_np) | np.isnan(pred_np) | np.isinf(actual_np) | np.isinf(pred_np))
                actual_clean = actual_np[valid_mask]
                pred_clean = pred_np[valid_mask]

                if len(actual_clean) > 0 and np.var(actual_clean) > 0:
                    r2 = 1 - np.sum((actual_clean - pred_clean)**2) / np.sum((actual_clean - np.mean(actual_clean))**2)
                    print(f"\n🎯 Global R² Score: {r2:.6f}")
                    print(f"📊 Total data points: {len(actual_clean)}")

            # Store results for this temperature
            results_dict[temp] = {
                'nrmse_total': sum(total_nrmse)/len(total_nrmse),
                'mape_total': sum(total_mape)/len(total_mape),
                'num_tasks': len(total_nrmse)
            }

            # Save results if requested
            if args.save_results:
                os.makedirs('data_result_npy_directory', exist_ok=True)
                if model_framework == 'mlp':
                    pred_filename = f"data_result_npy_directory/TSMC_voltage_variation_{corner}_{temp}_{data_type}_{model_type}_pred.npy"
                    act_filename = f"data_result_npy_directory/TSMC_voltage_variation_{corner}_{temp}_{data_type}_{model_type}_act.npy"
                else:  # maml
                    pred_filename = f"data_result_npy_directory/TSMC_voltage_variation_{corner}_{temp}_{data_type}_maml_innerdiv{args.innerdiv}_meta{args.meta}_layer{args.layer_length}_inner{args.inner_step}_pred.npy"
                    act_filename = f"data_result_npy_directory/TSMC_voltage_variation_{corner}_{temp}_{data_type}_maml_innerdiv{args.innerdiv}_meta{args.meta}_layer{args.layer_length}_inner{args.inner_step}_act.npy"

                np.save(pred_filename, all_predictions_global)
                np.save(act_filename, all_actuals_global)

                print(f"\n💾 Saved predictions to: {pred_filename}")
                print(f"💾 Saved actuals to: {act_filename}")
        else:
            print(f"\n⚠️ No valid results for {temp}°C")

    # Print summary for all temperatures
    print(f"\n{'='*80}")
    print("Summary Results for All Temperatures")
    print(f"{'='*80}\n")
    for temp, results in results_dict.items():
        print(f"{temp}°C:")
        print(f"  NRMSE: {results['nrmse_total']:.3f}%")
        print(f"  MAPE: {results['mape_total']:.3f}%")
        print(f"  Tasks: {results['num_tasks']}")


if __name__ == "__main__":
    main()
