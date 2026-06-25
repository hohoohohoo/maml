#!/usr/bin/env python
# coding: utf-8

"""
GNN MAML Evaluation Script

This script evaluates pretrained GNN MAML models on test datasets.
- Loads pretrained GNN model with normalization statistics
- Applies same evaluation metrics as MAML_topology_validation.py
- Uses test dataset from split_gnn_dataset_batch.py
- Performs fine-tuning and evaluation
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import argparse
from torch_geometric.data import Data, Batch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'model_code'))

from gnn_maml import (
    MAML_GNN_Model,
    create_maml_gcn_model
)

from split_gnn_dataset import GNNBatchDataLoader


def find_test_data_path(process_type, corner_type, data_type='cell', graph_mode='stage_aware'):
    """
    Find test data path based on process, corner, data_type, and graph_mode

    Args:
        process_type: Process type (RVT, LVT, SLVT, SRAM)
        corner_type: Corner type (TT, FF, SS)
        data_type: Data type ('cell' or 'transition')
        graph_mode: Graph mode ('stage_aware' or 'full_graph')

    Returns:
        Path to test_indices file

    Raises:
        FileNotFoundError: If no matching test data is found
    """
    base_path = "/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_temp"

    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Dataset base path not found: {base_path}")

    # Determine test indices filename based on data_type and graph_mode
    test_indices_filename = f"test_indices_{data_type}_{graph_mode}.pth"

    matching_folders = []

    # Find all matching folders for this condition
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if not os.path.isdir(item_path):
            continue

        parts = item.split('_')
        dataset_process = None
        for part in parts:
            if part in ['RVT', 'LVT', 'SLVT', 'SRAM']:
                dataset_process = part
                break

        if dataset_process != process_type:
            continue

        # Check corner type matching
        corner_match = False
        if corner_type == "TT" and not item.endswith(('_FF', '_SS')):
            corner_match = True
        elif corner_type == "FF" and item.endswith('_FF'):
            corner_match = True
        elif corner_type == "SS" and item.endswith('_SS'):
            corner_match = True

        if corner_match:
            test_data_path = os.path.join(base_path, item, "train_test_split", test_indices_filename)
            if os.path.exists(test_data_path):
                matching_folders.append((item, test_data_path))

    if not matching_folders:
        raise FileNotFoundError(
            f"No test data found for:\n"
            f"  Process: {process_type}\n"
            f"  Corner: {corner_type}\n"
            f"  Data type: {data_type}\n"
            f"  Graph mode: {graph_mode}\n"
            f"  Looking for: {test_indices_filename}"
        )

    if len(matching_folders) > 1:
        print(f"⚠️  Multiple matching datasets found:")
        for folder, path in matching_folders:
            print(f"   - {folder}")
        print(f"   Using first match: {matching_folders[0][0]}")

    selected_folder, selected_path = matching_folders[0]
    print(f"✅ Found test data: {selected_folder}")
    print(f"   Path: {selected_path}")

    return selected_path


def normalize_node_features(node_features, norm_stats):
    """
    Normalize node features using training statistics
    Uses MASK-BASED normalization (same as training code)

    Args:
        node_features: Tensor of shape [num_nodes, num_features]
        norm_stats: Dictionary containing normalization statistics

    Returns:
        Normalized node features
    """
    normalized = node_features.clone()

    # Column 4: voltage (mask-based normalization)
    if 'voltage' in norm_stats['node_features']:
        voltage_stats = norm_stats['node_features']['voltage']
        voltage_mask = normalized[:, 4] != 0
        if voltage_mask.any():
            normalized[voltage_mask, 4] = (
                normalized[voltage_mask, 4] - voltage_stats['mean']
            ) / voltage_stats['std']

    # Column 5: input_slew (mask-based normalization)
    if 'input_slew' in norm_stats['node_features']:
        slew_stats = norm_stats['node_features']['input_slew']
        slew_mask = normalized[:, 5] != 0
        if slew_mask.any():
            normalized[slew_mask, 5] = (
                normalized[slew_mask, 5] - slew_stats['mean']
            ) / slew_stats['std']

    # Column 6: output_load (mask-based normalization)
    if 'output_load' in norm_stats['node_features']:
        load_stats = norm_stats['node_features']['output_load']
        load_mask = normalized[:, 6] != 0
        if load_mask.any():
            normalized[load_mask, 6] = (
                normalized[load_mask, 6] - load_stats['mean']
            ) / load_stats['std']

    return normalized


def prepare_pyg_batch(graphs, norm_stats, device):
    """
    Prepare a batch of PyTorch Geometric Data objects
    Uses ADJACENCY MATRIX multiplication (same as training code)

    Args:
        graphs: List of graph dictionaries
        norm_stats: Normalization statistics
        device: torch device

    Returns:
        PyG Batch object
    """
    data_list = []

    for graph in graphs:
        # Normalize node features
        node_features = normalize_node_features(graph['node_features'], norm_stats)

        # Apply adjacency matrix multiplication (A × X) - SAME AS TRAINING
        adjacency_matrix = graph['adjacency_matrix']
        aggregated_features = torch.matmul(adjacency_matrix, node_features)

        # Create PyG Data object with aggregated features
        data = Data(
            x=aggregated_features.to(device),
            edge_index=graph['edge_index'].to(device),
            edge_attr=graph['edge_attr'].to(device) if 'edge_attr' in graph else None
        )
        data_list.append(data)

    # Create batch
    batch = Batch.from_data_list(data_list)
    return batch


def evaluate_model_performance_gnn(model, X_batch, y, test_X_batch, test_y,
                                   y_mean, y_std, grad, move, left_bound, right_bound, mode='extrapolation'):
    """
    Evaluate GNN model performance with GRAD and MOVE transformation
    (same as MAML_topology_validation)

    Args:
        model: GNN model
        X_batch: Support set batch (PyG Batch)
        y: Support set outputs (normalized with grad and move)
        test_X_batch: Test set batch (PyG Batch)
        test_y: Test set outputs (actual values, NOT normalized)
        y_mean: Mean used for normalization
        y_std: Std used for normalization
        grad: Scaling factor (range ratio)
        move: Center offset
        left_bound, right_bound: Interpolation region boundaries
        mode: 'extrapolation' or 'interpolation'

    Returns:
        Dictionary containing all evaluation metrics
    """
    model.eval()

    with torch.no_grad():
        # Get normalized predictions (in transformed space)
        predictions_norm = model(test_X_batch)
        predictions_norm = predictions_norm.cpu().numpy().flatten()

        # DENORMALIZE predictions to original scale (reverse transformation)
        # Formula: pred_actual = (pred_norm - move) * (y_std * grad) + y_mean
        predictions = (predictions_norm - move) * (y_std * grad) + y_mean

        # Actual values (already in original scale)
        actual_values = test_y.cpu().numpy().flatten() if torch.is_tensor(test_y) else test_y

        # Define regions
        inter_preds = predictions[left_bound:right_bound]
        inter_actuals = actual_values[left_bound:right_bound]

        # Calculate interpolation metrics
        inter_mse = np.mean((inter_preds - inter_actuals) ** 2)
        inter_mae = np.mean(np.abs(inter_preds - inter_actuals))
        inter_mape = np.mean(np.abs((inter_preds - inter_actuals) / (inter_actuals + 1e-8))) * 100

        # Total metrics
        total_mse = np.mean((predictions - actual_values) ** 2)
        total_mae = np.mean(np.abs(predictions - actual_values))
        total_mape = np.mean(np.abs((predictions - actual_values) / (actual_values + 1e-8))) * 100

        # Extrapolation metrics (if mode is extrapolation)
        if mode == 'extrapolation':
            leftex_preds = predictions[:left_bound]
            leftex_actuals = actual_values[:left_bound]
            rightex_preds = predictions[right_bound:]
            rightex_actuals = actual_values[right_bound:]

            leftex_mse = np.mean((leftex_preds - leftex_actuals) ** 2) if len(leftex_preds) > 0 else 0
            rightex_mse = np.mean((rightex_preds - rightex_actuals) ** 2) if len(rightex_preds) > 0 else 0
            leftex_mae = np.mean(np.abs(leftex_preds - leftex_actuals)) if len(leftex_preds) > 0 else 0
            rightex_mae = np.mean(np.abs(rightex_preds - rightex_actuals)) if len(rightex_preds) > 0 else 0
            leftex_mape = np.mean(np.abs((leftex_preds - leftex_actuals) / (leftex_actuals + 1e-8))) * 100 if len(leftex_preds) > 0 else 0
            rightex_mape = np.mean(np.abs((rightex_preds - rightex_actuals) / (rightex_actuals + 1e-8))) * 100 if len(rightex_preds) > 0 else 0
        else:
            leftex_mse = rightex_mse = 0
            leftex_mae = rightex_mae = 0
            leftex_mape = rightex_mape = 0

    return {
        'total_mse': total_mse,
        'inter_mse': inter_mse,
        'leftex_mse': leftex_mse,
        'rightex_mse': rightex_mse,
        'total_mae': total_mae,
        'inter_mae': inter_mae,
        'leftex_mae': leftex_mae,
        'rightex_mae': rightex_mae,
        'total_mape': total_mape,
        'inter_mape': inter_mape,
        'leftex_mape': leftex_mape,
        'rightex_mape': rightex_mape,
        'predictions': predictions,
        'actual_values': actual_values
    }


def fine_tune_gnn_model(model, X_batch, y, inner_steps=1, inner_lr=0.001, adam_steps=40):
    """
    Fine-tune GNN model on support set
    Uses selective Adam (same as maml_functions.py): Adam only if initial loss > 1e-4

    Args:
        model: GNN model
        X_batch: Support set batch (PyG Batch)
        y: Support set outputs (normalized)
        inner_steps: Number of SGD gradient steps (unused, kept for compatibility)
        inner_lr: Learning rate for SGD (unused, kept for compatibility)
        adam_steps: Number of Adam steps if initial loss is high

    Returns:
        Fine-tuned model, adam_used flag
    """
    model.train()
    criterion = nn.MSELoss()
    K = y.shape[0]

    # Check initial loss (no SGD, same as maml_functions.py selective_adam)
    adam_used = False
    with torch.no_grad():
        predictions = model(X_batch)
        initial_loss = criterion(predictions, y) / K

    # Adam fine-tuning if loss > 1e-4 (selective_adam behavior)
    if initial_loss > 1e-4:
        adam_used = True
        optimizer_adam = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)

        for step in range(adam_steps):
            optimizer_adam.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y) / K
            loss.backward()
            optimizer_adam.step()

    return model, adam_used


def main():
    parser = argparse.ArgumentParser(
        description='GNN MAML Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-load test data
  python gnn_maml_evaluation.py \\
    --model_path pretrained_models/gnn_maml_final/gnn_maml_LVT_FF_cell_stage_aware_innerdiv10_iter100000_inner1.pth \\
    --process LVT --corner FF --data_type cell --graph_mode stage_aware

  # Use specific test data path
  python gnn_maml_evaluation.py \\
    --model_path pretrained_models/gnn_maml_final/model.pth \\
    --process SLVT --corner TT \\
    --test_data_path /path/to/test_indices_cell_full_graph.pth

  # With custom indices and mode
  python gnn_maml_evaluation.py \\
    --model_path pretrained_models/gnn_maml_final/model.pth \\
    --process RVT --corner SS --graph_mode full_graph \\
    --mode interpolation --indices 5 15 25 35 45 55
        """
    )

    # Required arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to pretrained GNN MAML model')

    # Dataset configuration (auto-load test data)
    parser.add_argument('--process', type=str, required=True,
                        choices=['RVT', 'LVT', 'SLVT', 'SRAM'],
                        help='Process type: RVT, LVT, SLVT, or SRAM')
    parser.add_argument('--corner', type=str, required=True,
                        choices=['TT', 'FF', 'SS'],
                        help='Process corner: TT, FF, or SS')
    parser.add_argument('--data_type', type=str, default='cell',
                        choices=['cell', 'transition'],
                        help='Data type: cell or transition (default: cell)')
    parser.add_argument('--graph_mode', type=str, default='stage_aware',
                        choices=['stage_aware', 'full_graph'],
                        help='Graph mode: stage_aware or full_graph (default: stage_aware)')

    # Optional: Direct test data path (overrides auto-load)
    parser.add_argument('--test_data_path', type=str, default=None,
                        help='Path to test_indices.pth (optional, overrides auto-load)')

    # Model configuration
    parser.add_argument('--model_type', type=str, default='GCN', choices=['GCN', 'GraphSAGE', 'GAT'],
                        help='GNN model type (default: GCN)')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension (default: 128)')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of GNN layers (default: 3)')

    # Evaluation configuration
    parser.add_argument('--mode', type=str, choices=['extrapolation', 'interpolation'], default='extrapolation',
                        help='Testing mode (default: extrapolation)')
    parser.add_argument('--indices', type=int, nargs='+', default=None,
                        help='Support set indices (default: [5,30,55] for extrapolation, [0,13,30,45,60] for interpolation)')
    parser.add_argument('--total_points', type=int, default=61,
                        help='Total number of points (default: 61)')
    parser.add_argument('--inner_steps', type=int, default=1,
                        help='Inner loop steps for fine-tuning (default: 1)')
    parser.add_argument('--inner_lr', type=float, default=0.001,
                        help='Inner loop learning rate (default: 0.001)')

    # Output configuration
    parser.add_argument('--save_results', action='store_true',
                        help='Save prediction results to .npy files')
    parser.add_argument('--output_dir', type=str, default='gnn_evaluation_results',
                        help='Output directory for results (default: gnn_evaluation_results)')
    parser.add_argument('--num_test_samples', type=int, default=None,
                        help='Number of test samples to process (default: all)')

    args = parser.parse_args()

    # Set default indices based on mode if not provided
    if args.indices is None:
        if args.mode == 'extrapolation':
            args.indices = [5, 30, 55]
        else:  # interpolation
            args.indices = [0, 13, 30, 45, 60]

    # Auto-find test data path if not provided
    if args.test_data_path is None:
        print(f"\n📂 Auto-loading test data...")
        print(f"   Process: {args.process}")
        print(f"   Corner: {args.corner}")
        print(f"   Data type: {args.data_type}")
        print(f"   Graph mode: {args.graph_mode}")

        try:
            args.test_data_path = find_test_data_path(
                args.process, args.corner, args.data_type, args.graph_mode
            )
        except FileNotFoundError as e:
            print(f"\n❌ Error: {e}")
            return 1
    else:
        print(f"\n📂 Using provided test data path: {args.test_data_path}")

    # Calculate boundaries
    K = len(args.indices)
    left_bound = args.indices[0]
    right_bound = args.indices[-1] + 1

    print(f"\n{'='*80}")
    print(f"GNN MAML Evaluation")
    print(f"{'='*80}")
    print(f"⚙️ Configuration:")
    print(f"   Process: {args.process}")
    print(f"   Corner: {args.corner}")
    print(f"   Data type: {args.data_type}")
    print(f"   Graph mode: {args.graph_mode}")
    print(f"   Model path: {args.model_path}")
    print(f"   Test data: {args.test_data_path}")
    print(f"   Model type: {args.model_type}")
    print(f"   Hidden dim: {args.hidden_dim}")
    print(f"   Num layers: {args.num_layers}")
    print(f"   Mode: {args.mode}")
    print(f"   Support set indices: {args.indices}")
    print(f"   K (support samples): {K}")
    print(f"   Left bound: {left_bound}")
    print(f"   Right bound: {right_bound}")
    print(f"   Total points: {args.total_points}")
    print(f"   Inner steps: {args.inner_steps}")
    print(f"   Inner LR: {args.inner_lr}")

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")

    # Load pretrained model
    print(f"\n🤖 Loading pretrained model...")
    if not os.path.exists(args.model_path):
        print(f"❌ Model file not found: {args.model_path}")
        return 1

    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    norm_stats = checkpoint['norm_stats']
    config = checkpoint.get('config', {})

    print(f"✅ Loaded model checkpoint")

    # Print config without meta_losses (too long)
    config_display = {k: v for k, v in config.items() if k != 'meta_losses'}
    if 'meta_losses' in config:
        config_display['meta_losses'] = f"<{len(config['meta_losses'])} values>"

    print(f"   Config:")
    for key, value in config_display.items():
        print(f"     {key}: {value}")
    print(f"   Normalization stats keys: {list(norm_stats.keys())}")

    # Initialize model (using create_maml_gcn_model from gnn_maml)
    node_features = 7  # GNN node feature dimension
    model = create_maml_gcn_model(
        node_features=node_features,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        pooling='mean',
        output_dim=1
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"✅ Model initialized: {args.model_type}")

    # Load test data
    print(f"\n📊 Loading test dataset...")
    test_loader = GNNBatchDataLoader(args.test_data_path, batch_size=1)
    num_test_samples = args.num_test_samples if args.num_test_samples else len(test_loader.sample_indices)

    print(f"✅ Test data loaded")
    print(f"   Total test samples: {len(test_loader.sample_indices)}")
    print(f"   Processing: {num_test_samples} samples")
    print(f"   Lib files per sample: {test_loader.num_lib_files}")

    # Evaluation
    print(f"\n🔍 Starting evaluation...")

    all_results = []
    all_predictions_global = []
    all_actuals_global = []

    for sample_idx in range(min(num_test_samples, len(test_loader.sample_indices))):
        if sample_idx % 100 == 0:
            print(f"   Processing sample {sample_idx}/{num_test_samples}...")

        # Get sample data
        _, outputs = test_loader.get_batch(sample_idx)

        # Get the actual sample index to access all lib files directly
        actual_sample_idx = test_loader.sample_indices[sample_idx]

        # Select one lib file's output (e.g., first lib)
        randomlib = np.random.randint(0, outputs.shape[1])
        test_y = outputs[0, randomlib].item()  # Single output value for this sample

        # For GNN, we need to create support and test sets
        # Since each sample is one condition, we'll use the lib dimension as variations
        # Support set: select K lib files based on indices
        # IMPORTANT: Each lib file has a different graph with different voltage values
        support_graphs = []
        support_outputs = []

        for idx in args.indices:
            if idx < outputs.shape[1]:
                # Get graph for this specific lib file (not just lib_idx=0!)
                lib_graph = test_loader.graph_data[idx][actual_sample_idx]
                support_graphs.append(lib_graph)
                support_outputs.append(outputs[0, idx].item())

        # Debug: Check voltage values in support_graphs for first few samples
        if sample_idx < 5:
            print(f"\n   [DEBUG Sample {sample_idx}] Voltage values in support_graphs:")
            for i, g in enumerate(support_graphs):
                voltage_col = g['node_features'][:, 4]  # Column 4 is voltage
                non_zero_voltages = voltage_col[voltage_col != 0]
                if len(non_zero_voltages) > 0:
                    print(f"     Support graph {i} (lib idx {args.indices[i]}): "
                          f"voltage min={non_zero_voltages.min():.6f}, "
                          f"max={non_zero_voltages.max():.6f}, "
                          f"mean={non_zero_voltages.mean():.6f}, "
                          f"count={len(non_zero_voltages)}")
                else:
                    print(f"     Support graph {i} (lib idx {args.indices[i]}): No non-zero voltages")

        # Normalize outputs with grad and move (like MAML_topology_validation)
        support_y = np.array(support_outputs)
        y_mean = support_y.mean()
        y_std = support_y.std()

        if y_std > 0:
            # Calculate grad and move (same as ANN MAML validation)
            # 1. Get normalized support set range
            y_norm_temp = (support_y - y_mean) / y_std
            y_max = y_norm_temp.max()
            y_min = y_norm_temp.min()

            # 2. Get model's prediction range on interpolation region for scaling
            # Create temporary batch to get predictions
            temp_batch = prepare_pyg_batch(support_graphs, norm_stats, device)

            # Debug: Check normalized voltage values in temp_batch
            # if sample_idx < 5:
            #     print(f"\n   [DEBUG Sample {sample_idx}] Normalized voltage in temp_batch (after prepare_pyg_batch):")
            #     for i in range(len(temp_batch)):
            #         batch_data = temp_batch[i]
            #         voltage_col = batch_data.x[:, 4]  # Column 4 is voltage (after A*X multiplication)
            #         non_zero_voltages = voltage_col[voltage_col.abs() > 1e-8]
            #         if len(non_zero_voltages) > 0:
            #             print(f"     Batch item {i}: "
            #                   f"voltage min={non_zero_voltages.min():.6f}, "
            #                   f"max={non_zero_voltages.max():.6f}, "
            #                   f"mean={non_zero_voltages.mean():.6f}")
            #         else:
            #             print(f"     Batch item {i}: All voltages near zero")

            with torch.no_grad():
                temp_preds = model(temp_batch).cpu().numpy().flatten()

            min_val = temp_preds.min()
            max_val = temp_preds.max()

            # Debug logging for first few samples
            if sample_idx < 5:
                print(f"\n   [DEBUG Sample {sample_idx}] Prediction summary:")
                print(f"     Support outputs (actual): {support_y}")
                print(f"     y_norm_temp range: [{y_min:.4f}, {y_max:.4f}] (span: {y_max - y_min:.4f})")
                print(f"     Model preds: {temp_preds}")
                print(f"     Model preds range: [{min_val:.6f}, {max_val:.6f}] (span: {max_val - min_val:.6f})")

            # 3. Calculate grad (scaling factor)
            if abs(max_val - min_val) > 1e-8:
                grad = (y_max - y_min) / (max_val - min_val)
            else:
                grad = 1.0

            if sample_idx < 5:
                print(f"     → grad = {grad:.4f}")

            # 4. Calculate move (center adjustment) - using middle index
            middle_idx = len(args.indices) // 2
            # Note: For GNN, we use voltage=0 concept differently
            # We just use the middle point of support set for centering
            center = temp_preds[middle_idx]
            move = center - y_norm_temp[middle_idx] / grad

            # 5. Apply normalization with grad and move
            y_test = (support_y - y_mean) / (y_std * grad) + move
            y_norm_tensor = torch.FloatTensor(y_test).unsqueeze(1).to(device)

            # Log grad and move occasionally
            if sample_idx % 1000 == 0:
                print(f"   Sample {sample_idx}: grad={grad:.4f}, move={move:.4f}, y_mean={y_mean:.2e}, y_std={y_std:.2e}")

            # Prepare support batch
            support_batch = prepare_pyg_batch(support_graphs, norm_stats, device)

            # Fine-tune model on support set
            model_copy = create_maml_gcn_model(
                node_features=node_features,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                pooling='mean',
                output_dim=1
            ).to(device)
            model_copy.load_state_dict(model.state_dict())

            model_finetuned, adam_used = fine_tune_gnn_model(
                model_copy, support_batch, y_norm_tensor,
                inner_steps=args.inner_steps, inner_lr=args.inner_lr, adam_steps=40
            )

            # Prepare test batch (all lib files)
            # IMPORTANT: Get individual graph for each lib file (they have different voltage values)
            test_graphs = []
            for lib_idx in range(outputs.shape[1]):
                test_graphs.append(test_loader.graph_data[lib_idx][actual_sample_idx])
            test_batch = prepare_pyg_batch(test_graphs, norm_stats, device)
            test_y_full = outputs[0, :].numpy()

            # Evaluate with denormalization parameters (including grad and move)
            results = evaluate_model_performance_gnn(
                model_finetuned, support_batch, y_norm_tensor,
                test_batch, test_y_full,
                y_mean=y_mean, y_std=y_std, grad=grad, move=move,
                left_bound=left_bound, right_bound=right_bound,
                mode=args.mode
            )

            # Track adam usage and transformation parameters
            results['adam_used'] = adam_used
            results['grad'] = grad
            results['move'] = move
            all_results.append(results)
            all_predictions_global.extend(results['predictions'].tolist())
            all_actuals_global.extend(results['actual_values'].tolist())

    # Aggregate results
    print(f"\n📈 Evaluation Results:")
    print(f"{'='*80}")

    # Calculate Adam usage and grad/move statistics
    adam_count = sum([r.get('adam_used', False) for r in all_results])
    adam_usage_rate = adam_count / len(all_results) if all_results else 0
    print(f"\n🔧 Adam Fallback Usage: {adam_count}/{len(all_results)} tasks ({adam_usage_rate:.2%})")

    # Grad and move statistics
    grad_values = [r.get('grad', 1.0) for r in all_results]
    move_values = [r.get('move', 0.0) for r in all_results]
    print(f"\n📐 Transformation Parameters:")
    print(f"   Grad - Mean: {np.mean(grad_values):.4f}, Std: {np.std(grad_values):.4f}, Range: [{np.min(grad_values):.4f}, {np.max(grad_values):.4f}]")
    print(f"   Move - Mean: {np.mean(move_values):.4f}, Std: {np.std(move_values):.4f}, Range: [{np.min(move_values):.4f}, {np.max(move_values):.4f}]")

    avg_results = {}
    for key in ['total_mse', 'inter_mse', 'leftex_mse', 'rightex_mse',
                'total_mae', 'inter_mae', 'leftex_mae', 'rightex_mae',
                'total_mape', 'inter_mape', 'leftex_mape', 'rightex_mape']:
        values = [r[key] for r in all_results]
        avg_results[key] = np.mean(values)

    print(f"\n MSE Results:")
    print(f"   Total MSE: {avg_results['total_mse']:.6f}")
    print(f"   Interpolation MSE: {avg_results['inter_mse']:.6f}")
    if args.mode == 'extrapolation':
        print(f"   Left Extrapolation MSE: {avg_results['leftex_mse']:.6f}")
        print(f"   Right Extrapolation MSE: {avg_results['rightex_mse']:.6f}")

    print(f"\n📊 MAE Results:")
    print(f"   Total MAE: {avg_results['total_mae']:.6f}")
    print(f"   Interpolation MAE: {avg_results['inter_mae']:.6f}")
    if args.mode == 'extrapolation':
        print(f"   Left Extrapolation MAE: {avg_results['leftex_mae']:.6f}")
        print(f"   Right Extrapolation MAE: {avg_results['rightex_mae']:.6f}")

    print(f"\n📊 MAPE Results (%):")
    print(f"   Total MAPE: {avg_results['total_mape']:.2f}%")
    print(f"   Interpolation MAPE: {avg_results['inter_mape']:.2f}%")
    if args.mode == 'extrapolation':
        print(f"   Left Extrapolation MAPE: {avg_results['leftex_mape']:.2f}%")
        print(f"   Right Extrapolation MAPE: {avg_results['rightex_mape']:.2f}%")

    # Save results
    if args.save_results:
        os.makedirs(args.output_dir, exist_ok=True)

        # Create descriptive filename with model and test info
        # Extract model name from path (remove directory and .pth extension)
        model_name = os.path.basename(args.model_path).replace('.pth', '')

        # Create test set identifier
        test_id = f"{args.process}_{args.corner}_{args.data_type}_{args.graph_mode}"

        # Create mode identifier
        mode_id = args.mode
        indices_str = '_'.join(map(str, args.indices))

        # Final identifier
        result_prefix = f"{model_name}_{test_id}_{mode_id}_K{K}_indices{indices_str}"

        np.save(f"{args.output_dir}/{result_prefix}_predictions.npy", np.array(all_predictions_global))
        np.save(f"{args.output_dir}/{result_prefix}_actuals.npy", np.array(all_actuals_global))
        np.save(f"{args.output_dir}/{result_prefix}_avg_results.npy", avg_results)

        # Also save a summary text file
        summary_path = f"{args.output_dir}/{result_prefix}_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"GNN MAML Evaluation Summary\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Test Set: {test_id}\n")
            f.write(f"Mode: {args.mode}\n")
            f.write(f"Support Indices: {args.indices}\n")
            f.write(f"K (support samples): {K}\n")
            f.write(f"Total test samples: {num_test_samples}\n\n")

            f.write(f"Adam Fallback Usage: {adam_count}/{len(all_results)} ({adam_usage_rate:.2%})\n\n")

            f.write(f"Grad - Mean: {np.mean(grad_values):.4f}, Std: {np.std(grad_values):.4f}\n")
            f.write(f"Move - Mean: {np.mean(move_values):.4f}, Std: {np.std(move_values):.4f}\n\n")

            f.write(f"MSE Results:\n")
            f.write(f"  Total MSE: {avg_results['total_mse']:.6f}\n")
            f.write(f"  Interpolation MSE: {avg_results['inter_mse']:.6f}\n")
            if args.mode == 'extrapolation':
                f.write(f"  Left Extrapolation MSE: {avg_results['leftex_mse']:.6f}\n")
                f.write(f"  Right Extrapolation MSE: {avg_results['rightex_mse']:.6f}\n")

            f.write(f"\nMAE Results:\n")
            f.write(f"  Total MAE: {avg_results['total_mae']:.6f}\n")
            f.write(f"  Interpolation MAE: {avg_results['inter_mae']:.6f}\n")
            if args.mode == 'extrapolation':
                f.write(f"  Left Extrapolation MAE: {avg_results['leftex_mae']:.6f}\n")
                f.write(f"  Right Extrapolation MAE: {avg_results['rightex_mae']:.6f}\n")

            f.write(f"\nMAPE Results (%):\n")
            f.write(f"  Total MAPE: {avg_results['total_mape']:.2f}%\n")
            f.write(f"  Interpolation MAPE: {avg_results['inter_mape']:.2f}%\n")
            if args.mode == 'extrapolation':
                f.write(f"  Left Extrapolation MAPE: {avg_results['leftex_mape']:.2f}%\n")
                f.write(f"  Right Extrapolation MAPE: {avg_results['rightex_mape']:.2f}%\n")

        print(f"\n💾 Results saved to {args.output_dir}/")
        print(f"   Prefix: {result_prefix}")

    print(f"\n✅ Evaluation complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
