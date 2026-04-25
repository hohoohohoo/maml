"""
GNN-specific functions for extrapolation testing
Adapted from mlp_functions.py for GCN models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch


def model_functions_at_training_gnn(initial_model, X_samples, y, true_samples, true_function,
                                   topology_cache, cache_type, norm_stats, normalize_fn,
                                   optim=torch.optim.SGD, lr=0.003, adam_step=0, std=1, mean=10, move=0,
                                   left_bound=5, right_bound=56, total_points=61, mode='extrapolation'):
    """
    Trains the GNN model on X_samples, y and measures the loss curve.
    For each n in sampled_steps, records model(x_axis) after n gradient updates.
    mode: 'extrapolation' or 'interpolation' - determines whether to calculate left/right extrapolation metrics

    Args:
        initial_model: Pretrained GNN model
        X_samples: List of minimal samples for support set
        y: Support set outputs (normalized)
        true_samples: All 61 minimal samples for evaluation
        true_function: Ground truth outputs for all 61 points
        topology_cache: Pre-computed topology cache
        cache_type: 'stage_aware' or 'full_graph'
        norm_stats: Normalization statistics
        normalize_fn: Function to normalize node features
        optim: Optimizer class
        lr: Learning rate
        adam_step: Number of Adam optimization steps if SGD loss is high
        std: Standard deviation for denormalization
        mean: Mean for denormalization
        move: Move parameter for additional normalization
        left_bound: Left boundary for interpolation region
        right_bound: Right boundary for interpolation region
        total_points: Total number of data points (default: 61)
        mode: 'extrapolation' or 'interpolation'
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    y = y.to(device).view(-1, 1)
    true_function = true_function.to(device)
    std = std.to(device) if isinstance(std, torch.Tensor) else torch.tensor(std).to(device)
    mean = mean.to(device) if isinstance(mean, torch.Tensor) else torch.tensor(mean).to(device)
    move = move.to(device) if isinstance(move, torch.Tensor) else torch.tensor(move).to(device)

    # Import GNN model - detect model type (GCN vs GAT vs HeteroGNN)
    from gnn_maml import create_maml_gcn_model, create_maml_gat_model
    # Get node_features from initial_model's first conv layer weight shape
    # HeteroGNN uses input_linears instead of convs[0].lin
    if hasattr(initial_model, 'input_linears'):
        node_features = initial_model.input_linears[0].weight.shape[1]
    else:
        node_features = initial_model.convs[0].lin.weight.shape[1]

    # Check if model is GAT (has 'heads' attribute but not HeteroGNN)
    is_gat = hasattr(initial_model, 'heads') and not hasattr(initial_model, 'input_linears')

    # Check if model is HeteroGNN
    is_hetero = hasattr(initial_model, 'input_linears')

    if is_hetero:
        from hetero_gnn_maml import create_maml_hetero_gnn_model
        model = create_maml_hetero_gnn_model(
            node_features=node_features,
            conv_hidden_dim=initial_model.conv_hidden_dim,
            num_conv_layers=initial_model.num_conv_layers,
            fc_hidden_dim=initial_model.fc_hidden_dim,
            num_fc_layers=initial_model.num_fc_layers,
            pooling=initial_model.pooling_type,
            output_dim=1,
            dropout=0.0,
            num_node_types=initial_model.num_node_types,
            conv_type=initial_model.conv_type,
            heads=initial_model.heads if hasattr(initial_model, 'heads') else 4
        ).to(device)
    elif is_gat:
        model = create_maml_gat_model(
            node_features=node_features,
            conv_hidden_dim=initial_model.conv_hidden_dim,
            num_conv_layers=initial_model.num_conv_layers,
            fc_hidden_dim=initial_model.fc_hidden_dim,
            num_fc_layers=initial_model.num_fc_layers,
            heads=initial_model.heads,
            pooling=initial_model.pooling_type,
            output_dim=1,
            dropout=0.0
        ).to(device)
    else:
        model = create_maml_gcn_model(
            node_features=node_features,
            conv_hidden_dim=initial_model.conv_hidden_dim,
            num_conv_layers=initial_model.num_conv_layers,
            fc_hidden_dim=initial_model.fc_hidden_dim,
            num_fc_layers=initial_model.num_fc_layers,
            pooling=initial_model.pooling_type,
            output_dim=1,
            dropout=0.0
        ).to(device)
    model.load_state_dict(initial_model.state_dict())

    criterion = nn.MSELoss()
    optimiser = optim(model.parameters(), lr, weight_decay=1e-4)
    adam_condition_triggered = False

    # Convert minimal samples to PyG Data objects
    def create_pyg_data(minimal_sample):
        """Create PyG Data object from minimal sample (assumes data is already normalized)"""
        node_features = minimal_sample['node_features']
        cell_name = minimal_sample['cell_name']

        # Get adjacency matrix from topology cache
        if cell_name not in topology_cache:
            raise ValueError(f"Cell {cell_name} not found in topology cache")

        cell_cache = topology_cache[cell_name]

        if cache_type == 'stage_aware':
            output_name = minimal_sample['output_name']
            delay_type = minimal_sample['delay_type']

            if output_name not in cell_cache['output_topologies']:
                raise ValueError(f"Output {output_name} not found for cell {cell_name}")

            output_topo = cell_cache['output_topologies'][output_name]

            if 'rise' in delay_type:
                adjacency_matrix = output_topo['pull_up']['adjacency_matrix']
            else:
                adjacency_matrix = output_topo['pull_down']['adjacency_matrix']
        else:  # full_graph
            adjacency_matrix = cell_cache['adjacency_matrix']

        # Data is already normalized, use directly
        # Create edge_index from adjacency matrix for GCN convolution
        # GCNConv will perform A × X internally
        edge_index = adjacency_matrix.nonzero().t()

        # Validate edge_index doesn't exceed number of nodes
        num_nodes = node_features.shape[0]
        if edge_index.numel() > 0:
            max_idx = edge_index.max().item()
            if max_idx >= num_nodes:
                raise ValueError(f"Edge index out of bounds for cell {cell_name}: "
                               f"max_idx={max_idx}, num_nodes={num_nodes}")

        # Create PyG Data with edge_index for GCN convolution
        data = Data(x=node_features, edge_index=edge_index)

        return data

    # Train model on support set
    K = len(X_samples)

    losses = []
    outputs = {}

    # Create support batch
    support_batch_data = []
    for sample in X_samples:
        data = create_pyg_data(sample)
        support_batch_data.append(data)

    X_batch = Batch.from_data_list(support_batch_data).to(device)
    loss = criterion(model(X_batch), y) / K

    # Adam training if SGD loss is still high
    if loss > 1e-4:
        adam_condition_triggered = True
        optimiser2 = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
        for step in range(1, adam_step+1):
            loss = criterion(model(X_batch), y) / K
            losses.append(loss.item())

            # compute grad and update inner loop weights
            model.zero_grad()
            loss.backward()
            optimiser2.step()

    # Calculate losses on all 61 points
    total_loss = 0
    total_mape_loss = 0
    total_rmse_loss = 0

    # Store predictions and actual values for plotting
    predictions = []
    actual_values = []

    model.eval()
    with torch.no_grad():
        for i in range(total_points):
            # Create single sample batch
            sample_data = create_pyg_data(true_samples[i])
            sample_batch = Batch.from_data_list([sample_data]).to(device)

            # Predict
            pred_value = ((model(sample_batch).item() - move) * std + mean).item()
            actual_value = ((true_function[i] - move) * std + mean).item()
            predictions.append(pred_value)
            actual_values.append(actual_value)

            loss = criterion((model(sample_batch).item() - move) * std + mean,
                           (true_function[i] - move) * std + mean)

            # Calculate MAPE (Mean Absolute Percentage Error) and squared error for RMSE
            if abs(actual_value) > 1e-8:  # Avoid division by zero
                squared_error = (pred_value - actual_value) ** 2
                mape_loss = abs((pred_value - actual_value) / actual_value)
            else:
                squared_error = (pred_value - actual_value) ** 2
                mape_loss = 0

            total_loss += loss
            total_mape_loss += mape_loss
            total_rmse_loss += squared_error

    # Calculate average losses
    import math
    avg_total_loss = total_loss / total_points
    avg_total_mape = total_mape_loss / total_points
    avg_total_rmse = math.sqrt(total_rmse_loss / total_points)

    return (model, outputs, losses, avg_total_loss, avg_total_mape, predictions, actual_values,
            adam_condition_triggered, avg_total_rmse)


def evaluate_model_performance_gnn(initial_model, model_name, X_samples, y, true_samples, true_function,
                                   grad, move, topology_cache, cache_type, norm_stats, normalize_fn,
                                   optim=torch.optim.SGD, lr=0.001,
                                   left_bound=5, right_bound=56, total_points=61, mode='extrapolation',
                                   adaptation_method='selective_adam'):
    """
    Evaluate GNN model performance with grad/move normalization parameters

    Args:
        initial_model: Pretrained GNN model
        model_name: Name of the model (for logging)
        X_samples: Support set samples (minimal format)
        y: Support set outputs
        true_samples: All 61 minimal samples
        true_function: Ground truth outputs for all 61 points
        grad: Gradient scaling parameter
        move: Move parameter for additional normalization
        topology_cache: Pre-computed topology cache
        cache_type: 'stage_aware' or 'full_graph'
        norm_stats: Normalization statistics
        normalize_fn: Function to normalize node features
        optim: Optimizer class
        lr: Learning rate
        left_bound: Left boundary for interpolation region
        right_bound: Right boundary for interpolation region
        total_points: Total number of data points (default: 61)
        mode: 'extrapolation' or 'interpolation'
        adaptation_method: 'selective_adam' (grad/move + conditional Adam) or 'adam' (direct Adam, no grad/move)

    Returns:
        tuple: (total_loss, total_mape, predictions, actual_values, model, adam_used, total_rmse)
    """
    import numpy as np

    # If using 'adam' method, use direct Adam without grad/move scaling
    if adaptation_method == 'adam':
        result = model_functions_with_optim_mode_gnn(
            initial_model=initial_model,
            X_samples=X_samples,
            y=y,
            true_samples=true_samples,
            true_function=true_function,
            topology_cache=topology_cache,
            cache_type=cache_type,
            norm_stats=norm_stats,
            normalize_fn=normalize_fn,
            optim_mode='adam',
            num_steps=40,
            lr=0.003,
            std=1,
            mean=0,
            move=0,
            grad=1,
            left_bound=left_bound,
            right_bound=right_bound,
            total_points=total_points,
            mode=mode
        )

        return (result['total_loss'], result['total_mape'],
                result['predictions'], result['actual_values'], result['model'], True,
                result['total_rmse'])

    # Original selective_adam method with grad/move scaling
    y_mean = y.mean()
    y_std = y.std()
    mean_values = [y_mean]
    std_values = [y_std * grad]

    # Store all predictions and actuals for final plotting
    all_predictions = []
    all_actuals = []

    # For each combination of mean and std
    for mean in mean_values:
        for std in std_values:
            y_mean1 = mean  # Update mean
            y_std1 = std    # Update std

            y_test = (y-y_mean1) / y_std1 + move
            true_function1 = (true_function-y_mean1) / y_std1 + move

            # Pass the updated mean and std to model_functions_at_training
            (model, outputs, losses, total_loss, total_mape_loss, predictions, actual_values,
             adam_condition_triggered, avg_total_rmse) = model_functions_at_training_gnn(
                initial_model,
                X_samples,
                y=y_test,
                true_samples=true_samples,
                true_function=true_function1,
                topology_cache=topology_cache,
                cache_type=cache_type,
                norm_stats=norm_stats,
                normalize_fn=normalize_fn,
                optim=optim,
                lr=lr,
                adam_step=40,
                std=y_std1,
                mean=y_mean1,
                move=move,
                left_bound=left_bound,
                right_bound=right_bound,
                total_points=total_points,
                mode=mode
            )
            adam_used = adam_condition_triggered

            # Collect predictions and actuals
            all_predictions.extend(predictions)
            all_actuals.extend(actual_values)

            model_min = model
            loss_min = total_loss
            mape_min = total_mape_loss
            rmse_min = avg_total_rmse

    return (loss_min, mape_min, all_predictions, all_actuals, model_min, adam_used, rmse_min)


def model_functions_with_optim_mode_gnn(initial_model, X_samples, y, true_samples, true_function,
                                        topology_cache, cache_type, norm_stats, normalize_fn,
                                        optim_mode='selective_adam', num_steps=50, lr=0.003,
                                        std=1, mean=10, move=0, grad=1,
                                        left_bound=5, right_bound=56, total_points=61, mode='extrapolation'):
    """
    Train GNN model with different optimization modes for comparison.

    Args:
        optim_mode:
            - 'none': Grad+Move only (no optimization)
            - 'sgd': Direct SGD optimization (no grad/move)
            - 'adam': Direct Adam optimization (no grad/move)
            - 'selective_adam': Grad+Move + Adam if loss > threshold
    """
    import math
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Determine if using grad/move scaling
    use_grad_move = optim_mode in ['none', 'selective_adam']

    y_tensor = y.clone().to(device).view(-1, 1)
    true_function_tensor = true_function.clone().to(device)
    std_tensor = std.to(device) if isinstance(std, torch.Tensor) else torch.tensor(std).to(device)
    mean_tensor = mean.to(device) if isinstance(mean, torch.Tensor) else torch.tensor(mean).to(device)
    move_tensor = move.to(device) if isinstance(move, torch.Tensor) else torch.tensor(move).to(device)

    from gnn_maml import create_maml_gcn_model, create_maml_gat_model
    # Get node_features - HeteroGNN uses input_linears instead of convs[0].lin
    if hasattr(initial_model, 'input_linears'):
        node_features = initial_model.input_linears[0].weight.shape[1]
    else:
        node_features = initial_model.convs[0].lin.weight.shape[1]

    # Check if model is GAT (has 'heads' attribute but not HeteroGNN)
    is_gat = hasattr(initial_model, 'heads') and not hasattr(initial_model, 'input_linears')

    # Check if model is HeteroGNN
    is_hetero = hasattr(initial_model, 'input_linears')

    if is_hetero:
        from hetero_gnn_maml import create_maml_hetero_gnn_model
        model = create_maml_hetero_gnn_model(
            node_features=node_features,
            conv_hidden_dim=initial_model.conv_hidden_dim,
            num_conv_layers=initial_model.num_conv_layers,
            fc_hidden_dim=initial_model.fc_hidden_dim,
            num_fc_layers=initial_model.num_fc_layers,
            pooling=initial_model.pooling_type,
            output_dim=1,
            dropout=0.0,
            num_node_types=initial_model.num_node_types,
            conv_type=initial_model.conv_type,
            heads=initial_model.heads if hasattr(initial_model, 'heads') else 4
        ).to(device)
    elif is_gat:
        model = create_maml_gat_model(
            node_features=node_features,
            conv_hidden_dim=initial_model.conv_hidden_dim,
            num_conv_layers=initial_model.num_conv_layers,
            fc_hidden_dim=initial_model.fc_hidden_dim,
            num_fc_layers=initial_model.num_fc_layers,
            heads=initial_model.heads,
            pooling=initial_model.pooling_type,
            output_dim=1,
            dropout=0.0
        ).to(device)
    else:
        model = create_maml_gcn_model(
            node_features=node_features,
            conv_hidden_dim=initial_model.conv_hidden_dim,
            num_conv_layers=initial_model.num_conv_layers,
            fc_hidden_dim=initial_model.fc_hidden_dim,
            num_fc_layers=initial_model.num_fc_layers,
            pooling=initial_model.pooling_type,
            output_dim=1,
            dropout=0.0
        ).to(device)
    model.load_state_dict(initial_model.state_dict())

    criterion = nn.MSELoss()

    def create_pyg_data(minimal_sample):
        node_features = minimal_sample['node_features']
        cell_name = minimal_sample['cell_name']
        cell_cache = topology_cache[cell_name]

        if cache_type == 'stage_aware':
            output_name = minimal_sample['output_name']
            delay_type = minimal_sample['delay_type']
            output_topo = cell_cache['output_topologies'][output_name]
            if 'rise' in delay_type:
                adjacency_matrix = output_topo['pull_up']['adjacency_matrix']
            else:
                adjacency_matrix = output_topo['pull_down']['adjacency_matrix']
        else:
            adjacency_matrix = cell_cache['adjacency_matrix']

        edge_index = adjacency_matrix.nonzero().t()
        return Data(x=node_features, edge_index=edge_index)

    K = len(X_samples)
    losses = []

    support_batch_data = [create_pyg_data(sample) for sample in X_samples]
    X_batch = Batch.from_data_list(support_batch_data).to(device)

    # For SGD/Adam: use original y values directly
    # For grad_move methods: y is already normalized by caller
    if use_grad_move:
        y_train = y_tensor
    else:
        # Normalize y for training (simple mean/std normalization)
        y_mean_local = y_tensor.mean()
        y_std_local = y_tensor.std() + 1e-8
        y_train = (y_tensor - y_mean_local) / y_std_local

    initial_loss = criterion(model(X_batch), y_train) / K
    losses.append(initial_loss.item())

    # Apply optimization
    if optim_mode == 'none':
        pass  # No optimization

    elif optim_mode == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=1e-4)
        for step in range(num_steps):
            loss = criterion(model(X_batch), y_train) / K
            losses.append(loss.item())
            model.zero_grad()
            loss.backward()
            optimizer.step()

    elif optim_mode == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
        for step in range(num_steps):
            loss = criterion(model(X_batch), y_train) / K
            losses.append(loss.item())
            model.zero_grad()
            loss.backward()
            optimizer.step()

    elif optim_mode == 'selective_adam':
        if initial_loss > 1e-4:
            optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
            for step in range(num_steps):
                loss = criterion(model(X_batch), y_train) / K
                losses.append(loss.item())
                model.zero_grad()
                loss.backward()
                optimizer.step()

    # Evaluate
    predictions = []
    actual_values = []
    total_loss = 0
    total_mape_loss = 0
    total_rmse_loss = 0

    model.eval()
    with torch.no_grad():
        for i in range(total_points):
            sample_data = create_pyg_data(true_samples[i])
            sample_batch = Batch.from_data_list([sample_data]).to(device)

            raw_pred = model(sample_batch).item()

            if use_grad_move:
                # Denormalize using grad/move
                pred_value = ((raw_pred - move_tensor) * std_tensor + mean_tensor).item()
                actual_value = ((true_function_tensor[i] - move_tensor) * std_tensor + mean_tensor).item()
            else:
                # Denormalize using local mean/std
                pred_value = (raw_pred * y_std_local + y_mean_local).item()
                actual_value = true_function[i].item()

            predictions.append(pred_value)
            actual_values.append(actual_value)

            squared_error = (pred_value - actual_value) ** 2
            mape_loss = abs((pred_value - actual_value) / (actual_value + 1e-8))

            total_loss += squared_error
            total_mape_loss += mape_loss
            total_rmse_loss += squared_error

    # Calculate averages
    avg_total_loss = total_loss / total_points
    avg_total_mape = total_mape_loss / total_points
    avg_total_rmse = math.sqrt(total_rmse_loss / total_points)

    return {
        'model': model,
        'losses': losses,
        'predictions': predictions,
        'actual_values': actual_values,
        'total_loss': avg_total_loss,
        'total_mape': avg_total_mape,
        'total_rmse': avg_total_rmse,
        'optim_mode': optim_mode,
        'num_steps': num_steps,
        'use_grad_move': use_grad_move
    }


def compare_optimization_methods_gnn(initial_model, X_samples, y, true_samples, true_function,
                                     grad, move, topology_cache, cache_type, norm_stats, normalize_fn,
                                     num_steps=50, left_bound=5, right_bound=56, total_points=61,
                                     mode='extrapolation'):
    """
    Compare different optimization methods on the same task.

    Methods:
        - 'none': Grad+Move only (no optimization)
        - 'sgd': Direct SGD (no grad/move)
        - 'adam': Direct Adam (no grad/move)
        - 'selective_adam': Grad+Move + Adam if loss > threshold
    """
    y_mean = y.mean()
    y_std = y.std()
    std_scaled = y_std * grad

    # For grad_move methods: normalize targets
    y_norm = (y - y_mean) / std_scaled + move
    true_function_norm = (true_function - y_mean) / std_scaled + move

    methods = ['none', 'sgd', 'adam', 'selective_adam']
    method_names = {
        'none': 'Grad+Move Only',
        'sgd': f'SGD {num_steps} steps',
        'adam': f'Adam {num_steps} steps',
        'selective_adam': f'Selective Adam'
    }

    results = {}
    for method in methods:
        use_grad_move = method in ['none', 'selective_adam']

        if use_grad_move:
            y_input = y_norm
            true_fn_input = true_function_norm
        else:
            y_input = y  # Original scale
            true_fn_input = true_function  # Original scale

        result = model_functions_with_optim_mode_gnn(
            initial_model=initial_model,
            X_samples=X_samples,
            y=y_input,
            true_samples=true_samples,
            true_function=true_fn_input,
            topology_cache=topology_cache,
            cache_type=cache_type,
            norm_stats=norm_stats,
            normalize_fn=normalize_fn,
            optim_mode=method,
            num_steps=num_steps,
            lr=0.003,
            std=std_scaled,
            mean=y_mean,
            move=move,
            grad=grad,
            left_bound=left_bound,
            right_bound=right_bound,
            total_points=total_points,
            mode=mode
        )
        result['name'] = method_names[method]
        results[method] = result

    return results


def plot_optimization_comparison_gnn(results, indices, total_points=61, left_bound=5, right_bound=56,
                                     mode='extrapolation', cell_name='', task_id=0):
    """Plot comparison of different optimization methods."""
    import matplotlib.pyplot as plt
    import numpy as np

    methods = ['none', 'sgd', 'adam', 'selective_adam']
    colors = {'none': '#e74c3c', 'sgd': '#3498db', 'adam': '#2ecc71', 'selective_adam': '#9b59b6'}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Predictions
    ax = axes[0]
    x_axis = np.arange(total_points)
    actual = results['none']['actual_values']
    ax.plot(x_axis, actual, 'o-', label='Ground Truth', color='gray', alpha=0.5, markersize=3)

    for method in methods:
        preds = results[method]['predictions']
        ax.plot(x_axis, preds, '-', label=results[method]['name'], color=colors[method], alpha=0.8, linewidth=1.5)

    support_y = [actual[i] for i in indices]
    ax.scatter(indices, support_y, color='red', s=100, zorder=5, marker='x', label='Support Set')

    if mode == 'extrapolation':
        ax.axvspan(0, left_bound, alpha=0.1, color='blue')
        ax.axvspan(left_bound, right_bound, alpha=0.1, color='green')
        ax.axvspan(right_bound, total_points, alpha=0.1, color='orange')

    ax.set_xlabel('Sample Index', fontsize=11)
    ax.set_ylabel('Delay (s)', fontsize=11)
    ax.set_title(f'Predictions Comparison\nCell: {cell_name}, Task: {task_id}', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 2: Loss curves
    ax = axes[1]
    for method in methods:
        losses = results[method]['losses']
        if len(losses) > 1:
            ax.plot(losses, '-', label=results[method]['name'], color=colors[method], linewidth=1.5)
        else:
            ax.axhline(y=losses[0], linestyle='--', label=results[method]['name'], color=colors[method])
    ax.set_xlabel('Optimization Step', fontsize=11)
    ax.set_ylabel('MSE Loss', fontsize=11)
    ax.set_title('Loss Curves', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Plot 3: RMSE/NRMSE comparison (Total only)
    ax = axes[2]
    method_labels = [results[m]['name'] for m in methods]
    x_pos = np.arange(len(methods))
    width = 0.35

    rmse_total = [results[m]['total_rmse'] for m in methods]

    # Calculate NRMSE: RMSE / mean * 100
    actual_mean = np.mean(actual) if np.mean(actual) != 0 else 1e-8
    nrmse_total = [results[m]['total_rmse'] / actual_mean * 100 for m in methods]

    ax.bar(x_pos - width/2, rmse_total, width, label='RMSE', color='#34495e')
    ax2 = ax.twinx()
    ax2.bar(x_pos + width/2, nrmse_total, width, label='NRMSE (%)', color='#27ae60', alpha=0.7)

    ax.set_xlabel('Optimization Method', fontsize=11)
    ax.set_ylabel('RMSE', fontsize=11)
    ax2.set_ylabel('NRMSE (%)', fontsize=11)
    ax.set_title('Total RMSE/NRMSE Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([m.replace(' ', '\n') for m in method_labels], fontsize=8)
    ax.legend(loc='upper left', fontsize=9)
    ax2.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


def print_optimization_comparison_summary_gnn(results, mode='extrapolation'):
    """Print summary table."""
    methods = ['none', 'sgd', 'adam', 'selective_adam']

    print("\n" + "="*70)
    print("OPTIMIZATION METHOD COMPARISON SUMMARY")
    print("="*70)
    print(f"\n{'Method':<20} | {'RMSE Total':<12} | {'MAPE Total (%)':<14}")
    print("-"*70)

    for method in methods:
        r = results[method]
        name = r['name']
        print(f"{name:<20} | {r['total_rmse']:<12.6f} | {r['total_mape']*100:<14.3f}")

    print("="*70)
