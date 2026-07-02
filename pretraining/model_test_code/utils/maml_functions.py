"""
MAML-specific functions for extrapolation testing
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


def model_functions_at_training_maml(initial_model, X, y, true_x, true_function,
                                     optim=torch.optim.SGD, lr=0.003, adam_step=0, std=1, mean=10, move=0,
                                     left_bound=5, right_bound=56, total_points=61, mode='extrapolation',
                                     layer_length=40):
    """
    trains the model on X, y and measures the loss curve.
    for each n in sampled_steps, records model(x_axis) after n gradient updates.
    mode: 'extrapolation' or 'interpolation' - determines whether to calculate left/right extrapolation metrics
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X = X.to(device)
    y = y.to(device)
    true_x = true_x.to(device)
    true_function = true_function.to(device)
    std = std.to(device) if isinstance(std, torch.Tensor) else torch.tensor(std).to(device)
    mean = mean.to(device) if isinstance(mean, torch.Tensor) else torch.tensor(mean).to(device)
    move = move.to(device) if isinstance(move, torch.Tensor) else torch.tensor(move).to(device)
    # Copy MAML model into a new object to preserve MAML weights during training
    input_dim = X.shape[2] if len(X.shape) > 2 else X.shape[1]
    model = nn.Sequential(OrderedDict([
        ('l1', nn.Linear(input_dim, layer_length)),
        ('relu1', nn.ReLU()),
        ('l2', nn.Linear(layer_length, layer_length)),
        ('relu3', nn.ReLU()),
        ('l4', nn.Linear(layer_length, layer_length)),
        ('relu2', nn.ReLU()),
        ('l3', nn.Linear(layer_length, 1))
    ])).to(device)
    model.load_state_dict(initial_model.state_dict())
    criterion = nn.MSELoss()
    optimiser = optim(model.parameters(), lr, weight_decay=1e-4)
    adam_condition_triggered = False

    # Train model on a random task
    K = X.shape[0]

    losses = []
    outputs = {}
    loss = criterion(model(X), y) / K

    # Adam training if SGD loss is still high
    if loss > 1e-4:
        adam_condition_triggered = True
        optimiser2 = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
        for step in range(1, adam_step+1):
            loss = criterion(model(X), y) / K
            losses.append(loss.item())

            # compute grad and update inner loop weights
            model.zero_grad()
            loss.backward()
            optimiser2.step()

    # Calculate losses
    total_loss = 0
    total_mape_loss = 0
    total_mae_loss = 0

    # Store predictions and actual values for plotting
    predictions = []
    actual_values = []

    for i in range(total_points):
        pred_value = ((model(true_x[i])-move)*std+mean).item()
        actual_value = ((true_function[i]-move)*std+mean).item()
        predictions.append(pred_value)
        actual_values.append(actual_value)

        loss = criterion((model(true_x[i])-move)*std+mean, (true_function[i]-move)*std+mean)

        # Calculate MAPE (Mean Absolute Percentage Error)
        if abs(actual_value) > 1e-8:  # Avoid division by zero
            mae_loss = abs((pred_value - actual_value))
            mape_loss = abs(mae_loss / actual_value)
        else:
            mape_loss = 0

        total_loss += loss
        total_mape_loss += mape_loss
        total_mae_loss += mae_loss

    # Calculate average losses
    avg_total_loss = total_loss / total_points
    avg_total_mape = total_mape_loss / total_points
    avg_total_mae = total_mae_loss / total_points

    return (model, outputs, losses, avg_total_loss, avg_total_mape, predictions, actual_values,
            adam_condition_triggered, avg_total_mae)


def evaluate_model_performance_maml(initial_model, model_name, X, y, true_x, true_function, grad, move,
                                   optim=torch.optim.SGD, lr=0.001,
                                   left_bound=5, right_bound=56, total_points=61, mode='extrapolation',
                                   adaptation_method='selective_adam', layer_length=40):
    """
    Evaluate MAML model performance with grad/move normalization parameters.

    Args:
        adaptation_method: 'selective_adam' (grad/move + conditional Adam) or 'adam' (direct Adam, no grad/move)

    Returns:
        tuple: (total_loss, total_mape, predictions, actual_values, model, adam_used, total_rmse)
    """
    import numpy as np
    import math

    # If using 'adam' method, use direct Adam without grad/move scaling
    if adaptation_method == 'adam':
        result = model_functions_with_optim_mode_maml(
            initial_model=initial_model,
            X=X,
            y=y,
            true_x=true_x,
            true_function=true_function,
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
            mode=mode,
            layer_length=layer_length
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
             adam_condition_triggered, avg_total_mae) = model_functions_at_training_maml(
                initial_model,
                X, y=y_test,
                true_x=true_x,
                true_function=true_function1,
                optim=optim,
                lr=lr,
                adam_step=40,
                std=y_std1,
                mean=y_mean1,
                move = move,
                left_bound=left_bound,
                right_bound=right_bound,
                total_points=total_points,
                mode=mode,
                layer_length=layer_length
            )
            adam_used = adam_condition_triggered

            # Collect predictions and actuals
            all_predictions.extend(predictions)
            all_actuals.extend(actual_values)

            model_min = model
            loss_min = total_loss
            mape_min = total_mape_loss

    # Calculate RMSE from predictions and actuals
    rmse_total = math.sqrt(sum((p - a) ** 2 for p, a in zip(all_predictions, all_actuals)) / len(all_predictions)) if all_predictions else 0

    return (loss_min, mape_min, all_predictions, all_actuals, model_min, adam_used, rmse_total)


def model_functions_with_optim_mode_maml(initial_model, X, y, true_x, true_function,
                                          optim_mode='selective_adam', num_steps=50, lr=0.003,
                                          std=1, mean=10, move=0, grad=1,
                                          left_bound=5, right_bound=56, total_points=61, mode='extrapolation',
                                          layer_length=40):
    """
    Train MLP/MAML model with different optimization modes for comparison.

    Args:
        optim_mode:
            - 'none': Grad+Move only (no optimization)
            - 'sgd': Direct SGD optimization (no grad/move)
            - 'adam': Direct Adam optimization (no grad/move)
            - 'selective_adam': Grad+Move + Adam if loss > threshold
            - 'full_adam': Grad+Move + Adam always (no threshold)
    """
    import math
    # Use device from input tensor (allows CPU execution when measure_time is enabled)
    device = X.device

    # Determine if using grad/move scaling
    use_grad_move = optim_mode in ['none', 'selective_adam', 'full_adam']

    y_tensor = y.clone().to(device)
    true_x = true_x.to(device)
    true_function_tensor = true_function.clone().to(device)
    std_tensor = std.to(device) if isinstance(std, torch.Tensor) else torch.tensor(std).to(device)
    mean_tensor = mean.to(device) if isinstance(mean, torch.Tensor) else torch.tensor(mean).to(device)
    move_tensor = move.to(device) if isinstance(move, torch.Tensor) else torch.tensor(move).to(device)

    # Copy model
    input_dim = X.shape[2] if len(X.shape) > 2 else X.shape[1]
    model = nn.Sequential(OrderedDict([
        ('l1', nn.Linear(input_dim, layer_length)),
        ('relu1', nn.ReLU()),
        ('l2', nn.Linear(layer_length, layer_length)),
        ('relu3', nn.ReLU()),
        ('l4', nn.Linear(layer_length, layer_length)),
        ('relu2', nn.ReLU()),
        ('l3', nn.Linear(layer_length, 1))
    ])).to(device)
    model.load_state_dict(initial_model.state_dict())

    criterion = nn.MSELoss()
    K = X.shape[0]
    losses = []

    # For SGD/Adam: use original y values directly
    # For grad_move methods: y is already normalized by caller
    if use_grad_move:
        y_train = y_tensor
    else:
        # Normalize y for training (simple mean/std normalization)
        y_mean_local = y_tensor.mean()
        y_std_local = y_tensor.std() + 1e-8
        y_train = (y_tensor - y_mean_local) / y_std_local

    initial_loss = criterion(model(X), y_train) / K
    losses.append(initial_loss.item())

    # Track if Adam was triggered (for selective_adam)
    adam_triggered = False

    # Apply optimization
    if optim_mode == 'none':
        pass  # No optimization

    elif optim_mode == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=1e-4)
        for step in range(num_steps):
            loss = criterion(model(X), y_train) / K
            losses.append(loss.item())
            model.zero_grad()
            loss.backward()
            optimizer.step()

    elif optim_mode == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
        for step in range(num_steps):
            loss = criterion(model(X), y_train) / K
            losses.append(loss.item())
            model.zero_grad()
            loss.backward()
            optimizer.step()

    elif optim_mode == 'selective_adam':
        adam_triggered = initial_loss > 1e-4
        if adam_triggered:
            optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
            for step in range(num_steps):
                loss = criterion(model(X), y_train) / K
                losses.append(loss.item())
                model.zero_grad()
                loss.backward()
                optimizer.step()

    elif optim_mode == 'full_adam':
        # Grad+Move + Adam always (no threshold check)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
        for step in range(num_steps):
            loss = criterion(model(X), y_train) / K
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
            raw_pred = model(true_x[i]).item()

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
        'use_grad_move': use_grad_move,
        'adam_triggered': adam_triggered
    }


def compare_optimization_methods_maml(initial_model, X, y, true_x, true_function,
                                       grad, move, num_steps=50,
                                       left_bound=5, right_bound=56, total_points=61,
                                       mode='extrapolation', layer_length=40,
                                       use_cpu_for_timing=False):
    """
    Compare different optimization methods on the same task.

    Methods:
        - 'none': Grad+Move only (no optimization)
        - 'sgd': Direct SGD (no grad/move)
        - 'adam': Direct Adam (no grad/move)
        - 'selective_adam': Grad+Move + Adam if loss > threshold
        - 'full_adam': Grad+Move + Adam always (no threshold)

    Args:
        use_cpu_for_timing: If True, run optimization on CPU for consistent timing measurement
    """
    import time
    import copy

    # If using CPU for timing, create CPU copies of model and data
    if use_cpu_for_timing:
        cpu_device = torch.device('cpu')
        # Deep copy model to CPU
        initial_model_cpu = copy.deepcopy(initial_model).to(cpu_device)
        X_cpu = X.to(cpu_device)
        y_cpu = y.to(cpu_device)
        true_x_cpu = true_x.to(cpu_device)
        true_function_cpu = true_function.to(cpu_device)
        # Use CPU copies
        model_to_use = initial_model_cpu
        X_to_use = X_cpu
        y_to_use = y_cpu
        true_x_to_use = true_x_cpu
        true_function_to_use = true_function_cpu
    else:
        model_to_use = initial_model
        X_to_use = X
        y_to_use = y
        true_x_to_use = true_x
        true_function_to_use = true_function

    # Convert grad/move to Python floats if they are tensors (for CPU compatibility)
    grad_val = grad.item() if isinstance(grad, torch.Tensor) else grad
    move_val = move.item() if isinstance(move, torch.Tensor) else move

    y_mean = y_to_use.mean()
    y_std = y_to_use.std()
    std_scaled = y_std * grad_val

    # For grad_move methods: normalize targets
    y_norm = (y_to_use - y_mean) / std_scaled + move_val
    true_function_norm = (true_function_to_use - y_mean) / std_scaled + move_val

    methods = ['none', 'sgd', 'adam', 'selective_adam', 'full_adam']
    method_names = {
        'none': 'Grad+Move Only',
        'sgd': f'SGD {num_steps} steps',
        'adam': f'Adam {num_steps} steps',
        'selective_adam': 'Selective Adam',
        'full_adam': f'Full Adam {num_steps}'
    }

    results = {}
    for method in methods:
        use_grad_move = method in ['none', 'selective_adam', 'full_adam']

        if use_grad_move:
            y_input = y_norm
            true_fn_input = true_function_norm
        else:
            y_input = y_to_use  # Original scale
            true_fn_input = true_function_to_use  # Original scale

        # Synchronize GPU before timing (only if not using CPU for timing)
        if not use_cpu_for_timing and torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        result = model_functions_with_optim_mode_maml(
            initial_model=model_to_use,
            X=X_to_use,
            y=y_input,
            true_x=true_x_to_use,
            true_function=true_fn_input,
            optim_mode=method,
            num_steps=num_steps,
            lr=0.003,
            std=std_scaled,
            mean=y_mean,
            move=move_val,
            grad=grad_val,
            left_bound=left_bound,
            right_bound=right_bound,
            total_points=total_points,
            mode=mode,
            layer_length=layer_length
        )
        # Synchronize GPU after operation (only if not using CPU for timing)
        if not use_cpu_for_timing and torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_time = time.perf_counter() - start_time
        result['name'] = method_names[method]
        result['time_ms'] = elapsed_time * 1000  # Convert to milliseconds
        results[method] = result

    return results


def plot_optimization_comparison_maml(results, indices, total_points=61, left_bound=5, right_bound=56,
                                       mode='extrapolation', cell_name='', task_id=0):
    """Plot comparison of different optimization methods."""
    import matplotlib.pyplot as plt
    import numpy as np

    methods = ['none', 'sgd', 'adam', 'selective_adam', 'full_adam']
    colors = {'none': '#e74c3c', 'sgd': '#3498db', 'adam': '#2ecc71', 'selective_adam': '#9b59b6', 'full_adam': '#f39c12'}

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Predictions
    ax = axes[0, 0]
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
    ax.set_ylabel('Delay', fontsize=11)
    ax.set_title(f'Predictions Comparison\nCell: {cell_name}, Task: {task_id}', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 2: Loss curves
    ax = axes[0, 1]
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

    # Plot 3: RMSE comparison
    ax = axes[1, 0]
    method_labels = [results[m]['name'] for m in methods]
    x_pos = np.arange(len(methods))
    width = 0.2

    rmse_total = [results[m]['total_rmse'] for m in methods]
    rmse_inter = [results[m]['inter_rmse'] for m in methods]

    ax.bar(x_pos - 0.5*width, rmse_total, width, label='Total', color='#34495e')
    ax.bar(x_pos + 0.5*width, rmse_inter, width, label='Interpolation', color='#27ae60')

    if mode == 'extrapolation':
        rmse_leftex = [results[m]['leftex_rmse'] for m in methods]
        rmse_rightex = [results[m]['rightex_rmse'] for m in methods]
        ax.bar(x_pos + 1.5*width, rmse_leftex, width, label='Left Extrap', color='#3498db')
        ax.bar(x_pos + 2.5*width, rmse_rightex, width, label='Right Extrap', color='#e67e22')

    ax.set_xlabel('Optimization Method', fontsize=11)
    ax.set_ylabel('RMSE', fontsize=11)
    ax.set_title('RMSE Comparison by Region', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([m.replace(' ', '\n') for m in method_labels], fontsize=8)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: NRMSE comparison
    ax = axes[1, 1]

    # Calculate NRMSE: RMSE / mean * 100
    actual = results['none']['actual_values']
    actual_mean = np.mean(actual) if np.mean(actual) != 0 else 1e-8

    # Calculate regional means for NRMSE normalization
    if mode == 'extrapolation':
        leftex_mean = np.mean(actual[:left_bound]) if left_bound > 0 and np.mean(actual[:left_bound]) != 0 else actual_mean
        inter_mean = np.mean(actual[left_bound:right_bound]) if np.mean(actual[left_bound:right_bound]) != 0 else actual_mean
        rightex_mean = np.mean(actual[right_bound:]) if len(actual[right_bound:]) > 0 and np.mean(actual[right_bound:]) != 0 else actual_mean
    else:
        inter_mean = np.mean(actual[left_bound:right_bound]) if np.mean(actual[left_bound:right_bound]) != 0 else actual_mean

    nrmse_total = [results[m]['total_rmse'] / actual_mean * 100 for m in methods]
    nrmse_inter = [results[m]['inter_rmse'] / inter_mean * 100 for m in methods]

    ax.bar(x_pos - 0.5*width, nrmse_total, width, label='Total', color='#34495e')
    ax.bar(x_pos + 0.5*width, nrmse_inter, width, label='Interpolation', color='#27ae60')

    if mode == 'extrapolation':
        nrmse_leftex = [results[m]['leftex_rmse'] / leftex_mean * 100 for m in methods]
        nrmse_rightex = [results[m]['rightex_rmse'] / rightex_mean * 100 for m in methods]
        ax.bar(x_pos + 1.5*width, nrmse_leftex, width, label='Left Extrap', color='#3498db')
        ax.bar(x_pos + 2.5*width, nrmse_rightex, width, label='Right Extrap', color='#e67e22')

    ax.set_xlabel('Optimization Method', fontsize=11)
    ax.set_ylabel('NRMSE (%)', fontsize=11)
    ax.set_title('NRMSE Comparison by Region', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([m.replace(' ', '\n') for m in method_labels], fontsize=8)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


def print_optimization_comparison_summary_maml(results, mode='extrapolation'):
    """Print summary table."""
    methods = ['none', 'sgd', 'adam', 'selective_adam', 'full_adam']

    print("\n" + "="*100)
    print("OPTIMIZATION METHOD COMPARISON SUMMARY")
    print("="*100)
    print(f"\n{'Method':<20} | {'RMSE Total':<12} | {'RMSE Inter':<12} | {'MAPE Total (%)':<14} | {'MAPE Inter (%)':<14}")
    print("-"*100)

    for method in methods:
        r = results[method]
        name = r['name']
        print(f"{name:<20} | {r['total_rmse']:<12.6f} | {r['inter_rmse']:<12.6f} | {r['total_mape']*100:<14.3f} | {r['inter_mape']*100:<14.3f}")

    if mode == 'extrapolation':
        print("\n" + "-"*100)
        print(f"{'Method':<20} | {'RMSE Left':<12} | {'RMSE Right':<12} | {'MAPE Left (%)':<14} | {'MAPE Right (%)':<14}")
        print("-"*100)
        for method in methods:
            r = results[method]
            name = r['name']
            print(f"{name:<20} | {r['leftex_rmse']:<12.6f} | {r['rightex_rmse']:<12.6f} | {r['leftex_mape']*100:<14.3f} | {r['rightex_mape']*100:<14.3f}")

    print("="*100)
