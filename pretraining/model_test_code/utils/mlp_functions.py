"""
MLP-specific functions for extrapolation testing
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


def model_functions_at_training_mlp(initial_model, X, y, true_x, true_function,
                                   optim=torch.optim.SGD, lr=0.003, adam_step=0, std=1, mean=10,
                                   left_bound=5, right_bound=56, total_points=61, mode='extrapolation'):
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


    # Copy MLP model into a new object to preserve initial weights during training
    model = copy.deepcopy(initial_model).to(device)

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

    # Store predictions and actual values for plotting
    predictions = []
    actual_values = []

    for i in range(total_points):
        pred_value = ((model(true_x[i]))*std+mean).item()
        actual_value = ((true_function[i])*std+mean).item()

        predictions.append(pred_value)
        actual_values.append(actual_value)

        loss = criterion((model(true_x[i]))*std+mean, (true_function[i])*std+mean)

        # Calculate MAPE (Mean Absolute Percentage Error)
        if abs(actual_value) > 1e-8:  # Avoid division by zero
            mape_loss = abs((pred_value - actual_value) / actual_value)
        else:
            mape_loss = 0

        total_loss += loss
        total_mape_loss += mape_loss

    # Calculate average losses
    avg_total_loss = total_loss / total_points
    avg_total_mape = total_mape_loss / total_points

    return (model, outputs, losses, avg_total_loss, avg_total_mape, predictions, actual_values,
            adam_condition_triggered)


def evaluate_model_performance_mlp(initial_model, model_name, X, y, true_x, true_function, grad, move,
                                  optim=torch.optim.SGD, lr=0.001,
                                  left_bound=5, right_bound=56, total_points=61, mode='extrapolation',
                                  adaptation_method='selective_adam'):
    """
    Evaluate MLP model performance with grad/move normalization parameters.

    Args:
        adaptation_method: 'selective_adam' (grad/move + conditional Adam), 'adam' (direct Adam, no grad/move),
                           or 'sgd' (direct vanilla SGD, no grad/move — mirrors MAML inner-loop style).

    Returns:
        tuple: (total_loss, total_mape, predictions, actual_values, model, adam_used, mae_total)
    """
    import numpy as np

    # 'adam' / 'sgd': direct optimizer adaptation without grad/move scaling.
    # The shared helper `model_functions_with_optim_mode_mlp` handles both.
    if adaptation_method in ('adam', 'sgd'):
        result = model_functions_with_optim_mode_mlp(
            initial_model=initial_model,
            X=X,
            y=y,
            true_x=true_x,
            true_function=true_function,
            optim_mode=adaptation_method,
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
                result['total_mae'])

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
             adam_condition_triggered) = model_functions_at_training_mlp(
                initial_model,
                X, y=y_test,
                true_x=true_x,
                true_function=true_function1,
                optim=optim,
                lr=lr,
                adam_step=40,
                std=y_std1,
                mean=y_mean1,
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

    # Calculate MAE from predictions and actuals
    mae_total = sum(abs(p - a) for p, a in zip(all_predictions, all_actuals)) / len(all_predictions) if all_predictions else 0

    return (loss_min, mape_min, all_predictions, all_actuals, model_min, adam_used, mae_total)


def model_functions_with_optim_mode_mlp(initial_model, X, y, true_x, true_function,
                                         optim_mode='selective_adam', num_steps=50, lr=0.003,
                                         std=1, mean=10, move=0, grad=1,
                                         left_bound=5, right_bound=56, total_points=61, mode='extrapolation'):
    """
    Train MLP model with different optimization modes for comparison.

    Args:
        optim_mode:
            - 'adam': Direct Adam optimization (no grad/move)
            - 'sgd':  Direct vanilla SGD optimization (no grad/move, no momentum) —
                      mirrors MAML's inner-loop adaptation style (`w - lr * grad`).
            - 'selective_adam': Grad+Move + Adam if loss > threshold
    """
    import math
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Determine if using grad/move scaling (only selective_adam does)
    use_grad_move = optim_mode in ['selective_adam']

    X = X.to(device)
    y_tensor = y.clone().to(device)
    true_x = true_x.to(device)
    true_function_tensor = true_function.clone().to(device)
    std_tensor = std.to(device) if isinstance(std, torch.Tensor) else torch.tensor(std).to(device)
    mean_tensor = mean.to(device) if isinstance(mean, torch.Tensor) else torch.tensor(mean).to(device)
    move_tensor = move.to(device) if isinstance(move, torch.Tensor) else torch.tensor(move).to(device)

    # Copy model using deepcopy to preserve architecture
    model = copy.deepcopy(initial_model).to(device)

    criterion = nn.MSELoss()
    K = X.shape[0]
    losses = []

    # For adam: normalize y for training (simple mean/std normalization)
    if not use_grad_move:
        y_mean_local = y_tensor.mean()
        y_std_local = y_tensor.std() + 1e-8
        y_train = (y_tensor - y_mean_local) / y_std_local
    else:
        y_train = y_tensor

    initial_loss = criterion(model(X), y_train) / K
    losses.append(initial_loss.item())

    # Apply optimizer adaptation. SGD path uses vanilla SGD (no momentum) to
    # mirror MAML inner-loop `w - lr * grad`; other paths use Adam.
    if optim_mode == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    else:
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
    total_mae_loss = 0

    model.eval()
    with torch.no_grad():
        for i in range(total_points):
            raw_pred = model(true_x[i]).item()

            if use_grad_move:
                pred_value = ((raw_pred - move_tensor) * std_tensor + mean_tensor).item()
                actual_value = ((true_function_tensor[i] - move_tensor) * std_tensor + mean_tensor).item()
            else:
                pred_value = (raw_pred * y_std_local + y_mean_local).item()
                actual_value = true_function[i].item()

            predictions.append(pred_value)
            actual_values.append(actual_value)

            squared_error = (pred_value - actual_value) ** 2
            abs_error = abs(pred_value - actual_value)
            mape_loss = abs_error / (abs(actual_value) + 1e-8)

            total_loss += squared_error
            total_mape_loss += mape_loss
            total_mae_loss += abs_error

    # Calculate averages
    avg_total_loss = total_loss / total_points
    avg_total_mape = total_mape_loss / total_points
    avg_total_mae = total_mae_loss / total_points

    return {
        'model': model,
        'losses': losses,
        'predictions': predictions,
        'actual_values': actual_values,
        'total_loss': avg_total_loss,
        'total_mape': avg_total_mape,
        'total_mae': avg_total_mae,
        'optim_mode': optim_mode,
        'num_steps': num_steps,
        'use_grad_move': use_grad_move
    }
