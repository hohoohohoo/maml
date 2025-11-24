"""
MAML-specific functions for extrapolation testing
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


def model_functions_at_training_maml(initial_model, X, y, true_x, true_function,
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

    # Copy MAML model into a new object to preserve MAML weights during training
    input_dim = X.shape[2] if len(X.shape) > 2 else X.shape[1]
    model = nn.Sequential(OrderedDict([
        ('l1', nn.Linear(input_dim, 40)),
        ('relu1', nn.ReLU()),
        ('l2', nn.Linear(40, 40)),
        ('relu3', nn.ReLU()),
        ('l4', nn.Linear(40, 40)),
        ('relu2', nn.ReLU()),
        ('l3', nn.Linear(40, 1))
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
    total_inter_loss = 0
    total_rightex_loss = 0
    total_leftex_loss = 0
    total_mape_loss = 0
    total_leftex_mape = 0
    total_inter_mape = 0
    total_rightex_mape = 0
    total_mae_loss = 0
    total_leftex_mae = 0
    total_inter_mae = 0
    total_rightex_mae = 0

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
            mae_loss = abs((pred_value - actual_value))
            mape_loss = abs(mae_loss / actual_value)
        else:
            mape_loss = 0

        total_loss += loss
        total_mape_loss += mape_loss
        total_mae_loss += mae_loss

        # Regional calculations (only for extrapolation mode)
        if mode == 'extrapolation':
            if i < left_bound:  # Left extrapolation region
                total_leftex_loss += loss
                total_leftex_mape += mape_loss
                total_leftex_mae += mae_loss
            elif i < right_bound:  # Interpolation region
                total_inter_loss += loss
                total_inter_mape += mape_loss
                total_inter_mae += mae_loss
            else:  # Right extrapolation region
                total_rightex_loss += loss
                total_rightex_mape += mape_loss
                total_rightex_mae += mae_loss
        else:  # interpolation mode - only calculate interpolation region
            if i >= left_bound and i < right_bound:
                total_inter_loss += loss
                total_inter_mape += mape_loss
                total_inter_mae += mae_loss

    # Calculate average losses, MAPEs, and MAEs for each region
    avg_total_loss = total_loss / total_points
    avg_total_mape = total_mape_loss / total_points
    avg_total_mae = total_mae_loss / total_points

    # For interpolation mode, only calculate interpolation metrics
    if mode == 'interpolation':
        avg_inter_loss = total_inter_loss / (right_bound - left_bound) if (right_bound - left_bound) > 0 else 0
        avg_inter_mape = total_inter_mape / (right_bound - left_bound) if (right_bound - left_bound) > 0 else 0
        avg_inter_mae = total_inter_mae / (right_bound - left_bound) if (right_bound - left_bound) > 0 else 0
        avg_leftex_loss = 0
        avg_rightex_loss = 0
        avg_leftex_mape = 0
        avg_rightex_mape = 0
        avg_leftex_mae = 0
        avg_rightex_mae = 0
    else:  # extrapolation mode
        avg_inter_loss = total_inter_loss / (right_bound - left_bound) if (right_bound - left_bound) > 0 else 0
        avg_rightex_loss = total_rightex_loss / (total_points - right_bound) if (total_points - right_bound) > 0 else 0
        avg_leftex_loss = total_leftex_loss / left_bound if left_bound > 0 else 0
        avg_leftex_mape = total_leftex_mape / left_bound if left_bound > 0 else 0
        avg_inter_mape = total_inter_mape / (right_bound - left_bound) if (right_bound - left_bound) > 0 else 0
        avg_rightex_mape = total_rightex_mape / (total_points - right_bound) if (total_points - right_bound) > 0 else 0
        avg_leftex_mae = total_leftex_mae / left_bound if left_bound > 0 else 0
        avg_inter_mae = total_inter_mae / (right_bound - left_bound) if (right_bound - left_bound) > 0 else 0
        avg_rightex_mae = total_rightex_mae / (total_points - right_bound) if (total_points - right_bound) > 0 else 0

    return (model, outputs, losses, avg_total_loss, avg_inter_loss, avg_rightex_loss, avg_leftex_loss,
            avg_total_mape, avg_leftex_mape, avg_inter_mape, avg_rightex_mape, predictions, actual_values,
            adam_condition_triggered, avg_total_mae, avg_leftex_mae, avg_inter_mae, avg_rightex_mae)


def evaluate_model_performance_maml(initial_model, model_name, X, y, true_x, true_function, grad, move,
                                   optim=torch.optim.SGD, lr=0.001,
                                   left_bound=5, right_bound=56, total_points=61, mode='extrapolation'):
    import numpy as np

    y_mean = y.mean()
    y_std = y.std()
    mean_values = [y_mean]
    std_values = [y_std * grad]
    loss_min = 10000
    inter_loss_min = 10000
    rightex_loss_min = 10000
    leftex_loss_min = 10000
    mape_min = 10000

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
            (model, outputs, losses, total_loss, total_inter_loss, total_rightex_loss, total_leftex_loss,
             total_mape_loss, leftex_mape, inter_mape, rightex_mape, predictions, actual_values,
             adam_condition_triggered, avg_total_mae, avg_leftex_mae, avg_rightex_mae, avg_inter_mae) = model_functions_at_training_maml(
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
            mean_min = mean
            std_min = std
            output_min = outputs
            losses_min = losses
            inter_loss_min = total_inter_loss
            leftex_loss_min = total_leftex_loss
            rightex_loss_min = total_rightex_loss
            mape_min = total_mape_loss
            leftex_mape_min = leftex_mape
            inter_mape_min = inter_mape
            rightex_mape_min = rightex_mape

    return (loss_min, inter_loss_min, leftex_loss_min, rightex_loss_min,
            mape_min, leftex_mape_min, inter_mape_min, rightex_mape_min,
            all_predictions, all_actuals, output_min, model_min, mean_min, std_min, move, losses_min, adam_used,
            avg_total_mae, avg_leftex_mae, avg_inter_mae, avg_rightex_mae)
