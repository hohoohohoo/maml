#!/usr/bin/env python3
"""
Analyze adaptation loss over different number of steps
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

# Add paths for imports
sys.path.insert(0, '/home/tkdgn2907/Deepsets_test/MAML/Projects/model_code')
sys.path.insert(0, '/home/tkdgn2907/Deepsets_test/MAML/Projects/pretraining/model_test_code/utils')

from maml_optimized import OptimizedMAML, MAMLModel_3hidden

# GPU settings
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load model
model_path = '/home/tkdgn2907/Deepsets_test/MAML/Projects/pretrained_models/training_loss_taskdivide_all/cell_innerdiv100_meta32_combined_519traintask_full1DMAML_weights_3hidden_(40)_300000_inner1_upgraded_tsmc.pth'

maml_model = OptimizedMAML(
    model=MAMLModel_3hidden(in_features=9, layer_length=40),
    dataset_in=None,
    dataset_out=None,
    inner_lr=0.001,
    meta_lr=0.0001
)
state_dict = torch.load(model_path, map_location=device)
maml_model.model.load_state_dict(state_dict)
maml_model.model.to(device)
maml_model.model.eval()
print("Model loaded.")

# First task support data (normalized)
support_conditions = ['ff0p88vm40c', 'ss0p72vm40c', 'tt0p8v25c', 'ff0p99v125c', 'ss0p81v125c',
                      'tt0p9v25c', 'ff1p1v125c', 'ff1p1vm40c', 'ss0p9v125c', 'ss0p9vm40c']
X_support = torch.tensor([
    [1.4420, 0.0710, 2.0240, -1.0658, -0.2554, 2.0, 1.0, 0.157995, 3.359191],
    [1.4630, -0.0760, 2.0240, -1.0658, -1.6174, 2.0, 1.0, 0.157995, 3.359191],
    [1.4500, 0.0000, 2.0240, -0.1889, -0.9364, 2.0, 1.0, 0.157995, 3.359191],
    [1.4420, 0.0710, 2.0240, 1.1602, 0.6810, 2.0, 1.0, 0.157995, 3.359191],
    [1.4630, -0.0760, 2.0240, 1.1602, -0.8513, 2.0, 1.0, 0.157995, 3.359191],
    [1.4500, 0.0000, 2.0240, -0.1889, -0.0851, 2.0, 1.0, 0.157995, 3.359191],
    [1.4420, 0.0710, 2.0240, 1.1602, 1.6174, 2.0, 1.0, 0.157995, 3.359191],
    [1.4420, 0.0710, 2.0240, -1.0658, 1.6174, 2.0, 1.0, 0.157995, 3.359191],
    [1.4630, -0.0760, 2.0240, 1.1602, -0.0851, 2.0, 1.0, 0.157995, 3.359191],
    [1.4630, -0.0760, 2.0240, -1.0658, -0.0851, 2.0, 1.0, 0.157995, 3.359191],
], dtype=torch.float32).to(device)

y_support = torch.tensor([
    [0.264363], [2.527260], [0.492759], [0.198350], [0.744809],
    [0.363336], [0.168710], [0.174016], [0.540752], [0.651044]
], dtype=torch.float32).to(device)

# Test/Query data (5 conditions)
test_conditions = ['ff0p88v125c', 'ss0p72v125c', 'ff0p99vm40c', 'ss0p81vm40c', 'tt1p0v25c']
X_query = torch.tensor([
    [1.4420, 0.0710, 2.0240, 1.1602, -0.2554, 2.0, 1.0, 0.157995, 3.359191],
    [1.4630, -0.0760, 2.0240, 1.1602, -1.6174, 2.0, 1.0, 0.157995, 3.359191],
    [1.4420, 0.0710, 2.0240, -1.0658, 0.6810, 2.0, 1.0, 0.157995, 3.359191],
    [1.4630, -0.0760, 2.0240, -1.0658, -0.8513, 2.0, 1.0, 0.157995, 3.359191],
    [1.4500, 0.0000, 2.0240, -0.1889, 0.7661, 2.0, 1.0, 0.157995, 3.359191],
], dtype=torch.float32).to(device)

y_query = torch.tensor([
    [0.240873], [1.168940], [0.210771], [1.083850], [0.289540]
], dtype=torch.float32).to(device)

def run_adaptation_with_tracking(initial_model, X_support, y_support, X_query, y_query,
                                  max_steps=200, lr=3e-4):
    """Run adaptation and track loss at each step"""

    # Create a copy of the model
    model = nn.Sequential(OrderedDict([
        ('l1', nn.Linear(9, 40)),
        ('relu1', nn.ReLU()),
        ('l2', nn.Linear(40, 40)),
        ('relu3', nn.ReLU()),
        ('l4', nn.Linear(40, 40)),
        ('relu2', nn.ReLU()),
        ('l3', nn.Linear(40, 1))
    ])).to(device)
    model.load_state_dict(initial_model.state_dict())

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Normalize support data
    y_mean = y_support.mean()
    y_std = y_support.std()
    y_norm = (y_support - y_mean) / y_std

    # Calculate grad and move
    with torch.no_grad():
        pred_support = model(X_support)
        min_val = pred_support.min().item()
        max_val = pred_support.max().item()

    y_max = y_norm.max().item()
    y_min = y_norm.min().item()
    grad = (y_max - y_min) / (max_val - min_val) if abs(max_val - min_val) > 1e-8 else 1.0

    # Prepare normalized target
    y_target = y_norm

    # Track metrics
    support_losses = []
    query_losses = []
    query_nrmses = []
    query_mapes = []

    K = X_support.shape[0]

    for step in range(max_steps + 1):
        # Calculate support loss
        with torch.no_grad():
            pred = model(X_support)
            support_loss = criterion(pred, y_target).item()
            support_losses.append(support_loss)

            # Calculate query metrics (in original scale)
            pred_query = model(X_query)
            pred_query_orig = pred_query * y_std + y_mean

            # MSE on query
            query_mse = ((pred_query_orig - y_query) ** 2).mean().item()
            query_losses.append(query_mse)

            # NRMSE
            query_range = y_query.max().item() - y_query.min().item()
            query_rmse = np.sqrt(query_mse)
            query_nrmse = (query_rmse / query_range * 100) if query_range > 0 else 0
            query_nrmses.append(query_nrmse)

            # MAPE
            mape = (torch.abs(pred_query_orig - y_query) / (y_query + 1e-8)).mean().item() * 100
            query_mapes.append(mape)

        if step < max_steps:
            # Training step
            model.zero_grad()
            loss = criterion(model(X_support), y_target)
            loss.backward()
            optimizer.step()

    return {
        'support_losses': support_losses,
        'query_losses': query_losses,
        'query_nrmses': query_nrmses,
        'query_mapes': query_mapes
    }

# Run adaptation with tracking
print("\nRunning adaptation with loss tracking...")
max_steps = 200
results = run_adaptation_with_tracking(
    maml_model.model.model, X_support, y_support, X_query, y_query, max_steps=max_steps
)

# Print results at key steps
print("\n" + "=" * 80)
print("ADAPTATION LOSS OVER STEPS")
print("=" * 80)
print(f"{'Step':<8} {'Support Loss':<15} {'Query MSE':<15} {'Query NRMSE':<15} {'Query MAPE':<15}")
print("-" * 80)
key_steps = [0, 10, 20, 40, 60, 80, 100, 150, 200]
for step in key_steps:
    if step <= max_steps:
        print(f"{step:<8} {results['support_losses'][step]:<15.6f} {results['query_losses'][step]:<15.6f} "
              f"{results['query_nrmses'][step]:<15.2f}% {results['query_mapes'][step]:<15.2f}%")

# Find best step
best_step = np.argmin(results['query_nrmses'])
print("-" * 80)
print(f"Best step: {best_step} with Query NRMSE: {results['query_nrmses'][best_step]:.2f}%")

# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Support Loss
ax1 = axes[0, 0]
ax1.plot(results['support_losses'], 'b-', linewidth=1)
ax1.set_xlabel('Adaptation Step')
ax1.set_ylabel('Support Loss (MSE)')
ax1.set_title('Support Loss over Adaptation Steps')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)
ax1.axvline(x=40, color='r', linestyle='--', alpha=0.5, label='Default (40 steps)')
ax1.axvline(x=best_step, color='g', linestyle='--', alpha=0.5, label=f'Best ({best_step} steps)')
ax1.legend()

# Query Loss
ax2 = axes[0, 1]
ax2.plot(results['query_losses'], 'r-', linewidth=1)
ax2.set_xlabel('Adaptation Step')
ax2.set_ylabel('Query Loss (MSE)')
ax2.set_title('Query Loss over Adaptation Steps')
ax2.grid(True, alpha=0.3)
ax2.axvline(x=40, color='r', linestyle='--', alpha=0.5, label='Default (40 steps)')
ax2.axvline(x=best_step, color='g', linestyle='--', alpha=0.5, label=f'Best ({best_step} steps)')
ax2.legend()

# Query NRMSE
ax3 = axes[1, 0]
ax3.plot(results['query_nrmses'], 'g-', linewidth=1)
ax3.set_xlabel('Adaptation Step')
ax3.set_ylabel('Query NRMSE (%)')
ax3.set_title('Query NRMSE over Adaptation Steps')
ax3.grid(True, alpha=0.3)
ax3.axvline(x=40, color='r', linestyle='--', alpha=0.5, label='Default (40 steps)')
ax3.axvline(x=best_step, color='g', linestyle='--', alpha=0.5, label=f'Best ({best_step} steps)')
ax3.axhline(y=results['query_nrmses'][40], color='r', linestyle=':', alpha=0.5)
ax3.legend()

# Query MAPE
ax4 = axes[1, 1]
ax4.plot(results['query_mapes'], 'm-', linewidth=1)
ax4.set_xlabel('Adaptation Step')
ax4.set_ylabel('Query MAPE (%)')
ax4.set_title('Query MAPE over Adaptation Steps')
ax4.grid(True, alpha=0.3)
ax4.axvline(x=40, color='r', linestyle='--', alpha=0.5, label='Default (40 steps)')
ax4.axvline(x=best_step, color='g', linestyle='--', alpha=0.5, label=f'Best ({best_step} steps)')
ax4.legend()

plt.tight_layout()
plt.savefig('/home/tkdgn2907/Deepsets_test/MAML/Projects/CAD_TEST/adaptation_loss_curve.png', dpi=150, bbox_inches='tight')
plt.savefig('/home/tkdgn2907/Deepsets_test/MAML/Projects/CAD_TEST/adaptation_loss_curve.pdf', bbox_inches='tight')
print("\nSaved: adaptation_loss_curve.png, adaptation_loss_curve.pdf")
