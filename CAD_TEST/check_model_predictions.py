#!/usr/bin/env python3
"""
Check model predictions vs actual support values
"""

import os
import sys
import torch
import numpy as np

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

# First task support data (from debug output, already normalized)
# Task: AN4YM16 / cell_fall / pin A / slew_idx=4, load_idx=6
support_conditions = ['ff0p88vm40c', 'ss0p72vm40c', 'tt0p8v25c', 'ff0p99v125c', 'ss0p81v125c',
                      'tt0p9v25c', 'ff1p1v125c', 'ff1p1vm40c', 'ss0p9v125c', 'ss0p9vm40c']

# Normalized input features [a, b, c, temp, volt, dim, delay, slew, load]
X_support = torch.tensor([
    [1.4420, 0.0710, 2.0240, -1.0658, -0.2554, 2.0, 1.0, 0.157995, 3.359191],  # ff0p88vm40c
    [1.4630, -0.0760, 2.0240, -1.0658, -1.6174, 2.0, 1.0, 0.157995, 3.359191],  # ss0p72vm40c
    [1.4500, 0.0000, 2.0240, -0.1889, -0.9364, 2.0, 1.0, 0.157995, 3.359191],  # tt0p8v25c
    [1.4420, 0.0710, 2.0240, 1.1602, 0.6810, 2.0, 1.0, 0.157995, 3.359191],   # ff0p99v125c
    [1.4630, -0.0760, 2.0240, 1.1602, -0.8513, 2.0, 1.0, 0.157995, 3.359191],  # ss0p81v125c
    [1.4500, 0.0000, 2.0240, -0.1889, -0.0851, 2.0, 1.0, 0.157995, 3.359191],  # tt0p9v25c
    [1.4420, 0.0710, 2.0240, 1.1602, 1.6174, 2.0, 1.0, 0.157995, 3.359191],   # ff1p1v125c
    [1.4420, 0.0710, 2.0240, -1.0658, 1.6174, 2.0, 1.0, 0.157995, 3.359191],  # ff1p1vm40c
    [1.4630, -0.0760, 2.0240, 1.1602, -0.0851, 2.0, 1.0, 0.157995, 3.359191],  # ss0p9v125c
    [1.4630, -0.0760, 2.0240, -1.0658, -0.0851, 2.0, 1.0, 0.157995, 3.359191], # ss0p9vm40c
], dtype=torch.float32).to(device)

# Actual support outputs (ns)
y_support = np.array([0.264363, 2.527260, 0.492759, 0.198350, 0.744809,
                      0.363336, 0.168710, 0.174016, 0.540752, 0.651044])

# Get model predictions
with torch.no_grad():
    predictions = maml_model.model.model(X_support).cpu().numpy().flatten()

print("\n" + "=" * 90)
print("MODEL PREDICTIONS vs ACTUAL SUPPORT VALUES")
print("=" * 90)
print(f"{'Idx':<4} {'Condition':<15} {'Actual (ns)':<15} {'Model Pred':<15} {'Ratio':<10}")
print("-" * 90)

for i, cond in enumerate(support_conditions):
    ratio = predictions[i] / y_support[i] if y_support[i] != 0 else 0
    print(f"{i:<4} {cond:<15} {y_support[i]:<15.6f} {predictions[i]:<15.6f} {ratio:<10.4f}")

print("-" * 90)
print(f"\n{'SUMMARY':}")
print(f"  Actual min:      {y_support.min():.6f} ns ({support_conditions[np.argmin(y_support)]})")
print(f"  Actual max:      {y_support.max():.6f} ns ({support_conditions[np.argmax(y_support)]})")
print(f"  Actual range:    {y_support.max() - y_support.min():.6f} ns")
print()
print(f"  Model pred min:  {predictions.min():.6f} ({support_conditions[np.argmin(predictions)]})")
print(f"  Model pred max:  {predictions.max():.6f} ({support_conditions[np.argmax(predictions)]})")
print(f"  Model pred range:{predictions.max() - predictions.min():.6f}")
print()

# Calculate grad
y_mean = y_support.mean()
y_std = y_support.std()
y_norm = (y_support - y_mean) / y_std

y_max = y_norm.max()
y_min = y_norm.min()
max_val = predictions.max()
min_val = predictions.min()

grad = (y_max - y_min) / (max_val - min_val)

print(f"  y_norm range:    {y_max - y_min:.6f} (y_max={y_max:.4f}, y_min={y_min:.4f})")
print(f"  pred range:      {max_val - min_val:.6f} (max={max_val:.4f}, min={min_val:.4f})")
print(f"  grad:            {grad:.6f}")
print()

# Check if min/max indices match
print("=" * 90)
print("MIN/MAX INDEX COMPARISON")
print("=" * 90)
print(f"  Actual MIN at:   index {np.argmin(y_support)} ({support_conditions[np.argmin(y_support)]})")
print(f"  Model MIN at:    index {np.argmin(predictions)} ({support_conditions[np.argmin(predictions)]})")
print(f"  Match: {'YES' if np.argmin(y_support) == np.argmin(predictions) else 'NO'}")
print()
print(f"  Actual MAX at:   index {np.argmax(y_support)} ({support_conditions[np.argmax(y_support)]})")
print(f"  Model MAX at:    index {np.argmax(predictions)} ({support_conditions[np.argmax(predictions)]})")
print(f"  Match: {'YES' if np.argmax(y_support) == np.argmax(predictions) else 'NO'}")
