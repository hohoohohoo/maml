#!/usr/bin/env python3
"""Verify min/max in grad calculation"""
import numpy as np

# Support outputs (10 conditions)
support_conditions = ['ff0p88vm40c', 'ss0p72vm40c', 'tt0p8v25c', 'ff0p99v125c', 'ss0p81v125c',
                      'tt0p9v25c', 'ff1p1v125c', 'ff1p1vm40c', 'ss0p9v125c', 'ss0p9vm40c']
support_outputs = np.array([0.264363, 2.527260, 0.492759, 0.198350, 0.744809,
                            0.363336, 0.168710, 0.174016, 0.540752, 0.651044])

# Find min/max indices
min_idx = np.argmin(support_outputs)
max_idx = np.argmax(support_outputs)

print("=" * 60)
print("SUPPORT DATA MIN/MAX ANALYSIS")
print("=" * 60)
print(f"\nMIN: {support_conditions[min_idx]} = {support_outputs[min_idx]:.6f} ns (index {min_idx})")
print(f"MAX: {support_conditions[max_idx]} = {support_outputs[max_idx]:.6f} ns (index {max_idx})")

# Calculate normalized values (as done in code)
y_mean = support_outputs.mean()
y_std = support_outputs.std()
y_norm = (support_outputs - y_mean) / y_std

print(f"\ny_mean = {y_mean:.6f}")
print(f"y_std = {y_std:.6f}")
print(f"\nNormalized values:")
for i, (cond, val, norm) in enumerate(zip(support_conditions, support_outputs, y_norm)):
    marker = ""
    if i == min_idx:
        marker = " ← MIN"
    elif i == max_idx:
        marker = " ← MAX"
    print(f"  {cond}: {val:.4f} ns → normalized: {norm:.4f}{marker}")

print(f"\ny_norm.min() = {y_norm.min():.4f} (condition: {support_conditions[np.argmin(y_norm)]})")
print(f"y_norm.max() = {y_norm.max():.4f} (condition: {support_conditions[np.argmax(y_norm)]})")
print(f"\ngrad numerator = y_max - y_min = {y_norm.max() - y_norm.min():.4f}")
