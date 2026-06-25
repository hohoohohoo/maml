#!/usr/bin/env python3
"""
Plot single task results - Support and Query outputs
"""

import matplotlib.pyplot as plt
import numpy as np

# First task data from debug output
# Task: AN4YM16 / cell_fall / pin A / slew_idx=4, load_idx=6

# Support data (10 PVT conditions)
support_conditions = ['ff0p88vm40c', 'ss0p72vm40c', 'tt0p8v25c', 'ff0p99v125c', 'ss0p81v125c',
                      'tt0p9v25c', 'ff1p1v125c', 'ff1p1vm40c', 'ss0p9v125c', 'ss0p9vm40c']
support_outputs = [0.264363, 2.527260, 0.492759, 0.198350, 0.744809,
                   0.363336, 0.168710, 0.174016, 0.540752, 0.651044]

# Query data (5 test conditions)
test_conditions = ['ff0p88v125c', 'ss0p72v125c', 'ff0p99vm40c', 'ss0p81vm40c', 'tt1p0v25c']
actual_outputs = [0.240873, 1.168940, 0.210771, 1.083850, 0.289540]
predicted_outputs = [0.038699, 2.627887, 0.093601, 0.944991, 0.152382]

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Support data
ax1 = axes[0]
x_support = np.arange(len(support_conditions))
bars = ax1.bar(x_support, support_outputs, color='steelblue', alpha=0.8)
ax1.set_xticks(x_support)
ax1.set_xticklabels(support_conditions, rotation=45, ha='right', fontsize=8)
ax1.set_ylabel('Delay (ns)')
ax1.set_title('Support Data (10 PVT conditions)\nTask: AN4YM16 / cell_fall / pin A / slew_idx=4, load_idx=6')
ax1.set_ylim(0, max(support_outputs) * 1.15)

# Add value labels
for bar, val in zip(bars, support_outputs):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f'{val:.3f}', ha='center', va='bottom', fontsize=7)

# Highlight TT 0.9V 25C (nominal point, index 5)
bars[5].set_color('orange')
ax1.axhline(y=support_outputs[5], color='orange', linestyle='--', alpha=0.5, label=f'Nominal (tt0p9v25c): {support_outputs[5]:.3f} ns')
ax1.legend(fontsize=8)

# Plot 2: Query data - Actual vs Predicted
ax2 = axes[1]
x_query = np.arange(len(test_conditions))
width = 0.35

bars_actual = ax2.bar(x_query - width/2, actual_outputs, width, label='Actual', color='green', alpha=0.8)
bars_pred = ax2.bar(x_query + width/2, predicted_outputs, width, label='Predicted', color='red', alpha=0.8)

ax2.set_xticks(x_query)
ax2.set_xticklabels(test_conditions, rotation=45, ha='right', fontsize=9)
ax2.set_ylabel('Delay (ns)')
ax2.set_title('Query Data (5 Test conditions)\nActual vs Predicted')
ax2.legend()
ax2.set_ylim(0, max(max(actual_outputs), max(predicted_outputs)) * 1.15)

# Add value labels
for bar, val in zip(bars_actual, actual_outputs):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
             f'{val:.3f}', ha='center', va='bottom', fontsize=7, color='green')
for bar, val in zip(bars_pred, predicted_outputs):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
             f'{val:.3f}', ha='center', va='bottom', fontsize=7, color='red')

plt.tight_layout()
plt.savefig('/home/tkdgn2907/Deepsets_test/MAML/Projects/CAD_TEST/single_task_plot.png', dpi=150, bbox_inches='tight')
plt.savefig('/home/tkdgn2907/Deepsets_test/MAML/Projects/CAD_TEST/single_task_plot.pdf', bbox_inches='tight')
print("Saved: single_task_plot.png, single_task_plot.pdf")

# Also create a combined PVT curve plot
fig2, ax3 = plt.subplots(figsize=(12, 6))

# Combine all conditions for a full PVT curve view
all_conditions = support_conditions + test_conditions
all_support_outputs = support_outputs + [None]*5  # Support has values
all_test_actual = [None]*10 + actual_outputs  # Test actual
all_test_pred = [None]*10 + predicted_outputs  # Test predicted

x_all = np.arange(len(all_conditions))

# Plot support as circles
support_x = np.arange(len(support_conditions))
ax3.scatter(support_x, support_outputs, s=100, c='steelblue', marker='o', label='Support (known)', zorder=3)

# Plot test actual as squares
test_x = np.arange(len(support_conditions), len(all_conditions))
ax3.scatter(test_x, actual_outputs, s=100, c='green', marker='s', label='Test Actual', zorder=3)

# Plot test predicted as triangles
ax3.scatter(test_x, predicted_outputs, s=100, c='red', marker='^', label='Test Predicted', zorder=3)

# Draw error lines
for i, (actual, pred) in enumerate(zip(actual_outputs, predicted_outputs)):
    ax3.plot([test_x[i], test_x[i]], [actual, pred], 'r--', alpha=0.5, linewidth=1)

ax3.set_xticks(x_all)
ax3.set_xticklabels(all_conditions, rotation=45, ha='right', fontsize=8)
ax3.set_ylabel('Delay (ns)')
ax3.set_xlabel('PVT Condition')
ax3.set_title('PVT Curve View: Task AN4YM16 / cell_fall / pin A\n10 Support → 5 Query Prediction')
ax3.legend(loc='upper left')
ax3.axvline(x=9.5, color='gray', linestyle=':', alpha=0.5)
ax3.text(4.5, max(support_outputs)*0.95, 'Support', ha='center', fontsize=10, color='steelblue')
ax3.text(12, max(support_outputs)*0.95, 'Test', ha='center', fontsize=10, color='gray')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/tkdgn2907/Deepsets_test/MAML/Projects/CAD_TEST/single_task_pvt_curve.png', dpi=150, bbox_inches='tight')
plt.savefig('/home/tkdgn2907/Deepsets_test/MAML/Projects/CAD_TEST/single_task_pvt_curve.pdf', bbox_inches='tight')
print("Saved: single_task_pvt_curve.png, single_task_pvt_curve.pdf")
