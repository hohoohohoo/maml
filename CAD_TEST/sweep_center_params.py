#!/usr/bin/env python3
"""
Sweep center_steps and center_lr for LUT table validation.
Produces a summary table of results.

Usage:
    python sweep_center_params.py
    python sweep_center_params.py --num_tables 100 --gpu 1
"""

import subprocess
import re
import argparse
import numpy as np
from collections import defaultdict

# Parse arguments
parser = argparse.ArgumentParser(description='Sweep center_steps and center_lr')
parser.add_argument('--num_tables', type=int, default=50,
                    help='Number of tables per run (default: 50)')
parser.add_argument('--gpu', type=str, default='0',
                    help='GPU device ID (default: 0)')
parser.add_argument('--script', type=str, default='validate_lut_table_sweep.py',
                    choices=['validate_lut_table_sweep.py', 'validate_lut_table_sweep_full_cells.py'],
                    help='Which validation script to use')
parser.add_argument('--unit_convert', action='store_true', default=True,
                    help='Apply unit conversion')
args = parser.parse_args()

# Parameter grids to sweep
CENTER_STEPS_GRID = [20, 40, 60, 80, 100]
CENTER_LR_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]

# Store results
results = defaultdict(dict)


def run_validation(center_steps, center_lr):
    """Run validation with given parameters and extract key metrics"""
    cmd = [
        'python', args.script,
        '--num_tables', str(args.num_tables),
        '--gpu', args.gpu,
        '--center_steps', str(center_steps),
        '--center_lr', str(center_lr),
    ]
    if args.unit_convert:
        cmd.append('--unit_convert')

    print(f"  Running: steps={center_steps}, lr={center_lr}...")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        output = result.stdout + result.stderr

        # Extract metrics using regex
        metrics = {}

        # Look for OVERALL line in MAPE results section
        # Format: OVERALL           X.XX            Y.YY             Z.ZZ
        overall_match = re.search(
            r'OVERALL\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)',
            output
        )
        if overall_match:
            metrics['center_mape'] = float(overall_match.group(1))
            metrics['peripheral_mape'] = float(overall_match.group(2))
            metrics['all_mape'] = float(overall_match.group(3))

        # Look for summary section
        center_mean_match = re.search(r'Center Point.*?Mean MAPE:\s+(\d+\.?\d*)%', output, re.DOTALL)
        peripheral_mean_match = re.search(r'Peripheral Points.*?Mean MAPE:\s+(\d+\.?\d*)%', output, re.DOTALL)
        all_mean_match = re.search(r'All Points.*?Mean MAPE:\s+(\d+\.?\d*)%', output, re.DOTALL)

        if center_mean_match:
            metrics['center_mape'] = float(center_mean_match.group(1))
        if peripheral_mean_match:
            metrics['peripheral_mape'] = float(peripheral_mean_match.group(1))
        if all_mean_match:
            metrics['all_mape'] = float(all_mean_match.group(1))

        return metrics

    except subprocess.TimeoutExpired:
        print(f"    Timeout!")
        return None
    except Exception as e:
        print(f"    Error: {e}")
        return None


# Run sweep
print("=" * 80)
print("CENTER STEPS / LR SWEEP")
print("=" * 80)
print(f"Script: {args.script}")
print(f"Tables per run: {args.num_tables}")
print(f"GPU: {args.gpu}")
print(f"Steps grid: {CENTER_STEPS_GRID}")
print(f"LR grid: {CENTER_LR_GRID}")
print(f"Total runs: {len(CENTER_STEPS_GRID) * len(CENTER_LR_GRID)}")
print("=" * 80)

for steps in CENTER_STEPS_GRID:
    for lr in CENTER_LR_GRID:
        metrics = run_validation(steps, lr)
        if metrics:
            results[(steps, lr)] = metrics
            print(f"    → Center: {metrics.get('center_mape', 'N/A'):.2f}%, "
                  f"Peripheral: {metrics.get('peripheral_mape', 'N/A'):.2f}%, "
                  f"All: {metrics.get('all_mape', 'N/A'):.2f}%")

# Print results table
print("\n" + "=" * 100)
print("RESULTS SUMMARY - ALL MAPE (%)")
print("=" * 100)

# Header
header = f"{'Steps\\LR':<10}"
for lr in CENTER_LR_GRID:
    header += f"{lr:<12}"
print(header)
print("-" * (10 + 12 * len(CENTER_LR_GRID)))

# Rows
for steps in CENTER_STEPS_GRID:
    row = f"{steps:<10}"
    for lr in CENTER_LR_GRID:
        if (steps, lr) in results:
            val = results[(steps, lr)].get('all_mape', float('nan'))
            row += f"{val:<12.2f}"
        else:
            row += f"{'N/A':<12}"
    print(row)

# Find best parameters
print("\n" + "=" * 100)
print("BEST PARAMETERS")
print("=" * 100)

if results:
    # Best by all_mape
    best_all = min(results.items(), key=lambda x: x[1].get('all_mape', float('inf')))
    print(f"Best All MAPE:        steps={best_all[0][0]}, lr={best_all[0][1]} → {best_all[1]['all_mape']:.2f}%")

    # Best by center_mape
    best_center = min(results.items(), key=lambda x: x[1].get('center_mape', float('inf')))
    print(f"Best Center MAPE:     steps={best_center[0][0]}, lr={best_center[0][1]} → {best_center[1]['center_mape']:.2f}%")

    # Best by peripheral_mape
    best_peripheral = min(results.items(), key=lambda x: x[1].get('peripheral_mape', float('inf')))
    print(f"Best Peripheral MAPE: steps={best_peripheral[0][0]}, lr={best_peripheral[0][1]} → {best_peripheral[1]['peripheral_mape']:.2f}%")

# Detailed table for Center MAPE
print("\n" + "=" * 100)
print("CENTER MAPE (%)")
print("=" * 100)

header = f"{'Steps\\LR':<10}"
for lr in CENTER_LR_GRID:
    header += f"{lr:<12}"
print(header)
print("-" * (10 + 12 * len(CENTER_LR_GRID)))

for steps in CENTER_STEPS_GRID:
    row = f"{steps:<10}"
    for lr in CENTER_LR_GRID:
        if (steps, lr) in results:
            val = results[(steps, lr)].get('center_mape', float('nan'))
            row += f"{val:<12.2f}"
        else:
            row += f"{'N/A':<12}"
    print(row)

# Detailed table for Peripheral MAPE
print("\n" + "=" * 100)
print("PERIPHERAL MAPE (%)")
print("=" * 100)

header = f"{'Steps\\LR':<10}"
for lr in CENTER_LR_GRID:
    header += f"{lr:<12}"
print(header)
print("-" * (10 + 12 * len(CENTER_LR_GRID)))

for steps in CENTER_STEPS_GRID:
    row = f"{steps:<10}"
    for lr in CENTER_LR_GRID:
        if (steps, lr) in results:
            val = results[(steps, lr)].get('peripheral_mape', float('nan'))
            row += f"{val:<12.2f}"
        else:
            row += f"{'N/A':<12}"
    print(row)

print("\n" + "=" * 100)
print("SWEEP COMPLETE")
print("=" * 100)
