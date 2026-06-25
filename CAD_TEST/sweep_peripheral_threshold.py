#!/usr/bin/env python3
"""
Sweep peripheral_loss_threshold for LUT table validation.
Produces a summary table of results to find optimal threshold.

Usage:
    python sweep_peripheral_threshold.py
    python sweep_peripheral_threshold.py --num_tables 100 --gpu 1
    python sweep_peripheral_threshold.py --script validate_lut_table_cross_condition.py
"""

import subprocess
import re
import argparse
import numpy as np
from collections import defaultdict

# Parse arguments
parser = argparse.ArgumentParser(description='Sweep peripheral_loss_threshold')
parser.add_argument('--num_tables', type=int, default=50,
                    help='Number of tables per run (default: 50)')
parser.add_argument('--gpu', type=str, default='0',
                    help='GPU device ID (default: 0)')
parser.add_argument('--script', type=str, default='validate_lut_table_sweep_full_cells.py',
                    choices=['validate_lut_table_sweep_full_cells.py', 'validate_lut_table_cross_condition.py'],
                    help='Which validation script to use')
parser.add_argument('--unit_convert', action='store_true', default=True,
                    help='Apply unit conversion')
args = parser.parse_args()

# Parameter grid to sweep - peripheral_loss_threshold
# Based on typical loss ranges (grad/move normalized space)
THRESHOLD_GRID = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.2, 0.5]

# Store results
results = {}


def run_validation(threshold):
    """Run validation with given threshold and extract key metrics"""
    cmd = [
        'python', args.script,
        '--num_tables', str(args.num_tables),
        '--gpu', args.gpu,
        '--peripheral_loss_threshold', str(threshold),
    ]
    if args.unit_convert:
        cmd.append('--unit_convert')

    print(f"  Running: threshold={threshold}...")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
        output = result.stdout + result.stderr

        metrics = {}

        # Extract TRANSITION results (MAPE)
        # Look for: OVERALL         X.XX            Y.YY               Z.ZZ
        # In the MAPE section
        mape_section = re.search(r'RESULTS - TRANSITION - MAPE.*?OVERALL\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)',
                                  output, re.DOTALL)
        if mape_section:
            metrics['center_mape'] = float(mape_section.group(1))
            metrics['peripheral_mape'] = float(mape_section.group(2))
            metrics['all_mape'] = float(mape_section.group(3))

        # Extract peripheral Adam usage rate
        adam_match = re.search(r'PERIPHERAL ADAM USAGE.*?OVERALL\s+\d+\s+\d+\s+(\d+\.?\d*)', output, re.DOTALL)
        if adam_match:
            metrics['adam_rate'] = float(adam_match.group(1))

        # Extract peripheral support loss statistics
        loss_stats = re.search(r'PERIPHERAL SUPPORT LOSS.*?OVERALL\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)',
                               output, re.DOTALL)
        if loss_stats:
            metrics['mean_loss'] = float(loss_stats.group(1))
            metrics['median_loss'] = float(loss_stats.group(2))
            metrics['min_loss'] = float(loss_stats.group(3))
            metrics['max_loss'] = float(loss_stats.group(4))
            metrics['p90_loss'] = float(loss_stats.group(5))

        return metrics

    except subprocess.TimeoutExpired:
        print(f"    Timeout!")
        return None
    except Exception as e:
        print(f"    Error: {e}")
        return None


# Run sweep
print("=" * 100)
print("PERIPHERAL THRESHOLD SWEEP")
print("=" * 100)
print(f"Script: {args.script}")
print(f"Tables per run: {args.num_tables}")
print(f"GPU: {args.gpu}")
print(f"Threshold grid: {THRESHOLD_GRID}")
print(f"Total runs: {len(THRESHOLD_GRID)}")
print("=" * 100)

for threshold in THRESHOLD_GRID:
    metrics = run_validation(threshold)
    if metrics:
        results[threshold] = metrics
        print(f"    → All MAPE: {metrics.get('all_mape', 'N/A'):.2f}%, "
              f"Adam Rate: {metrics.get('adam_rate', 'N/A'):.1f}%")

# Print results table
print("\n" + "=" * 120)
print("RESULTS SUMMARY")
print("=" * 120)

# Header
print(f"\n{'Threshold':<15} {'Center MAPE%':<15} {'Periph MAPE%':<15} {'All MAPE%':<12} {'Adam Rate%':<12} {'Mean Loss':<12} {'Median Loss':<12}")
print("-" * 105)

# Rows
for threshold in THRESHOLD_GRID:
    if threshold in results:
        m = results[threshold]
        print(f"{threshold:<15.6f} {m.get('center_mape', float('nan')):<15.2f} "
              f"{m.get('peripheral_mape', float('nan')):<15.2f} {m.get('all_mape', float('nan')):<12.2f} "
              f"{m.get('adam_rate', float('nan')):<12.1f} {m.get('mean_loss', float('nan')):<12.6f} "
              f"{m.get('median_loss', float('nan')):<12.6f}")
    else:
        print(f"{threshold:<15.6f} {'N/A':<15} {'N/A':<15} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")

# Find best parameters
print("\n" + "=" * 120)
print("BEST PARAMETERS")
print("=" * 120)

if results:
    # Best by all_mape
    best_all = min(results.items(), key=lambda x: x[1].get('all_mape', float('inf')))
    print(f"Best All MAPE:        threshold={best_all[0]:.6f} → {best_all[1]['all_mape']:.2f}%")

    # Best by peripheral_mape
    best_peripheral = min(results.items(), key=lambda x: x[1].get('peripheral_mape', float('inf')))
    print(f"Best Peripheral MAPE: threshold={best_peripheral[0]:.6f} → {best_peripheral[1]['peripheral_mape']:.2f}%")

    # Best balance (low MAPE + low Adam usage)
    # Weighted score: mape + 0.01 * adam_rate (prefer lower Adam usage with similar MAPE)
    def score(item):
        m = item[1]
        return m.get('all_mape', float('inf')) + 0.01 * m.get('adam_rate', 0)

    best_balanced = min(results.items(), key=score)
    print(f"Best Balanced:        threshold={best_balanced[0]:.6f} → MAPE={best_balanced[1]['all_mape']:.2f}%, Adam={best_balanced[1]['adam_rate']:.1f}%")

print("\n" + "=" * 120)
print("SWEEP COMPLETE")
print("=" * 120)
