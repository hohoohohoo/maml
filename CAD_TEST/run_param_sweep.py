#!/usr/bin/env python3
"""
Parameter sweep script for validate_lut_table_sweep_full_cells.py
Runs multiple parameter combinations in parallel using multiprocessing.

Usage:
    python run_param_sweep.py --max-parallel 4
    python run_param_sweep.py --dry-run
    python run_param_sweep.py --max-parallel 2 --gpu 0,1
"""

import os
import sys
import argparse
import subprocess
import itertools
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import json

# ============================================================
# PARAMETER GRID - MODIFY THIS SECTION
# ============================================================

PARAM_GRID = {
    'center_steps': [50, 100, 150],
    'center_lr': [1e-3, 3e-3, 5e-3],
    'mape_threshold': [0.05, 0.1, 0.5, 1.0],
    'peripheral_adam_steps': [20, 40, 60],
    'peripheral_adam_lr': [1e-4, 3e-4, 1e-3],
}

# Fixed parameters (common to all runs)
# Note: ref_mode is added dynamically from command line argument
FIXED_PARAMS = {
    'unit_convert': True,
    'num_tables': 100,
}

# ============================================================

SCRIPT_DIR = Path(__file__).parent
PYTHON_SCRIPT = SCRIPT_DIR / 'validate_lut_table_sweep_full_cells.py'
LOG_DIR = SCRIPT_DIR / 'sweep_logs'


def generate_combinations(param_grid):
    """Generate all parameter combinations from grid."""
    keys = list(param_grid.keys())
    values = list(param_grid.values())

    combinations = []
    for combo in itertools.product(*values):
        combinations.append(dict(zip(keys, combo)))

    return combinations


def build_command(params, fixed_params, gpu_id):
    """Build command line for a single run."""
    cmd = ['python', str(PYTHON_SCRIPT)]

    # Add sweep parameters
    for key, value in params.items():
        cmd.extend([f'--{key}', str(value)])

    # Add fixed parameters
    for key, value in fixed_params.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f'--{key}')
        else:
            cmd.extend([f'--{key}', str(value)])

    # Add GPU
    cmd.extend(['--gpu', str(gpu_id)])

    return cmd


def run_single_experiment(args):
    """Run a single experiment (called by process pool)."""
    idx, total, params, fixed_params, gpu_id, log_file, dry_run = args

    cmd = build_command(params, fixed_params, gpu_id)

    if dry_run:
        return {
            'idx': idx,
            'params': params,
            'cmd': ' '.join(cmd),
            'status': 'dry_run',
            'log_file': str(log_file)
        }

    # Run the command
    try:
        with open(log_file, 'w') as f:
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Parameters: {json.dumps(params, indent=2)}\n")
            f.write("=" * 80 + "\n\n")
            f.flush()

            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=SCRIPT_DIR
            )

        return {
            'idx': idx,
            'params': params,
            'status': 'success' if result.returncode == 0 else 'failed',
            'returncode': result.returncode,
            'log_file': str(log_file)
        }

    except Exception as e:
        return {
            'idx': idx,
            'params': params,
            'status': 'error',
            'error': str(e),
            'log_file': str(log_file)
        }


def parse_log_for_results(log_file):
    """Parse log file to extract MAPE results."""
    results = {}
    try:
        with open(log_file, 'r') as f:
            content = f.read()

        # Look for typical result patterns (adjust based on actual output format)
        import re

        # Try to find MAPE values
        patterns = {
            'all_mape': r'All MAPE[:\s]+([0-9.]+)',
            'center_mape': r'Center MAPE[:\s]+([0-9.]+)',
            'peripheral_mape': r'Peripheral MAPE[:\s]+([0-9.]+)',
        }

        for key, pattern in patterns.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                results[key] = float(matches[-1])  # Take last match

    except Exception:
        pass

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Parameter sweep for validate_lut_table_sweep_full_cells.py',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_param_sweep.py --max-parallel 4
  python run_param_sweep.py --dry-run
  python run_param_sweep.py --gpu 0,1,2,3 --max-parallel 4
  python run_param_sweep.py --subset 10  # Run only first 10 combinations
        """
    )

    parser.add_argument('--max-parallel', '-p', type=int, default=4,
                        help='Maximum parallel jobs (default: 4)')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU IDs (comma-separated, e.g., "0,1,2,3")')
    parser.add_argument('--ref-mode', '-r', type=str, default='corner',
                        choices=['corner', 'middle', 'both'],
                        help='Reference voltage mode: corner, middle, both (default: corner)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print commands without executing')
    parser.add_argument('--subset', type=int, default=None,
                        help='Run only first N combinations')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Custom output directory for logs')

    args = parser.parse_args()

    # Parse GPU IDs
    gpu_ids = [int(x.strip()) for x in args.gpu.split(',')]

    # Setup log directory
    log_dir = Path(args.output_dir) if args.output_dir else LOG_DIR
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = log_dir / f'sweep_{timestamp}'
    log_dir.mkdir(parents=True, exist_ok=True)

    # Generate combinations
    combinations = generate_combinations(PARAM_GRID)

    if args.subset:
        combinations = combinations[:args.subset]

    total = len(combinations)

    # Create fixed params with ref_mode from command line
    fixed_params = FIXED_PARAMS.copy()
    fixed_params['ref_mode'] = args.ref_mode

    print("=" * 70)
    print(f"  Parameter Sweep: {total} combinations")
    print("=" * 70)
    print(f"Max parallel jobs: {args.max_parallel}")
    print(f"GPUs: {gpu_ids}")
    print(f"Ref mode: {args.ref_mode}")
    print(f"Log directory: {log_dir}")
    print()

    # Save configuration
    config = {
        'param_grid': PARAM_GRID,
        'fixed_params': fixed_params,
        'total_combinations': total,
        'gpu_ids': gpu_ids,
        'max_parallel': args.max_parallel,
        'ref_mode': args.ref_mode,
        'timestamp': timestamp
    }

    with open(log_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Prepare tasks
    tasks = []
    for idx, params in enumerate(combinations):
        gpu_id = gpu_ids[idx % len(gpu_ids)]

        # Create log filename from parameters
        param_str = '_'.join(f"{k[:3]}{v}" for k, v in params.items())
        log_file = log_dir / f'{idx:04d}_{param_str}.log'

        tasks.append((idx, total, params, fixed_params, gpu_id, log_file, args.dry_run))

    if args.dry_run:
        print("DRY RUN - Commands to execute:")
        print()
        for task in tasks:
            idx, _, params, fixed_params, gpu_id, log_file, _ = task
            cmd = build_command(params, fixed_params, gpu_id)
            print(f"[{idx+1}/{total}] GPU {gpu_id}")
            print(f"  {' '.join(cmd)}")
            print(f"  → {log_file}")
            print()
        return

    # Run experiments in parallel
    print(f"Starting {total} experiments...")
    print()

    results = []
    completed = 0

    with ProcessPoolExecutor(max_workers=args.max_parallel) as executor:
        futures = {executor.submit(run_single_experiment, task): task for task in tasks}

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1

            status_symbol = "✓" if result['status'] == 'success' else "✗"
            print(f"[{completed}/{total}] {status_symbol} Experiment {result['idx']+1}: {result['status']}")

    # Summary
    print()
    print("=" * 70)
    print("  Summary")
    print("=" * 70)

    success = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] != 'success')

    print(f"Completed: {success}/{total}")
    if failed:
        print(f"Failed: {failed}")

    # Parse results and create summary CSV
    print()
    print("Parsing results...")

    summary_file = log_dir / 'summary.csv'
    with open(summary_file, 'w') as f:
        # Header
        param_keys = list(PARAM_GRID.keys())
        f.write(','.join(param_keys + ['all_mape', 'center_mape', 'peripheral_mape', 'status']) + '\n')

        for result in sorted(results, key=lambda x: x['idx']):
            params = result['params']
            mape_results = parse_log_for_results(result['log_file']) if result['status'] == 'success' else {}

            row = [str(params[k]) for k in param_keys]
            row.extend([
                str(mape_results.get('all_mape', 'N/A')),
                str(mape_results.get('center_mape', 'N/A')),
                str(mape_results.get('peripheral_mape', 'N/A')),
                result['status']
            ])
            f.write(','.join(row) + '\n')

    print(f"Summary saved to: {summary_file}")

    # Save detailed results
    results_file = log_dir / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Detailed results saved to: {results_file}")
    print()
    print("Done!")


if __name__ == '__main__':
    main()
