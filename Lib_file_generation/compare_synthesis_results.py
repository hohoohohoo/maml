#!/usr/bin/env python3
"""
Compare synthesis results (WNS, TNS) between Original (SPICE) and Predicted libraries.

Usage:
    python compare_synthesis_results.py [--corner CORNER] [--output OUTPUT]

Example:
    python compare_synthesis_results.py --corner TT
    python compare_synthesis_results.py --corner FF --output results_FF_all_temps.csv

The script processes all 5 temperatures (0, 25, 50, 75, 100) by default.
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def extract_qor_metrics(filepath):
    """
    Extract QoR metrics from synthesis report.

    Returns:
        dict with WNS, TNS, num_violations, critical_path_length, area, cell_count
    """
    metrics = {
        'wns': None,
        'tns': None,
        'num_violations': None,
        'critical_path_length': None,
        'levels_of_logic': None,
        'cell_count': None,
        'area': None
    }

    try:
        with open(filepath, 'r') as f:
            content = f.read()

        # Critical Path Slack (WNS)
        match = re.search(r'Critical Path Slack:\s+([-\d.]+)', content)
        if match:
            metrics['wns'] = float(match.group(1))

        # Total Negative Slack (TNS)
        match = re.search(r'Total Negative Slack:\s+([-\d.]+)', content)
        if match:
            metrics['tns'] = float(match.group(1))

        # Number of Violating Paths
        match = re.search(r'No\. of Violating Paths:\s+([\d.]+)', content)
        if match:
            metrics['num_violations'] = int(float(match.group(1)))

        # Critical Path Length
        match = re.search(r'Critical Path Length:\s+([\d.]+)', content)
        if match:
            metrics['critical_path_length'] = float(match.group(1))

        # Levels of Logic
        match = re.search(r'Levels of Logic:\s+([\d.]+)', content)
        if match:
            metrics['levels_of_logic'] = int(float(match.group(1)))

        # Leaf Cell Count
        match = re.search(r'Leaf Cell Count:\s+([\d.]+)', content)
        if match:
            metrics['cell_count'] = int(float(match.group(1)))

        # Design Area
        match = re.search(r'Design Area:\s+([\d.]+)', content)
        if match:
            metrics['area'] = float(match.group(1))

    except FileNotFoundError:
        pass  # Silently skip missing files
    except Exception as e:
        print(f"Warning: Error reading {filepath}: {e}")

    return metrics


def extract_timing_metrics(filepath):
    """
    Extract timing metrics from timing report.

    Returns:
        dict with critical_path_delay (data arrival time)
    """
    metrics = {
        'critical_path_delay': None,
        'slack': None
    }

    try:
        with open(filepath, 'r') as f:
            content = f.read()

        # Data arrival time (Critical Path Delay)
        match = re.search(r'data arrival time\s+([\d.]+)', content)
        if match:
            metrics['critical_path_delay'] = float(match.group(1))

        # Slack from timing report
        slack_match = re.search(r'slack\s+\((?:VIOLATED|MET)\)\s+([-\d.]+)', content)
        if slack_match:
            metrics['slack'] = float(slack_match.group(1))

    except Exception as e:
        pass  # Silently ignore timing report errors

    return metrics


def compare_results(orig_dir, pred_dir, designs, voltages, temperature):
    """
    Compare synthesis results between original and predicted for a specific temperature.

    Returns:
        DataFrame with comparison results
    """
    results = []

    for design in designs:
        for v in voltages:
            voltage_str = f"V{v:03d}"
            voltage_val = v / 100.0  # Convert to actual voltage (e.g., 60 -> 0.60V)

            orig_qor = os.path.join(orig_dir, f"{design}_{voltage_str}", f"{design}_qor.rpt")
            pred_qor = os.path.join(pred_dir, f"{design}_{voltage_str}", f"{design}_qor.rpt")
            orig_timing = os.path.join(orig_dir, f"{design}_{voltage_str}", f"{design}_timing.rpt")
            pred_timing = os.path.join(pred_dir, f"{design}_{voltage_str}", f"{design}_timing.rpt")

            orig_metrics = extract_qor_metrics(orig_qor)
            pred_metrics = extract_qor_metrics(pred_qor)
            orig_timing_metrics = extract_timing_metrics(orig_timing)
            pred_timing_metrics = extract_timing_metrics(pred_timing)

            if orig_metrics['wns'] is not None and pred_metrics['wns'] is not None:
                result = {
                    'design': design,
                    'temperature': temperature,
                    'voltage': voltage_val,
                    'voltage_mv': v * 10,  # mV
                    # Original metrics
                    'orig_wns': orig_metrics['wns'],
                    'orig_tns': orig_metrics['tns'],
                    'orig_violations': orig_metrics['num_violations'],
                    'orig_path_length': orig_metrics['critical_path_length'],
                    'orig_cpd': orig_timing_metrics['critical_path_delay'],  # Critical Path Delay from timing.rpt
                    'orig_area': orig_metrics['area'],
                    # Predicted metrics
                    'pred_wns': pred_metrics['wns'],
                    'pred_tns': pred_metrics['tns'],
                    'pred_violations': pred_metrics['num_violations'],
                    'pred_path_length': pred_metrics['critical_path_length'],
                    'pred_cpd': pred_timing_metrics['critical_path_delay'],  # Critical Path Delay from timing.rpt
                    'pred_area': pred_metrics['area'],
                    # Differences
                    'wns_diff': pred_metrics['wns'] - orig_metrics['wns'],
                    'tns_diff': pred_metrics['tns'] - orig_metrics['tns'] if orig_metrics['tns'] and pred_metrics['tns'] else None,
                    'path_length_diff': pred_metrics['critical_path_length'] - orig_metrics['critical_path_length'] if orig_metrics['critical_path_length'] and pred_metrics['critical_path_length'] else None,
                    'cpd_diff': pred_timing_metrics['critical_path_delay'] - orig_timing_metrics['critical_path_delay'] if orig_timing_metrics['critical_path_delay'] and pred_timing_metrics['critical_path_delay'] else None,
                }

                # Calculate relative errors
                if orig_metrics['wns'] != 0:
                    result['wns_rel_err'] = abs(result['wns_diff'] / abs(orig_metrics['wns'])) * 100
                else:
                    result['wns_rel_err'] = 0 if result['wns_diff'] == 0 else None

                if orig_timing_metrics['critical_path_delay'] and orig_timing_metrics['critical_path_delay'] != 0:
                    result['cpd_rel_err'] = abs(result['cpd_diff'] / orig_timing_metrics['critical_path_delay']) * 100
                else:
                    result['cpd_rel_err'] = None

                results.append(result)

    return pd.DataFrame(results)


def print_summary(df):
    """Print summary statistics."""
    print("\n" + "=" * 120)
    print("SYNTHESIS RESULTS COMPARISON: Original (SPICE) vs Predicted")
    print("=" * 120)

    temperatures = sorted(df['temperature'].unique())
    designs = df['design'].unique()

    # Per-temperature summary
    print("\n" + "=" * 120)
    print("PER-TEMPERATURE SUMMARY")
    print("=" * 120)

    for temp in temperatures:
        temp_df = df[df['temperature'] == temp]
        # WNS/TNS: only include violated cases (orig < 0)
        wns_violated_df = temp_df[temp_df['orig_wns'] < 0]
        tns_violated_df = temp_df[temp_df['orig_tns'] < 0]
        wns_diffs = wns_violated_df['wns_diff'].dropna()
        tns_diffs = tns_violated_df['tns_diff'].dropna()
        # CPD: all cases
        cpd_diffs = temp_df['cpd_diff'].dropna()

        print(f"\n--- Temperature: {temp}C ---")
        print(f"  Data points: {len(temp_df)} (WNS violations: {len(wns_violated_df)}, TNS violations: {len(tns_violated_df)})")
        if len(wns_diffs) > 0:
            print(f"  WNS Diff (violated only) - MAE: {wns_diffs.abs().mean():.4f} ns, Std: {wns_diffs.std():.4f} ns, |Max|: {wns_diffs.abs().max():.4f} ns")
        else:
            print(f"  WNS Diff (violated only) - No violations")
        if len(cpd_diffs) > 0:
            print(f"  CPD Diff (all cases)     - MAE: {cpd_diffs.abs().mean():.4f} ns, Std: {cpd_diffs.std():.4f} ns, |Max|: {cpd_diffs.abs().max():.4f} ns")
        if len(tns_diffs) > 0:
            print(f"  TNS Diff (violated only) - MAE: {tns_diffs.abs().mean():.2f} ns, Std: {tns_diffs.std():.2f} ns, |Max|: {tns_diffs.abs().max():.2f} ns")

    # Per-design summary (across all temperatures)
    print("\n" + "=" * 120)
    print("PER-DESIGN SUMMARY (ALL TEMPERATURES)")
    print("=" * 120)

    for design in designs:
        design_df = df[df['design'] == design]
        # WNS/TNS: only include violated cases (orig < 0)
        wns_violated_df = design_df[design_df['orig_wns'] < 0]
        tns_violated_df = design_df[design_df['orig_tns'] < 0]
        wns_diffs = wns_violated_df['wns_diff'].dropna()
        tns_diffs = tns_violated_df['tns_diff'].dropna()
        # CPD: all cases
        cpd_diffs = design_df['cpd_diff'].dropna()
        orig_cpd = design_df['orig_cpd'].dropna()
        pred_cpd = design_df['pred_cpd'].dropna()

        print(f"\n--- Design: {design} ---")
        print(f"  Data points: {len(design_df)} (WNS violations: {len(wns_violated_df)}, TNS violations: {len(tns_violated_df)})")
        if len(orig_cpd) > 0 and len(pred_cpd) > 0:
            print(f"  CPD Average - SPICE: {orig_cpd.mean():.4f} ns, Prediction: {pred_cpd.mean():.4f} ns")
        if len(wns_diffs) > 0:
            print(f"  WNS Diff (violated only) - MAE: {wns_diffs.abs().mean():.4f} ns, Std: {wns_diffs.std():.4f} ns, |Max|: {wns_diffs.abs().max():.4f} ns")
        else:
            print(f"  WNS Diff (violated only) - No violations")
        if len(cpd_diffs) > 0:
            print(f"  CPD Diff (all cases)     - MAE: {cpd_diffs.abs().mean():.4f} ns, Std: {cpd_diffs.std():.4f} ns, |Max|: {cpd_diffs.abs().max():.4f} ns")
        if len(tns_diffs) > 0:
            print(f"  TNS Diff (violated only) - MAE: {tns_diffs.abs().mean():.2f} ns, Std: {tns_diffs.std():.2f} ns, |Max|: {tns_diffs.abs().max():.2f} ns")

    # Overall summary (all temperatures and designs)
    print("\n" + "=" * 120)
    print("OVERALL SUMMARY (ALL TEMPERATURES & DESIGNS)")
    print("=" * 120)

    # WNS/TNS: only include violated cases (orig < 0)
    wns_violated_df = df[df['orig_wns'] < 0]
    tns_violated_df = df[df['orig_tns'] < 0]
    all_wns_diff = wns_violated_df['wns_diff'].dropna()
    all_tns_diff = tns_violated_df['tns_diff'].dropna()
    # CPD: all cases
    all_cpd_diff = df['cpd_diff'].dropna()

    print(f"\nTotal data points: {len(df)}")
    print(f"WNS violations: {len(wns_violated_df)}, TNS violations: {len(tns_violated_df)}")
    print(f"Temperatures: {temperatures}")
    print(f"Designs: {list(designs)}")

    if len(all_wns_diff) > 0:
        print(f"\nWNS Difference (violated cases only, N={len(all_wns_diff)}):")
        print(f"  MAE:  {all_wns_diff.abs().mean():.4f} ns")
        print(f"  Std:  {all_wns_diff.std():.4f} ns")
        print(f"  Min:  {all_wns_diff.min():.4f} ns")
        print(f"  Max:  {all_wns_diff.max():.4f} ns")
        print(f"  |Max|: {all_wns_diff.abs().max():.4f} ns")
    else:
        print(f"\nWNS Difference: No violations found")

    if len(all_cpd_diff) > 0:
        print(f"\nCritical Path Delay Difference (all cases, N={len(all_cpd_diff)}):")
        print(f"  MAE:  {all_cpd_diff.abs().mean():.4f} ns")
        print(f"  Std:  {all_cpd_diff.std():.4f} ns")
        print(f"  Min:  {all_cpd_diff.min():.4f} ns")
        print(f"  Max:  {all_cpd_diff.max():.4f} ns")
        print(f"  |Max|: {all_cpd_diff.abs().max():.4f} ns")

    if len(all_tns_diff) > 0:
        print(f"\nTNS Difference (violated cases only, N={len(all_tns_diff)}):")
        print(f"  MAE:  {all_tns_diff.abs().mean():.2f} ns")
        print(f"  Std:  {all_tns_diff.std():.2f} ns")
        print(f"  Min:  {all_tns_diff.min():.2f} ns")
        print(f"  Max:  {all_tns_diff.max():.2f} ns")
        print(f"  |Max|: {all_tns_diff.abs().max():.2f} ns")
    else:
        print(f"\nTNS Difference: No violations found")


def main():
    parser = argparse.ArgumentParser(description='Compare WNS/TNS between Original and Predicted synthesis results')
    parser.add_argument('--corner', type=str, default='TT', help='Process corner (TT, FF, SS, etc.)')
    parser.add_argument('--orig-suffix', type=str, default='_spice', help='Suffix for original results directory')
    parser.add_argument('--output', type=str, default=None, help='Output CSV file path')
    parser.add_argument('--designs', type=str, nargs='+',
                        #default=['aes_ip', 's5378', 's38584', 'picorv32', 'vga_enh_top', 'darkriscv'],
                        default=['aes_ip', 's5378', 's38584', 'picorv32', 'vga_enh_top','darkriscv'],
                        help='Designs to compare')
    parser.add_argument('--voltages', type=str, default='60-120',
                        help='Voltage range (e.g., "60-120" for V060 to V120)')
    parser.add_argument('--plot', action='store_true',
                        help='Generate comparison plots')

    args = parser.parse_args()

    # All temperatures to process
    TEMPERATURES = [0, 25, 50, 75, 100]

    # Parse voltage range
    if '-' in args.voltages:
        v_start, v_end = map(int, args.voltages.split('-'))
        voltages = list(range(v_start, v_end + 1))
    else:
        voltages = [int(v) for v in args.voltages.split(',')]

    # Set up directories
    base_dir = Path(__file__).parent

    # Collect results from all temperatures
    all_dfs = []
    for temp in TEMPERATURES:
        orig_dir = base_dir / f"Projects_TSMC_SYN/syn/voltage_sweep_results_{args.corner}_{temp}{args.orig_suffix}"
        pred_dir = base_dir / f"Projects_TSMC_SYN_pred/syn/voltage_sweep_results_{args.corner}_{temp}"

        if not orig_dir.exists():
            print(f"Warning: Original directory not found: {orig_dir}")
            continue

        if not pred_dir.exists():
            print(f"Warning: Predicted directory not found: {pred_dir}")
            continue

        print(f"Processing temperature {temp}C...")
        print(f"  Original directory: {orig_dir}")
        print(f"  Predicted directory: {pred_dir}")

        temp_df = compare_results(str(orig_dir), str(pred_dir), args.designs, voltages, temp)
        if len(temp_df) > 0:
            all_dfs.append(temp_df)
            print(f"  Found {len(temp_df)} data points")

    if len(all_dfs) == 0:
        print("No results found to compare.")
        return

    # Combine all temperature results
    df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal data points across all temperatures: {len(df)}")

    # Print summary
    print_summary(df)

    # Save to CSV if requested
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to: {args.output}")
    else:
        # Default output filename
        output_file = base_dir / f"synthesis_comparison_{args.corner}_all_temps.csv"
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")

    # Generate plots if requested
    if args.plot:
        plot_comparison(df, base_dir)


def plot_comparison(df, output_dir=None):
    """Generate comparison plots with temperature-based coloring."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    designs = df['design'].unique()
    temperatures = sorted(df['temperature'].unique())

    # Color map for temperatures
    temp_colors = {0: 'blue', 25: 'green', 50: 'orange', 75: 'red', 100: 'purple'}

    # Create figure with subplots - WNS difference per temperature
    fig, axes = plt.subplots(len(designs), 2, figsize=(14, 4 * len(designs)))
    if len(designs) == 1:
        axes = axes.reshape(1, -1)

    for idx, design in enumerate(designs):
        design_df = df[df['design'] == design]

        # WNS difference by temperature (scatter plot)
        ax1 = axes[idx, 0]
        for temp in temperatures:
            temp_df = design_df[design_df['temperature'] == temp].sort_values('voltage')
            color = temp_colors.get(temp, 'gray')
            ax1.scatter(temp_df['voltage'], temp_df['wns_diff'], c=color, label=f'{temp}C', alpha=0.7, s=20)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax1.set_xlabel('Voltage (V)')
        ax1.set_ylabel('WNS Difference (ns)')
        ax1.set_title(f'{design} - WNS Difference by Temperature')
        ax1.legend(title='Temp', loc='best', fontsize=8)
        ax1.grid(True, alpha=0.3)

        # WNS difference histogram (aggregated)
        ax2 = axes[idx, 1]
        wns_diffs = design_df['wns_diff'].dropna()
        ax2.hist(wns_diffs, bins=30, color='purple', alpha=0.7, edgecolor='black')
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax2.axvline(x=wns_diffs.mean(), color='red', linestyle='--', linewidth=1, label=f'Mean: {wns_diffs.mean():.4f}')
        ax2.set_xlabel('WNS Difference (ns)')
        ax2.set_ylabel('Count')
        ax2.set_title(f'{design} - WNS Difference Distribution (All Temps)')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_dir:
        plot_path = Path(output_dir) / 'wns_comparison_all_temps.png'
    else:
        plot_path = Path(__file__).parent / 'wns_comparison_all_temps.png'

    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")
    plt.close()

    # Summary plot: box plot by temperature
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

    # WNS difference box plot by temperature
    wns_by_temp = [df[df['temperature'] == t]['wns_diff'].dropna().values for t in temperatures]
    bp1 = axes2[0].boxplot(wns_by_temp, labels=[f'{t}C' for t in temperatures], patch_artist=True)
    for patch, temp in zip(bp1['boxes'], temperatures):
        patch.set_facecolor(temp_colors.get(temp, 'gray'))
        patch.set_alpha(0.7)
    axes2[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes2[0].set_xlabel('Temperature')
    axes2[0].set_ylabel('WNS Difference (ns)')
    axes2[0].set_title('WNS Difference Distribution by Temperature (All Designs)')
    axes2[0].grid(True, alpha=0.3)

    # WNS difference box plot by design
    wns_by_design = [df[df['design'] == d]['wns_diff'].dropna().values for d in designs]
    bp2 = axes2[1].boxplot(wns_by_design, labels=designs, patch_artist=True)
    for patch in bp2['boxes']:
        patch.set_facecolor('steelblue')
        patch.set_alpha(0.7)
    axes2[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes2[1].set_xlabel('Design')
    axes2[1].set_ylabel('WNS Difference (ns)')
    axes2[1].set_title('WNS Difference Distribution by Design (All Temps)')
    axes2[1].tick_params(axis='x', rotation=45)
    axes2[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_dir:
        plot_path = Path(output_dir) / 'wns_summary_all_temps.png'
    else:
        plot_path = Path(__file__).parent / 'wns_summary_all_temps.png'

    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    plt.close()


if __name__ == "__main__":
    main()
