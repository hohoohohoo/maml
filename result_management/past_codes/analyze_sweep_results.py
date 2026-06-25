#!/usr/bin/env python3
"""
Analyze sweep results - automatically group and compare results by sweep parameter
- Automatically detects which parameter was swept
- Groups results with same config/cell/data_type/mode but different sweep parameter values
- Generates comparison plots and analysis
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import glob
import re
from pathlib import Path
from collections import defaultdict

# Add utils directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pretraining', 'model_test_code', 'utils'))
from test_dataset_config import TEST_CONFIGS

# Set matplotlib style
plt.style.use('ggplot')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3


def parse_filename(filename):
    """
    Parse filename to extract metadata

    Returns:
        dict with metadata or None if not parseable
    """
    basename = os.path.basename(filename)

    # Try MAML pattern first
    maml_pattern = r'(\w+)_([\w_]+)_(\w+)_(cell|transition)_(extrapolation|interpolation)_MAML_innerdiv(\d+)_meta(\d+)_layer(\d+)_(\d+)_(pred|act)\.npy'
    match = re.match(maml_pattern, basename)

    if match:
        return {
            'prefix': match.group(1),
            'topology': match.group(2),
            'cell': match.group(3),
            'data_type': match.group(4),
            'mode': match.group(5),
            'model_type': 'MAML',
            'innerdiv': int(match.group(6)),
            'meta': int(match.group(7)),
            'layer_length': int(match.group(8)),
            'iterations': int(match.group(9)),
            'file_type': match.group(10),
            'filename': basename
        }

    # Try MLP pattern
    mlp_pattern = r'(\w+)_([\w_]+)_(\w+)_(cell|transition)_(extrapolation|interpolation)_(aadam|mlp)_(\d+)_(pred|act)\.npy'
    match = re.match(mlp_pattern, basename)

    if match:
        return {
            'prefix': match.group(1),
            'topology': match.group(2),
            'cell': match.group(3),
            'data_type': match.group(4),
            'mode': match.group(5),
            'model_type': match.group(6).upper(),
            'innerdiv': None,
            'meta': None,
            'layer_length': None,
            'iterations': int(match.group(7)),
            'file_type': match.group(8),
            'filename': basename
        }

    return None


def calculate_metrics(predictions, actuals):
    """Calculate NRMSE, SMAPE, MAE metrics"""
    predictions = np.array(predictions)
    actuals = np.array(actuals)

    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mean_actual = np.mean(np.abs(actuals))
    nrmse = (rmse / (mean_actual + 1e-8)) * 100
    # SMAPE: Symmetric Mean Absolute Percentage Error
    smape = np.mean(2 * np.abs(predictions - actuals) / (np.abs(predictions) + np.abs(actuals) + 1e-8)) * 100
    mae = np.mean(np.abs(predictions - actuals))

    return {
        'NRMSE': nrmse,
        'SMAPE': smape,
        'MAE': mae,
        'RMSE': rmse,
        'MSE': mse
    }


def load_all_results(data_dir):
    """Load all results and return DataFrame"""
    pred_files = glob.glob(os.path.join(data_dir, '*_pred.npy'))
    results = []

    for pred_file in pred_files:
        act_file = pred_file.replace('_pred.npy', '_act.npy')
        if not os.path.exists(act_file):
            continue

        metadata = parse_filename(pred_file)
        if metadata is None:
            continue

        try:
            predictions = np.load(pred_file)
            actuals = np.load(act_file)

            if len(predictions) != len(actuals):
                continue

            metrics = calculate_metrics(predictions, actuals)
            result = {**metadata, **metrics}
            results.append(result)
        except Exception as e:
            print(f"❌ Error loading {pred_file}: {e}")
            continue

    if not results:
        return None

    return pd.DataFrame(results)


def detect_sweep_parameter(df):
    """
    Automatically detect which parameter was swept

    Returns:
        str: 'innerdiv', 'meta', 'num_iterations', or 'layer_length'
    """
    maml_data = df[df['model_type'] == 'MAML']

    if len(maml_data) == 0:
        return None

    # Count unique values for each parameter
    param_counts = {
        'innerdiv': maml_data['innerdiv'].nunique(),
        'meta': maml_data['meta'].nunique(),
        'num_iterations': maml_data['iterations'].nunique(),
        'layer_length': maml_data['layer_length'].nunique()
    }

    # Find parameter with most variation
    sweep_param = max(param_counts, key=param_counts.get)

    if param_counts[sweep_param] <= 1:
        return None

    return sweep_param


def group_by_combination(df, sweep_param):
    """
    Group results by combination (same everything except sweep parameter)
    Groups MAML and MLP/AADAM together for comparison

    AADAM only has: cell, data_type, mode, iterations
    MAML has: cell, data_type, mode, innerdiv, meta, layer_length, iterations

    Returns:
        dict: {group_key: DataFrame}
    """
    # Group keys: everything except sweep parameter and model_type
    # (We want MAML and AADAM in the same group)
    # Only group by cell, data_type, mode for base matching
    base_group_cols = ['cell', 'data_type', 'mode']

    # Add non-sweep MAML parameters to grouping (only for MAML rows)
    # AADAM will match on base_group_cols only
    maml_param_cols = []
    if sweep_param == 'innerdiv':
        maml_param_cols = ['meta', 'iterations', 'layer_length']
    elif sweep_param == 'meta':
        maml_param_cols = ['innerdiv', 'iterations', 'layer_length']
    elif sweep_param == 'num_iterations':
        maml_param_cols = ['innerdiv', 'meta', 'layer_length']
    elif sweep_param == 'layer_length':
        maml_param_cols = ['innerdiv', 'meta', 'iterations']

    groups = {}

    # First, group MAML data by all parameters
    maml_df = df[df['model_type'] == 'MAML'].copy()
    aadam_df = df[df['model_type'].isin(['AADAM', 'MLP'])].copy()

    group_cols = base_group_cols + maml_param_cols

    for name, maml_group in maml_df.groupby(group_cols, dropna=False):
        # Skip groups with only 1 MAML result
        if len(maml_group) < 2:
            continue

        # Create readable key
        key_parts = []
        for col, val in zip(group_cols, name):
            if pd.notna(val):
                # Convert to int if it's a numeric parameter
                if col in ['innerdiv', 'meta', 'iterations', 'layer_length'] and isinstance(val, (int, float)):
                    val = int(val)
                key_parts.append(f"{col}={val}")
        key = "_".join(key_parts)

        # Find matching AADAM data (only match on base_group_cols)
        aadam_match = aadam_df.copy()
        for i, col in enumerate(base_group_cols):
            val = name[i]
            if pd.notna(val):
                aadam_match = aadam_match[aadam_match[col] == val]

        # Combine MAML and AADAM data
        combined = pd.concat([maml_group, aadam_match], ignore_index=True)
        combined = combined.sort_values(sweep_param if sweep_param != 'num_iterations' else 'iterations')

        groups[key] = combined

    return groups


def plot_comparison(group_df, sweep_param, group_name, output_dir, png_dir):
    """Create comparison plot for a group"""
    metrics = ['NRMSE', 'SMAPE', 'MAE']

    # Get actual column name
    sweep_col = sweep_param if sweep_param != 'num_iterations' else 'iterations'

    maml_data = group_df[group_df['model_type'] == 'MAML'].sort_values(sweep_col)
    mlp_data = group_df[group_df['model_type'].isin(['AADAM', 'MLP'])]

    if len(maml_data) == 0:
        return

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Extract cell and data info for title
    cell = maml_data.iloc[0]['cell']
    data_type = maml_data.iloc[0]['data_type']
    mode = maml_data.iloc[0]['mode']

    fig.suptitle(f'{cell} - {data_type} - {mode}\nVarying {sweep_param}',
                 fontsize=16, fontweight='bold')

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        # MAML data - convert to int if possible
        x_vals = maml_data[sweep_col].values
        if sweep_col in ['innerdiv', 'meta', 'iterations', 'layer_length']:
            x_vals = x_vals.astype(int)
        y_vals = maml_data[metric].values

        # Plot MAML bars
        bars = ax.bar(range(len(x_vals)), y_vals, alpha=0.7, label='MAML', edgecolor='black')
        colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        # Value labels
        for i, (x, y) in enumerate(zip(range(len(x_vals)), y_vals)):
            ax.text(i, y, f'{y:.3f}', ha='center', va='bottom', fontsize=14, fontweight='bold')

        ax.set_xticks(range(len(x_vals)))
        ax.set_xticklabels(x_vals, rotation=45 if len(x_vals) > 3 else 0)

        # AADAM/MLP baseline
        if len(mlp_data) > 0:
            mlp_val = mlp_data[metric].values[0]
            mlp_type = mlp_data.iloc[0]['model_type']
            ax.axhline(y=mlp_val, color='red', linestyle='--', linewidth=2,
                      label=f'{mlp_type} (baseline)', alpha=0.7)
            ax.text(0.02, mlp_val, f'{mlp_type}: {mlp_val:.3f}', transform=ax.get_yaxis_transform(),
                   fontsize=14, color='red', va='bottom')

        ax.set_xlabel(sweep_param, fontsize=12, fontweight='bold')
        ax.set_ylabel(metric, fontsize=12, fontweight='bold')
        ax.set_title(f'{metric} Comparison', fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

    plt.tight_layout()

    # Save to PNG directory
    safe_name = group_name.replace('/', '_').replace('=', '_')
    # Remove "cell_" prefix if present
    if safe_name.startswith('cell_'):
        safe_name = safe_name[5:]
    plot_path = os.path.join(png_dir, f'{safe_name}_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved: {plot_path}")
    plt.close()


def plot_trend(group_df, sweep_param, group_name, output_dir):
    """Create trend analysis plot"""
    metrics = ['NRMSE', 'SMAPE', 'MAE']
    sweep_col = sweep_param if sweep_param != 'num_iterations' else 'iterations'

    maml_data = group_df[group_df['model_type'] == 'MAML'].sort_values(sweep_col)

    if len(maml_data) < 2:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    cell = maml_data.iloc[0]['cell']
    data_type = maml_data.iloc[0]['data_type']
    mode = maml_data.iloc[0]['mode']

    fig.suptitle(f'{cell} - {data_type} - {mode}\nTrend Analysis: {sweep_param}',
                 fontsize=16, fontweight='bold')

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        x_vals = maml_data[sweep_col].values
        y_vals = maml_data[metric].values

        # Line plot
        ax.plot(x_vals, y_vals, marker='o', linewidth=2, markersize=10,
               label='MAML', color='steelblue')

        # Value labels
        for x, y in zip(x_vals, y_vals):
            ax.text(x, y, f'{y:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Linear regression
        if len(x_vals) >= 2:
            z = np.polyfit(x_vals, y_vals, 1)
            p = np.poly1d(z)
            ax.plot(x_vals, p(x_vals), "--", alpha=0.5, color='orange',
                   label=f'Trend (slope={z[0]:.4f})')

            # Improvement
            if y_vals[0] != 0:
                improvement = ((y_vals[0] - y_vals[-1]) / y_vals[0]) * 100
                ax.text(0.5, 0.95, f'Improvement: {improvement:.1f}%',
                       transform=ax.transAxes, ha='center', va='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                       fontsize=10, fontweight='bold')

        ax.set_xlabel(sweep_param, fontsize=12, fontweight='bold')
        ax.set_ylabel(metric, fontsize=12, fontweight='bold')
        ax.set_title(f'{metric} Trend', fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

    plt.tight_layout()

    safe_name = group_name.replace('/', '_').replace('=', '_')
    plot_path = os.path.join(output_dir, f'{safe_name}_trend.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved: {plot_path}")
    plt.close()


def export_to_csv(df, sweep_param, output_file):
    """Export results to CSV"""
    cols_order = ['cell', 'data_type', 'mode', 'model_type',
                  'innerdiv', 'meta', 'layer_length', 'iterations',
                  'NRMSE', 'SMAPE', 'MAE', 'RMSE', 'MSE']

    cols_order = [c for c in cols_order if c in df.columns]
    df_export = df[cols_order].copy()

    metric_cols = ['NRMSE', 'SMAPE', 'MAE', 'RMSE', 'MSE']
    for col in metric_cols:
        if col in df_export.columns:
            df_export[col] = df_export[col].round(4)

    df_export.to_csv(output_file, index=False)
    print(f"💾 Exported: {output_file}")


def create_summary_report(groups, sweep_param, output_file):
    """Create text summary report"""
    sweep_col = sweep_param if sweep_param != 'num_iterations' else 'iterations'

    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SWEEP RESULTS ANALYSIS SUMMARY\n")
        f.write("="*80 + "\n\n")

        f.write(f"Sweep Parameter: {sweep_param}\n")
        f.write(f"Total Combinations: {len(groups)}\n\n")

        for group_name, group_df in groups.items():
            f.write(f"\n{'='*80}\n")
            f.write(f"COMBINATION: {group_name}\n")
            f.write(f"{'='*80}\n\n")

            maml_data = group_df[group_df['model_type'] == 'MAML'].sort_values(sweep_col)
            mlp_data = group_df[group_df['model_type'].isin(['AADAM', 'MLP'])]

            if len(maml_data) > 0:
                f.write("MAML Results:\n")
                for _, row in maml_data.iterrows():
                    f.write(f"  {sweep_param}={row[sweep_col]}: ")
                    f.write(f"NRMSE={row['NRMSE']:.3f}%, SMAPE={row['SMAPE']:.3f}%, MAE={row['MAE']:.4f}\n")

                best_idx = maml_data['NRMSE'].idxmin()
                best = maml_data.loc[best_idx]
                f.write(f"\n  ✅ Best: {sweep_param}={best[sweep_col]} (NRMSE={best['NRMSE']:.3f}%)\n")

                if len(maml_data) >= 2:
                    first = maml_data.iloc[0]
                    last = maml_data.iloc[-1]
                    improvement = ((first['NRMSE'] - last['NRMSE']) / first['NRMSE']) * 100
                    f.write(f"  📈 Improvement from {first[sweep_col]} to {last[sweep_col]}: {improvement:.1f}%\n")

            if len(mlp_data) > 0:
                mlp = mlp_data.iloc[0]
                f.write(f"\nMLP Baseline:\n")
                f.write(f"  NRMSE={mlp['NRMSE']:.3f}%, SMAPE={mlp['SMAPE']:.3f}%, MAE={mlp['MAE']:.4f}\n")

                if len(maml_data) > 0:
                    best_maml = maml_data.loc[maml_data['NRMSE'].idxmin()]
                    vs_mlp = ((mlp['NRMSE'] - best_maml['NRMSE']) / mlp['NRMSE']) * 100
                    if vs_mlp > 0:
                        f.write(f"  🏆 Best MAML is {vs_mlp:.1f}% better than MLP\n")
                    else:
                        f.write(f"  ⚠️  MLP is {-vs_mlp:.1f}% better than best MAML\n")

            f.write("\n")

    print(f"💾 Created: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze sweep results - automatically detect and group by sweep parameter',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all results in directory (auto-detects sweep parameter)
  python analyze_sweep_results_v2.py

  # Filter by config (only analyze cells from specific config)
  python analyze_sweep_results_v2.py --config 0  # ASAP7 Intra Topology

  # Specify sweep parameter and config
  python analyze_sweep_results_v2.py --vary innerdiv --config 0

  # Available configs:
  #   0: ASAP7 Intra Topology (NAND3x2, OR2x6, NOR2xp67, AND2x6)
  #   1: ASAP7 Topology Agnostic (MAJIxp5, MAJx2, MAJx3, HAxp5, FAx1, XOR2xp5, etc.)
  #   2: TSMC Intra Topology (NR3D1BWP30P140, OR4D0BWP30P140, etc.)
  #   3: TSMC Topology Agnostic (HA1D0BWP30P140, FA1D0BWP30P140, etc.)
        """
    )

    parser.add_argument('--data_dir', type=str, default='../pretraining/model_test_code/data_result_npy_directory',
                       help='Directory containing .npy result files')
    parser.add_argument('--output_dir', type=str, default='./result_summary',
                       help='Directory for output plots and CSV')
    parser.add_argument('--vary', type=str, default=None,
                       choices=['innerdiv', 'meta', 'num_iterations', 'layer_length'],
                       help='Force specific sweep parameter (auto-detected if not specified)')
    parser.add_argument('--config', type=int, default=None, choices=[0, 1, 2, 3],
                       help='Filter by dataset config (0=ASAP7 Intra, 1=ASAP7 Topo Agnostic, 2=TSMC Intra, 3=TSMC Topo Agnostic)')
    parser.add_argument('--aggregate', action='store_true',
                       help='Aggregate all cells and show average results instead of per-cell plots')

    args = parser.parse_args()

    print("="*80)
    print("SWEEP RESULTS ANALYSIS")
    print("="*80)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print()

    # Load all results
    print("📂 Loading result files...")
    df = load_all_results(args.data_dir)

    if df is None or len(df) == 0:
        print("❌ No results found.")
        return 1

    print(f"✅ Loaded {len(df)} result files")
    print()

    # Filter by config if specified
    if args.config is not None:
        config = TEST_CONFIGS[args.config]
        config_cells = config['default_cells']
        df = df[df['cell'].isin(config_cells)]
        print(f"🔍 Filtered to config {args.config}: {config['name']}")
        print(f"   Cells: {', '.join(config_cells)}")
        print(f"   Results after filter: {len(df)}")
        print()

        if len(df) == 0:
            print("❌ No results found for this config.")
            return 1

    # Detect or use specified sweep parameter
    if args.vary:
        sweep_param = args.vary
        print(f"Using specified sweep parameter: {sweep_param}")
    else:
        sweep_param = detect_sweep_parameter(df)
        if sweep_param is None:
            print("❌ Could not detect sweep parameter.")
            return 1
        print(f"🔍 Auto-detected sweep parameter: {sweep_param}")

    print()

    # Group by combination
    print("📊 Grouping results by combination...")
    groups = group_by_combination(df, sweep_param)

    if len(groups) == 0:
        print("❌ No valid groups found.")
        return 1

    print(f"✅ Found {len(groups)} combinations to compare")
    print()

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)

    if args.aggregate:
        # Aggregate mode: separate directories
        png_dir = os.path.join(args.output_dir, 'png_aggregate')
        csv_dir = os.path.join(args.output_dir, 'csv_aggregate')
        os.makedirs(png_dir, exist_ok=True)
        os.makedirs(csv_dir, exist_ok=True)

        # Aggregate mode: Calculate average across all cells
        print("📊 Calculating aggregated results across all cells...")
        create_aggregated_plots(groups, sweep_param, png_dir, csv_dir, df, args.config)
    else:
        # Normal mode: separate directories
        png_dir = os.path.join(args.output_dir, 'png_comparison')
        csv_dir = os.path.join(args.output_dir, 'csv_comparison')
        os.makedirs(png_dir, exist_ok=True)
        os.makedirs(csv_dir, exist_ok=True)

        # Normal mode: Generate plots for each group
        print("📊 Generating comparison plots...")
        for group_name, group_df in groups.items():
            print(f"  Processing: {group_name}")
            plot_comparison(group_df, sweep_param, group_name, args.output_dir, png_dir)

        # Export CSV
        print("\n💾 Exporting to CSV...")
        csv_path = os.path.join(csv_dir, f'results_{sweep_param}_sweep.csv')
        export_to_csv(df, sweep_param, output_file=csv_path)

        # Create summary
        print("\n📝 Creating summary report...")
        report_path = os.path.join(csv_dir, f'summary_{sweep_param}_sweep.txt')
        create_summary_report(groups, sweep_param, output_file=report_path)

    print("\n" + "="*80)
    print("✅ Analysis complete!")
    print(f"📁 PNG files saved to: {png_dir}")
    print(f"📁 CSV/Report files saved to: {csv_dir}")
    print("="*80)

    return 0


def create_aggregated_plots(groups, sweep_param, png_dir, csv_dir, df, config_num=None):
    """
    모든 cell의 평균 결과를 계산하여 plot과 CSV 생성
    extrapolation과 interpolation을 구분하여 처리

    Parameters:
    -----------
    groups : dict
        각 조합별 결과 그룹
    sweep_param : str
        Sweep parameter 이름
    png_dir : str
        PNG 저장 디렉토리
    csv_dir : str
        CSV 저장 디렉토리
    df : DataFrame
        전체 데이터프레임
    config_num : int, optional
        Config number for filename (0, 1, 2, 3)
    """
    sweep_col = sweep_param if sweep_param != 'num_iterations' else 'iterations'
    metrics = ['NRMSE', 'SMAPE', 'MAE']

    # mode별로 처리 (extrapolation, interpolation)
    modes = df['mode'].unique()

    for mode in modes:
        print(f"\n  Processing mode: {mode}")

        # 해당 mode의 groups만 필터링
        mode_groups = {}
        for group_name, group_df in groups.items():
            if mode in group_name:  # group_name에 mode가 포함되어 있음
                mode_df = group_df[group_df['mode'] == mode]
                if len(mode_df) > 0:
                    mode_groups[group_name] = mode_df

        if len(mode_groups) == 0:
            print(f"    ⚠️  No data found for mode: {mode}")
            continue

        # 각 sweep 값별로 데이터 수집
        sweep_values_set = set()
        for group_name, group_df in mode_groups.items():
            maml_data = group_df[group_df['model_type'] == 'MAML']
            sweep_values_set.update(maml_data[sweep_col].unique())

        sweep_values = sorted(list(sweep_values_set))
        if len(sweep_values) == 0:
            print(f"    ❌ No sweep values found for mode: {mode}")
            continue

        # 각 sweep 값과 메트릭별로 평균 계산
        aggregated_results = {metric: {'MAML': [], 'AADAM': []} for metric in metrics}

        for sweep_val in sweep_values:
            for metric in metrics:
                maml_values = []
                aadam_values = []

                for group_name, group_df in mode_groups.items():
                    # MAML data
                    maml_data = group_df[group_df['model_type'] == 'MAML']
                    maml_row = maml_data[maml_data[sweep_col] == sweep_val]
                    if len(maml_row) > 0 and metric in maml_row.columns:
                        val = maml_row[metric].values[0]
                        if not np.isnan(val):
                            maml_values.append(val)

                    # AADAM data (baseline, 같은 값)
                    aadam_data = group_df[group_df['model_type'].isin(['AADAM', 'MLP'])]
                    if len(aadam_data) > 0 and metric in aadam_data.columns:
                        val = aadam_data[metric].values[0]
                        if not np.isnan(val):
                            aadam_values.append(val)

                # 평균 계산
                if maml_values:
                    aggregated_results[metric]['MAML'].append(np.mean(maml_values))
                else:
                    aggregated_results[metric]['MAML'].append(np.nan)

                if aadam_values:
                    aggregated_results[metric]['AADAM'].append(np.mean(aadam_values))
                else:
                    aggregated_results[metric]['AADAM'].append(np.nan)

        # Plot 생성
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Aggregated Results - {mode.upper()} (Average across all cells)\nVarying {sweep_param}',
                     fontsize=16, fontweight='bold')

        # Convert sweep values to int if possible
        x_vals = np.array(sweep_values)
        if sweep_col in ['innerdiv', 'meta', 'iterations', 'layer_length']:
            x_vals = x_vals.astype(int)

        for idx, metric in enumerate(metrics):
            ax = axes[idx]

            maml_means = aggregated_results[metric]['MAML']
            aadam_means = aggregated_results[metric]['AADAM']

            # MAML bars
            bars = ax.bar(range(len(x_vals)), maml_means, alpha=0.7, label='MAML (avg)', edgecolor='black')
            colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)

            # Value labels
            for i, y in enumerate(maml_means):
                if not np.isnan(y):
                    ax.text(i, y, f'{y:.3f}', ha='center', va='bottom', fontsize=14, fontweight='bold')

            ax.set_xticks(range(len(x_vals)))
            ax.set_xticklabels(x_vals, rotation=45 if len(x_vals) > 3 else 0)

            # AADAM baseline (평균)
            if not all(np.isnan(aadam_means)):
                aadam_avg = np.nanmean(aadam_means)
                ax.axhline(y=aadam_avg, color='red', linestyle='--', linewidth=2,
                          label='AADAM (avg baseline)', alpha=0.7)
                ax.text(0.02, aadam_avg, f'AADAM: {aadam_avg:.3f}', transform=ax.get_yaxis_transform(),
                       fontsize=14, color='red', va='bottom')

            ax.set_xlabel(sweep_param, fontsize=12, fontweight='bold')
            ax.set_ylabel(metric, fontsize=12, fontweight='bold')
            ax.set_title(f'{metric} Comparison (Aggregated)', fontsize=13)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best')

        plt.tight_layout()

        # Build descriptive filename
        # Extract common parameters from first group to describe what's constant
        first_group_df = next(iter(mode_groups.values()))
        data_type = first_group_df.iloc[0]['data_type'] if 'data_type' in first_group_df.columns else 'cell'

        # Get constant parameters (non-sweep MAML parameters)
        constant_params = []
        if sweep_param != 'innerdiv' and 'innerdiv' in first_group_df.columns:
            innerdiv_val = first_group_df[first_group_df['model_type'] == 'MAML']['innerdiv'].iloc[0]
            if pd.notna(innerdiv_val):
                constant_params.append(f"innerdiv{int(innerdiv_val)}")
        if sweep_param != 'meta' and 'meta' in first_group_df.columns:
            meta_val = first_group_df[first_group_df['model_type'] == 'MAML']['meta'].iloc[0]
            if pd.notna(meta_val):
                constant_params.append(f"meta{int(meta_val)}")
        if sweep_param != 'layer_length' and 'layer_length' in first_group_df.columns:
            layer_val = first_group_df[first_group_df['model_type'] == 'MAML']['layer_length'].iloc[0]
            if pd.notna(layer_val):
                constant_params.append(f"layer{int(layer_val)}")
        if sweep_param != 'num_iterations' and 'iterations' in first_group_df.columns:
            iter_val = first_group_df[first_group_df['model_type'] == 'MAML']['iterations'].iloc[0]
            if pd.notna(iter_val):
                constant_params.append(f"iter{int(iter_val)}")

        # Build filename parts
        filename_parts = []

        # Add config if provided
        if config_num is not None:
            filename_parts.append(f"config{config_num}")

        # Add data_type only if it's NOT 'cell' (to avoid "cell_" prefix)
        if data_type != 'cell':
            filename_parts.append(f"datatype_{data_type}")

        # Add mode
        filename_parts.append(f"mode_{mode}")

        # Add constant parameters
        if constant_params:
            filename_parts.extend(constant_params)

        # Add sweep parameter
        filename_parts.append(f"sweep_{sweep_param}")

        # Create final filename
        filename_base = "_".join(filename_parts)

        # Save plot
        plot_path = os.path.join(png_dir, f'{filename_base}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"    💾 Saved aggregated plot: {plot_path}")
        plt.close()

        # CSV 생성 (mode별로)
        csv_data = []
        for i, sweep_val in enumerate(sweep_values):
            row = {sweep_param: int(sweep_val) if sweep_col in ['innerdiv', 'meta', 'iterations', 'layer_length'] else sweep_val}
            for metric in metrics:
                row[f'{metric}_MAML_mean'] = aggregated_results[metric]['MAML'][i]
                row[f'{metric}_AADAM_mean'] = aggregated_results[metric]['AADAM'][i]

                # Diff% 계산
                maml_val = aggregated_results[metric]['MAML'][i]
                aadam_val = aggregated_results[metric]['AADAM'][i]
                if not np.isnan(maml_val) and not np.isnan(aadam_val) and aadam_val != 0:
                    diff_pct = ((aadam_val - maml_val) / aadam_val) * 100
                    row[f'{metric}_Diff%'] = diff_pct
                else:
                    row[f'{metric}_Diff%'] = np.nan

            csv_data.append(row)

        agg_df = pd.DataFrame(csv_data)
        csv_path = os.path.join(csv_dir, f'{filename_base}.csv')
        agg_df.to_csv(csv_path, index=False)
        print(f"    💾 Exported aggregated CSV: {csv_path}")

        # 요약 출력
        print(f"\n    {'='*80}")
        print(f"    AGGREGATED SUMMARY - {mode.upper()}")
        print(f"    {'='*80}")
        print("    " + agg_df.to_string(index=False).replace('\n', '\n    '))
        print(f"    {'='*80}")


if __name__ == "__main__":
    sys.exit(main())
