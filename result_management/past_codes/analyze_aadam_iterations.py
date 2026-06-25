#!/usr/bin/env python3
"""
Analyze AADAM/MLP iteration sweep results
- Groups AADAM results by cell, data_type, mode
- Compares different iteration values
- Generates bar plot comparison
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
    Parse AADAM/MLP filename to extract metadata

    Returns:
        dict with metadata or None if not parseable
    """
    basename = os.path.basename(filename)

    # AADAM/MLP pattern: prefix_topology_cell_datatype_mode_modeltype_iterations_pred/act.npy
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
            'iterations': int(match.group(7)),
            'file_type': match.group(8),
            'filename': basename
        }

    return None


def calculate_metrics(predictions, actuals, group_size=61):
    """
    Calculate NRMSE, SMAPE, MAPE, MAE metrics with 61-group averaging.

    Same methodology as compare_gcn_topology_validation_results.py

    Args:
        predictions: predicted values
        actuals: actual values
        group_size: number of samples per group (default: 61 for voltage variations)

    Returns:
        dict: averaged metrics across all groups
    """
    predictions = np.array(predictions).flatten()
    actuals = np.array(actuals).flatten()

    # Filter out invalid values
    valid_mask = ~(np.isnan(predictions) | np.isnan(actuals) | np.isinf(predictions) | np.isinf(actuals))
    predictions = predictions[valid_mask]
    actuals = actuals[valid_mask]

    if len(predictions) == 0:
        return None

    # Group by 61 samples
    n_groups = len(predictions) // group_size

    if n_groups == 0:
        n_groups = 1
        group_size = len(predictions)

    # Trim to exact group multiples
    predictions = predictions[:n_groups * group_size]
    actuals = actuals[:n_groups * group_size]

    # Reshape to (n_groups, group_size)
    pred_grouped = predictions.reshape(n_groups, group_size)
    act_grouped = actuals.reshape(n_groups, group_size)

    # Calculate metrics per group
    group_metrics = []

    for i in range(n_groups):
        y_p = pred_grouped[i]
        y_t = act_grouped[i]

        # MAE
        mae = np.mean(np.abs(y_p - y_t))

        # MAPE (with masking for zero values)
        mask = y_t != 0
        mape = np.mean(np.abs((y_t[mask] - y_p[mask]) / y_t[mask])) * 100 if np.any(mask) else 0

        # SMAPE (with masking)
        denom = np.abs(y_t) + np.abs(y_p)
        mask_smape = denom != 0
        smape = np.mean(
            2.0 * np.abs(y_t[mask_smape] - y_p[mask_smape]) / denom[mask_smape]
        ) * 100 if np.any(mask_smape) else 0

        # RMSE
        mse = np.mean((y_p - y_t) ** 2)
        rmse = np.sqrt(mse)

        # NRMSE (range normalization)
        y_range = np.max(y_t) - np.min(y_t)
        nrmse = (rmse / y_range * 100) if y_range > 0 else 0

        group_metrics.append({
            'mse': mse,
            'rmse': rmse,
            'nrmse': nrmse,
            'mae': mae,
            'mape': mape,
            'smape': smape
        })

    # Average across groups
    return {
        'NRMSE': np.mean([g['nrmse'] for g in group_metrics]),
        'SMAPE': np.mean([g['smape'] for g in group_metrics]),
        'MAPE': np.mean([g['mape'] for g in group_metrics]),
        'MAE': np.mean([g['mae'] for g in group_metrics]),
        'RMSE': np.mean([g['rmse'] for g in group_metrics]),
        'MSE': np.mean([g['mse'] for g in group_metrics]),
        'num_samples': len(predictions),
        'num_groups': n_groups
    }


def load_all_results(data_dir):
    """Load all AADAM/MLP results and return DataFrame"""
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
            if metrics is None:
                print(f"⚠️  Warning: No valid data in {pred_file}")
                continue
            result = {**metadata, **metrics}
            results.append(result)
        except Exception as e:
            print(f"❌ Error loading {pred_file}: {e}")
            continue

    if not results:
        return None

    return pd.DataFrame(results)


def group_by_combination(df):
    """
    Group AADAM results by cell, data_type, mode
    Each group will have multiple iteration values

    Returns:
        dict: {group_key: DataFrame}
    """
    group_cols = ['cell', 'data_type', 'mode', 'model_type']

    groups = {}
    for name, group in df.groupby(group_cols, dropna=False):
        # Skip groups with only 1 result
        if len(group) < 2:
            continue

        # Create readable key
        key_parts = []
        for col, val in zip(group_cols, name):
            if pd.notna(val):
                key_parts.append(f"{col}={val}")
        key = "_".join(key_parts)

        # Sort by iterations
        groups[key] = group.sort_values('iterations')

    return groups


def plot_iteration_comparison(group_df, group_name, png_dir, config_num=None):
    """Create iteration comparison plot for a group"""
    metrics = ['NRMSE', 'SMAPE', 'MAE']

    # MAE x1000 scaling only for config 2, 3 (TSMC)
    scale_mae = config_num in [2, 3]

    if len(group_df) == 0:
        return

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Extract info for title
    cell = group_df.iloc[0]['cell']
    data_type = group_df.iloc[0]['data_type']
    mode = group_df.iloc[0]['mode']
    model_type = group_df.iloc[0]['model_type']

    fig.suptitle(f'{cell} - {data_type} - {mode} ({model_type})\nIteration Sweep',
                 fontsize=16, fontweight='bold')

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        # Get iteration values and metrics
        x_vals = group_df['iterations'].values.astype(int)
        y_vals = group_df[metric].values

        # MAE x1000 scaling (only for config 2, 3)
        if metric == 'MAE' and scale_mae:
            y_vals = y_vals * 1000

        # Plot bars
        bars = ax.bar(range(len(x_vals)), y_vals, alpha=0.7, label=model_type, edgecolor='black')
        colors = plt.cm.plasma(np.linspace(0, 1, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        # Value labels
        for i, (x, y) in enumerate(zip(range(len(x_vals)), y_vals)):
            ax.text(i, y, f'{y:.3f}', ha='center', va='bottom', fontsize=14, fontweight='bold')

        ax.set_xticks(range(len(x_vals)))
        ax.set_xticklabels(x_vals, rotation=45 if len(x_vals) > 3 else 0)

        ax.set_xlabel('Iterations', fontsize=12, fontweight='bold')
        ylabel = 'MAE (x1000)' if (metric == 'MAE' and scale_mae) else metric
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(f'{ylabel} vs Iterations', fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

        # Add trend info
        if len(x_vals) >= 2:
            improvement = ((y_vals[0] - y_vals[-1]) / y_vals[0]) * 100
            trend_text = f'Change: {improvement:.1f}%'
            ax.text(0.5, 0.95, trend_text,
                   transform=ax.transAxes, ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=10, fontweight='bold')

    plt.tight_layout()

    # Save to PNG directory
    safe_name = group_name.replace('/', '_').replace('=', '_')
    plot_path = os.path.join(png_dir, f'{safe_name}_iterations.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved: {plot_path}")
    plt.close()


def export_to_csv(df, output_file):
    """Export results to CSV"""
    cols_order = ['cell', 'data_type', 'mode', 'model_type', 'iterations',
                  'NRMSE', 'MAPE', 'SMAPE', 'MAE', 'RMSE', 'MSE',
                  'num_samples', 'num_groups']

    cols_order = [c for c in cols_order if c in df.columns]
    df_export = df[cols_order].copy()

    metric_cols = ['NRMSE', 'MAPE', 'SMAPE', 'MAE', 'RMSE', 'MSE']
    for col in metric_cols:
        if col in df_export.columns:
            df_export[col] = df_export[col].round(4)

    df_export.to_csv(output_file, index=False)
    print(f"💾 Exported: {output_file}")


def create_summary_report(groups, output_file):
    """Create text summary report"""
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("AADAM/MLP ITERATION SWEEP ANALYSIS\n")
        f.write("="*80 + "\n\n")

        f.write(f"Total Combinations: {len(groups)}\n\n")

        for group_name, group_df in groups.items():
            f.write(f"\n{'='*80}\n")
            f.write(f"COMBINATION: {group_name}\n")
            f.write(f"{'='*80}\n\n")

            group_df = group_df.sort_values('iterations')

            f.write("Results:\n")
            for _, row in group_df.iterrows():
                f.write(f"  iterations={int(row['iterations'])}: ")
                f.write(f"NRMSE={row['NRMSE']:.3f}%, MAPE={row['MAPE']:.3f}%, SMAPE={row['SMAPE']:.3f}%, MAE={row['MAE']:.4f}\n")

            # Best result
            best_idx = group_df['NRMSE'].idxmin()
            best = group_df.loc[best_idx]
            f.write(f"\n  ✅ Best: iterations={int(best['iterations'])} (NRMSE={best['NRMSE']:.3f}%)\n")

            # Trend analysis
            if len(group_df) >= 2:
                first = group_df.iloc[0]
                last = group_df.iloc[-1]
                improvement = ((first['NRMSE'] - last['NRMSE']) / first['NRMSE']) * 100
                f.write(f"  📈 Change from {int(first['iterations'])} to {int(last['iterations'])}: {improvement:.1f}%\n")

            f.write("\n")

    print(f"💾 Created: {output_file}")


def create_aggregated_plots(groups, png_dir, csv_dir, df, config_num=None):
    """
    모든 cell의 평균 결과를 계산하여 plot과 CSV 생성
    extrapolation과 interpolation, cell과 transition을 구분하여 처리

    Parameters:
    -----------
    groups : dict
        각 조합별 결과 그룹
    png_dir : str
        PNG 저장 디렉토리
    csv_dir : str
        CSV 저장 디렉토리
    df : DataFrame
        전체 데이터프레임
    config_num : int, optional
        Config number for filename (0, 1, 2, 3)
    """
    metrics = ['NRMSE', 'SMAPE', 'MAE']  # For plotting (3 columns)
    csv_metrics = ['NRMSE', 'MAPE', 'SMAPE', 'MAE']  # For CSV output (includes MAPE)

    # mode와 data_type별로 처리 (extrapolation/interpolation x cell/transition)
    modes = df['mode'].unique()
    data_types = df['data_type'].unique()
    model_types = df['model_type'].unique()

    print(f"\n🔍 Found modes: {modes}")
    print(f"🔍 Found data_types: {data_types}")
    print(f"🔍 Found model_types: {model_types}")

    for mode in modes:
        for data_type in data_types:
            for model_type in model_types:
                print(f"\n  Processing mode: {mode}, data_type: {data_type}, model_type: {model_type}")

                # 해당 mode, data_type, model_type의 groups만 필터링
                mode_data_groups = {}
                for group_name, group_df in groups.items():
                    if mode in group_name and model_type in group_name:
                        filtered_df = group_df[(group_df['mode'] == mode) &
                                              (group_df['data_type'] == data_type) &
                                              (group_df['model_type'] == model_type)]
                        if len(filtered_df) > 0:
                            mode_data_groups[group_name] = filtered_df

                print(f"    Found {len(mode_data_groups)} groups for this combination")

                if len(mode_data_groups) == 0:
                    print(f"    ⚠️  No data found for mode: {mode}, data_type: {data_type}, model_type: {model_type}")
                    continue

                # 각 iteration 값별로 데이터 수집
                iteration_values_set = set()
                for group_name, group_df in mode_data_groups.items():
                    iteration_values_set.update(group_df['iterations'].unique())

                iteration_values = sorted(list(iteration_values_set))
                if len(iteration_values) == 0:
                    print(f"    ❌ No iteration values found")
                    continue

                # 각 iteration 값과 메트릭별로 평균 계산 (csv_metrics includes MAPE)
                aggregated_results = {metric: [] for metric in csv_metrics}

                for iter_val in iteration_values:
                    for metric in csv_metrics:
                        values = []

                        for group_name, group_df in mode_data_groups.items():
                            rows = group_df[group_df['iterations'] == iter_val]
                            if len(rows) > 0 and metric in rows.columns:
                                val = rows[metric].values[0]
                                if not np.isnan(val):
                                    values.append(val)

                        # 평균 계산
                        if values:
                            aggregated_results[metric].append(np.mean(values))
                        else:
                            aggregated_results[metric].append(np.nan)

                # Plot 생성
                fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                fig.suptitle(f'Aggregated Results - {data_type.upper()} - {mode.upper()} ({model_type})\n(Average across all cells) Iteration Sweep',
                             fontsize=16, fontweight='bold')

                # Convert iteration values to int
                x_vals = np.array(iteration_values).astype(int)

                for idx, metric in enumerate(metrics):
                    ax = axes[idx]

                    means = aggregated_results[metric]

                    # MAE x1000 scaling (only for config 2, 3 - TSMC)
                    scale_mae = config_num in [2, 3]
                    if metric == 'MAE' and scale_mae:
                        means_display = [m * 1000 if not np.isnan(m) else np.nan for m in means]
                    else:
                        means_display = means

                    # Bars
                    bars = ax.bar(range(len(x_vals)), means_display, alpha=0.7, label=f'{model_type} (avg)', edgecolor='black')
                    colors = plt.cm.plasma(np.linspace(0, 1, len(bars)))
                    for bar, color in zip(bars, colors):
                        bar.set_color(color)

                    # Value labels
                    for i, y in enumerate(means_display):
                        if not np.isnan(y):
                            ax.text(i, y, f'{y:.3f}', ha='center', va='bottom', fontsize=14, fontweight='bold')

                    ax.set_xticks(range(len(x_vals)))
                    ax.set_xticklabels(x_vals, rotation=45 if len(x_vals) > 3 else 0)

                    ax.set_xlabel('Iterations', fontsize=12, fontweight='bold')
                    ylabel = 'MAE (x1000)' if (metric == 'MAE' and scale_mae) else metric
                    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
                    ax.set_title(f'{ylabel} vs Iterations (Aggregated)', fontsize=13)
                    ax.grid(True, alpha=0.3)
                    ax.legend(loc='best')

                    # Add trend info
                    if len(x_vals) >= 2 and not any(np.isnan(means)):
                        improvement = ((means[0] - means[-1]) / means[0]) * 100
                        trend_text = f'Change: {improvement:.1f}%'
                        ax.text(0.5, 0.95, trend_text,
                               transform=ax.transAxes, ha='center', va='top',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                               fontsize=10, fontweight='bold')

                plt.tight_layout()

                # Build descriptive filename
                filename_parts = []

                # Add config if provided
                if config_num is not None:
                    filename_parts.append(f"config{config_num}")

                # Add data_type
                if data_type == 'cell':
                    filename_parts.append(f"{data_type}")
                else:
                    filename_parts.append(f"{data_type}")

                # Add mode
                filename_parts.append(f"mode_{mode}")

                # Add model_type
                filename_parts.append(f"{model_type.lower()}")

                # Add sweep indicator
                filename_parts.append("sweep_iterations")

                # Create final filename
                filename_base = "_".join(filename_parts)

                # Save plot
                plot_path = os.path.join(png_dir, f'{filename_base}.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"    💾 Saved aggregated plot: {plot_path}")
                plt.close()

                # CSV 생성 - includes MAPE
                csv_data = []
                for i, iter_val in enumerate(iteration_values):
                    row = {'iterations': int(iter_val)}
                    for metric in csv_metrics:
                        row[f'{metric}_mean'] = aggregated_results[metric][i]
                    csv_data.append(row)

                agg_df = pd.DataFrame(csv_data)
                csv_path = os.path.join(csv_dir, f'{filename_base}.csv')
                agg_df.to_csv(csv_path, index=False)
                print(f"    💾 Exported aggregated CSV: {csv_path}")

                # 요약 출력
                print(f"\n    {'='*80}")
                print(f"    AGGREGATED SUMMARY - {data_type.upper()} - {mode.upper()} ({model_type})")
                print(f"    {'='*80}")
                print("    " + agg_df.to_string(index=False).replace('\n', '\n    '))
                print(f"    {'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze AADAM/MLP iteration sweep results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all AADAM results in directory
  python analyze_aadam_iterations.py

  # Filter by config (only analyze cells from specific config)
  python analyze_aadam_iterations.py --config 0  # ASAP7 Intra Topology

  # Filter by model type
  python analyze_aadam_iterations.py --model_type aadam

  # Aggregate all cells and show average results
  python analyze_aadam_iterations.py --config 0 --aggregate

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
    parser.add_argument('--config', type=int, default=None, choices=[0, 1, 2, 3],
                       help='Filter by dataset config')
    parser.add_argument('--model_type', type=str, default=None, choices=['aadam', 'mlp'],
                       help='Filter by model type (aadam or mlp)')
    parser.add_argument('--aggregate', action='store_true',
                       help='Aggregate all cells and show average results instead of per-cell plots')

    args = parser.parse_args()

    print("="*80)
    print("AADAM/MLP ITERATION SWEEP ANALYSIS")
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

    # Filter by model type if specified
    if args.model_type:
        df = df[df['model_type'] == args.model_type.upper()]
        print(f"🔍 Filtered to model type: {args.model_type.upper()}")
        print(f"   Results after filter: {len(df)}")
        print()

        if len(df) == 0:
            print("❌ No results found for this model type.")
            return 1

    # Group by combination
    print("📊 Grouping results by combination...")
    groups = group_by_combination(df)

    if len(groups) == 0:
        print("❌ No valid groups found.")
        return 1

    print(f"✅ Found {len(groups)} combinations to compare")
    print()

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)

    if args.aggregate:
        # Aggregate mode: separate directories
        png_dir = os.path.join(args.output_dir, 'png_aadam_aggregate')
        csv_dir = os.path.join(args.output_dir, 'csv_aadam_aggregate')
        os.makedirs(png_dir, exist_ok=True)
        os.makedirs(csv_dir, exist_ok=True)

        # Aggregate mode: Calculate average across all cells
        print("📊 Calculating aggregated results across all cells...")
        create_aggregated_plots(groups, png_dir, csv_dir, df, args.config)
    else:
        # Normal mode: separate directories
        png_dir = os.path.join(args.output_dir, 'png_aadam_iterations')
        csv_dir = os.path.join(args.output_dir, 'csv_aadam_iterations')
        os.makedirs(png_dir, exist_ok=True)
        os.makedirs(csv_dir, exist_ok=True)

        # Normal mode: Generate plots for each group
        print("📊 Generating iteration comparison plots...")
        for group_name, group_df in groups.items():
            print(f"  Processing: {group_name}")
            plot_iteration_comparison(group_df, group_name, png_dir, config_num=args.config)

        # Export CSV
        print("\n💾 Exporting to CSV...")
        csv_path = os.path.join(csv_dir, 'aadam_iterations_sweep.csv')
        export_to_csv(df, output_file=csv_path)

        # Create summary
        print("\n📝 Creating summary report...")
        report_path = os.path.join(csv_dir, 'aadam_iterations_summary.txt')
        create_summary_report(groups, output_file=report_path)

    print("\n" + "="*80)
    print("✅ Analysis complete!")
    print(f"📁 PNG files saved to: {png_dir}")
    print(f"📁 CSV/Report files saved to: {csv_dir}")
    print("="*80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
