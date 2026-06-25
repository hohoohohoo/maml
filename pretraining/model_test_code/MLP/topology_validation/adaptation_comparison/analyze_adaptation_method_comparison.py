#!/usr/bin/env python3
"""
Optimization Comparison Analysis Script
Visualizes MAML optimization method comparison results from JSON files.

Generates:
1. Combined summary figure with all key metrics
2. Selective Adam trigger ratio per cell and overall average
"""

import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any


def load_json_results(json_path: str) -> Dict[str, Any]:
    """Load optimization comparison results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def extract_metrics(data: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Extract relevant metrics from loaded JSON data.

    Returns:
        Dict with structure: {cell_name: {method: {metric: value}}}
    """
    results = data['results']
    metrics = {}

    for cell_name, cell_data in results.items():
        metrics[cell_name] = {}
        for method, method_data in cell_data.items():
            metrics[cell_name][method] = {
                'total_rmse': method_data.get('avg_total_rmse', 0),
                'total_nrmse': method_data.get('avg_total_nrmse', 0),
                'adam_triggered_ratio': method_data.get('adam_triggered_ratio', None),
                'num_tasks': method_data.get('num_tasks', 0),
                'name': method_data.get('name', method),
                'time_ms': method_data.get('avg_time_ms', 0),
            }

    return metrics


def calculate_overall_metrics(metrics: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    """
    Calculate weighted average metrics across all cells.

    Weights by num_tasks for each cell.
    """
    methods = ['none', 'sgd', 'adam', 'selective_adam', 'full_adam']
    overall = {}

    for method in methods:
        total_rmse_sum = 0
        total_nrmse_sum = 0
        total_time_sum = 0
        total_tasks = 0
        adam_ratio_sum = 0
        adam_ratio_count = 0

        for cell_name, cell_data in metrics.items():
            if method in cell_data:
                num_tasks = cell_data[method]['num_tasks']
                total_rmse_sum += cell_data[method]['total_rmse'] * num_tasks
                total_nrmse_sum += cell_data[method]['total_nrmse'] * num_tasks
                total_time_sum += cell_data[method].get('time_ms', 0) * num_tasks
                total_tasks += num_tasks

                if cell_data[method]['adam_triggered_ratio'] is not None:
                    adam_ratio_sum += cell_data[method]['adam_triggered_ratio']
                    adam_ratio_count += 1

        if total_tasks > 0:
            overall[method] = {
                'total_rmse': total_rmse_sum / total_tasks,
                'total_nrmse': total_nrmse_sum / total_tasks,
                'time_ms': total_time_sum / total_tasks,
                'adam_triggered_ratio': adam_ratio_sum / adam_ratio_count if adam_ratio_count > 0 else None,
                'name': metrics[list(metrics.keys())[0]][method]['name']
            }

    return overall


def plot_adam_trigger_ratio(metrics: Dict, overall: Dict, config: Dict, output_dir: str):
    """
    Plot selective adam trigger ratio per cell and overall average.
    """
    cells = list(metrics.keys())

    # Get per-cell trigger ratios
    trigger_ratios = []
    for cell in cells:
        ratio = metrics[cell]['selective_adam']['adam_triggered_ratio']
        trigger_ratios.append(ratio if ratio is not None else 0)

    overall_ratio = overall['selective_adam']['adam_triggered_ratio']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Per-cell trigger ratio
    colors = plt.cm.RdYlGn_r(np.array(trigger_ratios) / 100)
    x_pos = np.arange(len(cells))
    bars = axes[0].bar(x_pos, trigger_ratios, color=colors)
    axes[0].axhline(y=overall_ratio, color='red', linestyle='--', linewidth=2,
                    label=f'Overall Avg: {overall_ratio:.1f}%')
    axes[0].set_xlabel('Cell', fontsize=12)
    axes[0].set_ylabel('Adam Triggered Ratio (%)', fontsize=12)
    axes[0].set_title('Per-Cell Selective Adam Trigger Ratio', fontsize=14)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(cells, rotation=45, ha='right')
    axes[0].legend(loc='upper right')
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].set_ylim(0, 100)

    # Add value labels
    for bar, val in zip(bars, trigger_ratios):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

    # Overall summary (pie chart showing triggered vs not triggered)
    triggered = overall_ratio
    not_triggered = 100 - triggered
    sizes = [triggered, not_triggered]
    labels = [f'Adam Triggered\n({triggered:.1f}%)', f'Gradient Only\n({not_triggered:.1f}%)']
    colors_pie = ['#d62728', '#2ca02c']

    axes[1].pie(sizes, labels=labels, colors=colors_pie, autopct='',
               startangle=90, explode=(0.05, 0))
    axes[1].set_title(f'Overall Selective Adam Trigger Distribution', fontsize=14)

    plt.suptitle(f'{config["config_name"]} - {config["mode"]} - {config["data_type"]}',
                fontsize=14, y=1.02)
    plt.tight_layout()

    filename = f'{config["config_name"].replace(" ", "_")}_{config["mode"]}_{config["data_type"]}_adam_trigger_ratio.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_combined_summary(metrics: Dict, overall: Dict, config: Dict, output_dir: str):
    """
    Create a combined summary figure with all key metrics.
    """
    cells = list(metrics.keys())
    methods = ['none', 'sgd', 'adam', 'selective_adam', 'full_adam']
    method_names = ['Delta Scale + Offset', 'SGD', 'Adam', 'Sel. Adam', 'Full Adam']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    fig = plt.figure(figsize=(20, 12))

    # Grid spec for layout
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # 1. Per-cell RMSE (top left) - convert to ps
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(cells))
    width = 0.15
    for i, method in enumerate(methods):
        values = [metrics[cell][method]['total_rmse'] * 1000 for cell in cells]
        ax1.bar(x + i * width - width * 2, values, width, label=method_names[i], color=colors[i])
    ax1.set_xlabel('Cell')
    ax1.set_ylabel('RMSE (ps)')
    ax1.set_title('Per-Cell RMSE')
    ax1.set_xticks(x)
    ax1.set_xticklabels(cells, rotation=45, ha='right', fontsize=8)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(axis='y', alpha=0.3)

    # 2. Per-cell NRMSE (top center)
    ax2 = fig.add_subplot(gs[0, 1])
    for i, method in enumerate(methods):
        values = [metrics[cell][method]['total_nrmse'] for cell in cells]
        ax2.bar(x + i * width - width * 2, values, width, label=method_names[i], color=colors[i])
    ax2.set_xlabel('Cell')
    ax2.set_ylabel('NRMSE (%)')
    ax2.set_title('Per-Cell NRMSE')
    ax2.set_xticks(x)
    ax2.set_xticklabels(cells, rotation=45, ha='right', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(axis='y', alpha=0.3)

    # 3. Adam trigger ratio per cell (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    trigger_ratios = [metrics[cell]['selective_adam']['adam_triggered_ratio'] or 0 for cell in cells]
    overall_ratio = overall['selective_adam']['adam_triggered_ratio']
    bar_colors = plt.cm.RdYlGn_r(np.array(trigger_ratios) / 100)
    x_pos_3 = np.arange(len(cells))
    bars = ax3.bar(x_pos_3, trigger_ratios, color=bar_colors)
    ax3.axhline(y=overall_ratio, color='red', linestyle='--', linewidth=2,
               label=f'Avg: {overall_ratio:.1f}%')
    ax3.set_xlabel('Cell')
    ax3.set_ylabel('Adam Triggered (%)')
    ax3.set_title('Per-Cell Adam Trigger Ratio')
    ax3.set_xticks(x_pos_3)
    ax3.set_xticklabels(cells, rotation=45, ha='right', fontsize=8)
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(axis='y', alpha=0.3)
    ax3.set_ylim(0, 100)

    # 4. Overall RMSE comparison (bottom left) - convert to ps
    ax4 = fig.add_subplot(gs[1, 0])
    rmse_values = [overall[m]['total_rmse'] * 1000 for m in methods]
    bars4 = ax4.bar(method_names, rmse_values, color=colors)
    ax4.set_ylabel('RMSE (ps)')
    ax4.set_title('Overall Average RMSE')
    ax4.grid(axis='y', alpha=0.3)
    max_rmse4 = max(rmse_values) if rmse_values else 1
    offset_rmse4 = max_rmse4 * 0.02
    for bar, val in zip(bars4, rmse_values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset_rmse4,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # 5. Overall NRMSE comparison (bottom center)
    ax5 = fig.add_subplot(gs[1, 1])
    nrmse_values = [overall[m]['total_nrmse'] for m in methods]
    bars5 = ax5.bar(method_names, nrmse_values, color=colors)
    ax5.set_ylabel('NRMSE (%)')
    ax5.set_title('Overall Average NRMSE')
    ax5.grid(axis='y', alpha=0.3)
    max_nrmse5 = max(nrmse_values) if nrmse_values else 1
    offset_nrmse5 = max_nrmse5 * 0.02
    for bar, val in zip(bars5, nrmse_values):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset_nrmse5,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # 6. Summary table (bottom right)
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    # Create summary text
    summary_text = "Summary Statistics\n" + "=" * 30 + "\n\n"
    summary_text += f"Config: {config['config_name']}\n"
    summary_text += f"Mode: {config['mode']}\n"
    summary_text += f"Data Type: {config['data_type']}\n"
    summary_text += f"Cells: {len(cells)}\n\n"

    summary_text += "Best Method by RMSE:\n"
    best_rmse_method = min(methods, key=lambda m: overall[m]['total_rmse'])
    best_rmse_ps = overall[best_rmse_method]['total_rmse'] * 1000
    summary_text += f"  {method_names[methods.index(best_rmse_method)]}: {best_rmse_ps:.4f} ps\n\n"

    summary_text += "Best Method by NRMSE:\n"
    best_nrmse_method = min(methods, key=lambda m: overall[m]['total_nrmse'])
    summary_text += f"  {method_names[methods.index(best_nrmse_method)]}: {overall[best_nrmse_method]['total_nrmse']:.4f}%\n\n"

    summary_text += f"Selective Adam Trigger Rate:\n"
    summary_text += f"  Overall Avg: {overall_ratio:.1f}%\n"
    summary_text += f"  Min: {min(trigger_ratios):.1f}%\n"
    summary_text += f"  Max: {max(trigger_ratios):.1f}%\n"

    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'Optimization Comparison Summary: {config["config_name"]} - {config["mode"]} - {config["data_type"]}',
                fontsize=16, y=0.98)

    filename = f'{config["config_name"].replace(" ", "_")}_{config["mode"]}_{config["data_type"]}_combined_summary.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def analyze_single_file(json_path: str, output_dir: str):
    """Analyze a single JSON file and generate plots."""
    print(f"\nAnalyzing: {os.path.basename(json_path)}")

    data = load_json_results(json_path)
    config = data['config']
    metrics = extract_metrics(data)
    overall = calculate_overall_metrics(metrics)

    # Generate only combined_summary and adam_trigger_ratio plots
    plot_adam_trigger_ratio(metrics, overall, config, output_dir)
    plot_combined_summary(metrics, overall, config, output_dir)

    return metrics, overall, config


def main():
    parser = argparse.ArgumentParser(description='Analyze MAML optimization comparison results')
    parser.add_argument('--input', '-i', type=str,
                       default='/home/tkdgn2907/Deepsets_test/MAML/Projects/pretraining/model_test_code/MLP/adaptation_method_comparison_results',
                       help='Input directory containing JSON result files or single JSON file path')
    parser.add_argument('--output', '-o', type=str,
                       default=None,
                       help='Output directory for plots (default: same as input)')
    parser.add_argument('--file', '-f', type=str,
                       default=None,
                       help='Specific JSON file to analyze (optional)')

    args = parser.parse_args()

    # Determine input path
    if args.file:
        input_path = args.file
    else:
        input_path = args.input

    # Determine output directory
    if args.output:
        output_dir = args.output
    elif os.path.isfile(input_path):
        output_dir = os.path.dirname(input_path)
    else:
        output_dir = input_path

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("MAML Optimization Comparison Analysis")
    print("=" * 60)
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")

    if os.path.isfile(input_path):
        # Single file analysis
        analyze_single_file(input_path, output_dir)
    else:
        # Directory analysis - process all JSON files
        json_files = sorted(Path(input_path).glob('*.json'))

        if not json_files:
            print(f"No JSON files found in {input_path}")
            return

        print(f"Found {len(json_files)} JSON file(s)")

        for json_file in json_files:
            analyze_single_file(str(json_file), output_dir)

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
