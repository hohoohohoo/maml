#!/usr/bin/env python3
"""
MLP vs GCN Comparison Script for TSMC Dataset

Compare 4 model types:
1. MLP + Aadam (baseline)
2. MLP + MAML
3. GCN + Baseline
4. GCN + MAML

Usage:
    python compare_mlp_gcn_tsmc.py --corner TT --temperature 25 --data_type cell
    python compare_mlp_gcn_tsmc.py --all_configs --data_type cell
    python compare_mlp_gcn_tsmc.py --corner FF --temperature 0 --gcn_arch conv64x2_fc128x2 --graph_mode stage_aware
    python compare_mlp_gcn_tsmc.py --all_configs --pooling output --graph_mode full_graph
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import argparse
import os
import re
from typing import Dict, Optional, List
from dataclasses import dataclass


# Default directories
MLP_RESULTS_DIR = '/home/tkdgn2907/Deepsets_test/MAML/Projects/pretraining/model_test_code/data_result_npy_directory'
GCN_RESULTS_DIR = '/home/tkdgn2907/Deepsets_test/MAML/Projects/pretraining/model_test_code/gnn/data_result_npy_directory'

# TSMC configurations
TSMC_CORNERS = ['TT', 'FF', 'SS']
TSMC_TEMPERATURES = [0, 25, 50, 75, 100]


@dataclass
class ModelResult:
    """Model result container"""
    model_type: str  # 'MLP_Aadam', 'MLP_MAML', 'GCN_Baseline', 'GCN_MAML'
    predictions: np.ndarray
    actuals: np.ndarray
    source_file: str
    config: Dict


def compute_metrics_grouped(predictions: np.ndarray, actuals: np.ndarray,
                            group_size: int = 61) -> Dict:
    """
    Compute metrics using 61-group averaging methodology.

    Args:
        predictions: predicted values
        actuals: actual values
        group_size: samples per group (default: 61 for voltage variations)

    Returns:
        dict: averaged metrics
    """
    predictions = np.array(predictions).flatten()
    actuals = np.array(actuals).flatten()

    # Filter invalid values
    valid_mask = ~(np.isnan(predictions) | np.isnan(actuals) |
                   np.isinf(predictions) | np.isinf(actuals))
    predictions = predictions[valid_mask]
    actuals = actuals[valid_mask]

    if len(predictions) == 0:
        return None

    # Group calculation
    n_groups = len(predictions) // group_size
    if n_groups == 0:
        n_groups = 1
        group_size = len(predictions)

    predictions = predictions[:n_groups * group_size]
    actuals = actuals[:n_groups * group_size]

    pred_grouped = predictions.reshape(n_groups, group_size)
    act_grouped = actuals.reshape(n_groups, group_size)

    group_metrics = []
    for i in range(n_groups):
        y_p = pred_grouped[i]
        y_t = act_grouped[i]

        # MAE
        mae = np.mean(np.abs(y_p - y_t))

        # MAPE
        mask = y_t != 0
        mape = np.mean(np.abs((y_t[mask] - y_p[mask]) / y_t[mask])) * 100 if np.any(mask) else 0

        # SMAPE
        denom = np.abs(y_t) + np.abs(y_p)
        mask_smape = denom != 0
        smape = np.mean(2.0 * np.abs(y_t[mask_smape] - y_p[mask_smape]) / denom[mask_smape]) * 100 if np.any(mask_smape) else 0

        # RMSE & NRMSE
        rmse = np.sqrt(np.mean((y_p - y_t) ** 2))
        y_range = np.max(y_t) - np.min(y_t)
        nrmse = (rmse / y_range * 100) if y_range > 0 else 0

        group_metrics.append({
            'mae': mae, 'mape': mape, 'smape': smape,
            'rmse': rmse, 'nrmse': nrmse
        })

    return {
        'mae': np.mean([g['mae'] for g in group_metrics]),
        'mape': np.mean([g['mape'] for g in group_metrics]),
        'smape': np.mean([g['smape'] for g in group_metrics]),
        'rmse': np.mean([g['rmse'] for g in group_metrics]),
        'nrmse': np.mean([g['nrmse'] for g in group_metrics]),
        'n_groups': n_groups
    }


class MLPGCNComparator:
    """Compare MLP and GCN models for TSMC dataset"""

    def __init__(self, mlp_dir: str = MLP_RESULTS_DIR,
                 gcn_dir: str = GCN_RESULTS_DIR,
                 group_size: int = 61):
        self.mlp_dir = Path(mlp_dir)
        self.gcn_dir = Path(gcn_dir)
        self.group_size = group_size

    def find_mlp_aadam(self, corner: str, temperature: int,
                       data_type: str = 'cell') -> Optional[ModelResult]:
        """Find MLP Aadam result files"""
        corner_lower = corner.lower()
        pattern = f"TSMC_voltage_variation_{corner_lower}_{temperature}_{data_type}_aadam_*_pred.npy"

        # Also try with 'adam' spelling
        pred_files = list(self.mlp_dir.glob(pattern))
        if not pred_files:
            pattern_alt = f"TSMC_voltage_variation_{corner_lower}_{temperature}_{data_type}_adam_*_pred.npy"
            pred_files = list(self.mlp_dir.glob(pattern_alt))

        if not pred_files:
            return None

        pred_file = pred_files[0]
        act_file = Path(str(pred_file).replace('_pred.npy', '_act.npy'))

        if not act_file.exists():
            return None

        try:
            predictions = np.load(pred_file)
            actuals = np.load(act_file)
            return ModelResult(
                model_type='MLP_Aadam',
                predictions=predictions,
                actuals=actuals,
                source_file=pred_file.name,
                config={'corner': corner, 'temperature': temperature, 'data_type': data_type}
            )
        except Exception as e:
            print(f"Error loading MLP Aadam: {e}")
            return None

    def find_mlp_maml(self, corner: str, temperature: int,
                      data_type: str = 'cell',
                      innerdiv: int = None, meta: int = None) -> Optional[ModelResult]:
        """Find MLP MAML result files"""
        corner_lower = corner.lower()
        pattern = f"TSMC_voltage_variation_{corner_lower}_{temperature}_{data_type}_maml_*_pred.npy"

        pred_files = list(self.mlp_dir.glob(pattern))

        # Apply filters if specified
        if innerdiv is not None:
            pred_files = [f for f in pred_files if f"innerdiv{innerdiv}" in f.name]
        if meta is not None:
            pred_files = [f for f in pred_files if f"meta{meta}" in f.name]

        if not pred_files:
            return None

        pred_file = pred_files[0]
        act_file = Path(str(pred_file).replace('_pred.npy', '_act.npy'))

        if not act_file.exists():
            return None

        try:
            predictions = np.load(pred_file)
            actuals = np.load(act_file)
            return ModelResult(
                model_type='MLP_MAML',
                predictions=predictions,
                actuals=actuals,
                source_file=pred_file.name,
                config={'corner': corner, 'temperature': temperature, 'data_type': data_type}
            )
        except Exception as e:
            print(f"Error loading MLP MAML: {e}")
            return None

    def find_gcn_baseline(self, corner: str, temperature: int,
                          data_type: str = 'cell',
                          graph_mode: str = 'stage_aware',
                          conv_dim: int = None, fc_dim: int = None,
                          num_conv: int = None, num_fc: int = None,
                          pooling: str = 'mean',
                          subdir: str = None) -> Optional[ModelResult]:
        """Find GCN Baseline result files"""
        # Build search pattern
        corner_upper = corner.upper()
        base_pattern = f"*TSMC_{corner_upper}_{temperature}_{data_type}_{graph_mode}_*baseline*_pred.npy"

        search_dir = self.gcn_dir / subdir if subdir else self.gcn_dir
        pred_files = list(search_dir.glob(base_pattern))

        # Filter by pooling suffix
        pool_suffix = f"_pool{pooling}" if pooling != 'mean' else ""
        if pool_suffix:
            pred_files = [f for f in pred_files if pool_suffix in f.name]
        else:
            # For mean pooling, exclude files with _pool prefix
            pred_files = [f for f in pred_files if '_pool' not in f.name or '_poolmean' in f.name]

        # Apply architecture filters
        if conv_dim is not None:
            pred_files = [f for f in pred_files if f"conv{conv_dim}x" in f.name]
        if fc_dim is not None:
            pred_files = [f for f in pred_files if f"fc{fc_dim}x" in f.name]
        if num_conv is not None:
            pred_files = [f for f in pred_files if f"conv{conv_dim if conv_dim else ''}x{num_conv}" in f.name or re.search(rf'conv\d+x{num_conv}', f.name)]
        if num_fc is not None:
            pred_files = [f for f in pred_files if f"fc{fc_dim if fc_dim else ''}x{num_fc}" in f.name or re.search(rf'fc\d+x{num_fc}', f.name)]

        if not pred_files:
            return None

        pred_file = pred_files[0]
        act_file = Path(str(pred_file).replace('_pred.npy', '_act.npy'))

        if not act_file.exists():
            return None

        try:
            predictions = np.load(pred_file)
            actuals = np.load(act_file)
            return ModelResult(
                model_type='GCN_Baseline',
                predictions=predictions,
                actuals=actuals,
                source_file=pred_file.name,
                config={
                    'corner': corner, 'temperature': temperature,
                    'data_type': data_type, 'graph_mode': graph_mode
                }
            )
        except Exception as e:
            print(f"Error loading GCN Baseline: {e}")
            return None

    def find_gcn_maml(self, corner: str, temperature: int,
                      data_type: str = 'cell',
                      graph_mode: str = 'stage_aware',
                      conv_dim: int = None, fc_dim: int = None,
                      num_conv: int = None, num_fc: int = None,
                      pooling: str = 'mean',
                      subdir: str = None) -> Optional[ModelResult]:
        """Find GCN MAML result files"""
        corner_upper = corner.upper()
        base_pattern = f"*TSMC_{corner_upper}_{temperature}_{data_type}_{graph_mode}_*maml*_pred.npy"

        search_dir = self.gcn_dir / subdir if subdir else self.gcn_dir
        pred_files = list(search_dir.glob(base_pattern))

        # Filter by pooling suffix
        pool_suffix = f"_pool{pooling}" if pooling != 'mean' else ""
        if pool_suffix:
            pred_files = [f for f in pred_files if pool_suffix in f.name]
        else:
            # For mean pooling, exclude files with _pool prefix
            pred_files = [f for f in pred_files if '_pool' not in f.name or '_poolmean' in f.name]

        # Apply architecture filters
        if conv_dim is not None:
            pred_files = [f for f in pred_files if f"conv{conv_dim}x" in f.name]
        if fc_dim is not None:
            pred_files = [f for f in pred_files if f"fc{fc_dim}x" in f.name]
        if num_conv is not None:
            pred_files = [f for f in pred_files if re.search(rf'conv\d+x{num_conv}', f.name)]
        if num_fc is not None:
            pred_files = [f for f in pred_files if re.search(rf'fc\d+x{num_fc}', f.name)]

        if not pred_files:
            return None

        pred_file = pred_files[0]
        act_file = Path(str(pred_file).replace('_pred.npy', '_act.npy'))

        if not act_file.exists():
            return None

        try:
            predictions = np.load(pred_file)
            actuals = np.load(act_file)
            return ModelResult(
                model_type='GCN_MAML',
                predictions=predictions,
                actuals=actuals,
                source_file=pred_file.name,
                config={
                    'corner': corner, 'temperature': temperature,
                    'data_type': data_type, 'graph_mode': graph_mode
                }
            )
        except Exception as e:
            print(f"Error loading GCN MAML: {e}")
            return None

    def compare_single_config(self, corner: str, temperature: int,
                              data_type: str = 'cell',
                              graph_mode: str = 'stage_aware',
                              gcn_conv_dim: int = 32,
                              gcn_num_conv: int = 3,
                              gcn_fc_dim: int = 128,
                              gcn_num_fc: int = 2,
                              pooling: str = 'mean',
                              gcn_subdir: str = None) -> Optional[Dict]:
        """
        Compare all 4 models for a single configuration.

        Returns dict with metrics for each model found.
        """
        print(f"\n{'='*70}")
        print(f"Configuration: {corner}_{temperature} ({data_type}, {graph_mode}, pooling={pooling})")
        print(f"GCN Architecture: conv{gcn_conv_dim}x{gcn_num_conv}_fc{gcn_fc_dim}x{gcn_num_fc}")
        print(f"{'='*70}")

        results = {}

        # 1. MLP Aadam
        mlp_aadam = self.find_mlp_aadam(corner, temperature, data_type)
        if mlp_aadam:
            metrics = compute_metrics_grouped(mlp_aadam.predictions, mlp_aadam.actuals, self.group_size)
            if metrics:
                results['MLP_Aadam'] = {
                    'metrics': metrics,
                    'source': mlp_aadam.source_file
                }
                print(f"  MLP_Aadam: NRMSE={metrics['nrmse']:.4f}%, SMAPE={metrics['smape']:.4f}%")
        else:
            print(f"  MLP_Aadam: Not found")

        # 2. MLP MAML
        mlp_maml = self.find_mlp_maml(corner, temperature, data_type)
        if mlp_maml:
            metrics = compute_metrics_grouped(mlp_maml.predictions, mlp_maml.actuals, self.group_size)
            if metrics:
                results['MLP_MAML'] = {
                    'metrics': metrics,
                    'source': mlp_maml.source_file
                }
                print(f"  MLP_MAML:  NRMSE={metrics['nrmse']:.4f}%, SMAPE={metrics['smape']:.4f}%")
        else:
            print(f"  MLP_MAML:  Not found")

        # 3. GCN Baseline
        gcn_baseline = self.find_gcn_baseline(
            corner, temperature, data_type, graph_mode,
            conv_dim=gcn_conv_dim, fc_dim=gcn_fc_dim,
            num_conv=gcn_num_conv, num_fc=gcn_num_fc,
            pooling=pooling, subdir=gcn_subdir
        )
        if gcn_baseline:
            metrics = compute_metrics_grouped(gcn_baseline.predictions, gcn_baseline.actuals, self.group_size)
            if metrics:
                results['GCN_Baseline'] = {
                    'metrics': metrics,
                    'source': gcn_baseline.source_file
                }
                print(f"  GCN_Baseline: NRMSE={metrics['nrmse']:.4f}%, SMAPE={metrics['smape']:.4f}%")
        else:
            print(f"  GCN_Baseline: Not found")

        # 4. GCN MAML
        gcn_maml = self.find_gcn_maml(
            corner, temperature, data_type, graph_mode,
            conv_dim=gcn_conv_dim, fc_dim=gcn_fc_dim,
            num_conv=gcn_num_conv, num_fc=gcn_num_fc,
            pooling=pooling, subdir=gcn_subdir
        )
        if gcn_maml:
            metrics = compute_metrics_grouped(gcn_maml.predictions, gcn_maml.actuals, self.group_size)
            if metrics:
                results['GCN_MAML'] = {
                    'metrics': metrics,
                    'source': gcn_maml.source_file
                }
                print(f"  GCN_MAML:  NRMSE={metrics['nrmse']:.4f}%, SMAPE={metrics['smape']:.4f}%")
        else:
            print(f"  GCN_MAML:  Not found")

        if len(results) < 2:
            print(f"  Warning: Need at least 2 models to compare (found {len(results)})")
            return None

        return {
            'corner': corner,
            'temperature': temperature,
            'data_type': data_type,
            'graph_mode': graph_mode,
            'results': results
        }

    def compare_all_configs(self, data_type: str = 'cell',
                            graph_mode: str = 'stage_aware',
                            gcn_conv_dim: int = 32,
                            gcn_num_conv: int = 3,
                            gcn_fc_dim: int = 128,
                            gcn_num_fc: int = 2,
                            pooling: str = 'mean',
                            gcn_subdir: str = None) -> List[Dict]:
        """Compare all corner x temperature configurations"""
        all_results = []

        for corner in TSMC_CORNERS:
            for temp in TSMC_TEMPERATURES:
                result = self.compare_single_config(
                    corner, temp, data_type, graph_mode,
                    gcn_conv_dim, gcn_num_conv, gcn_fc_dim, gcn_num_fc,
                    pooling, gcn_subdir
                )
                if result:
                    all_results.append(result)

        return all_results


def plot_comparison_bar(comparison_results: List[Dict], output_dir: str = None,
                        metric: str = 'nrmse', data_type: str = 'cell'):
    """
    Create bar chart comparing all 4 models across configurations.
    """
    if not comparison_results:
        print("No results to plot")
        return

    model_types = ['MLP_Aadam', 'MLP_MAML', 'GCN_Baseline', 'GCN_MAML']
    model_colors = {
        'MLP_Aadam': '#1f77b4',      # Blue
        'MLP_MAML': '#2ca02c',        # Green
        'GCN_Baseline': '#ff7f0e',    # Orange
        'GCN_MAML': '#d62728'         # Red
    }
    model_labels = {
        'MLP_Aadam': 'MLP (Aadam)',
        'MLP_MAML': 'MLP (MAML)',
        'GCN_Baseline': 'GCN (Baseline)',
        'GCN_MAML': 'GCN (MAML)'
    }

    metric_labels = {
        'nrmse': 'NRMSE (%)',
        'smape': 'SMAPE (%)',
        'mae': 'MAE',
        'mape': 'MAPE (%)'
    }

    # Prepare data
    configs = []
    data = {m: [] for m in model_types}

    for result in comparison_results:
        config_label = f"{result['corner']}_{result['temperature']}"
        configs.append(config_label)

        for model in model_types:
            if model in result['results']:
                val = result['results'][model]['metrics'][metric]
                if metric == 'mae':
                    val = val * 1000  # Scale MAE
                data[model].append(val)
            else:
                data[model].append(np.nan)

    # Create figure with more width for better separation
    _fig, ax = plt.subplots(figsize=(18, 7))

    # Increase group spacing: multiply x positions by a factor
    group_spacing = 1.5  # Space between configuration groups
    x = np.arange(len(configs)) * group_spacing
    width = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]

    for i, model in enumerate(model_types):
        values = data[model]
        valid_mask = ~np.isnan(values)
        valid_x = x[valid_mask]
        valid_vals = np.array(values)[valid_mask]

        bars = ax.bar(valid_x + offsets[i] * width, valid_vals, width,
                      label=model_labels[model], color=model_colors[model], alpha=0.85,
                      edgecolor='black', linewidth=0.5)

        # Add value labels
        for bar, val in zip(bars, valid_vals):
            height = bar.get_height()
            ax.annotate(f'{val:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8, rotation=90)

    # Add vertical separator lines between configurations
    for i in range(1, len(configs)):
        separator_x = (x[i-1] + x[i]) / 2
        ax.axvline(x=separator_x, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    # Alternate background shading for better distinction
    for i in range(len(configs)):
        if i % 2 == 0:
            left = x[i] - group_spacing * 0.4
            right = x[i] + group_spacing * 0.4
            ax.axvspan(left, right, alpha=0.08, color='gray')

    # Calculate and display improvement % (GCN_MAML vs MLP_MAML)
    mlp_maml_vals = data['MLP_MAML']
    gcn_maml_vals = data['GCN_MAML']
    y_max = ax.get_ylim()[1]

    for i in range(len(configs)):
        mlp_val = mlp_maml_vals[i]
        gcn_val = gcn_maml_vals[i]

        if not np.isnan(mlp_val) and not np.isnan(gcn_val) and mlp_val > 0:
            # Calculate improvement: positive means GCN is better (lower error)
            improvement = (mlp_val - gcn_val) / mlp_val * 100

            # Position for the improvement text (above the bars)
            text_x = x[i]
            text_y = y_max * 0.95

            # Color based on improvement direction
            if improvement > 0:
                color = 'green'
                sign = '-'  # Lower is better for error metrics
            else:
                color = 'red'
                sign = '+'
                improvement = abs(improvement)

            ax.annotate(f'{sign}{improvement:.1f}%',
                       xy=(text_x, text_y),
                       ha='center', va='top', fontsize=8, fontweight='bold',
                       color=color,
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=color, alpha=0.8))

    ax.set_xlabel('Configuration (Corner_Temperature)', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric_labels.get(metric, metric), fontsize=12, fontweight='bold')
    ax.set_title(f'MLP vs GCN Comparison: {metric_labels.get(metric, metric)} (TSMC, {data_type})', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right', fontsize=10)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')

    # Set x-axis limits to add padding
    ax.set_xlim(x[0] - group_spacing * 0.6, x[-1] + group_spacing * 0.6)

    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'mlp_gcn_comparison_{data_type}_{metric}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    plt.show()


def plot_comparison_grouped(comparison_results: List[Dict], output_dir: str = None,
                            data_type: str = 'cell'):
    """
    Create grouped comparison plot with multiple metrics.
    """
    if not comparison_results:
        print("No results to plot")
        return

    model_types = ['MLP_Aadam', 'MLP_MAML', 'GCN_Baseline', 'GCN_MAML']
    metrics = ['nrmse', 'smape', 'mae']
    metric_labels = {'nrmse': 'NRMSE (%)', 'smape': 'SMAPE (%)', 'mae': 'MAE (x1000)'}

    model_colors = {
        'MLP_Aadam': '#1f77b4',
        'MLP_MAML': '#2ca02c',
        'GCN_Baseline': '#ff7f0e',
        'GCN_MAML': '#d62728'
    }

    # Aggregate metrics across all configs
    aggregated = {m: {metric: [] for metric in metrics} for m in model_types}

    for result in comparison_results:
        for model in model_types:
            if model in result['results']:
                for metric in metrics:
                    val = result['results'][model]['metrics'][metric]
                    if metric == 'mae':
                        val = val * 1000
                    aggregated[model][metric].append(val)

    # Calculate mean and std for each model
    summary = {m: {} for m in model_types}
    for model in model_types:
        for metric in metrics:
            vals = aggregated[model][metric]
            if vals:
                summary[model][metric] = {
                    'mean': np.mean(vals),
                    'std': np.std(vals)
                }

    # Create subplot for each metric
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        models_with_data = [m for m in model_types if metric in summary[m]]
        x = np.arange(len(models_with_data))

        means = [summary[m][metric]['mean'] for m in models_with_data]
        stds = [summary[m][metric]['std'] for m in models_with_data]
        colors = [model_colors[m] for m in models_with_data]

        bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.8)

        # Add value labels
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax.annotate(f'{mean:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height + std),
                       xytext=(0, 5),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_ylabel(metric_labels[metric], fontsize=11)
        ax.set_title(metric_labels[metric], fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', '\n') for m in models_with_data], fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(f'MLP vs GCN Performance Comparison (TSMC, {data_type}, All Configurations)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'mlp_gcn_summary_comparison_{data_type}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    plt.show()


def save_comparison_table(comparison_results: List[Dict], output_dir: str = None,
                          data_type: str = 'cell'):
    """
    Save comparison results as a table.
    """
    if not comparison_results:
        print("No results to save")
        return

    model_types = ['MLP_Aadam', 'MLP_MAML', 'GCN_Baseline', 'GCN_MAML']
    metrics = ['nrmse', 'smape', 'mae']

    rows = []
    for result in comparison_results:
        row = {
            'Corner': result['corner'],
            'Temperature': result['temperature'],
            'Data_Type': result['data_type'],
            'Graph_Mode': result['graph_mode']
        }

        for model in model_types:
            if model in result['results']:
                for metric in metrics:
                    val = result['results'][model]['metrics'][metric]
                    if metric == 'mae':
                        val = val * 1000
                    row[f'{model}_{metric}'] = val
            else:
                for metric in metrics:
                    row[f'{model}_{metric}'] = np.nan

        rows.append(row)

    df = pd.DataFrame(rows)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, f'mlp_gcn_comparison_table_{data_type}.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

    print(f"\nComparison Table ({data_type}):")
    print(df.to_string(index=False))

    return df


def save_improvement_table(comparison_results: List[Dict], output_dir: str = None,
                           data_type: str = 'cell'):
    """
    Save improvement table showing how much each model improved compared to MLP_Aadam baseline.

    Calculates:
    - Absolute difference (baseline - model)
    - Improvement percentage ((baseline - model) / baseline * 100)
    """
    if not comparison_results:
        print("No results to save")
        return

    baseline_model = 'MLP_Aadam'
    compare_models = ['MLP_MAML', 'GCN_Baseline', 'GCN_MAML']
    metrics = ['nrmse', 'smape', 'mae']

    rows = []
    for result in comparison_results:
        row = {
            'Corner': result['corner'],
            'Temperature': result['temperature'],
        }

        # Get baseline values
        baseline_metrics = {}
        if baseline_model in result['results']:
            for metric in metrics:
                val = result['results'][baseline_model]['metrics'][metric]
                if metric == 'mae':
                    val = val * 1000
                baseline_metrics[metric] = val
                row[f'{baseline_model}_{metric}'] = val
        else:
            rows.append(row)
            continue

        # Calculate improvement for each model
        for model in compare_models:
            if model in result['results']:
                for metric in metrics:
                    val = result['results'][model]['metrics'][metric]
                    if metric == 'mae':
                        val = val * 1000
                    row[f'{model}_{metric}'] = val

                    # Calculate improvement
                    baseline_val = baseline_metrics.get(metric)
                    if baseline_val and baseline_val > 0:
                        improvement = ((baseline_val - val) / baseline_val) * 100
                        row[f'{model}_{metric}_impr%'] = improvement
                    else:
                        row[f'{model}_{metric}_impr%'] = np.nan
            else:
                for metric in metrics:
                    row[f'{model}_{metric}'] = np.nan
                    row[f'{model}_{metric}_impr%'] = np.nan

        rows.append(row)

    df = pd.DataFrame(rows)

    # Print summary statistics
    print("\n" + "="*80)
    print("IMPROVEMENT SUMMARY vs MLP_Aadam (positive = better)")
    print("="*80)

    for model in compare_models:
        print(f"\n{model}:")
        for metric in metrics:
            col = f'{model}_{metric}_impr%'
            if col in df.columns:
                valid_vals = df[col].dropna()
                if len(valid_vals) > 0:
                    mean_impr = valid_vals.mean()
                    min_impr = valid_vals.min()
                    max_impr = valid_vals.max()
                    print(f"  {metric.upper():>6}: avg={mean_impr:+.2f}%  (min={min_impr:+.2f}%, max={max_impr:+.2f}%)")

    # Create summary table with only improvement percentages
    summary_cols = ['Corner', 'Temperature']
    for model in compare_models:
        for metric in metrics:
            summary_cols.append(f'{model}_{metric}_impr%')

    df_summary = df[[c for c in summary_cols if c in df.columns]].copy()

    # Rename columns for clarity
    rename_map = {}
    for model in compare_models:
        for metric in metrics:
            old_name = f'{model}_{metric}_impr%'
            new_name = f'{model.replace("_", " ")} {metric.upper()} (%)'
            rename_map[old_name] = new_name
    df_summary = df_summary.rename(columns=rename_map)

    print("\n" + "-"*80)
    print("Improvement Table (% better than MLP_Aadam):")
    print("-"*80)
    print(df_summary.to_string(index=False, float_format='{:+.2f}'.format))

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        # Save full table
        csv_path = os.path.join(output_dir, f'mlp_gcn_improvement_table_{data_type}.csv')
        df.to_csv(csv_path, index=False)
        print(f"\nSaved: {csv_path}")

        # Save summary table
        summary_path = os.path.join(output_dir, f'mlp_gcn_improvement_summary_{data_type}.csv')
        df_summary.to_csv(summary_path, index=False)
        print(f"Saved: {summary_path}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description='Compare MLP vs GCN models for TSMC dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single configuration
  python compare_mlp_gcn_tsmc.py --corner TT --temperature 25

  # All configurations
  python compare_mlp_gcn_tsmc.py --all_configs

  # With specific GCN architecture
  python compare_mlp_gcn_tsmc.py --all_configs --gcn_conv_dim 64 --gcn_fc_dim 128

  # With specific graph mode
  python compare_mlp_gcn_tsmc.py --all_configs --graph_mode full_graph

  # With output pooling (output-node-only)
  python compare_mlp_gcn_tsmc.py --all_configs --pooling output --graph_mode full_graph
"""
    )

    parser.add_argument('--corner', type=str, choices=['TT', 'FF', 'SS'],
                       help='TSMC corner')
    parser.add_argument('--temperature', type=int, choices=[0, 25, 50, 75, 100],
                       help='Temperature')
    parser.add_argument('--all_configs', action='store_true',
                       help='Compare all corner x temperature configurations')
    parser.add_argument('--data_type', type=str, default='cell',
                       choices=['cell', 'transition'],
                       help='Data type (default: cell)')
    parser.add_argument('--graph_mode', type=str, default='stage_aware',
                       choices=['stage_aware', 'full_graph'],
                       help='GCN graph mode (default: stage_aware)')
    parser.add_argument('--pooling', type=str, default='mean',
                       choices=['mean', 'max', 'add', 'output'],
                       help='GCN pooling method (default: mean)')
    parser.add_argument('--gcn_conv_dim', type=int, default=32,
                       help='GCN conv hidden dimension (default: 32)')
    parser.add_argument('--gcn_num_conv', type=int, default=2,
                       help='GCN number of conv layers (default: 3)')
    parser.add_argument('--gcn_fc_dim', type=int, default=256,
                       help='GCN fc hidden dimension (default: 256)')
    parser.add_argument('--gcn_num_fc', type=int, default=2,
                       help='GCN number of fc layers (default: 2)')
    parser.add_argument('--gcn_subdir', type=str, default=None,
                       help='GCN results subdirectory (e.g., 10000samples)')
    parser.add_argument('--mlp_dir', type=str, default=MLP_RESULTS_DIR,
                       help='MLP results directory')
    parser.add_argument('--gcn_dir', type=str, default=GCN_RESULTS_DIR,
                       help='GCN results directory')
    parser.add_argument('--output_dir', type=str,
                       default='/home/tkdgn2907/Deepsets_test/MAML/Projects/result_management/result_summary/mlp_vs_gcn_comparison_outputs',
                       help='Output directory for plots and tables')

    args = parser.parse_args()

    # Validate arguments
    if not args.all_configs and (args.corner is None or args.temperature is None):
        parser.error("Either --all_configs or both --corner and --temperature are required")

    # Create comparator
    comparator = MLPGCNComparator(
        mlp_dir=args.mlp_dir,
        gcn_dir=args.gcn_dir
    )

    # Run comparison
    if args.all_configs:
        results = comparator.compare_all_configs(
            data_type=args.data_type,
            graph_mode=args.graph_mode,
            gcn_conv_dim=args.gcn_conv_dim,
            gcn_num_conv=args.gcn_num_conv,
            gcn_fc_dim=args.gcn_fc_dim,
            gcn_num_fc=args.gcn_num_fc,
            pooling=args.pooling,
            gcn_subdir=args.gcn_subdir
        )
    else:
        result = comparator.compare_single_config(
            corner=args.corner,
            temperature=args.temperature,
            data_type=args.data_type,
            graph_mode=args.graph_mode,
            gcn_conv_dim=args.gcn_conv_dim,
            gcn_num_conv=args.gcn_num_conv,
            gcn_fc_dim=args.gcn_fc_dim,
            gcn_num_fc=args.gcn_num_fc,
            pooling=args.pooling,
            gcn_subdir=args.gcn_subdir
        )
        results = [result] if result else []

    if not results:
        print("\nNo valid comparison results found")
        return 1

    # Create output directory with data_type, graph_mode, pooling and architecture info
    arch_str = f"conv{args.gcn_conv_dim}x{args.gcn_num_conv}_fc{args.gcn_fc_dim}x{args.gcn_num_fc}"
    pool_str = f"_pool{args.pooling}" if args.pooling != 'mean' else ""
    output_subdir = f"{args.data_type}_{args.graph_mode}_{arch_str}{pool_str}"
    output_dir = os.path.join(args.output_dir, output_subdir)

    print(f"\nOutput directory: {output_dir}")

    # Generate plots and tables
    print("\n" + "="*70)
    print("Generating comparison plots...")
    print("="*70)

    plot_comparison_bar(results, output_dir, metric='nrmse', data_type=args.data_type)
    plot_comparison_bar(results, output_dir, metric='smape', data_type=args.data_type)
    plot_comparison_bar(results, output_dir, metric='mae', data_type=args.data_type)
    plot_comparison_grouped(results, output_dir, data_type=args.data_type)
    save_comparison_table(results, output_dir, data_type=args.data_type)
    save_improvement_table(results, output_dir, data_type=args.data_type)

    print("\nComparison complete!")
    return 0


if __name__ == "__main__":
    exit(main())
