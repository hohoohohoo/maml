#!/usr/bin/env python3
"""
Adam vs MAML Comparison Script

61개씩 그룹으로 나누어 Adam과 MAML의 결과를 비교합니다.
특정 조건 (PDK, topology, task, cell)에 맞는 파일을 자동으로 검색하여 비교합니다.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import argparse
from typing import Dict, Tuple, Optional
import sys


class ModelComparator:
    """Adam과 MAML 모델 결과를 비교하는 클래스"""

    def __init__(self, data_dir: str, group_size: int = 61):
        self.data_dir = Path(data_dir)
        self.group_size = group_size

        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {data_dir}")

    def find_all_cells(self, pdk: str, topology: str, variation_type: str, task: str, num_iterations: int = 300000,
                      innerdiv: Optional[int] = None, meta: Optional[int] = None) -> set:
        """
        조건에 맞는 모든 cell 이름을 찾습니다.

        Parameters:
        -----------
        pdk, topology, variation_type, task : str
            검색 조건

        Returns:
        --------
        set : 찾은 모든 cell 이름들
        """
        all_files = list(self.data_dir.glob('*.npy'))
        cells = set()

        for file in all_files:
            filename = file.name
            # 조건 체크 (num_iterations도 포함)
            if (pdk.lower() in filename.lower() and
                topology.lower() in filename.lower() and
                variation_type.lower() in filename.lower() and
                task.lower() in filename.lower() and
                str(num_iterations) in filename):

                # MAML 파일인 경우 innerdiv, meta 체크
                if 'maml' in filename.lower():
                    if innerdiv is not None and f"innerdiv{innerdiv}" not in filename.lower():
                        continue
                    if meta is not None and f"meta{meta}" not in filename.lower():
                        continue

                # cell 이름 추출 (패턴: PDK_topology_CELL_variation_task_...)
                parts = filename.split('_')
                # cell은 보통 3번째 위치에 있음 (ASAP7_topology_agnostic_FAx1_...)
                for i, part in enumerate(parts):
                    if part.lower() == topology.split('_')[-1].lower():  # topology_agnostic의 agnostic 찾기
                        if i + 1 < len(parts):
                            cell_candidate = parts[i + 1]
                            # cell 이름 검증 (보통 대문자+숫자+x+숫자 형태)
                            if cell_candidate and not cell_candidate.lower() in ['cell', 'transition', 'interpolation', 'extrapolation']:
                                cells.add(cell_candidate)

        return cells

    def find_matching_files(self, pdk: str, topology: str, variation_type: str, task: str, cell: str,
                           num_iterations: Optional[list] = None,
                           innerdiv: Optional[list] = None, meta: Optional[list] = None) -> Dict[str, Optional[Tuple[Path, Path]]]:
        """
        조건에 맞는 Adam과 MAML 파일을 검색합니다.

        Parameters:
        -----------
        pdk : str
            PDK 이름 (예: 'ASAP7', 'TSMC')
        topology : str
            Topology 타입 (예: 'intra_topology', 'topology_agnostic')
        variation_type : str
            Variation 타입 (예: 'cell', 'transition')
        task : str
            Task 타입 (예: 'interpolation', 'extrapolation')
        cell : str
            Cell 이름 (예: 'AND2x6', 'NAND2x1', 'FAx1')
        num_iterations : list of int, optional
            학습 반복 횟수 리스트 (여러 값 가능, 기본값: [300000])
        innerdiv : list of int, optional
            MAML inner loop division factors (여러 값 가능)
        meta : list of int, optional
            MAML meta batch sizes (여러 값 가능)

        Returns:
        --------
        dict : {'Adam_iter300000': (pred_path, act_path), 'MAML_innerdivX_metaY_iter300000': (pred_path, act_path), ...}
        """
        results = {}

        # num_iterations 리스트 처리
        iterations_list = num_iterations if num_iterations is not None else [300000]

        # 패턴 매칭을 위한 기본 키워드 (num_iterations 제외)
        base_keywords = {
            'pdk': pdk,
            'topology': topology,
            'variation_type': variation_type,
            'task': task,
            'cell': cell
        }

        # 모든 .npy 파일 검색
        all_files = list(self.data_dir.glob('*.npy'))

        # Method 매칭 패턴 정의 (Aadam도 Adam으로 인식)
        adam_patterns = ['Adam', 'Aadam', 'adam', 'aadam']
        maml_patterns = ['MAML', 'maml']

        # innerdiv, meta가 None이면 모든 MAML 파일 찾기
        innerdiv_list = innerdiv if innerdiv is not None else [None]
        meta_list = meta if meta is not None else [None]

        # 각 iteration에 대해 파일 찾기
        for iter_val in iterations_list:
            # 현재 iteration에 대한 키워드
            keywords = {**base_keywords, 'num_iterations': str(iter_val)}

            # Adam 파일 찾기
            adam_matching_pairs = []
            for file in all_files:
                filename = file.name
                if all(keyword.lower() in filename.lower() for keyword in keywords.values()):
                    if any(pattern in filename for pattern in adam_patterns):
                        if '_pred.npy' in filename:
                            expected_act_name = filename.replace('_pred.npy', '_act.npy')
                            expected_act_path = file.parent / expected_act_name
                            if expected_act_path.exists():
                                adam_matching_pairs.append((file, expected_act_path))

            if adam_matching_pairs:
                key = f"Adam_iter{iter_val}" if len(iterations_list) > 1 else "Adam"
                # 중복 키 처리
                if key in results:
                    print(f"  ⚠ Warning: Duplicate iteration {iter_val} found, skipping")
                else:
                    results[key] = adam_matching_pairs[0]
                    print(f"✓ Found {key} files:")
                    print(f"  Pred: {adam_matching_pairs[0][0].name}")
                    print(f"  Act:  {adam_matching_pairs[0][1].name}")
                    if len(adam_matching_pairs) > 1:
                        print(f"  ⚠ Warning: Found {len(adam_matching_pairs)} matching Adam file pairs, using first one")
            else:
                print(f"✗ Adam files not found for iteration {iter_val}")

            # MAML 파일들 찾기 (innerdiv, meta 조합별로)
            for inner_val in innerdiv_list:
                for meta_val in meta_list:
                    maml_matching_pairs = []

                    for file in all_files:
                        filename = file.name
                        if all(keyword.lower() in filename.lower() for keyword in keywords.values()):
                            if any(pattern in filename for pattern in maml_patterns):
                                # innerdiv, meta 체크
                                if inner_val is not None and f"innerdiv{inner_val}" not in filename.lower():
                                    continue
                                if meta_val is not None and f"meta{meta_val}" not in filename.lower():
                                    continue

                                if '_pred.npy' in filename:
                                    expected_act_name = filename.replace('_pred.npy', '_act.npy')
                                    expected_act_path = file.parent / expected_act_name
                                    if expected_act_path.exists():
                                        maml_matching_pairs.append((file, expected_act_path))

                    if maml_matching_pairs:
                        # 키 이름 생성
                        key_parts = ["MAML"]
                        if inner_val is not None:
                            key_parts.append(f"innerdiv{inner_val}")
                        if meta_val is not None:
                            key_parts.append(f"meta{meta_val}")
                        if len(iterations_list) > 1:
                            key_parts.append(f"iter{iter_val}")

                        key = "_".join(key_parts)

                        # 중복 키 처리
                        if key in results:
                            print(f"  ⚠ Warning: Duplicate key {key} found, skipping")
                        else:
                            results[key] = maml_matching_pairs[0]
                            print(f"✓ Found {key} files:")
                            print(f"  Pred: {maml_matching_pairs[0][0].name}")
                            print(f"  Act:  {maml_matching_pairs[0][1].name}")
                            if len(maml_matching_pairs) > 1:
                                print(f"  ⚠ Warning: Found {len(maml_matching_pairs)} matching file pairs, using first one")

        return results

    def load_data(self, pred_file: Path, act_file: Path) -> Tuple[np.ndarray, np.ndarray]:
        """예측값과 실제값을 로드합니다."""
        y_pred = np.load(pred_file)
        y_true = np.load(act_file)

        if y_pred.shape != y_true.shape:
            raise ValueError(f"Shape mismatch: pred={y_pred.shape}, act={y_true.shape}")

        return y_pred, y_true

    def calculate_metrics_grouped(self, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
        """
        61개씩 끊어서 그룹별 메트릭을 계산합니다.

        Returns:
        --------
        pd.DataFrame : 그룹별 메트릭이 담긴 데이터프레임
        """
        n_groups = len(y_true) // self.group_size

        # 남는 샘플 제외
        y_true_trimmed = y_true[:n_groups * self.group_size]
        y_pred_trimmed = y_pred[:n_groups * self.group_size]

        # Reshape to (n_groups, group_size)
        y_true_grouped = y_true_trimmed.reshape(n_groups, self.group_size)
        y_pred_grouped = y_pred_trimmed.reshape(n_groups, self.group_size)

        # 각 그룹별 계산
        group_metrics = []

        for i in range(n_groups):
            y_t = y_true_grouped[i]
            y_p = y_pred_grouped[i]

            # MAE
            mae = np.mean(np.abs(y_t - y_p))

            # MAPE
            mask = y_t != 0
            mape = np.mean(np.abs((y_t[mask] - y_p[mask]) / y_t[mask])) * 100 if np.any(mask) else 0

            # SMAPE (Symmetric MAPE)
            denom = np.abs(y_t) + np.abs(y_p)
            mask_smape = denom != 0   # 0 division 방지
            smape = np.mean(
                2.0 * np.abs(y_t[mask_smape] - y_p[mask_smape]) / denom[mask_smape]
            ) * 100 if np.any(mask_smape) else 0

            # RMSE
            rmse = np.sqrt(np.mean((y_t - y_p) ** 2))

            # NRMSE (Range normalization)
            y_range = np.max(y_t) - np.min(y_t)
            nrmse = (rmse / y_range * 100) if y_range > 0 else 0

            # Error statistics
            errors = y_t - y_p
            std_error = np.std(errors)
            mean_error = np.mean(errors)

            group_metrics.append({
                'group': i,
                'MAE': mae,
                'MAPE': mape,
                'SMAPE': smape,
                'RMSE': rmse,
                'NRMSE': nrmse,
                'STD': std_error,
                'Mean_Error': mean_error,
                'Range': y_range
            })

        return pd.DataFrame(group_metrics)

    def compare_methods(self, files_dict: Dict[str, Tuple[Path, Path]]) -> pd.DataFrame:
        """
        Adam과 MAML의 메트릭을 비교합니다.

        Parameters:
        -----------
        files_dict : dict
            {'Adam': (pred_path, act_path), 'MAML': (pred_path, act_path)}

        Returns:
        --------
        pd.DataFrame : 비교 결과 테이블
        """
        results = {}
        detailed_results = {}

        for method, files in files_dict.items():
            if files is None:
                print(f"⚠ Skipping {method}: files not found")
                continue

            pred_file, act_file = files
            print(f"\n{'='*70}")
            print(f"Processing {method}...")
            print(f"{'='*70}")

            # 데이터 로드
            y_pred, y_true = self.load_data(pred_file, act_file)
            print(f"Data shape: {y_true.shape}")
            print(f"Number of groups: {len(y_true) // self.group_size}")

            # 그룹별 메트릭 계산
            df_grouped = self.calculate_metrics_grouped(y_true, y_pred)
            detailed_results[method] = df_grouped

            # 평균 및 표준편차 계산
            metrics_mean = {
                'MAE': df_grouped['MAE'].mean(),
                'MAPE': df_grouped['MAPE'].mean(),
                'SMAPE': df_grouped['SMAPE'].mean(),
                'RMSE': df_grouped['RMSE'].mean(),
                'NRMSE': df_grouped['NRMSE'].mean(),
                'STD': df_grouped['STD'].mean(),
                'Mean_Error': df_grouped['Mean_Error'].mean()
            }

            metrics_std = {
                'MAE_std': df_grouped['MAE'].std(),
                'MAPE_std': df_grouped['MAPE'].std(),
                'SMAPE_std': df_grouped['SMAPE'].std(),
                'RMSE_std': df_grouped['RMSE'].std(),
                'NRMSE_std': df_grouped['NRMSE'].std(),
                'STD_std': df_grouped['STD'].std(),
                'Mean_Error_std': df_grouped['Mean_Error'].std()
            }

            results[method] = {**metrics_mean, **metrics_std}

            # 결과 출력
            print(f"\n{method} Results (Mean ± Std):")
            print("-" * 60)
            for key in ['MAE', 'MAPE', 'SMAPE', 'RMSE', 'NRMSE', 'STD', 'Mean_Error']:
                if 'MAPE' in key or 'SMAPE' in key or 'NRMSE' in key:
                    print(f"  {key:12s}: {metrics_mean[key]:8.4f}% ± {metrics_std[key+'_std']:8.4f}%")
                else:
                    print(f"  {key:12s}: {metrics_mean[key]:8.4f} ± {metrics_std[key+'_std']:8.4f}")

        # 비교 테이블 생성
        if len(results) >= 2:
            comparison_df = self._create_comparison_table(results)
            return comparison_df, detailed_results
        else:
            print("\n⚠ Warning: Need at least 2 methods to compare")
            return None, detailed_results

    def _create_comparison_table(self, results: Dict) -> pd.DataFrame:
        """비교 테이블을 생성합니다."""
        methods = list(results.keys())
        metrics = ['MAE', 'MAPE', 'SMAPE', 'RMSE', 'NRMSE', 'STD', 'Mean_Error']

        data = []
        for metric in metrics:
            row = {'Metric': metric}
            for method in methods:
                mean_val = results[method][metric]
                std_val = results[method][f"{metric}_std"]
                row[f'{method}_mean'] = mean_val
                row[f'{method}_std'] = std_val

            # Adam과 각 MAML 버전 간의 차이 계산
            if 'Adam' in methods:
                adam_mean = results['Adam'][metric]
                for method in methods:
                    if method.startswith('MAML'):
                        maml_mean = results[method][metric]
                        diff = adam_mean - maml_mean
                        diff_pct = (diff / adam_mean * 100) if adam_mean != 0 else 0
                        row[f'Diff_Adam-{method}'] = diff
                        row[f'Diff%_Adam-{method}'] = diff_pct

            data.append(row)

        return pd.DataFrame(data)

    def plot_comparison(self, comparison_df: pd.DataFrame, save_path: Optional[str] = None):
        """비교 결과를 시각화합니다."""
        metrics = comparison_df['Metric'].tolist()

        _fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()

        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            row = comparison_df[comparison_df['Metric'] == metric].iloc[0]

            # Adam과 MAML 값 추출
            adam_mean = row['Adam_mean']
            adam_std = row['Adam_std']
            maml_mean = row['MAML_mean']
            maml_std = row['MAML_std']

            # Bar plot
            x = np.arange(2)
            means = [adam_mean, maml_mean]
            stds = [adam_std, maml_std]

            bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7,
                         color=['#FF6B6B', '#4ECDC4'], edgecolor='black', linewidth=1.5)

            ax.set_xticks(x)
            ax.set_xticklabels(['Adam', 'MAML'], fontsize=12, fontweight='bold')
            ax.set_ylabel('Value', fontsize=11)

            # 제목 설정 (% 표시)
            if 'MAPE' in metric or 'SMAPE' in metric or 'NRMSE' in metric:
                ax.set_title(f'{metric} (%)', fontsize=13, fontweight='bold')
            else:
                ax.set_title(f'{metric}', fontsize=13, fontweight='bold')

            ax.grid(True, alpha=0.3, axis='y')

            # 값 표시
            for bar, mean, std in zip(bars, means, stds):
                height = bar.get_height()
                if 'MAPE' in metric or 'SMAPE' in metric or 'NRMSE' in metric:
                    ax.text(bar.get_x() + bar.get_width()/2., height + std,
                           f'{mean:.3f}%\n±{std:.3f}%',
                           ha='center', va='bottom', fontsize=9)
                else:
                    ax.text(bar.get_x() + bar.get_width()/2., height + std,
                           f'{mean:.4f}\n±{std:.4f}',
                           ha='center', va='bottom', fontsize=9)

        # 마지막 빈 subplot 숨기기 (7개 메트릭이므로)
        axes[-1].set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Plot saved: {save_path}")

        plt.show()

    def plot_distribution_comparison(self, detailed_results: Dict, metric: str = 'MAPE',
                                     save_path: Optional[str] = None):
        """그룹별 메트릭 분포를 비교합니다."""
        _fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        methods = list(detailed_results.keys())
        colors = {'Adam': '#FF6B6B', 'MAML': '#4ECDC4'}

        # Histogram
        ax = axes[0]
        for method in methods:
            df = detailed_results[method]
            ax.hist(df[metric], bins=50, alpha=0.6, label=method,
                   color=colors.get(method, 'gray'), edgecolor='black', linewidth=0.5)

        ax.set_xlabel(f'{metric}', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'{metric} Distribution Comparison', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # Box plot
        ax = axes[1]
        data_to_plot = [detailed_results[method][metric] for method in methods]
        bp = ax.boxplot(data_to_plot, tick_labels=methods, patch_artist=True,
                       showmeans=True, meanline=True)

        # 색상 설정
        for patch, method in zip(bp['boxes'], methods):
            patch.set_facecolor(colors.get(method, 'gray'))
            patch.set_alpha(0.7)

        ax.set_ylabel(f'{metric}', fontsize=12)
        ax.set_title(f'{metric} Box Plot Comparison', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Distribution plot saved: {save_path}")

        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Compare Adam and MAML model results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 특정 cell만 비교
  python compare_adam_maml.py --pdk ASAP7 --topology topology_agnostic --variation-type cell --task extrapolation --cell FAx1

  # 모든 cell 비교
  python compare_adam_maml.py --pdk ASAP7 --topology topology_agnostic --variation-type cell --task extrapolation

  # 여러 cell 비교
  python compare_adam_maml.py --pdk ASAP7 --topology topology_agnostic --variation-type cell --task extrapolation --cell FAx1 AND2x6

  # 결과 저장
  python compare_adam_maml.py --pdk ASAP7 --topology intra_topology --variation-type cell --task interpolation --save

  # JSON config 사용
  python compare_adam_maml.py --config json_configs/comparison_config.json
        """
    )

    parser.add_argument('--config', type=str, default=None,
                       help='Path to JSON configuration file (overrides other arguments)')
    parser.add_argument('--data-dir', type=str,
                       default='/home/tkdgn2907/Deepsets_test/MAML/Projects/pretraining/model_test_code/data_result_npy_directory',
                       help='Directory containing .npy result files')
    parser.add_argument('--pdk', type=str,
                       help='PDK name (e.g., ASAP7, TSMC)')
    parser.add_argument('--topology', type=str,
                       help='Topology type (e.g., intra_topology, topology_agnostic)')
    parser.add_argument('--variation-type', type=str,
                       choices=['cell', 'transition'],
                       help='Variation type: cell or transition')
    parser.add_argument('--task', type=str,
                       help='Task type (e.g., interpolation, extrapolation)')
    parser.add_argument('--cell', type=str, nargs='*', default=None,
                       help='Cell name(s) (e.g., FAx1 AND2x6). If not specified, all matching cells will be processed.')
    parser.add_argument('--num-iterations', type=int, nargs='*', default=None,
                       help='Number of training iterations used (can specify multiple, default: [300000])')
    parser.add_argument('--group-size', type=int, default=61,
                       help='Group size for calculation (default: 61)')
    parser.add_argument('--innerdiv', type=int, nargs='*', default=None,
                       help='MAML inner loop division factor(s) (for MAML file search, can specify multiple)')
    parser.add_argument('--meta', type=int, nargs='*', default=None,
                       help='MAML meta batch size(s) (for MAML file search, can specify multiple)')
    parser.add_argument('--save', action='store_true',
                       help='Save plots to files')
    parser.add_argument('--metric', type=str, default='MAPE',
                       choices=['MAE', 'MAPE', 'SMAPE', 'RMSE', 'NRMSE', 'STD', 'Mean_Error'],
                       help='Metric for distribution comparison (default: MAPE)')

    args = parser.parse_args()

    # Load from JSON config if provided
    if args.config is not None:
        print(f"📋 Loading configuration from: {args.config}")
        import json
        with open(args.config, 'r') as f:
            json_config = json.load(f)

        # Override args with JSON config values
        args.data_dir = json_config.get('data_dir', args.data_dir)
        args.pdk = json_config.get('pdk', args.pdk)
        args.topology = json_config.get('topology', args.topology)
        args.variation_type = json_config.get('variation_type', args.variation_type)
        args.task = json_config.get('task', args.task)
        args.cell = json_config.get('cell', args.cell)
        args.group_size = json_config.get('group_size', args.group_size)

        # num_iterations, innerdiv, meta를 리스트로 처리
        num_iterations_config = json_config.get('num_iterations', args.num_iterations)
        if num_iterations_config is not None:
            args.num_iterations = num_iterations_config if isinstance(num_iterations_config, list) else [num_iterations_config]

        innerdiv_config = json_config.get('innerdiv', args.innerdiv)
        if innerdiv_config is not None:
            args.innerdiv = innerdiv_config if isinstance(innerdiv_config, list) else [innerdiv_config]

        meta_config = json_config.get('meta', args.meta)
        if meta_config is not None:
            args.meta = meta_config if isinstance(meta_config, list) else [meta_config]

        args.save = json_config.get('save', args.save)
        args.metric = json_config.get('metric', args.metric)

        print(f"✅ Configuration loaded successfully")
        print(f"   PDK: {args.pdk}")
        print(f"   Topology: {args.topology}")
        print(f"   Variation type: {args.variation_type}")
        print(f"   Task: {args.task}")
        print()

    # Validate required arguments
    if not args.pdk:
        parser.error("--pdk is required (either via --config or as direct argument)")
    if not args.topology:
        parser.error("--topology is required (either via --config or as direct argument)")
    if not args.variation_type:
        parser.error("--variation-type is required (either via --config or as direct argument)")
    if not args.task:
        parser.error("--task is required (either via --config or as direct argument)")

    # num_iterations 기본값 처리
    if args.num_iterations is None or len(args.num_iterations) == 0:
        args.num_iterations = [300000]

    comparator = ModelComparator(args.data_dir, args.group_size)

    # Cell 리스트 결정
    if args.cell is None or len(args.cell) == 0:
        # 모든 cell 찾기
        print("\n🔍 Searching for all available cells...")
        # find_all_cells는 단일 num_iterations만 받으므로 첫 번째 값 사용
        cells = comparator.find_all_cells(args.pdk, args.topology, args.variation_type, args.task,
                                         args.num_iterations[0] if args.num_iterations else 300000,
                                         args.innerdiv[0] if args.innerdiv and len(args.innerdiv) > 0 else None,
                                         args.meta[0] if args.meta and len(args.meta) > 0 else None)
        if not cells:
            print("❌ No cells found matching the given conditions")
            sys.exit(1)
        cells = sorted(list(cells))
        print(f"✓ Found {len(cells)} cells: {', '.join(cells)}")
    else:
        cells = args.cell

    # 각 cell에 대해 비교 수행
    all_results = {}

    for cell_idx, cell in enumerate(cells, 1):
        print("\n" + "="*80)
        print(f"Processing Cell {cell_idx}/{len(cells)}: {cell}")
        print("="*80)
        print(f"PDK:           {args.pdk}")
        print(f"Topology:      {args.topology}")
        print(f"Variation:     {args.variation_type}")
        print(f"Task:          {args.task}")
        print(f"Cell:          {cell}")
        if args.num_iterations is not None and len(args.num_iterations) > 0:
            print(f"Iterations:    {args.num_iterations}")
        print(f"Group size:    {args.group_size}")
        if args.innerdiv is not None and len(args.innerdiv) > 0:
            print(f"MAML innerdiv: {args.innerdiv}")
        if args.meta is not None and len(args.meta) > 0:
            print(f"MAML meta:     {args.meta}")
        print("="*80)

        # 파일 검색
        print("\nSearching for matching files...")
        files_dict = comparator.find_matching_files(args.pdk, args.topology, args.variation_type, args.task, cell, args.num_iterations,
                                                    args.innerdiv, args.meta)

        # 유효한 파일이 있는지 확인
        valid_files = {k: v for k, v in files_dict.items() if v is not None}
        if len(valid_files) < 1:
            print(f"\n⚠️  Warning: Skipping {cell} - No valid files found")
            continue
        # Adam 파일이 하나라도 있는지 확인 (Adam 또는 Adam_iterXXX)
        has_adam = any(k == 'Adam' or k.startswith('Adam_') for k in valid_files.keys())
        if not has_adam:
            print(f"\n⚠️  Warning: Skipping {cell} - Adam files not found")
            continue
        if not any(k.startswith('MAML') for k in valid_files.keys()):
            print(f"\n⚠️  Warning: Skipping {cell} - No MAML files found")
            continue

        # 비교 수행
        comparison_df, detailed_results = comparator.compare_methods(valid_files)

        if comparison_df is None:
            print(f"\n⚠️  Warning: Skipping {cell} - Comparison failed")
            continue

        # 결과 저장
        all_results[cell] = {
            'comparison_df': comparison_df,
            'detailed_results': detailed_results
        }

        # 각 cell별 결과 출력
        print("\n" + "="*80)
        print(f"COMPARISON RESULTS FOR {cell}")
        print("="*80)
        print(comparison_df.to_string(index=False))
        print("="*80)

        # 해석
        print(f"\n📊 INTERPRETATION FOR {cell}:")
        print("-" * 80)

        # Get all MAML methods
        maml_methods = [m for m in all_results[cell]['comparison_df'].columns if m.startswith('MAML') and m.endswith('_mean')]
        maml_methods = [m.replace('_mean', '') for m in maml_methods]

        for _, row in comparison_df.iterrows():
            metric = row['Metric']
            print(f"\n  {metric}:")

            for maml_method in maml_methods:
                diff_pct_col = f'Diff%_Adam-{maml_method}'
                if diff_pct_col in row:
                    diff_pct = row[diff_pct_col]

                    if abs(diff_pct) < 1:
                        status = "≈ Similar"
                        symbol = "🟡"
                    elif diff_pct > 0:
                        status = f"Adam is {abs(diff_pct):.2f}% worse ({maml_method} is better)"
                        symbol = "🟢"
                    else:
                        status = f"{maml_method} is {abs(diff_pct):.2f}% worse (Adam is better)"
                        symbol = "🔴"

                    print(f"    {symbol} vs {maml_method:30s}: {status}")

        print("-" * 80)

        # 시각화 및 저장
        # if args.save:
        #     save_prefix = f"{args.pdk}_{args.topology}_{args.variation_type}_{args.task}_{cell}_{args.num_iterations}"
        #     plot_path = f"result_summary/{save_prefix}_comparison.png"
        #     dist_path = f"result_summary/{save_prefix}_distribution_{args.metric}.png"

        #     print("\n📈 Generating comparison plots...")
        #     comparator.plot_comparison(comparison_df, save_path=plot_path)

        #     print(f"\n📊 Generating {args.metric} distribution plots...")
        #     comparator.plot_distribution_comparison(detailed_results, metric=args.metric,
        #                                            save_path=dist_path)

            # 결과를 CSV로 저장
            # csv_path = f"{save_prefix}_comparison.csv"
            # comparison_df.to_csv(csv_path, index=False)
            # print(f"✓ Comparison table saved: {csv_path}")

            # 상세 결과도 저장
            # for method, df in detailed_results.items():
            #     detail_path = f"{save_prefix}_{method}_detailed.csv"
            #     df.to_csv(detail_path, index=False)
            #     print(f"✓ {method} detailed results saved: {detail_path}")

    # 전체 요약 출력
    if len(all_results) > 0:
        print("\n" + "="*80)
        print(f"SUMMARY: Processed {len(all_results)} cell(s)")
        print("="*80)

        # 각 메트릭별로 cell 간 비교
        summary_data = []
        for cell, results in all_results.items():
            comparison_df = results['comparison_df']
            row_data = {'Cell': cell}

            # Get all MAML diff columns
            diff_cols = [col for col in comparison_df.columns if col.startswith('Diff%_Adam-MAML')]

            for _, metric_row in comparison_df.iterrows():
                metric = metric_row['Metric']
                for diff_col in diff_cols:
                    maml_method = diff_col.replace('Diff%_Adam-', '')
                    row_data[f'{metric}_{maml_method}'] = metric_row[diff_col]

            summary_data.append(row_data)

        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            print("\nPerformance Difference (%) by Cell (Positive = MAML is better):")
            print(summary_df.to_string(index=False))

        # 전체 요약 저장
        if args.save:
            summary_path = f"result_summary/{args.pdk}_{args.topology}_{args.variation_type}_{args.task}_{args.num_iterations}_all_cells_summary.csv"
            summary_df.to_csv(summary_path, index=False)
            print(f"\n✓ Overall summary saved: {summary_path}")
    else:
        print("\n❌ No valid results to display")


if __name__ == '__main__':
    main()
