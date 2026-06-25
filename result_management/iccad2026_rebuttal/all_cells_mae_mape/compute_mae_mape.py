#!/usr/bin/env python3
"""
Rebuttal P0-1 (B-Q2 intuitive metrics) + P0-4 (pessimism distribution) — driver.

Reuses the notebook's loader (load_dfs_from_npy.py) so the (PDK × experiment ×
data_type × mode) buckets are *identical* to compare_topology.ipynb. Adds
MAE_ps_scaled, MAPE_pct, and signed-error / pessimism columns per cell.

Outputs:
  rebuttal_metrics_per_cell.csv     one row per cell × config
  rebuttal_metrics_geomean.csv      bucket-level geomean (matches notebook layout)
  pessimism_overall.json            summary pessimism stats for A-Q2
"""
import os, sys, json
import numpy as np
import pandas as pd
from scipy.stats import gmean

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from load_dfs_from_npy import (
    load_gcn_results, load_mlp_results, add_engineering_metrics_to_df,
    CELL_FILTER,
)
from types import SimpleNamespace

# Paths mirror notebook cell 2 (compare_topology.ipynb) — keep consistent.
args = SimpleNamespace(
    gcn_dir='/home/tkdgn2907/Deepsets_test/MAML/Projects/pretraining/model_test_code/gnn/data_result_npy_directory_final',
    mlp_maml_dir='/home/tkdgn2907/Deepsets_test/MAML/Projects/pretraining/model_test_code/MLP/data_result_npy_directory_maml',
    aadam_dir='/home/tkdgn2907/Deepsets_test/MAML/Projects/pretraining/model_test_code/MLP/data_result_npy_directory_baseline',
    arch='conv64x2_fc256x2',
    aadam_iter=300000,
    aadam_adapt_method='adam',
    mlp_maml_iter=300000,
    mlp_maml_innerdiv=100,
    mlp_maml_meta=32,
    mlp_maml_layer=40,
    gcn_iter=300000,
    sa_innerdiv=10,
    sa_meta=16,
    gcn_adapt_method='selective_adam',
    gnn_model_type='GCN',
    gcn_baseline_iter=300000,
    vdd_only=False,
    relpin=False,
    exclude_parasitic=False,
)

OUT_DIR = HERE
os.makedirs(OUT_DIR, exist_ok=True)

gcn_adapt_suffix = '_adam' if args.gcn_adapt_method == 'adam' else ''
aadam_adapt_suffix = '_adam' if args.aadam_adapt_method == 'adam' else ''

print('=' * 80)
print('Loading dataframes (mirrors compare_topology.ipynb cell 3)...')
print('=' * 80)

per_cell_frames = []
for pdk in ['ASAP7', 'TSMC']:
    print(f'\n--- {pdk} ---')

    # GCN MAML + baseline
    pdk_gcn_df = load_gcn_results(
        args.gcn_dir, arch_filter=args.arch,
        gcn_adapt_suffix=gcn_adapt_suffix,
        cells_filter=CELL_FILTER if pdk == 'ASAP7' else None,
    )
    if pdk_gcn_df is None or len(pdk_gcn_df) == 0:
        print(f'  GCN: no results'); continue
    pdk_gcn_df = pdk_gcn_df[pdk_gcn_df['prefix'] == pdk]
    if args.gnn_model_type:
        pdk_gcn_df = pdk_gcn_df[pdk_gcn_df['gnn_model_type'] == args.gnn_model_type]

    # Split MAML / baseline
    gcn_maml = pdk_gcn_df[pdk_gcn_df['training_type'] == 'maml']
    if args.gcn_iter:
        gcn_maml = gcn_maml[gcn_maml['iterations'] == args.gcn_iter]
    # MAML — match notebook filtering: FG accepts any, SA needs innerdiv+meta
    fg_mask = (gcn_maml['graph_mode'] == 'full_graph')
    sa_mask = ((gcn_maml['graph_mode'] == 'stage_aware') &
               (gcn_maml['innerdiv'] == args.sa_innerdiv) &
               (gcn_maml['meta'] == args.sa_meta))
    gcn_maml = gcn_maml[fg_mask | sa_mask]
    print(f'  GCN MAML rows: {len(gcn_maml)}')

    gcn_base = pdk_gcn_df[pdk_gcn_df['training_type'] == 'baseline']
    if args.gcn_baseline_iter:
        gcn_base = gcn_base[gcn_base['iterations'] == args.gcn_baseline_iter]
    # Match notebook cell 4: baseline uses full_graph + mean pooling
    if 'graph_mode' in gcn_base.columns:
        gcn_base = gcn_base[gcn_base['graph_mode'] == 'full_graph']
    if 'pooling' in gcn_base.columns:
        gcn_base = gcn_base[gcn_base['pooling'] == 'mean']
    print(f'  GCN baseline rows: {len(gcn_base)}')

    # GCN_MAML — match notebook cell 4: pooling depends on data_type
    # (transition → output, cell → mean), graph_mode == stage_aware
    if 'graph_mode' in gcn_maml.columns:
        gcn_maml = gcn_maml[gcn_maml['graph_mode'] == 'stage_aware']
    if 'pooling' in gcn_maml.columns and 'data_type' in gcn_maml.columns:
        gcn_maml = gcn_maml[
            ((gcn_maml['data_type'] == 'transition') & (gcn_maml['pooling'] == 'output')) |
            ((gcn_maml['data_type'] == 'cell') & (gcn_maml['pooling'] == 'mean'))
        ]
    print(f'  GCN MAML rows (after pooling/graph_mode filter): {len(gcn_maml)}')

    # MLP — loader returns AADAM + MLP_MAML rows together; split + filter per
    # notebook cell 3 conventions before attaching engineering metrics.
    pdk_mlp_df = load_mlp_results(
        args.mlp_maml_dir, aadam_dir=args.aadam_dir,
        prefix_filter=pdk,
        aadam_adapt_suffix=aadam_adapt_suffix,
        cells_filter=CELL_FILTER if pdk == 'ASAP7' else None,
    )
    if pdk_mlp_df is not None and len(pdk_mlp_df) > 0 and 'model_type' in pdk_mlp_df.columns:
        # AADAM: iter==aadam_iter
        aadam_df = pdk_mlp_df[
            (pdk_mlp_df['model_type'] == 'AADAM') &
            (pdk_mlp_df['iterations'] == args.aadam_iter)
        ]
        # MLP_MAML: innerdiv/meta/layer/iter all must match
        mlp_maml_cols = ['innerdiv', 'meta', 'layer_length', 'iterations']
        if all(c in pdk_mlp_df.columns for c in mlp_maml_cols):
            mlp_maml_df = pdk_mlp_df[
                (pdk_mlp_df['model_type'] == 'MLP_MAML') &
                (pdk_mlp_df['innerdiv'] == args.mlp_maml_innerdiv) &
                (pdk_mlp_df['meta'] == args.mlp_maml_meta) &
                (pdk_mlp_df['layer_length'] == args.mlp_maml_layer) &
                (pdk_mlp_df['iterations'] == args.mlp_maml_iter)
            ]
        else:
            mlp_maml_df = pdk_mlp_df[pdk_mlp_df['model_type'] == 'MLP_MAML']
    else:
        aadam_df = None
        mlp_maml_df = None
    print(f'  AADAM rows: {len(aadam_df) if aadam_df is not None else 0}')
    print(f'  MLP_MAML rows: {len(mlp_maml_df) if mlp_maml_df is not None else 0}')

    # Append engineering metrics (MAE_scaled, MAPE_pct, pessimism)
    for label, df in [
        ('GCN_MAML', gcn_maml),
        ('GCN_Baseline', gcn_base),
        ('AADAM', aadam_df),
        ('MLP_MAML', mlp_maml_df),
    ]:
        if df is None or len(df) == 0:
            continue
        mlp_dirs = (args.mlp_maml_dir, args.aadam_dir)
        df_with_eng = add_engineering_metrics_to_df(df, args.gcn_dir, mlp_dirs)
        df_with_eng['pdk'] = pdk
        df_with_eng['source_label'] = label
        per_cell_frames.append(df_with_eng)

if not per_cell_frames:
    print('No data collected. Exiting.')
    sys.exit(1)

per_cell = pd.concat(per_cell_frames, ignore_index=True)
print(f'\nTotal per-cell rows collected: {len(per_cell)}')
per_cell_csv = os.path.join(OUT_DIR, 'rebuttal_metrics_per_cell.csv')
per_cell.to_csv(per_cell_csv, index=False)
print(f'  saved: {per_cell_csv}')

# ============================================================================
# Bucket-level geomean — mirrors notebook table layout
# ============================================================================
print('\n' + '=' * 80)
print('Bucket-level geomean (PDK × experiment × data_type × mode × source_label)')
print('=' * 80)

needed = ['pdk', 'source_label', 'experiment', 'data_type', 'mode']
have = [c for c in needed if c in per_cell.columns]
print(f'  group keys: {have}')

if 'topology' in per_cell.columns:
    if 'experiment' not in per_cell.columns:
        per_cell['experiment'] = per_cell['topology']
    else:
        per_cell['experiment'] = per_cell['experiment'].fillna(per_cell['topology'])
    have = [c for c in needed if c in per_cell.columns]

# For each bucket × each metric, compute geomean over cells.
metric_cols = ['NRMSE', 'RMSE', 'MAE', 'MAE_scaled', 'MAPE_pct',
               'UnderPred_frac', 'PessSafe_p50', 'PessSafe_p95',
               'MaxUnderPred']
metric_cols = [m for m in metric_cols if m in per_cell.columns]

rows = []
for keys, grp in per_cell.groupby(have):
    d = dict(zip(have, keys))
    d['n_cells'] = len(grp)
    for c in metric_cols:
        vals = grp[c].dropna()
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            d[c + '_geomean'] = np.nan
            d[c + '_mean'] = np.nan
            continue
        d[c + '_mean'] = float(vals.mean())
        # geomean needs positive values for NRMSE/MAE/MAPE; for signed pessimism use mean
        if c in ('NRMSE', 'RMSE', 'MAE', 'MAE_scaled', 'MAPE_pct'):
            positive = vals[vals > 0]
            d[c + '_geomean'] = float(gmean(positive)) if len(positive) > 0 else np.nan
        else:
            d[c + '_geomean'] = float(vals.mean())
    rows.append(d)

geomean_df = pd.DataFrame(rows)
geomean_csv = os.path.join(OUT_DIR, 'rebuttal_metrics_geomean.csv')
geomean_df.to_csv(geomean_csv, index=False)
print(f'  saved: {geomean_csv}')

# Print headline table
print('\n--- Headline (300k iter, mode=extrapolation, data_type=cell) ---')
view = geomean_df[(geomean_df.get('mode', '') == 'extrapolation') &
                  (geomean_df.get('data_type', '') == 'cell')]
if len(view) > 0:
    show = view[['pdk', 'source_label', 'experiment', 'n_cells',
                 'NRMSE_geomean', 'MAE_scaled_geomean', 'MAPE_pct_geomean',
                 'UnderPred_frac_mean', 'PessSafe_p95_mean',
                 'MaxUnderPred_mean']].copy()
    show = show.sort_values(['pdk', 'experiment', 'source_label'])
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 160)
    print(show.to_string(index=False))
else:
    print('  (empty — check df columns)')

# ============================================================================
# Pessimism summary for A-Q2 — overall stats per (PDK, source_label)
# ============================================================================
pessimism = {}
for (pdk, src), grp in per_cell.groupby(['pdk', 'source_label']):
    sub = grp.dropna(subset=['UnderPred_frac'])
    if len(sub) == 0: continue
    pessimism[f'{pdk}/{src}'] = {
        'n_cells': int(len(sub)),
        'mean_underpred_frac': float(sub['UnderPred_frac'].mean()),
        'max_underpred_frac':  float(sub['UnderPred_frac'].max()),
        'mean_pess_p95':       float(sub['PessSafe_p95'].mean()),
        'mean_max_underpred':  float(sub['MaxUnderPred'].mean()),
        'max_max_underpred':   float(sub['MaxUnderPred'].max()),
    }
with open(os.path.join(OUT_DIR, 'pessimism_overall.json'), 'w') as f:
    json.dump(pessimism, f, indent=2)
print(f'\n  saved: pessimism_overall.json')

print('\n=' * 80)
print('DONE.')
