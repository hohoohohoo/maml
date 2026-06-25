"""
Per-task NRMSE / MAE for the two sequential cells held out from training:
  - DFCNQD1BWP30P140  (D-FF with async clear; C2MOS)
  - SDFSNQD0BWP30P140 (Scan D-FF with async set; C2MOS)

Each *task* is one voltage-to-delay curve under a fixed (cell, timing
arc, input slew, output load, P-T) condition, sampled at 61 voltage
points (group_size=61). NRMSE is computed per task using that task's
own actual-curve range (max - min over the 61 points), then aggregated
across tasks (mean and geomean). This matches the convention of
TAMEL's calculate_metrics() and Fig. 4 evaluation.

We sweep two graph constructions x two training schemes for both cells,
both targets (cell delay / output transition), and both regimes
(interpolation / extrapolation) — a 32-condition grid.

Outputs (in this script's directory):
  seq_cell_metrics.csv             long-form per-task metrics
  seq_cell_nrmse_pivot_mean.csv    pivot: per-task mean NRMSE (%)
  seq_cell_nrmse_pivot_geomean.csv pivot: per-task geomean NRMSE (%)
"""
from pathlib import Path
import numpy as np
import pandas as pd

GCN_DIR = Path('/home/tkdgn2907/Deepsets_test/MAML/Projects/'
               'pretraining/model_test_code/gnn/data_result_npy_directory_final')
OUT_DIR = Path(__file__).resolve().parent  # = seq_cell/

CELLS = {
    'DFCNQD1BWP30P140':  'D-FF (DFCNQD1)',
    'SDFSNQD0BWP30P140': 'Scan D-FF (SDFSNQD0)',
}


def fname(cell, target, graph, regime, train):
    """Reproduce the exact filename conventions used by the model_test pipeline.

    For cell delay, full_graph variants carry `_vddonly_relpin_` and an
    `_adam_` suffix on the baseline run; stage_aware variants do not.
    For transition, both use `_pooloutput_filtered_` and full_graph
    baseline additionally has `_adam`.
    """
    base = f'TSMC_GCN_topology_agnostic_{cell}_{target}_{graph}_{regime}_'
    if target == 'cell':
        if graph == 'full_graph':
            if train == 'baseline':
                suffix = ('baseline_iter300000_conv64x2_fc256x2_'
                          'filtered_adam_vddonly_relpin')
            else:
                suffix = ('maml_innerdiv10_meta16_iter300000_inner1_'
                          'conv64x2_fc256x2_filtered_vddonly_relpin')
        else:  # stage_aware
            if train == 'baseline':
                suffix = 'baseline_iter300000_conv64x2_fc256x2_filtered'
            else:
                suffix = ('maml_innerdiv10_meta16_iter300000_inner1_'
                          'conv64x2_fc256x2_filtered')
    else:  # transition
        if train == 'baseline':
            if graph == 'full_graph':
                suffix = ('baseline_iter300000_conv64x2_fc256x2_'
                          'pooloutput_filtered_adam')
            else:
                suffix = ('baseline_iter300000_conv64x2_fc256x2_'
                          'pooloutput_filtered')
        else:
            suffix = ('maml_innerdiv10_meta16_iter300000_inner1_'
                      'conv64x2_fc256x2_pooloutput_filtered')
    return base + suffix


def metrics(pred, act, group_size=61):
    """Per-task NRMSE/MAE consistent with TAMEL evaluation."""
    pred = pred.flatten().astype(np.float64)
    act = act.flatten().astype(np.float64)
    mask = np.isfinite(pred) & np.isfinite(act)
    pred, act = pred[mask], act[mask]
    n_tasks = len(pred) // group_size
    if n_tasks == 0:
        return dict(status='TOO_SHORT', N=int(len(act)))
    pred = pred[:n_tasks * group_size].reshape(n_tasks, group_size)
    act = act[:n_tasks * group_size].reshape(n_tasks, group_size)
    err = pred - act
    rmse_t = np.sqrt(np.mean(err ** 2, axis=1))
    rng_t = np.max(act, axis=1) - np.min(act, axis=1)
    rng_t = np.where(rng_t > 0, rng_t, np.nan)
    nrmse_t = (rmse_t / rng_t) * 100.0
    mae_t = np.mean(np.abs(err), axis=1)
    ok = np.isfinite(nrmse_t) & (nrmse_t > 0)
    if ok.sum() == 0:
        return dict(status='NO_VALID_TASKS', n_tasks=int(n_tasks))
    nrmse_pos = nrmse_t[ok]
    return dict(
        n_tasks=int(n_tasks),
        NRMSE_pct_mean=float(np.nanmean(nrmse_t)),
        NRMSE_pct_geomean=float(np.exp(np.mean(np.log(nrmse_pos)))),
        NRMSE_pct_median=float(np.nanmedian(nrmse_t)),
        RMSE_mean=float(np.mean(rmse_t)),
        MAE_mean=float(np.mean(mae_t)),
        act_min=float(act.min()), act_max=float(act.max()),
    )


rows = []
for cell_key, cell_label in CELLS.items():
    for target in ('cell', 'transition'):
        for regime in ('interpolation', 'extrapolation'):
            for graph in ('full_graph', 'stage_aware'):
                for train in ('baseline', 'maml'):
                    base = fname(cell_key, target, graph, regime, train)
                    p_file = GCN_DIR / f'{base}_pred.npy'
                    a_file = GCN_DIR / f'{base}_act.npy'
                    if not p_file.exists() or not a_file.exists():
                        rows.append(dict(cell=cell_label, target=target,
                                         regime=regime, graph=graph,
                                         train=train, status='MISSING',
                                         file=p_file.name))
                        continue
                    p = np.load(p_file)
                    a = np.load(a_file)
                    m = metrics(p, a, group_size=61)
                    rows.append(dict(cell=cell_label, target=target,
                                     regime=regime, graph=graph,
                                     train=train, status='OK', **m))

df = pd.DataFrame(rows)
df_ok = df[df['status'] == 'OK'].copy()
df.to_csv(OUT_DIR / 'seq_cell_metrics.csv', index=False)
print(f'Saved: {OUT_DIR / "seq_cell_metrics.csv"}')

piv_mean = (df_ok.pivot_table(index=['cell', 'target', 'regime'],
                              columns=['graph', 'train'],
                              values='NRMSE_pct_mean')
            .round(3))
piv_gmean = (df_ok.pivot_table(index=['cell', 'target', 'regime'],
                               columns=['graph', 'train'],
                               values='NRMSE_pct_geomean')
             .round(3))
piv_mean.to_csv(OUT_DIR / 'seq_cell_nrmse_pivot_mean.csv')
piv_gmean.to_csv(OUT_DIR / 'seq_cell_nrmse_pivot_geomean.csv')
print('Saved: seq_cell_nrmse_pivot_mean.csv, seq_cell_nrmse_pivot_geomean.csv')

print('\n=== Per-task mean NRMSE (%) — sequential cells ===')
print(piv_mean.to_string())
print('\n=== Per-task geomean NRMSE (%) — sequential cells ===')
print(piv_gmean.to_string())
print('\n=== task counts per (cell, target, regime) ===')
print(df_ok.groupby(['cell', 'target', 'regime'])['n_tasks'].first().to_string())

miss = df[df['status'] == 'MISSING']
if not miss.empty:
    print('\n=== MISSING files ===')
    print(miss[['cell', 'target', 'regime', 'graph', 'train', 'file']]
          .to_string(index=False))
