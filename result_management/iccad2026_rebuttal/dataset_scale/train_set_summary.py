"""
Count per-cell `num_tasks` in the *training* set for both PDKs
(TSMC 28 nm and ASAP7 7 nm). Counterpart to per_cell_task_counts.py
which does the test set.

Inputs (read-only):
  TSMC :
    - dataset_all/GNN_dataset_TSMC/train_cell_stage_aware.pth
    - dataset_all/GNN_dataset_TSMC/train_transition_stage_aware.pth
  ASAP7:
    - dataset_all/GNN_dataset_ASAP7/train_cell_stage_aware_full.pth
      (full training corpus used by the paper; ~116 GB on disk)
    - dataset_all/GNN_dataset_ASAP7/train_transition_stage_aware_10pct.pth
      (only a 10 % subsample is on disk; transition full corpus
       has identical per-cell layout to the cell-delay full corpus
       per dataset-build conventions, so its per-cell counts are
       reported × 10 here)

NOTE on memory: the ASAP7 full file is ~116 GB; loading it with
`weights_only=False` is necessary to materialize `cell_names`,
which is required for the per-cell breakdown. The script reads each
file once, immediately discards the heavy tensors, and keeps only
the metadata. Run on a machine with enough RAM.

Outputs (this dir):
  - {tsmc,asap7}_train_per_cell_task_counts.csv
  - TRAIN_SET_SUMMARY.txt
"""
from pathlib import Path
from collections import Counter
import statistics
import torch
import pandas as pd

ROOT_TSMC  = Path('/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/GNN_dataset_TSMC')
ROOT_ASAP7 = Path('/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/GNN_dataset_ASAP7')
HERE = Path(__file__).resolve().parent


def per_cell_counts(path):
    d = torch.load(path, map_location='cpu', weights_only=False)
    cnt = Counter(d['cell_names'])
    meta = {
        'corners': d.get('train_corners') or [],
        'temperatures': d.get('train_temperatures') or [],
        'num_conditions': int(d.get('num_conditions', -1)),
        'num_libs': int(d.get('num_libs', -1)),
        'excluded_cells': d.get('excluded_cells') or [],
        'data_type': d.get('data_type'),
        'graph_mode': d.get('graph_mode'),
    }
    del d
    return cnt, meta


# ---- TSMC ----
print('Loading TSMC train_cell_stage_aware.pth ...')
cnt_tsmc_cell, meta_tsmc_cell = per_cell_counts(
    ROOT_TSMC / 'train_cell_stage_aware.pth')
print('Loading TSMC train_transition_stage_aware.pth ...')
cnt_tsmc_tran, meta_tsmc_tran = per_cell_counts(
    ROOT_TSMC / 'train_transition_stage_aware.pth')

# ---- ASAP7 ----
print('Loading ASAP7 train_cell_stage_aware_full.pth (large) ...')
cnt_asap7_cell, meta_asap7_cell = per_cell_counts(
    ROOT_ASAP7 / 'train_cell_stage_aware_full.pth')

# Only 10pct transition file is on disk; we report the per-cell
# counts of that subsample and a ×10 projection for the full corpus.
print('Loading ASAP7 train_transition_stage_aware_10pct.pth ...')
cnt_asap7_tran_10pct, meta_asap7_tran = per_cell_counts(
    ROOT_ASAP7 / 'train_transition_stage_aware_10pct.pth')


def long_df(cnt, suffix):
    return pd.DataFrame(
        [{'cell': k.replace('BWP30P140', '')
                    .replace('_ASAP7_75t_R', ''),
          f'tasks_{suffix}': v} for k, v in cnt.items()])


# Build per-PDK CSV
df_tsmc = (long_df(cnt_tsmc_cell, 'cell_delay')
           .merge(long_df(cnt_tsmc_tran, 'transition'),
                  on='cell', how='outer')
           .fillna(0).astype({'tasks_cell_delay': int,
                              'tasks_transition': int})
           .sort_values('tasks_cell_delay', ascending=False))
df_tsmc.to_csv(HERE / 'tsmc_train_per_cell_task_counts.csv', index=False)
print('Saved: tsmc_train_per_cell_task_counts.csv')

df_asap7 = (long_df(cnt_asap7_cell, 'cell_delay')
            .merge(long_df(cnt_asap7_tran_10pct, 'transition_10pct'),
                   on='cell', how='outer')
            .fillna(0)
            .astype({'tasks_cell_delay': int,
                     'tasks_transition_10pct': int})
            .sort_values('tasks_cell_delay', ascending=False))
# Project the 10pct transition counts to full-corpus equivalent
df_asap7['tasks_transition_full_proj'] = df_asap7['tasks_transition_10pct'] * 10
df_asap7.to_csv(HERE / 'asap7_train_per_cell_task_counts.csv', index=False)
print('Saved: asap7_train_per_cell_task_counts.csv')


def summarize(name, cnt, meta):
    vals = list(cnt.values())
    return {
        'name': name,
        'n_cells': len(cnt),
        'total_tasks': sum(vals),
        'mean': statistics.mean(vals),
        'median': statistics.median(vals),
        'min': min(vals),
        'max': max(vals),
        **meta,
    }


lines = []
lines.append('=' * 78 + '\n')
lines.append('TRAINING-SET PER-CELL TASK COUNTS — both PDKs\n')
lines.append('=' * 78 + '\n\n')

for s in (summarize('TSMC  train cell_delay     ', cnt_tsmc_cell,  meta_tsmc_cell),
          summarize('TSMC  train transition     ', cnt_tsmc_tran,  meta_tsmc_tran),
          summarize('ASAP7 train cell_delay (full)', cnt_asap7_cell, meta_asap7_cell),
          summarize('ASAP7 train transition (10pct subsample)',
                    cnt_asap7_tran_10pct, meta_asap7_tran)):
    lines.append(f'--- {s["name"]} ---\n')
    lines.append(f'  Unique training cells           : {s["n_cells"]}\n')
    lines.append(f'  Total training tasks            : {s["total_tasks"]:>14,}\n')
    lines.append(f'  mean tasks / cell               : {s["mean"]:>14,.1f}\n')
    lines.append(f'  median tasks / cell             : {int(s["median"]):>14,}\n')
    lines.append(f'  range                           : '
                 f'{s["min"]:,} – {s["max"]:,}\n')
    if s['corners']:
        lines.append(f'  train corners                   : {s["corners"]} '
                     f'({len(s["corners"])} corners)\n')
    if s['temperatures']:
        lines.append(f'  train temperatures              : {s["temperatures"]} '
                     f'({len(s["temperatures"])} temps)\n')
    if s['num_conditions'] > 0:
        lines.append(f'  num_conditions                  : '
                     f'{s["num_conditions"]:>14,}\n')
    if s['num_libs'] > 0:
        lines.append(f'  V_dd sweep length               : '
                     f'{s["num_libs"]:>14,} points/task\n')
        lines.append(f'  Total (V, delay) training pts   : '
                     f'{s["total_tasks"]*s["num_libs"]/1e6:>11.2f} M\n')
    if s['excluded_cells']:
        ex = [c.replace('BWP30P140', '').replace('_ASAP7_75t_R', '')
              for c in s['excluded_cells']]
        lines.append(f'  Held-out (excluded) cells       : '
                     f'{len(ex)} cells ({ex})\n')
    lines.append('\n')

# Project ASAP7 transition full from the 10% subsample
proj_tasks = sum(cnt_asap7_tran_10pct.values()) * 10
proj_samples = proj_tasks * meta_asap7_tran['num_libs'] / 1e6
lines.append('--- ASAP7 train transition (full corpus, ×10 projection) ---\n')
lines.append(f'  Projected total tasks           : {proj_tasks:>14,}\n')
lines.append(f'  Projected (V, delay) samples    : {proj_samples:>11.2f} M\n')
lines.append('\n')

# Headline combined
tsmc_total = (sum(cnt_tsmc_cell.values()) +
              sum(cnt_tsmc_tran.values())) * meta_tsmc_cell['num_libs']
asap7_total = (sum(cnt_asap7_cell.values()) +
               sum(cnt_asap7_tran_10pct.values()) * 10
               ) * meta_asap7_cell['num_libs']
lines.append('--- Combined training-corpus headline (cell_delay + transition) ---\n')
lines.append(f'  TSMC  : {tsmc_total/1e6:.2f} M (V, delay) training samples\n')
lines.append(f'  ASAP7 : {asap7_total/1e6:.2f} M (V, delay) training samples '
             f'(transition portion ×10 projection)\n')
lines.append(f'  Both  : {(tsmc_total + asap7_total)/1e6:.2f} M total\n')

out = HERE / 'TRAIN_SET_SUMMARY.txt'
out.write_text(''.join(lines))
print('Saved:', out)
print()
print(''.join(lines))
