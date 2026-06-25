"""
Count per-cell `num_tasks` for the exact set of TSMC cells that
appear in the test buckets used by the paper / rebuttal evaluation
(extracted from rebuttal_metrics_per_cell.csv), pulling the
authoritative `num_tasks` field directly from the per-cell
PyTorch dataset files.

A *task* here = one voltage-to-delay curve under a fixed
(timing arc × related pin × `when` condition × drive variant ×
input slew × output load × process corner × temperature)
combination — i.e., one unique row in a Liberty LUT replicated across
PVT conditions.

Inputs (read-only):
  - dataset_all/GNN_dataset_TSMC/test_by_cell_stage_aware/<CELL>.pth
  - dataset_all/GNN_dataset_TSMC/test_by_transition_stage_aware/<CELL>.pth

Outputs (this dir):
  - tsmc_per_cell_task_counts.csv
  - PER_CELL_TASK_SUMMARY.txt
"""
from pathlib import Path
import torch
import pandas as pd

ROOT = Path('/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/GNN_dataset_TSMC')
HERE = Path(__file__).resolve().parent

TOPOLOGY_AGNOSTIC = [
    'AO211D0BWP30P140', 'AO211D1BWP30P140',
    'AO21D0BWP30P140',  'AO21D1BWP30P140',
    'IAO21D0BWP30P140', 'IAO21D1BWP30P140',
    'OA211D0BWP30P140', 'OA211D1BWP30P140',
    'OA21D0BWP30P140',  'OA21D1BWP30P140',
    'IOA21D0BWP30P140', 'IOA21D1BWP30P140',
    'FA1D0BWP30P140',   'HA1D0BWP30P140',
    'DFCNQD1BWP30P140', 'SDFSNQD0BWP30P140',
]
INTRA_TOPOLOGY = [
    'AN4D0BWP30P140',  'ND3D0BWP30P140',
    'NR3D1BWP30P140',  'OR4D0BWP30P140',
    'XNR3D1BWP30P140', 'XOR3D1BWP30P140',
]


def load_num_tasks(subdir, cell):
    f = ROOT / subdir / f'{cell}.pth'
    if not f.exists():
        return None
    d = torch.load(f, map_location='cpu', weights_only=False)
    return int(d.get('num_tasks', -1))


rows = []
for scenario, cells in (('topology_agnostic', TOPOLOGY_AGNOSTIC),
                        ('intra_topology',    INTRA_TOPOLOGY)):
    for c in cells:
        nt_cell = load_num_tasks('test_by_cell_stage_aware', c)
        nt_tran = load_num_tasks('test_by_transition_stage_aware', c)
        rows.append({
            'scenario': scenario,
            'cell': c.replace('BWP30P140', ''),
            'tasks_cell_delay': nt_cell,
            'tasks_transition': nt_tran,
        })

df = pd.DataFrame(rows)
out_csv = HERE / 'tsmc_per_cell_task_counts.csv'
df.to_csv(out_csv, index=False)
print('Saved:', out_csv)
print()
print('=== Per-cell task counts (TSMC test set) ===')
print(df.to_string(index=False))
print()

# Aggregate stats
lines = []
lines.append('=' * 72 + '\n')
lines.append('Per-cell task counts for the TSMC test set (verified via\n')
lines.append('`num_tasks` field of the per-cell .pth dataset files).\n')
lines.append('=' * 72 + '\n\n')

for scenario in ('topology_agnostic', 'intra_topology'):
    sub = df[df.scenario == scenario]
    nc = len(sub)
    for tgt, col in (('cell_delay', 'tasks_cell_delay'),
                     ('transition', 'tasks_transition')):
        vals = sub[col].dropna()
        if not len(vals):
            continue
        lines.append(f'  {scenario:<18} / {tgt:<10} :  '
                     f'{nc} cells, '
                     f'mean = {vals.mean():>9,.1f} tasks/cell, '
                     f'median = {int(vals.median()):>7,} tasks/cell, '
                     f'min = {int(vals.min()):>6,}, '
                     f'max = {int(vals.max()):>7,}, '
                     f'total = {int(vals.sum()):>9,}\n')
    lines.append('\n')

lines.append('Per-cell table:\n')
for _, r in df.iterrows():
    lines.append(f'  {r["scenario"]:<18} {r["cell"]:<8} '
                 f'cell-delay tasks = {r["tasks_cell_delay"]:>6,}, '
                 f'transition tasks = {r["tasks_transition"]:>6,}\n')

out_txt = HERE / 'PER_CELL_TASK_SUMMARY.txt'
out_txt.write_text(''.join(lines))
print('Saved:', out_txt)
print()
print(''.join(lines))
