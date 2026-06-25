"""
Per-task NRMSE figures for the sequential-cell rebuttal experiment.

Inputs : seq_cell_metrics.csv (produced by extract_seq_cell_results.py)
Outputs: figures/seq_cell_nrmse_grouped.png      (2x2 grid grouped bars)
         figures/seq_cell_improvement_factors.png (full_graph / stage_aware ratios)
         figures/seq_cell_summary_geomean.txt     (text summary)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent  # = seq_cell/
FIG_DIR = BASE / 'figures'
FIG_DIR.mkdir(exist_ok=True)

df = pd.read_csv(BASE / 'seq_cell_metrics.csv')
df = df[df['status'] == 'OK'].copy()
# Primary metric: per-task mean NRMSE (group_size=61), matches the
# notebook's calculate_metrics convention.
df['NRMSE_pct'] = df['NRMSE_pct_mean']

CELLS = ['D-FF (DFCNQD1)', 'Scan D-FF (SDFSNQD0)']
TARGETS = ['cell', 'transition']
TARGET_LABEL = {'cell': 'Cell Delay', 'transition': 'Output Transition'}
REGIMES = ['interpolation', 'extrapolation']

COLORS = {
    ('full_graph',  'baseline'): '#c0c0c0',
    ('full_graph',  'maml'):     '#7f8c8d',
    ('stage_aware', 'baseline'): '#5dade2',
    ('stage_aware', 'maml'):     '#2471a3',
}
LABEL = {
    ('full_graph',  'baseline'): 'full_graph baseline',
    ('full_graph',  'maml'):     'full_graph + MAML',
    ('stage_aware', 'baseline'): 'stage_aware baseline',
    ('stage_aware', 'maml'):     'stage_aware + MAML',
}

# ----- Figure 1: 2x2 grid grouped bars -----
fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharey=False)
bar_groups = [('full_graph',  'baseline'), ('full_graph',  'maml'),
              ('stage_aware', 'baseline'), ('stage_aware', 'maml')]
width = 0.18
x_centers = np.arange(len(REGIMES))
offsets = (np.arange(len(bar_groups)) - (len(bar_groups) - 1) / 2.0) * width

for r, cell in enumerate(CELLS):
    for c, target in enumerate(TARGETS):
        ax = axes[r, c]
        sub = df[(df['cell'] == cell) & (df['target'] == target)]
        for i, (g, t) in enumerate(bar_groups):
            vals = []
            for reg in REGIMES:
                row = sub[(sub['graph'] == g) & (sub['train'] == t) &
                          (sub['regime'] == reg)]
                vals.append(row['NRMSE_pct'].values[0] if len(row) else np.nan)
            bars = ax.bar(x_centers + offsets[i], vals, width,
                          color=COLORS[(g, t)], label=LABEL[(g, t)],
                          edgecolor='black', linewidth=0.4)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        v + 0.02 * max(0.5, max(vals)),
                        f'{v:.2f}', ha='center', va='bottom', fontsize=7.4)
        ax.set_xticks(x_centers)
        ax.set_xticklabels(['Interpolation', 'Extrapolation'])
        ax.set_ylabel('NRMSE (%)')
        ax.set_title(f'{cell} — {TARGET_LABEL[target]}', fontsize=10.5)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        ymax = sub['NRMSE_pct'].max()
        ax.set_ylim(0, ymax * 1.22)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center',
           bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False, fontsize=9.5)
fig.suptitle('Sequential-cell per-task NRMSE: current-path (stage_aware) vs full netlist (full_graph)',
             y=1.06, fontsize=12.5, fontweight='bold')
plt.tight_layout()
out1 = FIG_DIR / 'seq_cell_nrmse_grouped.png'
plt.savefig(out1, dpi=200, bbox_inches='tight')
plt.close()
print(f'Saved: {out1}')

# ----- Figure 2: improvement factors (full_graph / stage_aware) -----
records = []
for cell in CELLS:
    for target in TARGETS:
        for reg in REGIMES:
            for train in ('baseline', 'maml'):
                fg = df[(df.cell == cell) & (df.target == target) &
                        (df.regime == reg) & (df.train == train) &
                        (df.graph == 'full_graph')]['NRMSE_pct'].values[0]
                sa = df[(df.cell == cell) & (df.target == target) &
                        (df.regime == reg) & (df.train == train) &
                        (df.graph == 'stage_aware')]['NRMSE_pct'].values[0]
                records.append({'cell': cell, 'target': target, 'regime': reg,
                                'train': train, 'improvement': fg / sa})
imp = pd.DataFrame(records)

fig2, ax2 = plt.subplots(figsize=(11, 5))
labels_x, imp_base, imp_maml = [], [], []
for cell in CELLS:
    for target in TARGETS:
        for reg in REGIMES:
            labels_x.append(
                f'{cell.split(" ")[0]}\n{TARGET_LABEL[target]}\n{reg[:6]}')
            imp_base.append(imp[(imp.cell == cell) & (imp.target == target) &
                                (imp.regime == reg) & (imp.train == 'baseline')
                                ]['improvement'].values[0])
            imp_maml.append(imp[(imp.cell == cell) & (imp.target == target) &
                                (imp.regime == reg) & (imp.train == 'maml')
                                ]['improvement'].values[0])

x = np.arange(len(labels_x))
w = 0.36
b1 = ax2.bar(x - w / 2, imp_base, w, color='#a9cce3', edgecolor='black',
             linewidth=0.5, label='baseline training')
b2 = ax2.bar(x + w / 2, imp_maml, w, color='#2471a3', edgecolor='black',
             linewidth=0.5, label='MAML training')
for bars in (b1, b2):
    for bar in bars:
        v = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.05,
                 f'{v:.2f}×', ha='center', va='bottom', fontsize=8)

ax2.axhline(1.0, color='red', linewidth=0.8, linestyle='--',
            label='parity (no improvement)')
ax2.set_xticks(x); ax2.set_xticklabels(labels_x, fontsize=8.2)
ax2.set_ylabel('NRMSE improvement factor\n(full_graph / stage_aware)')
ax2.set_title('Current-path graph improvement over full-netlist baseline on sequential cells',
              fontsize=12, fontweight='bold')
ax2.grid(axis='y', alpha=0.3, linestyle='--'); ax2.set_axisbelow(True)
ax2.legend(loc='upper right', frameon=False)
ax2.set_ylim(0, max(max(imp_base), max(imp_maml)) * 1.18)
plt.tight_layout()
out2 = FIG_DIR / 'seq_cell_improvement_factors.png'
plt.savefig(out2, dpi=200, bbox_inches='tight')
plt.close()
print(f'Saved: {out2}')

# ----- Text summary: geomean across (cell × regime) per (target, graph, train) -----
def gmean(x):
    x = np.asarray([v for v in x if v > 0])
    return float(np.exp(np.mean(np.log(x)))) if len(x) else float('nan')

lines = ['Geomean per-task NRMSE (%) — sequential cells (D-FF, Scan D-FF)\n',
         '-' * 64 + '\n']
for target in TARGETS:
    for graph in ('full_graph', 'stage_aware'):
        for train in ('baseline', 'maml'):
            sub = df[(df.target == target) & (df.graph == graph) &
                     (df.train == train)]
            g = gmean(sub['NRMSE_pct'].values)
            lines.append(f'  {TARGET_LABEL[target]:<20} {graph:<12} {train:<10} '
                         f'geomean NRMSE = {g:.3f}%\n')
    lines.append('\n')

lines.append('-' * 64 + '\n')
lines.append('Geomean improvement factor (full_graph / stage_aware), '
             'matched by training:\n')
for target in TARGETS:
    for train in ('baseline', 'maml'):
        sub = imp[(imp.target == target) & (imp.train == train)]
        g = gmean(sub['improvement'].values)
        lines.append(f'  {TARGET_LABEL[target]:<20} {train:<10} '
                     f'geomean speedup = {g:.2f}×\n')

out3 = FIG_DIR / 'seq_cell_summary_geomean.txt'
out3.write_text(''.join(lines))
print(f'Saved: {out3}')
print()
print(''.join(lines))
