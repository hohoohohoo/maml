"""
Build an intuitive-metrics summary (MAE in ps, MAPE %, NRMSE %) from the
canonical rebuttal_metrics_geomean.csv produced by compute_mae_mape.py,
in response to Reviewer B (Q9) — who asked for MAE / WNS / TNS-style
metrics alongside NRMSE.

Per-PDK unit:
  - ASAP7 raw values are stored in picoseconds (lib time_unit ~ ps;
    typical cell-delay actuals_range ~ 600 ps)
  - TSMC  raw values are stored in nanoseconds (typical cell-delay
    actuals_range ~ 1.9 ns)
So we convert MAE to picoseconds with PDK-aware scaling:
  MAE_ps = MAE_raw * 1   for ASAP7
  MAE_ps = MAE_raw * 1000 for TSMC

WNS / TNS / path-delay metrics are not in this CSV; they live in
paper Table 5 (Section 5.2) — referenced in the rebuttal md.

Outputs:
  intuitive_metrics_topology_agnostic.csv   pivoted summary used in the
                                            Q9 rebuttal table
"""
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
g = pd.read_csv(HERE / 'rebuttal_metrics_geomean.csv')

g['MAE_ps_geomean'] = g.apply(
    lambda r: r['MAE_geomean'] * (1.0 if r['pdk'] == 'ASAP7' else 1000.0),
    axis=1)

ORDER = ['AADAM', 'MLP_MAML', 'GCN_Baseline', 'GCN_MAML']
sub = g[g['experiment'] == 'topology_agnostic'].copy()

rows = []
for dt in ('cell', 'transition'):
    for pdk in ('ASAP7', 'TSMC'):
        for mode in ('interpolation', 'extrapolation'):
            for src in ORDER:
                r = sub[(sub.data_type == dt) & (sub.pdk == pdk) &
                        (sub['mode'] == mode) & (sub.source_label == src)]
                if not len(r):
                    continue
                r = r.iloc[0]
                rows.append(dict(
                    data_type=dt, pdk=pdk, mode=mode, model=src,
                    n_cells=int(r['n_cells']),
                    NRMSE_pct=round(r['NRMSE_geomean'], 3),
                    MAE_ps=round(r['MAE_ps_geomean'], 3),
                    MAPE_pct=round(r['MAPE_pct_geomean'], 3),
                ))

df = pd.DataFrame(rows)
out = HERE / 'intuitive_metrics_topology_agnostic.csv'
df.to_csv(out, index=False)
print('Saved:', out)

# Pretty-print pivot for cell delay and transition
for dt in ('cell', 'transition'):
    print()
    print(f'=== TOPOLOGY-AGNOSTIC — {dt.upper()} ===')
    pv = (df[df.data_type == dt]
          .pivot_table(index=['pdk', 'mode'], columns='model',
                       values=['NRMSE_pct', 'MAE_ps', 'MAPE_pct'])
          .round(3))
    pv = pv.reindex(columns=pd.MultiIndex.from_product(
        [['NRMSE_pct', 'MAE_ps', 'MAPE_pct'], ORDER]))
    print(pv.to_string())
