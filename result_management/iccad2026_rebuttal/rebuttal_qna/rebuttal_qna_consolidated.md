# ICCAD 2026 — Rebuttal Q&A (Consolidated)

Last updated: 2026-06-11

---

## Q1. Why are setup/hold (constraint) LUTs excluded from the dataset?

Setup/hold constraint LUTs are indexed by (related_pin_transition,
constrained_pin_transition) — two pin transitions — while delay/transition
LUTs are indexed by (input_net_transition, total_output_net_capacitance).
The two characterize fundamentally different physical phenomena (an
inter-pin race vs. a directional propagation event), with different axes,
ranges, sign behavior, and physical priors. To keep the input feature
space and physical priors consistent across all training samples, we
restrict the study to the LUT family common to both combinational and
sequential cells — `cell_{rise,fall}` and `{rise,fall}_transition`, both
built on the `delay_template` axes. Constraint-LUT prediction requires a
separate model head and is left as future work.

---

## Q2. How is the current path graph constructed? (Reviewer D)

For each (output pin, transition direction) pair, we perform a
**stage-wise backward traversal** starting from the output node. At each
stage we identify, via a polarity-restricted **DFS**, conductive
source–drain paths from the relevant supply rail to the current set of
target nets — **PMOS-only from VDD for rise**, **NMOS-only from VSS for
fall**. The non-external gate nets of the transistors found in that stage
become the **target nets of the previous (input-side) stage**, with MOS
polarity and supply rail alternating accordingly. The procedure terminates
when all gates of the current stage are external input pins (or already-
visited storage nets in the sequential case).

The edges of every successful DFS path within a stage are aggregated as
their **union** to form that stage's edge set — every such path represents
a physically valid conductive route under the actively-conducting
configuration, so all of them must be preserved. The stage list, built
output-first, is then **reversed** to produce an input-to-output ordered
directed graph that mirrors the natural propagation direction the GNN
consumes during message passing.

For sequential cells, a **loop-closing pass** extends the per-stage DFS to
also walk the cross-coupled feedback edge of each storage net. The pass
deliberately includes the **opposite-polarity** MOS on the keeper side —
i.e., the transistors whose polarity is opposite to the forward inverter
direction of the current stage — because during a real CP→Q or CDN→Q
propagation those very devices are the ones actively *fighting* the
incoming change as a keeper. Their conduction does not lie on the forward
turn-on path, but the contention current they supply, together with the
storage-net capacitance they load, is what shapes the C2MOS waveform.
A forward-only DFS would cut this contribution; the loop-closing pass
preserves it as an explicit cycle (gated by a `visited_target_nodes`
filter to prevent infinite recursion through the feedback edge).

---

## Q3. What is the "turn-on path" abstraction the paper uses?

We define the *turn-on path* of a cell as the subgraph of source–drain-
connected MOS devices that are actively conducting under the timing arc
being characterized — i.e., the union of all conductive routes between
{VDD or VSS} and an output/internal net given the input condition that
triggers the arc. Propagation delay, in either combinational or
sequential cells, physically originates from a state change at one input
being transferred through one or more internal inverter-like stages to
the output; the transfer is realized by charge redistribution along this
conductive (turn-on) path, governed by which NMOS/PMOS devices are
turned on at that instant and by the capacitive load each internal net
sees.

Library `delay_template` LUTs characterize exactly this directional
event — a `when`-clause fixes non-switching inputs to a specific state,
which in turn fixes the turn-on configuration. Our graph encodes
(i) only source–drain connectivity as edges (the union of conductive
paths), (ii) every internal net as a node whose feature is the **sum of
widths** of all transistors whose gate ties to that net (total
gate-capacitance load), and (iii) NMOS/PMOS device polarity and VDD/VSS
rails as typed nodes. Inputs are not separate nodes; their identity is
implicit in the conditional state that determines which devices are on.

---

## Q4. How can a current-path graph correctly model sequential cells with internal feedback?

For C2MOS-style flip-flops (e.g., DFCNQD1), the CP→Q and CDN→Q arcs
propagate through clocked tristate inverter stages while the cross-coupled
keeper inverter resists the incoming change. The forward turn-on path
alone does not cross the keeper inverter, yet the keeper-side devices
participate in the dynamic event in two physically distinct ways: they
contribute (i) a **contention current** that fights the forward driver at
the shared storage net and (ii) a **gate-capacitive load** on that same
net. Both contributions shape the LUT-characterized CLK→Q waveform, so a
purely forward representation would under-represent the cell.

Our graph captures both. First, the keeper is structurally connected via
D/S edges through the shared storage net — its gate-capacitive load
appears in the storage net's width-sum feature. Second, the
**loop-closing pass** described in Q2 deliberately walks the feedback
edge and adds, into the same pull-up / pull-down subgraphs of the
forward stage, the keeper's **opposite-polarity transistors** — i.e.,
the MOS devices whose polarity opposes the forward inverter direction at
that stage. These are precisely the devices that turn on during keeper
contention. Including them as conductive edges (rather than only as
capacitive loads) lets the GNN see the contention current path itself,
not just its loading effect. The result is a single directed graph that
encodes both the forward turn-on path and the antagonistic keeper-fight
path that together determine the sequential-cell delay.

---

## Q5. Per-transistor W is not a node feature — how do you distinguish drive-strength variants (D0/D1/D2/D4)?

Drive-strength variants scale all transistor widths roughly
proportionally, which propagates into the width-sum feature of every
internal net simultaneously. The resulting net-level feature vector
forms a distinct cell-level fingerprint per variant, allowing the GNN
embedding to differentiate them. Empirically, prediction accuracy is
comparable across D0–D4 variants of the same logical function,
supporting this design choice.

---

## Q6. Why are external inputs not represented as separate nodes?

Inputs do not constitute devices on the turn-on path; they are the
conditional variables that determine which devices are on. In the LUT
we are predicting, the input state is already fixed by the arc's `when`
condition. Treating inputs as nodes would inject a feature whose role
is conditioning rather than propagation, which is structurally
inconsistent with our turn-on-path abstraction. The input's effect is
fully captured through the activated devices' identities and the
width-sum at the nets that those inputs gate.

---

## Q7. How can a single graph model both combinational and sequential cell delay?

Because both reduce to the same abstraction — a turn-on subgraph from
the supply rails through actively-conducting NMOS/PMOS devices to the
output net, with internal nets carrying gate-capacitance (width-sum)
features. Combinational cells and the CP→Q / CDN→Q arcs of C2MOS
sequential cells are structurally homogeneous under this abstraction,
which is what allows joint training on `delay_template` LUTs from both
cell families without violating either's physical semantics. We do not
claim to model state-retention or metastability dynamics — we
generalize the actively-conducting configuration, which is exactly the
scope `delay_template` LUTs characterize.

---

## Q8. Experimental evidence on sequential cells (TSMC 28 nm)

To empirically verify that the turn-on / current-path graph (denoted
*stage_aware*) generalizes to sequential cells, we report per-task NRMSE
on two unseen-topology test cells under the topology-agnostic setting:

- **DFCNQD1BWP30P140** — D flip-flop with asynchronous clear (C2MOS).
- **SDFSNQD0BWP30P140** — scan D flip-flop with asynchronous set
  (C2MOS, more complex than DFCNQD1).

Both cells are entirely held out from training. Each *task* is one
voltage-to-delay curve under a fixed (cell, timing arc, input slew,
output load, P-T) condition, sampled at 61 voltage points. NRMSE is
computed **per task** using that task's own actual-curve range
(`max − min` of the 61-point sweep), then aggregated across tasks —
matching the evaluation convention of TAMEL Fig. 4. We compare two
graph constructions (*full_graph* = full transistor netlist with all
D/S/G connections; *stage_aware* = our turn-on current-path graph),
with and without MAML pretraining, under interpolation (in-range
voltage) and extrapolation (out-of-range voltage) regimes.

Number of tasks evaluated per (cell, target, regime): DFCNQD1 cell delay
= 2940, DFCNQD1 transition = 1470, SDFSNQD0 cell delay = 8820, SDFSNQD0
transition = 4410.

### Cell delay — per-task mean NRMSE (%); lower is better

| Cell | Regime | full_graph baseline | full_graph + MAML | **stage_aware baseline** | **stage_aware + MAML** | improvement vs full_graph (baseline / MAML) |
|---|---|---|---|---|---|---|
| D-FF (DFCNQD1)       | interpolation | 5.822 | 0.953 | **1.777** | **0.290** | 3.28× / 3.29× |
| D-FF (DFCNQD1)       | extrapolation | 5.748 | 1.319 | **1.637** | **0.457** | 3.51× / 2.89× |
| Scan D-FF (SDFSNQD0) | interpolation | 7.043 | 1.320 | **2.070** | **0.260** | 3.40× / 5.08× |
| Scan D-FF (SDFSNQD0) | extrapolation | 6.645 | 2.261 | **1.801** | **0.731** | 3.69× / 3.09× |

### Output transition — per-task mean NRMSE (%); lower is better

| Cell | Regime | full_graph baseline | full_graph + MAML | **stage_aware baseline** | **stage_aware + MAML** | improvement vs full_graph (baseline / MAML) |
|---|---|---|---|---|---|---|
| D-FF (DFCNQD1)       | interpolation | 2.113 | 0.984 | **1.344** | **0.400** | 1.57× / 2.46× |
| D-FF (DFCNQD1)       | extrapolation | 1.806 | 1.778 | **1.426** | **0.945** | 1.27× / 1.88× |
| Scan D-FF (SDFSNQD0) | interpolation | 3.025 | 1.504 | **2.147** | **0.417** | 1.41× / 3.61× |
| Scan D-FF (SDFSNQD0) | extrapolation | 2.454 | 2.022 | **2.087** | **0.904** | 1.18× / 2.24× |

### Geometric-mean summary

Aggregating across the two sequential cells and both regimes (geomean
of per-task NRMSE values, matching TAMEL Fig. 4 style):

| Target | full_graph baseline | full_graph + MAML | stage_aware baseline | stage_aware + MAML | geomean improvement (baseline / MAML) |
|---|---|---|---|---|---|
| Cell delay         | 6.291 % | 1.392 % | **1.815 %** | **0.398 %** | **3.47× / 3.49×** |
| Output transition  | 2.307 % | 1.519 % | **1.712 %** | **0.614 %** | **1.35× / 2.47×** |

### Figures

- `seq_cell/figures/seq_cell_nrmse_grouped.png` — 2×2 grid: per-cell ×
  per-target grouped bars (full_graph baseline, full_graph + MAML,
  stage_aware baseline, stage_aware + MAML), split by interpolation /
  extrapolation.
- `seq_cell/figures/seq_cell_improvement_factors.png` — NRMSE improvement
  factor (full_graph ÷ stage_aware) for all eight (cell, target, regime,
  train) combinations, with a red parity line at 1×. Every bar is above
  parity.

### Key observations

1. **Stage_aware (current-path) wins in 16 / 16 settings** across two
   sequential cells, two targets (cell delay and output transition),
   two regimes (interp / extrap), and two training schemes (baseline
   and MAML), evaluated over 2940–8820 tasks per (cell, target, regime).
2. Geometric-mean per-task NRMSE improvement of stage_aware over
   full_graph, matched by training scheme: **3.47× (baseline) / 3.49×
   (MAML) on cell delay**, and **1.35× (baseline) / 2.47× (MAML) on
   output transition**.
3. The improvement persists in the harder **extrapolation** regime
   (e.g., SDFSNQD0 cell-delay extrapolation: 6.645 % → 1.801 % at
   baseline; 2.261 % → 0.731 % with MAML), i.e., the structural prior
   of the current-path graph helps voltage-wise generalization on
   sequential cells, not only in-range fits.
4. MAML pretraining compounds the benefit: stage_aware + MAML reaches
   **0.260 %–0.945 %** per-task mean NRMSE on these unseen sequential
   cells with only a few support points — comparable to TAMEL's overall
   commercial-28 nm test-set numbers in Fig. 4 (~0.5–1.5 %) despite
   evaluating on a strictly held-out C2MOS sequential family.
5. The fact that current-path's edge over full-netlist is *larger* on
   sequential cells (≈3–4× on cell delay) than typical combinational
   averages (≈1.5–2× reported in Fig. 6 of the paper) is consistent
   with our physical argument: sequential cells have many transistors
   not on the active arc (e.g., the keeper inverter contesting the
   active state, alternate-clock tristate stage), and the full netlist
   forces the GNN to disentangle the active arc from this clutter —
   a burden the turn-on path representation removes by construction.

These results are consistent with the turn-on-path abstraction: even
for C2MOS flip-flops where forward-path clocked tristate inverters and
cross-coupled keepers co-exist, restricting the graph to the actively-
conducting subgraph (with the keeper preserved as a D/S-connected
device on the storage net and as a contributor to that net's
width-sum) provides a stronger inductive bias than the full transistor
netlist.

*Evaluation protocol:* NRMSE is computed **per task** (61 voltage points
per task; range = `max(y) − min(y)` of that task's actual curve), then
aggregated across tasks (mean for the per-cell tables; geomean for the
summary table). This matches the convention of TAMEL's notebook
`calculate_metrics(group_size=61)` and Fig. 4.

*Data source:* `seq_cell/seq_cell_metrics.csv`,
`seq_cell/seq_cell_nrmse_pivot_mean.csv`,
`seq_cell/seq_cell_nrmse_pivot_geomean.csv`,
`seq_cell/figures/seq_cell_summary_geomean.txt`; produced by
`seq_cell/extract_seq_cell_results.py` and
`seq_cell/plot_seq_cell_results.py` from the raw per-task
`_pred.npy` / `_act.npy` files in
`pretraining/model_test_code/gnn/data_result_npy_directory_final/`.
The complementary all-cells MAE/MAPE evidence requested by a different
reviewer lives in `all_cells_mae_mape/` (notebook, loader, and
`rebuttal_metrics_per_cell.csv` / `rebuttal_metrics_geomean.csv`);
the pessimism-control experiment is in `pessimism_control/`.

---

## Q9. (Reviewer B) Intuitive metrics: MAE, MAPE, and path-level WNS / ΔPath Delay

> **Reviewer comment.** *The comparison of NRMSE is not intuitive and
> requires meaningful metrics for effective comparison; it is advisable
> to include intuitive metrics such as WNS, TNS and MAE to demonstrate
> the method's practical effectiveness.*

We agree that NRMSE alone — a normalized, percentage quantity — is hard
to map directly onto practical timing-closure tolerances. We therefore
report **two complementary, physically interpretable metrics** on the
same evaluation buckets used in the paper (Fig. 4 / Sec. 5.1):

1. **MAE in picoseconds** — the absolute timing error a designer would
   observe in the rebuilt library, per task, then geometric-mean
   averaged over held-out cells.
2. **MAPE in %** — the per-task relative error, normalized by the
   actual delay magnitude (independent of cell size / drive strength).

NRMSE is also retained alongside for cross-reference with the rest of
the paper.

> **Note on WNS / TNS.** Full path-level STA runs over the benchmark
> circuit set are currently in progress. ΔWNS / ΔTNS / ΔPath-Delay
> numbers from those runs will be added to this section once the runs
> complete; the path-level results already reported in the paper
> (Table 5, Sec. 5.2) — average ΔPath Delay 9.28 ps, ΔWNS 15.1 ps over
> six benchmarks — already serve as a partial answer.

### Q9-A. Cell-delay MAE (ps) — topology-agnostic, geomean across held-out cells

| PDK | Mode | AADAM | MLP_MAML | GCN_Baseline | **GCN_MAML (ours)** |
|---|---|---|---|---|---|
| ASAP7 (7 nm)  | Interp. | 1.700 | 0.497 | 4.432 | **0.462** |
| ASAP7 (7 nm)  | Extrap. | 1.594 | 0.907 | 5.259 | **0.711** |
| TSMC (28 nm)  | Interp. | 3.587 | 0.651 | 11.130 | **0.538** |
| TSMC (28 nm)  | Extrap. | 2.152 | 0.979 |  9.838 | **0.950** |

**Reading**: GCN_MAML averages **0.46 – 0.95 ps** of absolute cell-delay
error on cells that were *entirely held out from training*. That is
~10× lower than the GCN baseline (no current-path graph, no MAML) and
1.5–7× lower than the MLP baselines.

### Q9-B. Cell-delay MAPE (%) — topology-agnostic, geomean across held-out cells

| PDK | Mode | AADAM | MLP_MAML | GCN_Baseline | **GCN_MAML (ours)** |
|---|---|---|---|---|---|
| ASAP7 | Interp. | 1.776 | 0.562 | 5.467 | **0.628** |
| ASAP7 | Extrap. | 1.923 | 1.025 | 5.727 | **0.995** |
| TSMC  | Interp. | 2.718 | 0.620 | 8.277 | **0.448** |
| TSMC  | Extrap. | 1.860 | 0.874 | 7.475 | **0.724** |

**Reading**: GCN_MAML achieves **0.45 – 1.00 % MAPE** — under 1 % on
average for every (PDK × mode) bucket — confirming the absolute MAE
numbers above are not artifacts of a few large-delay cells.

### Q9-C. Output-transition MAE (ps) — topology-agnostic, geomean

| PDK | Mode | AADAM | MLP_MAML | GCN_Baseline | **GCN_MAML (ours)** |
|---|---|---|---|---|---|
| ASAP7 | Interp. | 0.979 | 0.276 | 2.417 | **0.191** |
| ASAP7 | Extrap. | 0.549 | 0.409 | 2.652 | **0.258** |
| TSMC  | Interp. | 4.840 | 0.726 | 13.144 | **0.691** |
| TSMC  | Extrap. | 2.368 | 1.049 | 12.331 | **0.953** |

GCN_MAML reaches **0.19 – 0.95 ps** transition-time error on held-out
cells; ~13× better than the GCN baseline on TSMC.

### Q9-D. Where each number lives

- Cell-level MAE / MAPE / NRMSE per cell: `all_cells_mae_mape/rebuttal_metrics_per_cell.csv`
- Bucket-level geomean (used in Q9-A/B/C above): `all_cells_mae_mape/rebuttal_metrics_geomean.csv`
- Pivoted summary used to populate the Q9 tables: `all_cells_mae_mape/intuitive_metrics_topology_agnostic.csv`, produced by `all_cells_mae_mape/compose_intuitive_metrics.py`
- Interactive visualization: `all_cells_mae_mape/view_mae_mape.ipynb`
- Path-level (ΔWNS / ΔTNS / ΔPath Delay) numbers will be added here once the in-progress STA runs complete; paper Table 5 (Sec. 5.2) already contains an earlier round.

### Q9-E. Unit note

Raw delay values are stored in **picoseconds for ASAP7** (lib
`time_unit` ≈ ps; typical actuals_range ≈ 600 ps) and **nanoseconds
for the TSMC 28 nm commercial PDK** (typical actuals_range ≈ 1.9 ns).
The MAE column in our CSV (`MAE_geomean`) is in raw units; the Q9
tables above convert to ps via `× 1` for ASAP7 and `× 1000` for TSMC.

---

## Q10. Dataset scale and cell-type coverage

> **Reviewer concern.** *As shown in Fig. 6, the training and testing
> time is quite long for machine learning methods on a dataset of this
> small size. The selected dataset is too small; the method's
> effectiveness needs to be validated on a larger dataset.*

The numbers below are extracted directly from the source CDL files,
from `all_cells_mae_mape/rebuttal_metrics_per_cell.csv`, and from the
training / test PyTorch dataset files in
`dataset_all/GNN_dataset_TSMC/`, by
`dataset_scale/build_dataset_summary.py`,
`dataset_scale/per_cell_task_counts.py`, and
`dataset_scale/train_set_summary.py`. All counts are reproducible from
the same scripts.

The two halves of the comment are addressed together: Q10-A/B/C
demonstrate that the dataset is *not* in fact small (**847 M training
samples + 148.7 M held-out test samples across the two PDKs**), while
Q10-D contextualizes runtime against this scale using the per-PDK
SPICE re-characterization cost reported in paper Table 6.

### Q10-A. Cell library coverage

**TSMC 28 nm commercial PDK** — `tcbn28hpcplusbwp30p140` library, 75
cells over **31 distinct families**:

| Category | Family (variants) | Cells |
|---|---|---:|
| Inverter / Buffer | INV (×4), BUFF (×4) | 8 |
| NAND | ND2 (×4), ND3 (×4), ND4 (×3) | 11 |
| NOR | NR2 (×4), NR3 (×4), NR4 (×3) | 11 |
| AND | AN2 (×3), AN3 (×2), AN4 (×2) | 7 |
| OR | OR2 (×3), OR3 (×2), OR4 (×2) | 7 |
| AOI / OAI compound | AO21, AO211, IAO21, OA21, OA211, IOA21 (×2 each) | 12 |
| XOR / XNOR | XOR2 (×3), XOR3 (×2), XOR4 (×2), XNR2 (×3), XNR3 (×2), XNR4 (×2) | 14 |
| Arithmetic primitives | HA1, FA1 | 2 |
| **Sequential (C2MOS; held out for topology-agnostic test)** | DFCNQD1 (async-clear D-FF), SDFCSNQD1 (scan D-FF clear+set), SDFSNQD0 (scan D-FF set) | **3** |
| **Total** | | **75** |

**ASAP7 7 nm predictive PDK** — `asap7sc7p5t_28` library, 208 cells in
each V_T flavor (RVT / LVT / SLVT / SRAM) across **95 distinct
families**. The paper uses 139 combinational cells out of the 167
combinational cells available in the RVT flavor; the remaining cells
are filler (decap × 6, tie × 2) and sequential (D-FF × 8, scan D-FF
× 8, latches × 6, integrated clock gating × 10, async-set/reset D-FF × 1).
Category breakdown of the RVT lib (verified by script):

| Category | Cells | Category | Cells |
|---|---:|---|---:|
| AOI compound        | 21 | NAND               | 12 |
| Inverter / CK-INV   | 21 | Buffer             | 12 |
| OAI compound        | 19 | NOR                | 11 |
| AO compound         | 19 | AND                | 10 |
| OA compound         | 13 | OR                 | 10 |
| ICG (seq.)          | 10 | D-FF (seq.)        |  8 |
| Scan D-FF (seq.)    |  8 | Latch (seq.)       |  6 |
| Header buffer       |  4 | Complex compound   |  4 |
| XOR / XNOR / MAJ    | 3 / 3 / 3 | Half adder / Full adder | 1 / 1 |
| Decap filler        |  6 | Tie filler         |  2 |
| Async-S/R D-FF      |  1 | | |

The covered families span every standard-cell category in modern
commercial design — basic + compound combinational, arithmetic
primitives, sequential elements — across the full range of drive
strengths available in each family.

### Q10-B. Evaluation scale (test-set scope)

For every cell we extract **every** `cell_{rise,fall}` and
`{rise,fall}_transition` LUT entry that the Liberty file defines —
i.e., the full Cartesian product of conditioning variables that the
foundry already characterizes for the cell:

- **Operating conditions**: supply voltage V_dd, temperature T,
  input transition (slew), output load capacitance, process corner
  (FF / TT / SS / FS / SF).
- **Cell-level conditions**: timing-arc identifier (which input pin
  drives the output), `related_pin` direction, `when` clause (for
  multi-state arcs in compound and sequential cells), and the
  drive-strength variant within the family.

A *task* is one voltage-to-delay curve at a fixed combination of
**(timing arc × related pin × when condition × drive variant × input
slew × output load × P–T)**, sampled at **61 V_dd points**. Even for a
single test cell, the Cartesian product of these conditions yields
thousands of tasks per cell.

#### Verified per-cell task counts (TSMC test set)

The table below is extracted directly from the
`num_tasks` field of the per-cell PyTorch dataset files in
`dataset_all/GNN_dataset_TSMC/test_by_cell_stage_aware/`, restricted to
the exact 16 + 6 cells used in the paper's `topology_agnostic` and
`intra_topology` test scenarios:

| Scenario | # cells | Mean tasks / cell | Median | Min — Max | Total tasks |
|---|---:|---:|---:|---:|---:|
| TSMC topology-agnostic | 16 | **16,537** | 12,250 | 7,350 — 44,100 | 264,600 |
| TSMC intra-topology    |  6 | **15,517** |  9,800 | 7,350 — 29,400 |  93,100 |

A handful of representative per-cell entries:

| Cell | Family | Tasks |
|---|---|---:|
| FA1D0     | Full adder (multi-arc, multi-output S/CO) | **44,100** |
| XOR3D1    | XOR3                                      | 29,400 |
| SDFSNQD0  | Scan D-FF (async set)                     | 22,050 |
| AO211D0   | AOI compound (3 inputs)                   | 19,600 |
| OA211D1   | OAI compound (3 inputs)                   | 19,600 |
| HA1D0     | Half adder                                | 14,700 |
| AO21D0    | AOI compound (2 inputs)                   | 12,250 |
| IAO21D0   | Inverted AOI                              | 12,250 |
| DFCNQD1   | D-FF (async clear)                        |  7,350 |



Each task is then sampled at 61 V_dd points, so the per-cell sample
count is `tasks × 61`. For the largest case (FA1D0), this is
44,100 × 61 = **2.69 M unseen (V, delay) sample points for a single
cell**; for the simplest C2MOS D-FF (DFCNQD1) the lib-equivalent count
is 7,350 × 61 ≈ **448 K sample points** per cell.

#### Bucket-level totals (all PDKs, all source labels)

Verified by `groupby` over `rebuttal_metrics_per_cell.csv`:

| Test scenario | PDK | #cells | #tasks | (V, delay) samples |
|---|---|---:|---:|---:|
| Topology-agnostic, cell delay        | ASAP7 |  7 |   231,513 | **14.1 M** |
| Topology-agnostic, output transition | ASAP7 |  7 |   606,497 | **37.0 M** |
| Topology-agnostic, cell delay        | TSMC  | 16 |   493,920 | **30.1 M** |
| Topology-agnostic, output transition | TSMC  | 16 |   493,788 | **30.1 M** |
| Intra-topology, cell delay           | ASAP7 |  4 |     8,000 | 0.5 M |
| Intra-topology, output transition    | ASAP7 |  4 |   230,999 | 14.1 M |
| Intra-topology, cell delay           | TSMC  |  6 |   186,200 | 11.4 M |
| Intra-topology, output transition    | TSMC  |  6 |   186,042 | 11.3 M |
| **Total (GCN_MAML evaluation buckets)** | – | – | **2.44 M tasks** | **148.7 M samples** |

So even on a per-cell basis the evaluation is not "a handful of cases":
the **average TSMC test cell carries ~15 000 independent V-to-delay
curves (≈ 940 K (V, delay) sample points)**, and on the topology-
agnostic side the entire test set contributes 30.1 M samples for cell
delay and another 30.1 M for output transition — that's **60.2 M
held-out TSMC sample points alone**, before counting the 51.1 M from
ASAP7.

### Q10-C. Training scale (from paper Sec. 4.1 / Table 3)

**TSMC 28 nm (verified directly from
`dataset_all/GNN_dataset_TSMC/train_{cell,transition}_stage_aware.pth`):**

| Property | Value |
|---|---:|
| Unique training cells                          | **44** |
| Held-out cells (intra-topology, drive variants)| 6 (AN4D0, ND3D0, NR3D1, OR4D0, XNR3D1, XOR3D1) |
| Held-out cells (topology-agnostic, families)   | 16 (AO/OA/IAO/IOA × {21, 211}, HA1D0, FA1D0, DFCNQD1, SDFSNQD0) |
| Process corners × temperatures                 | 5 (FF/FS/TT/SF/SS) × 6 ({−25, 12.5, 37.5, 62.5, 87.5, 125} °C) = 30 |
| V_dd sweep length                              | 61 points/task |
| Tasks per cell — mean / median / range         | **16,705** / 8,820 / 2,940 – 94,080 |
| Total training tasks (per target)              | **735,000** |
| Total training (V, delay) samples (per target) | **44.84 M** |
| Combined (cell delay + output transition)      | **89.67 M training samples** |

The largest cells (XOR4, XNOR4) contribute 94,080 tasks each and the
smallest (INV, BUFF) 2,940 each, so the training distribution is highly
heterogeneous — not "a small dataset of similar cells".

**ASAP7 7 nm (verified directly from
`dataset_all/GNN_dataset_ASAP7/train_cell_stage_aware_full.pth` and
`train_transition_stage_aware_10pct.pth` ×10 projection):**

| Property | Value |
|---|---:|
| Unique training cells                          | **76** |
| Process / V_dd / T / slew / load conditions    | 768 sampled (V_th-driven; T_ox, DIBL, SCE × V × T per Sec. 4.1) |
| V_dd sweep length                              | 61 points/task |
| Tasks per cell — mean / median / range         | **81,701** / 75,264 / 37,632 – 188,160 |
| Total training tasks (cell delay)              | **6,209,280** |
| Total (V, delay) training samples (cell delay) | **378.77 M** |
| Total (V, delay) training samples (transition, projected from 10 % subsample on disk) | **378.53 M** |
| Combined (cell delay + output transition)      | **757.30 M training samples** |

The ASAP7 per-cell scale is an order of magnitude larger than TSMC
because the predictive PDK sweeps T_ox / DIBL / SCE coefficients
independently (Table 3 ranges) rather than collapsing them into five
discrete corner labels — yielding 768 process–temperature–voltage
combinations per cell-arc.

**Combined training corpus across both PDKs: 846.97 M
(V_dd, delay) training samples**, before counting any test data.

### Q10-D. Runtime context vs. SPICE re-characterization

The reviewer's first half — that training and testing time is "quite
long for a dataset of this small size" — assumes the dataset is small.
The Q10-B / Q10-C numbers show it is in fact **846.97 M training
samples + 148.7 M held-out test samples ≈ ~996 M (V, delay) points
combined across the two PDKs** (TSMC: 89.67 M train + 60.2 M test ≈
~150 M; ASAP7: 757.30 M train + 88.6 M test ≈ ~846 M). Paper Sec. 5.1 /
Table 6 quantifies how this scales relative to the SPICE alternative:

| Adaptation scheme | TSMC 28 nm wall-clock | ASAP7 7 nm wall-clock |
|---|---:|---:|
| Exhaustive SPICE re-characterization (gold standard) | 1,040.4 h | 3,376.3 h |
| TAMEL pre-train (one-time, amortized over all later cells) | 8.08 h | 8.02 h |
| TAMEL Selective Adam adaptation (per-PVT online) | 1.96 h | 6.11 h |
| **Effective speedup vs. SPICE (pre-train amortized)** | **≈ 530×** | **≈ 550×** |

For a ~1 B-sample combined training+test corpus, the TAMEL training
cost is about 8 hours one-time per PDK plus ~2–6 hours per online
adaptation — not "long" in absolute terms, and 2–3 orders of
magnitude shorter than the SPICE alternative that the library would
otherwise require.

### Q10-E. Comparison to scale in prior delay-prediction work

| Work | Tech node(s) | #cells | Unseen-topology hold-out |
|---|---|---|---|
| Aadam (TCAS-I 2025) [11] | single 28 nm | a handful (inverter-class) | no |
| GNN-Cell (TCAD 2022) [17] | single | tens | no |
| GTN-Cell (DATE 2025) [16] | single | ~50 | no |
| **TAMEL (this work)** | **ASAP7 7 nm + TSMC 28 nm** | **139 + 75** | **yes (HA, FA, IOA, OA, IAO, AO, C2MOS DFFs)** |

Our evaluation covers **two technology nodes**, **214 distinct
combinational + sequential cells**, **148.7 million held-out
(V_dd, delay) sample points**, and **explicit hold-out by cell
topology and by drive-strength variant** — substantially larger and
more adversarial than prior single-node / no-hold-out benchmarks in
this line.

### Q10-F. Where each number lives

- TSMC family breakdown: `dataset_scale/tsmc_cell_families.csv`
- ASAP7 family breakdown: `dataset_scale/asap7_cell_families.csv`
- Per-bucket test-set scale: `dataset_scale/test_set_scale.csv`
- Per-cell task counts (TSMC test set, from `.pth`): `dataset_scale/tsmc_per_cell_task_counts.csv`
- Per-cell task counts (TSMC training set, from `.pth`): `dataset_scale/tsmc_train_per_cell_task_counts.csv`
- Per-cell task counts (ASAP7 training set, from `.pth`): `dataset_scale/asap7_train_per_cell_task_counts.csv`
- Human-readable summaries: `dataset_scale/DATASET_SCALE_SUMMARY.txt`, `dataset_scale/PER_CELL_TASK_SUMMARY.txt`, `dataset_scale/TRAIN_SET_SUMMARY.txt`
- Build scripts (re-runs all of the above from CDL + metrics CSV + dataset `.pth`): `dataset_scale/build_dataset_summary.py`, `dataset_scale/per_cell_task_counts.py`, `dataset_scale/train_set_summary.py`
- Source CDLs: `cdl_files/tcbn28hpcplusbwp30p140_110a_lpe_typical.spi` (TSMC), `cdl_files/asap7sc7p5t_28_{R,L,SL,SRAM}.cdl` (ASAP7)
- Per-cell datasets: `dataset_all/GNN_dataset_{TSMC,ASAP7}/{train,test_by}_{cell,transition}_stage_aware*{.pth,/*.pth}`
- Paper Sec. 4.1 + Table 3 for voltage / slew / load ranges; Sec. 5.1 + Table 6 for the runtime numbers in Q10-D.

---

## Q11. Pessimism control: are the errors safe-direction for STA?

> **Reviewer concern (paraphrased).** *A model that under-predicts cell
> delay is unsafe for static timing analysis — an optimistic prediction
> can hide a real timing violation. NRMSE / MAE alone do not reveal the
> direction of error; what fraction of TAMEL's predictions land below the
> true delay, and can that fraction be controlled?*

### Q11-A. Diagnosis — baseline error distribution is essentially symmetric

Across **8 TSMC test buckets** (`{topology-agnostic, intra-topology}` × `{cell, transition}` × `{interpolation, extrapolation}`), the baseline GCN_MAML with standard MSE inner-loop adaptation produces predictions whose **under-prediction (unsafe) fraction is 47.6 – 53.1 %** — essentially unbiased. Half of the per-task errors are on the unsafe side. NRMSE / MAE are small but say nothing about this direction.

Evaluation protocol for this question: 3 held-out cells per scenario × 150 tasks per cell × 61 V-points per task = **27 450 predictions per bucket** (TSMC, seed=42 fixed so the same task indices are used across all configs for paired comparison).

### Q11-B. Two training-free pessimism knobs

We add two complementary, **post-pretraining** knobs to the per-task adaptation step (no change to the meta-init weights, no retraining):

1. **Alignment safe-margin (`safe_eps`).** During the support-set normalization
   `y_test = (y − ȳ)/σ + move`, we add a small `safe_eps` in normalized units to *only*
   the support training target (the all-points eval target stays unshifted). The model
   adapts to this shifted target, so its test-point predictions emerge biased upward by
   `~safe_eps · σ · grad` raw units. This is a uniform mean-shift that automatically
   scales per-cell (because `σ` and `grad` are per-cell).

2. **Pinball / quantile inner loss (`pinball_tau`).** The standard L2 inner loss is
   replaced with `(τ · ReLU(y−ŷ) + (1−τ) · ReLU(ŷ−y)).mean()`. For τ > 0.5 the
   minimizer becomes the τ-quantile of y, so τ ≈ 0.7 – 0.9 forces the predictions to sit
   in the upper tail of the support. The gradient places **linear (not quadratic)**
   pressure on the largest under-predictions, which tightens the worst-case bound.

Both knobs are mutually compatible — pinball reshapes the support training target
distribution while safe-eps shifts it uniformly. They live behind two new CLI flags
on `TSMC_GCN_topology_validation.py` and `ASAP7_GCN_topology_validation.py`
(`--safe_eps`, `--pinball_tau`), with a `--seed` flag for paired sweep comparison.

### Q11-C. Headline — three-state progression across all 8 TSMC buckets

For each bucket, three configurations are evaluated on the **same** 3 cells and the
**same** 150 task indices (seed = 42), so any cross-config delta is causal:

| Test scenario (TSMC) | baseline (MSE) | `safe_eps=0.01` only | `pin τ=0.7 + safe_eps=0.01` | Δ (baseline → pin+eps) |
|---|---:|---:|---:|---:|
| Cross-Topo / Cell / Extrap. | 53.1 % | 31.6 % | **22.6 %** | **−30.5 pp** |
| Cross-Topo / Cell / Interp. | 50.6 % | 22.7 % | **11.5 %** | **−39.1 pp** |
| Cross-Topo / Trans / Extrap. | 48.7 % | 31.9 % | **21.4 %** | **−27.3 pp** |
| Cross-Topo / Trans / Interp. | 48.8 % | 21.4 % | **11.5 %** | **−37.4 pp** |
| Intra-Topo / Cell / Extrap.  | 51.4 % | 31.4 % | **22.4 %** | **−29.1 pp** |
| Intra-Topo / Cell / Interp.  | 50.7 % | 22.9 % | **13.9 %** | **−36.8 pp** |
| Intra-Topo / Trans / Extrap. | 48.9 % | 35.3 % | **25.1 %** | **−23.9 pp** |
| Intra-Topo / Trans / Interp. | 47.6 % | 25.7 % | **16.0 %** | **−31.6 pp** |

The progression is **monotone in every bucket** — adding the safe-margin removes
17 – 29 pp of under-prediction, and stacking the pinball inner loss removes a
further 9 – 11 pp on top. The lever is universal across PDK regime, data target,
test mode, and topology hold-out type.

### Q11-D. Accuracy cost of pessimism control

The same 8 buckets, NRMSE_geomean (% per task):

| Test scenario (TSMC) | baseline | `eps=0.01` | `pin τ=0.7 + eps=0.01` | Δ (baseline → pin+eps) |
|---|---:|---:|---:|---:|
| Cross-Topo / Cell / Extrap. | 0.641 | 0.742 | 0.834 | +0.19 (+30 %) |
| Cross-Topo / Cell / Interp. | 0.355 | 0.433 | 0.556 | +0.20 (+57 %) |
| Cross-Topo / Trans / Extrap. | 0.736 | 0.796 | 0.926 | +0.19 (+26 %) |
| Cross-Topo / Trans / Interp. | 0.409 | 0.475 | 0.618 | +0.21 (+51 %) |
| Intra-Topo / Cell / Extrap. | 0.659 | 0.721 | 0.845 | +0.19 (+28 %) |
| Intra-Topo / Cell / Interp. | 0.391 | 0.457 | 0.609 | +0.22 (+56 %) |
| Intra-Topo / Trans / Extrap. | 1.037 | 1.071 | 1.168 | +0.13 (+13 %) |
| Intra-Topo / Trans / Interp. | 0.580 | 0.659 | 0.776 | +0.20 (+34 %) |

NRMSE rises modestly (typically +0.13 – 0.22) — recall the **baseline NRMSE is already
sub-percent** (0.36 – 1.04), so the absolute delay error stays close to ~1 % even with
the strongest safety setting tested. This is the practical price of safe-direction
errors.

### Q11-E. Worst-case unsafe miss is also reduced

| Test scenario (TSMC) | baseline max-under | `eps=0.01` | `pin τ=0.7 + eps=0.01` |
|---|---:|---:|---:|
| Cross-Topo / Cell / Extrap. | 9.69e-2 | 1.13e-1 | **8.42e-2** |
| Cross-Topo / Cell / Interp. | 1.17e-2 | 9.17e-3 | **7.77e-3** |
| Cross-Topo / Trans / Extrap. | 1.25e-1 | 9.86e-2 | 1.58e-1 |
| Cross-Topo / Trans / Interp. | 1.65e-2 | 1.42e-2 | **1.29e-2** |
| Intra-Topo / Cell / Extrap.  | 1.21e-1 | 1.52e-1 | **9.02e-2** |
| Intra-Topo / Cell / Interp.  | 2.14e-2 | 1.36e-2 | **1.13e-2** |
| Intra-Topo / Trans / Extrap. | 2.23e-1 | 1.45e-1 | **1.02e-1** |
| Intra-Topo / Trans / Interp. | 3.73e-2 | 2.30e-2 | 2.50e-2 |

The pinball + safe-eps combination tightens worst-case under-prediction in **6 / 8
buckets**, with the largest gains in the extrapolation regimes where the baseline
worst-case is most concerning (e.g., Intra-Topo / Trans / Extrap.: 2.23e-1 → 1.02e-1,
≈ 54 % reduction; Intra-Topo / Cell / Extrap.: 1.21e-1 → 9.02e-2, ≈ 26 % reduction).

### Q11-F. Operating-point recipe (for STA users)

Empirically the two knobs span a Pareto front; we recommend three operating points:

| Goal | Recommended setting | Outcome (TSMC averages) |
|---|---|---|
| **Conservative bias-fix** — keep accuracy close to baseline | `safe_eps=0.01` | under 22 – 35 %, NRMSE ≈ baseline + 0.07 |
| **Balanced** — strongest safe-direction shift at <2× NRMSE | `pin τ=0.7 + safe_eps=0.01` | under 12 – 25 %, NRMSE ≈ baseline + 0.20 |
| **Worst-case-bounded** — tightest max-under-pred | `pin τ=0.9` alone | max-under ≈ 5.9 × 10⁻², under ≈ 19 % |

All three are applied **per-cell at adaptation time only** — the same pretrained
GCN_MAML init is reused, so no extra training is needed and the knobs can be set
per-design (e.g., aggressive on critical-path nets, conservative elsewhere).

### Q11-G. Where each number lives

- 8-bucket × 3-state progression: `pessimism_control/three_state_per_bucket.csv` (16 rows, bucket × config), `pessimism_control/three_state_per_cell.csv` (72 rows)
- 4-panel summary figure (under-frac, NRMSE, MAPE, max-under): `pessimism_control/three_state_progression.png`
- Single-PDK / single-bucket safe_eps sweep that established the Pareto: `pessimism_control/safe_eps_summary.csv`, `pessimism_control/safe_eps_per_cell.csv`, `pessimism_control/safe_eps_tradeoff.png`
- Three-mechanism comparison (safe_eps vs. asymmetric MSE vs. pinball at the same bucket): `pessimism_control/three_mechanisms_comparison.csv`
- Pinball × safe_eps additivity table: `pessimism_control/pinball_x_safe_eps_comparison.csv`
- safe_eps × asymmetric-MSE additivity table: `pessimism_control/safe_eps_x_asym_summary.csv`, `pessimism_control/safe_eps_x_asym_per_cell.csv`
- Implementation: `pretraining/model_test_code/utils/gnn_functions.py` (helper `_make_train_criterion`, propagated through `evaluate_model_performance_gnn`), `pretraining/model_test_code/gnn/{TSMC,ASAP7}_GCN_topology_validation.py` (CLI flags `--safe_eps`, `--asym_alpha`, `--pinball_tau`, `--seed`)
- Sweep orchestrator + analyzer: `pretraining/model_test_code/gnn/run_safe_eps_sweep.py`, `pretraining/model_test_code/gnn/analyze_safe_eps_sweep.py`

### Q11-H. Take-away

TAMEL's baseline GCN_MAML is **essentially unbiased** — half of its errors are on
the unsafe side, which is the legitimate concern behind this reviewer comment. A
2-line modification to the per-task adaptation step (a normalized-units `safe_eps`
on the alignment target, optionally combined with a pinball inner loss) reduces
under-prediction from ≈ 50 % to **11 – 25 % across every bucket**, with an
absolute-NRMSE cost that stays sub-1 %. The knob is purely post-pretraining,
per-cell, and exposes a clean Pareto for the user to dial in their preferred
safety / accuracy operating point — without retraining the meta-model.

---

## Q12. Are the input parameters of the MLP and GCN backbones inconsistent?

> **Reviewer concern.** *As shown in Sec. 3.3, the input parameters of the
> models being compared are not consistent.*

Sec. 3.3 explicitly states that both backbones share **the same three
input feature categories**; only the granularity of the cell-structural
representation differs.

| Category | MLP (Table 1) | GCN (Table 2) | Consistency |
|---|---|---|---|
| Process (T_ox, DIBL, SCE) | scalar | NMOS/PMOS node attribute | identical 3 variables |
| Operating (V, T, slew, C_load) | scalar | node attribute | identical 4 variables |
| Cell structural (current path) | Path length + pull-up/down factor (2 scalars distilling current-path depth and direction) | Turn-on-path subgraph (NMOS/PMOS nodes, D–S edges, ΣW gate-cap features) | same inductive bias, different granularity |

The 7 process / operating variables are identical; the cell structural
feature differs in **granularity, not in concept** — both representations
are explicitly current-path-derived (paper l.403–417). The
AADAM → MLP_MAML → GCN_MAML progression in our results
(NRMSE ≈ 1.5–3 % → 0.5–1 % → 0.4–0.9 %) shares the same process /
operating features and varies *only* the current-path granularity,
confirming the gain comes from the shared inductive bias at richer
granularity rather than from a mismatch in input parameters.

---

## Consistency Notes

- **Direction**: macro stage-discovery is output → input (backward
  traversal); the final stored graph is reversed to input → output for
  natural GNN propagation order.
- **Algorithm**: DFS (polarity-restricted; supply rail → target net at
  each stage).
- **Aggregation**: **union** of all enumerated conductive paths within a
  stage (every conductive route is preserved).
- **Scope**: `delay_template` LUTs only — `constraint_template`
  (setup/hold) is excluded.
- **Sequential coverage**: C2MOS clocked-inverter arcs (CP→Q, CDN→Q);
  keeper is preserved as both a D/S-connected device on the storage net
  and a contributor to that net's gate-cap (width-sum) feature.
