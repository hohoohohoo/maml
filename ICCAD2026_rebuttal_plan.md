# ICCAD 2026 Paper #642 — Rebuttal Plan

> Authoritative working doc for the rebuttal effort. Read this first if you are landing on this task cold.
>
> **Source review**: `./ICCAD2026_review.txt`
> **Paper title**: TAMEL: Topology-Aware Meta-Learning for Few-shot Delay Re-characterization Across Unseen Cell Topologies
> **Deadline**: 2026-06-18 (Thu) 21:00 KST
> **Plan drafted**: 2026-06-10 (Wed)
> **Last updated**: 2026-06-12 (Fri) — pessimism_control asset inventory refreshed with full sweep/orchestrator/analyzer trail; Q11 numerics in `rebuttal_qna_consolidated.md` now sourced from `three_state_per_bucket.csv`
> **Owner**: tkdgn2907

---

## 1. Reviewer Mood Summary

| Reviewer | Likely standing | Tone | Key signal |
|---|---|---|---|
| **A** | Borderline Accept ~ Weak Accept | Positive, constructive | "excellent physical insight", "physically rigorous" — really values Tox/DIBL/SCE decoupling |
| **B** | Weak Reject | Most critical, novelty attack | "novelty of the manuscript is limited ... without new technical insight" + 6 weaknesses + experiment-coverage attacks |
| **C** | Mixed (meta) | Synthesizer | Explicit quote: *"The paper receives mixed ratings ranging from Weak Reject to Borderline."* Confirms A-vs-B split. |
| **D** | Implementation-curious | Method-focused | Asks how current-path graph is built; addressed by Q2 in `rebuttal_qna_consolidated.md`. |

**Net read**: Borderline ~ Weak Reject. Surviving this requires moving **Reviewer B up one notch** while keeping A from drifting down and D satisfied with implementation clarity.

**Suggested word budget**: ~55% to Reviewer B, ~25% to Reviewer A (especially D-FF), ~10% to Reviewer D (Q2 implementation), ~10% to Reviewer C (cross-reference short).

---

## 2. Concern Priority Matrix

| Priority | Concern | Reviewer ref | Mapped Q in rebuttal md | Status |
|---|---|---|---|---|
| **P0** | WNS/TNS/MAE intuitive STA metrics missing | B-W2, B-Q2 | **Q9** (Q9-A/B/C done; Q9-D pending STA) | MAE/MAPE in ps ✅. WNS/TNS deferred until STA runs finish (see §5 Q2). |
| **P0** | Only TT corner shown in Table 5 | B-W5 | (not yet drafted; ties to Q9-D STA run) | **PENDING** — need non-TT corner cell-level table; synthesis-level if time. |
| **P0** | MLP+MAML vs GCN+MAML similar accuracy | B-Q1 | partially Q12 | Conceptual answer drafted (granularity vs concept). Per-bucket ablation table TODO. |
| **P0** | Pessimism bound / under-estimation risk for STA sign-off | A-W2, A-Q2 | **Q11** | ✅ DONE. `pessimism_control/` subdir has safe_eps + pinball results. |
| **P1** | Sequential cell (D-FF) / feedback loop modeling | A-W1, A-Q1, D-Q | **Q2 + Q4 + Q8** | ✅ DONE. C2MOS framing in Q4, DFS construction in Q2, per-task NRMSE evidence for DFCNQD1 / SDFSNQD0 in Q8. |
| **P1** | Dataset too small + training time long | B-W4, B-W6 | **Q10** | ✅ DONE. ~847 M training + ~149 M test samples verified across both PDKs; runtime contextualized vs SPICE (Table 6). |
| **P1** | Input parameters inconsistent across compared models | B-W1 | **Q12** | ✅ DONE (concise). |
| **P2** | Setup/hold (constraint LUT) scope of paper | A clarification | **Q1** | ✅ DONE. Excluded by axis-mismatch argument, future work. |
| **P2** | Turn-on path abstraction definition | (clarification) | **Q3 + Q5 + Q6 + Q7** | ✅ DONE. |
| **P2** | Novelty too limited | B-W3 | (not Qx; cover in narrative) | TODO — argue 3-pillar: (a) physically-decoupled features, (b) current-path topology graph, (c) cross-topology generalization. |
| **P2** | Meta-training overhead | A-W3 | covered in Q10-D | ✅ DONE via Table 6 amortization argument (530×–550× speedup). |
| **P2** | Cross-PDK validation | B-W4 (supporting) | (not yet drafted) | Optional — scripts exist in `pretraining/model_test_code/gnn/`. Defer unless time permits. |

**Insight**: 7 of 12 priority items are now resolved in `rebuttal_qna_consolidated.md`. The remaining items (B-Q1 ablation table, B-W5 non-TT corner table, WNS/TNS from STA runs, novelty narrative) are the gating work for the rebuttal text.

---

## 3. Day-by-Day Timeline (updated)

### ✅ Day 1 — Wed 2026-06-10 (done)
- ✅ Audited per-cell prediction npy files for both backbones × both PDKs
- ✅ Determined D-FF handling: C2MOS / clocked-tristate-inverter, keeper preserved as load + storage-net D/S edge
- ✅ W5 path decision: cell-level guaranteed; synthesis-level opportunistic (in progress, §5 Q2)

### ✅ Day 2 — Thu 2026-06-11 (done)
- ✅ All-cells MAE/MAPE re-computation pipeline (`all_cells_mae_mape/compute_mae_mape.py` + `view_mae_mape.ipynb`)
- ✅ Pessimism distribution analysis + safe_eps / pinball Pareto (`pessimism_control/`)
- ✅ Sequential-cell per-task NRMSE evidence (`seq_cell/` — DFCNQD1, SDFSNQD0; 16/16 stage_aware win)
- ✅ Dataset-scale audit (`dataset_scale/` — verified per-cell train/test task counts for both PDKs)
- ✅ Q1–Q10 drafted in `rebuttal_qna_consolidated.md`

### 🟡 Day 3 — Fri 2026-06-12 (today, partial)
- ✅ Q11 (pessimism narrative) and Q12 (input-parameter consistency) added to `rebuttal_qna_consolidated.md`
- ✅ This plan refreshed
- [ ] B-Q1 ablation: MLP vs GCN × baseline vs MAML, paired-on-tasks table (use existing npy under `all_cells_mae_mape/rebuttal_metrics_per_cell.csv`)
- [ ] B-W5: non-TT corner cell-level table draft (re-process per-corner npy if available; flag missing buckets)

### Day 4 — Sat 2026-06-13
- [ ] First-cut rebuttal *narrative* (paragraphs per reviewer) — translate Q1–Q12 into reviewer-addressed prose
- [ ] B-W3 novelty paragraph: 3-pillar framing
- [ ] Monitor STA runs (see §5 Q2); if finished, drop ΔWNS / ΔTNS / ΔPath-Delay numbers into Q9-D

### Day 5 — Sun 2026-06-14
- [ ] Draft Reviewer B response (P0 + B-W1 + B-W3, longest section)
- [ ] Draft Reviewer A response (D-FF + pessimism + overhead)

### Day 6 — Mon 2026-06-15
- [ ] Draft Reviewer C response (mostly cross-reference) + Reviewer D response (lean on Q2)
- [ ] Lock numbers in all supporting tables/figures
- [ ] Self-review with reviewer-question checklist (every Q1/Q2 explicitly addressed?)

### Day 7 — Tue 2026-06-16
- [ ] Send full draft to advisors/co-authors (target: ≥24-hour review window)
- [ ] While waiting: handle remaining experiment backlog (STA cleanup, cross-PDK if time)

### Day 8 — Wed 2026-06-17
- [ ] Incorporate feedback + trim to word limit
- [ ] Polish figures/tables (caption, fonts, color)
- [ ] Evening: end-to-end read-through #1

### Day 9 — Thu 2026-06-18
- [ ] Morning: final read + typo/format pass
- [ ] Afternoon: submission prep + buffer
- [ ] **21:00 KST: SUBMIT**

---

## 4. Asset Inventory (verify paths before acting)

### Master rebuttal document
- `result_management/iccad2026_rebuttal/rebuttal_qna_consolidated.md` — **Q1–Q12 consolidated answers** (single source of truth for prose). Q1: setup/hold scope; Q2: current-path graph construction (Reviewer D); Q3: turn-on path; Q4: sequential cell C2MOS handling; Q5: drive-strength variants; Q6: no input nodes; Q7: joint comb/seq training; Q8: per-task NRMSE on DFCNQD1/SDFSNQD0; Q9: MAE/MAPE in ps + WNS/TNS placeholder; Q10: dataset scale (847 M train + 149 M test); Q11: pessimism control; Q12: input-feature consistency between MLP and GCN.

### Per-topic subdirectories (under `result_management/iccad2026_rebuttal/`)
- **`seq_cell/`** — Reviewer D's Q on current-path validity for sequential cells.
  - `extract_seq_cell_results.py`, `plot_seq_cell_results.py`
  - `seq_cell_metrics.csv`, `seq_cell_nrmse_pivot_{mean,geomean}.csv`
  - `figures/seq_cell_nrmse_grouped.png`, `figures/seq_cell_improvement_factors.png`, `figures/seq_cell_summary_geomean.txt`
  - `D-FF_handling_findings.md`
- **`all_cells_mae_mape/`** — Reviewer B's Q9 (MAE/MAPE) source.
  - `compute_mae_mape.py`, `view_mae_mape.ipynb`, `load_dfs_from_npy.py`
  - `rebuttal_metrics_per_cell.csv` (per-cell), `rebuttal_metrics_geomean.csv` (bucket)
  - `compose_intuitive_metrics.py`, `intuitive_metrics_topology_agnostic.csv`
  - `pessimism_overall.json`
- **`dataset_scale/`** — Reviewer B's Q10 (dataset size).
  - `build_dataset_summary.py`, `per_cell_task_counts.py`, `train_set_summary.py`
  - `tsmc_cell_families.csv`, `asap7_cell_families.csv`
  - `tsmc_per_cell_task_counts.csv` (test), `tsmc_train_per_cell_task_counts.csv`, `asap7_train_per_cell_task_counts.csv`
  - `DATASET_SCALE_SUMMARY.txt`, `PER_CELL_TASK_SUMMARY.txt`, `TRAIN_SET_SUMMARY.txt`
- **`pessimism_control/`** — Reviewer A's Q11 (under-prediction risk). Final state 2026-06-12.
  - **Headline figure (used in Q11)**: `three_state_progression.png` — 4-panel grid, 8 TSMC buckets × 3 configs (baseline / `safe_eps=0.01` / `pin τ=0.7 + safe_eps=0.01`), metrics: under-pred fraction, NRMSE, MAPE, max-under-pred.
  - **Headline tables (used in Q11-C/D/E)**: `three_state_per_bucket.csv` (24 rows, bucket × config aggregate), `three_state_per_cell.csv` (72 rows, cell × bucket × config; seed=42 paired-on-tasks).
  - **safe_eps Pareto sweep (1 bucket × 6 eps values 0.0–0.2)**: `safe_eps_per_cell.csv`, `safe_eps_summary.csv`, `safe_eps_tradeoff.png` (3-panel: under/over vs ε, NRMSE/MAPE cost, Pareto curve).
  - **Mechanism comparison at same bucket**: `three_mechanisms_comparison.csv` (safe_eps × asym_alpha × pinball_tau, all single-knob configs).
  - **safe_eps × asym_alpha additivity**: `safe_eps_x_asym_per_cell.csv`, `safe_eps_x_asym_summary.csv`.
  - **safe_eps × pinball_tau additivity**: `pinball_x_safe_eps_comparison.csv`.
  - **8-bucket fair comparison (`eps=0.01` vs `pin τ=0.7 + eps=0.01`, no baseline)**: `fair_8bucket_per_cell.csv` (48 rows), `fair_8bucket_summary.csv` (16 rows). Note: `three_state_*.csv` supersedes this with baseline included.
  - **Implementation (knobs)**:
    - `pretraining/model_test_code/utils/gnn_functions.py` — helper `_make_train_criterion(asym_alpha, pinball_tau)` (mutually exclusive); kwargs propagated through `model_functions_at_training_gnn`, `model_functions_with_optim_mode_gnn`, and `evaluate_model_performance_gnn`. `safe_eps` shifts only the support-training target in `evaluate_model_performance_gnn` (true_function1 stays unshifted so reported actuals are unbiased).
    - `pretraining/model_test_code/gnn/{TSMC,ASAP7}_GCN_topology_validation.py` — CLI flags `--safe_eps`, `--asym_alpha`, `--pinball_tau`, `--seed`. Filename suffixes `_safeE{eps}`, `_asymA{a}`, `_pinT{tau}` keep new outputs from clobbering the existing canonical npy files.
  - **Sweep tooling (re-runnable)**:
    - `pretraining/model_test_code/gnn/run_safe_eps_sweep.py` — orchestrator. Args: `--pdk`, `--experiment`, `--data_type`, `--mode`, `--graph_mode`, `--cells`, `--safe_eps_values`, `--asym_alpha`, `--pinball_tau`, `--gpus`, `--num_test_samples`, `--seed`, `--output_prefix`, `--output_dir`, `--dry_run`. Round-robins jobs across GPUs.
    - `pretraining/model_test_code/gnn/analyze_safe_eps_sweep.py` — aggregator. Walks `data_result_npy_directory{,_final}/<prefix>_*_pred.npy`, parses suffix-encoded knobs, emits per-cell CSV and bucket-summary CSV.
  - **Raw run logs**: `pretraining/model_test_code/gnn/safe_eps_sweep_logs/` (one log per validation invocation, cell × eps × gpu).
  - **Raw npy outputs from sweeps** (under `pretraining/model_test_code/gnn/data_result_npy_directory/`, prefixed):
    - `SAFESWEEP_main_*` — safe_eps Pareto (one bucket, 6 eps).
    - `SAFEASYM_e{00,01}a{07,08,09}_*` — safe_eps × asym_alpha grid.
    - `PINBALL_t{07,08,09}_*` — pinball-only sweep.
    - `PINEPS_t{07,08,09}_e01_*` — pinball × safe_eps=0.01 combo.
    - `FAIR_{ag,in}{c,t}{e,i}_{baseline,eps01,pineps}_*` — 8-bucket × 3-config fair comparison (the data behind `three_state_*.csv` and `three_state_progression.png`).
  - **Open extensions if we resume**: (a) ASAP7 equivalents (same sweep on ASAP7 — script supports `--pdk ASAP7` with `--cells` from CELL_FILTER list, ~25 min/bucket on a free GPU); (b) finer `safe_eps` grid around 0.015–0.025 for a smoother Pareto knee; (c) `pin τ=0.5` (= L1-only, no shift) as an additional reference to disentangle "pinball gradient shape" from "asymmetry"; (d) add the safe_eps/pinball knobs to the `_2d.py` validation scripts if 2D V×T runs are needed for the rebuttal too.

### Saved predictions (canonical, used by `all_cells_mae_mape/`)
- `pretraining/model_test_code/gnn/data_result_npy_directory_final/` — GCN per-cell `_pred.npy` / `_act.npy`
- `pretraining/model_test_code/MLP/data_result_npy_directory_{maml,baseline}/` — MLP per-cell pred/act

### Lib file generation (synthesis-level path for Q9-D / B-W5)
- `Lib_file_generation/maml_conv64x2_fc256x2_all_stage_aware_5shot/` — predicted .lib for 5 corners × 5 temps × 61 voltages (1525 files, 22 cells)
- `Lib_file_generation/maml_conv64x2_fc256x2_all_stage_aware_3shot/` — TT_75 only
- `Lib_file_generation/Projects_TSMC_SYN_pred/syn/run_voltage_sweep.sh` — voltage-sweep synthesis driver (verified 2026-06-11; runs DC with TAMEL-predicted .db files; cell setup/hold come from the .lib, not parameters)
- `Lib_file_generation/compare_synthesis_results.py` — QoR diff vs SPICE-golden synthesis (output dir paths still need to be located on the synthesis host)

### Cross-PDK validation (optional, P2)
- `pretraining/model_test_code/gnn/ASAP7_to_TSMC_GCN_validation.py`
- `pretraining/model_test_code/gnn/TSMC_to_ASAP7_GCN_validation.py`
- Opt-in PDK-scale flags: `--pdk_scale_factor`, `--voltage_shift`, `--use_target_norm`

### Datasets
- `dataset_all/GNN_dataset_TSMC/` — train (44 cells, 735 K tasks/target) + test_by_{cell,transition}_stage_aware/ (75 .pth files)
- `dataset_all/GNN_dataset_ASAP7/` — train_cell_stage_aware_full.pth (76 cells, 6.21 M tasks; 116 GB) + train_transition_stage_aware_10pct.pth + test_by_{cell,transition}_stage_aware/

### Pretrained models
- `pretrained_models/gnn_maml_{asap7|tsmc}_process_checkpoints*/`
- `pretrained_models/gnn_baseline_{asap7|tsmc}_process_checkpoints*/`
- Inspect for matching arch suffix `_conv{H}x{L}_fc{H}x{L}` before assuming a checkpoint exists.

---

## 5. Open Questions / Pending Decisions

| # | Question | Why it matters | Status |
|---|---|---|---|
| 1 | B-W5 — synthesis-level vs cell-level for non-TT corners? | Synthesis (WNS/TNS) is the strongest answer to Reviewer B but needs DC access on non-TT corners (~600 runs across 6 designs × 5 V × 5 T × 4 non-TT corners). Cell-level NRMSE/MAPE per corner is 1–2 days. Recommendation stands: **(C) cell-level guaranteed + synthesis opportunistic**. | **PENDING DECISION** — see §3 Day 3 task. |
| 2 | STA runs for WNS / TNS / ΔPath-Delay (Q9-D placeholder) | User noted on 2026-06-11 that STA simulation is still running; Q9-D currently has a placeholder pointing to paper Table 5 partial numbers. | **IN PROGRESS** — drop numbers into Q9-D when ready. |
| 3 | B-Q1 ablation table (MLP vs GCN × baseline vs MAML) | Q12 gives the conceptual answer; reviewer probably also wants a numerical paired-on-tasks comparison. | TODO — build from `rebuttal_metrics_per_cell.csv`, expose a 4-way table per (PDK × data_type × mode). |
| 4 | DFCNQD1 / SDFSNQD0 dataset coverage | Verified 2026-06-11 that these two C2MOS sequential cells were extracted with only 2 of the 5 standard temperatures; Q10-B already annotates the 5/2 = ×2.5 scaled equivalent. No further action needed unless reviewer asks. | RESOLVED. |
| 5 | Novelty narrative (B-W3) | Need a paragraph in Reviewer B's response that frames the contribution as 3 pillars (decoupled physical features + current-path graph + cross-topology generalization), not just "MAML on a GNN." | TODO — draft on Day 4–5. |
| 6 | Where are the TT synthesis baseline results that `compare_synthesis_results.py` reads? | Likely on `ids-sim1` synthesis host (lib file metadata). Need to confirm whether non-TT runs can be launched there and TT results imported. | **PENDING USER DECISION** (depends on #1). |

---

## 5.5. Pretrained-Model Compatibility Conventions (DO NOT mix)

The "ours" GCN_MAML pretrained checkpoints are tied to a specific
`(graph_mode × slew_mode × voltage_mode)` triple. If you build a new
dataset (e.g. constraint LUTs in `pessimism_control/` or a follow-up
ablation), match the convention exactly or the graph shape / feature
slot semantics will mismatch the checkpoint and inference will produce
garbage even if the script doesn't error.

| Backbone           | graph_mode    | slew_mode             | voltage_mode | Canonical PTH dir suffix |
|---|---|---|---|---|
| **stage_aware (ours)** | `stage_aware` | `all` (default)        | `all_nodes` | **no suffix** — e.g. `test_by_cell_stage_aware/`, `train_cell_stage_aware.pth` |
| **full_graph (ours)**  | `full_graph`  | `related_pin_only`     | `vdd_only`  | **`_vddonly_relpin`** — e.g. `test_by_cell_full_graph_vdd_only_relpin/`, `train_cell_full_graph_vdd_only_relpin.pth` |

Notes:
- The bare `_full_graph` (no suffix) variant is NOT used by any current ours-model checkpoint.
- Topology cache: stage_aware ours uses `stage_aware_topology_cache_tsmc_tcbn28hpcplusbwp30p140_110a_lpe_typical.pth` (no `_gatectrl` / `_directmos` / `_weighted` / etc.). The train-PTH metadata may reference a `_gatectrl` path historically; ignore — at run-time the cache_path is overridden via CLI / sweep config to the no-suffix cache.
- For the constraint pessimism follow-up (Phase 2 of `pessimism_control/`), the stage_aware build correctly uses `slew_mode='all'` and the no-suffix cache; output dirs are `test_by_{setup,hold,recovery,removal,non_seq_setup,non_seq_hold}_stage_aware/`.

---

## 6. Conventions for Other Agents

- **Date format**: YYYY-MM-DD KST.
- **Rebuttal writing style**: terse, evidence-led, one paragraph per reviewer concern. No filler. The Q1–Q12 entries in `rebuttal_qna_consolidated.md` are already in this style and should be the prose source.
- **New files for the rebuttal**: place under the topic-specific subdirs in `result_management/iccad2026_rebuttal/` (`seq_cell/`, `all_cells_mae_mape/`, `dataset_scale/`, `pessimism_control/`). The root contains only `rebuttal_qna_consolidated.md`. Do not collapse subdirs back into the root.
- **Reproducibility**: every CSV / figure in this effort is produced by a script in the same subdir. Re-running the script regenerates the artifact. Don't hand-edit derived CSVs.
- **Branch / commit prefix**: `iccad2026-rebuttal-` for any git activity.
- **Trust but verify**: paths and asset locations in this doc are snapshots from 2026-06-12. Re-check via `ls`/`find` before acting. Re-check open questions in §5 before assuming the answer is settled.
- **Update this doc** when an open question is resolved or a Day's tasks complete. Bump "Last updated" at the top.

---

## 7. Strategy Summary (one screen for a fresh agent)

1. **B is the bottleneck.** B sees no novelty and not enough experiments. The rebuttal must (a) add intuitive STA metrics (MAE/MAPE done in Q9-A/B/C; WNS/TNS still queued in Q9-D), (b) show *why* MLP+MAML and GCN+MAML look similar (Q12 conceptual; per-bucket ablation table still TODO), (c) demonstrate dataset is **not** small (Q10: 847 M training + 149 M test samples verified), (d) reframe novelty as the 3-pillar contribution.
2. **A is the upside.** A already likes the physical decoupling. Q4 / Q8 settle the D-FF question (C2MOS keeper preserved via shared storage net + width-sum, empirically validated by 16/16 stage_aware win on DFCNQD1 and SDFSNQD0). Q11 addresses A's pessimism question.
3. **D needs implementation detail.** Q2 covers the stage-wise backward DFS, polarity-restricted traversal, union aggregation, and reverse-to-input-output edge orientation.
4. **C will follow whoever is more convincing.** C's meta-summary already pegged the score band; cross-reference the strongest A/B answers in C's response and keep it short.
5. **Most evidence is already on disk and processed.** The remaining gating work is (a) STA runs landing for Q9-D, (b) the B-Q1 ablation table, (c) B-W5 non-TT corner table, (d) the B-W3 novelty paragraph. Everything else is prose translation of Q1–Q12.

---

## 8. Status Snapshot (2026-06-12)

| Block | State |
|---|---|
| `rebuttal_qna_consolidated.md` Q1–Q12 | **Complete** (placeholder in Q9-D awaiting STA) |
| Per-topic subdirs (4) | **Complete** — every figure/CSV reproducible from in-dir scripts |
| Day 1–2 tasks | **Complete** |
| Day 3 tasks | Partial — Q11/Q12 added; B-Q1 ablation + B-W5 non-TT outstanding |
| STA runs (Q9-D) | In progress |
| Reviewer-prose drafting | Not started (planned Day 4–5) |
