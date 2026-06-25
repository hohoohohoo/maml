# Background experiment snapshot — 2026-06-10 (rebuttal start)

> Frozen record of all background jobs running at the moment we pivot to ICCAD 2026 #642 rebuttal preparation. Do NOT kill — let them finish in the background while rebuttal work proceeds. Results will fold into the rebuttal where applicable (especially MLP vs GCN ablation for B-Q1).

## 1. Why these experiments are running

The 2-D V×T extension work (TSMC + ASAP7, GCN + MLP backbones, K-scaling sweep with two V-support patterns) was started before the review came back. The rebuttal review then arrived. The runs are now **secondary** to the rebuttal but most of them still produce useful evidence:

| Run | Rebuttal usefulness |
|---|---|
| **TSMC MLP intra + agnostic 300k training** | Direct input to B-Q1 (MLP vs GCN similarity explanation) once validation runs |
| **ASAP7 MLP intra + agnostic 300k training** | Same as above for ASAP7 |
| **GCN TSMC V_5-30-55 K=5/9/15 validation** | K-scaling sensitivity (paper Sec. 4.1 extension) — secondary |
| **GCN ASAP7 V_0-30-60 + V_5-30-55 validation (6 configs)** | Same — secondary |
| **ASAP7 MLP 2-D dataset builds** | Prerequisite for ASAP7 MLP training |

## 2. Process inventory (PIDs, log paths, expected completion)

> Last sync 2026-06-10 ~02:30 KST. Re-check with `ps -ef | grep python | grep -v grep` before relying on PIDs.

### GCN validations (background)
| Run | GPU | Cells done | Log | Output dir |
|---|---|---|---|---|
| TSMC V_5-30-55 K=5 sel_adam | 0 | 6/16 (~6h to go) | `/tmp/agn2_K5_seladam_V5-30-55.log` | `pretraining/model_test_code/gnn/data_result_npy_directory_final/V_5-30-55_300k/` |
| TSMC V_5-30-55 K=9 sel_adam | 2 | 6/16 | `/tmp/agn2_K9_seladam_V5-30-55.log` | same |
| TSMC V_5-30-55 K=15 sel_adam | 3 | 6/16 | `/tmp/agn2_K15_seladam_V5-30-55.log` | same |
| ASAP7 V_0-30-60 × K=5/9/15 | 1 | 5/7 each (~2h to go) | `/tmp/asap7_agn2_K{5,9,15}_seladam_V0-30-60.log` | `…_final/ASAP7_V_0-30-60_300k/` |
| ASAP7 V_5-30-55 × K=5/9/15 | 1 | 5/7 each | `/tmp/asap7_agn2_K{5,9,15}_seladam_V5-30-55.log` | `…_final/ASAP7_V_5-30-55_300k/` |

Output saved at cell end → final `.npy` files appear only after each script finishes its full cell sweep. **Do NOT inspect partial output, trust the log.**

### MLP trainings (background)
| Run | GPU | Iter | Log | Final ckpt path |
|---|---|---|---|---|
| TSMC MLP intra 2-D 300k | 0 | chunk 18/30 (180k @ 02:30) | `/tmp/mlp_tsmc_2d_intra_300k.log` | `pretrained_models/<dirname>` |
| TSMC MLP agnostic 2-D 300k | 2 | chunk 18/30 | `/tmp/mlp_tsmc_2d_agn_300k.log` | same |
| ASAP7 MLP intra 2-D 300k | **NOT LAUNCHED** | dataset just finished | — | needs launch when GPU is free |
| ASAP7 MLP agnostic 2-D 300k | **NOT LAUNCHED** | dataset still building | — | needs dataset finish first |

Naming convention (matches `maml_mlp_training_2d.py`):
`{data_type}_innerdiv{ID}_meta{M}_{tech}_{topology_suffix}_train_input_519traintask_full1DMAML_weights_3hidden_({L})_{iter}_inner{i}_upgraded_2d.pth`

### Dataset builds (background)
| Build | Status | Output |
|---|---|---|
| TSMC MLP intra 2-D cell train | ✅ done @ 00:53 | `dataset_TSMC_2d/intra_topology_data/tsmc_intra_topology_train_{input,output}_cell_2d.pth` (1.9 GB) |
| TSMC MLP agnostic 2-D cell train | ✅ done @ 00:53 | `dataset_TSMC_2d/topology_agnostic_data/tsmc_topology_agnostic_train_{input,output}_cell_2d.pth` (1.9 GB) |
| ASAP7 MLP intra 2-D cell train | ✅ done @ 02:14 | `dataset_ASAP7_2d/intra_topology_data/asap7_intra_topology_train_{input,output}_cell_2d.pth` |
| ASAP7 MLP agnostic 2-D cell train | ⏳ ~30min to go | `dataset_ASAP7_2d/topology_agnostic_data/asap7_topology_agnostic_train_{input,output}_cell_2d.pth` |

All MLP train datasets enforce a **TRAIN_TEMPS filter** = `{-25, 12.5, 37.5, 62.5, 87.5, 125}` °C so test temps `{0, 25, 50, 75, 100}` °C remain unseen.

### MLP test data + validation scripts — **NOT WRITTEN YET**
Needed before any MLP 2-D validation can run. Plan parked for the post-rebuttal cycle unless rebuttal explicitly needs MLP 2-D numbers (B-Q1 could be answered from existing 1-D MLP predictions — see §3).

## 3. Mapping to rebuttal concerns

Per `Projects/ICCAD2026_rebuttal_plan.md` §2:

| Rebuttal concern | Existing background experiment | Action item for rebuttal |
|---|---|---|
| **P0 STA metrics (B-Q2)** | None — derived from already-saved 1-D predictions | Re-process saved `.npy` files in `pretraining/model_test_code/gnn/data_result_npy_directory_final/` |
| **P0 Non-TT corner (B-W5)** | None — separate non-TT inference path | New inference, cell-level guaranteed / synthesis opportunistic |
| **P0 MLP vs GCN ablation (B-Q1)** | Partly — 1-D existing predictions; 2-D MLP training in progress | Start with 1-D paired analysis on existing predictions, augment with 2-D MLP when available |
| **P0 Pessimism (A-Q2)** | None | Signed-error distribution from saved `.npy` |
| **P1 Cross-PDK (B-W4)** | None — separate cross-PDK scripts in plan §4 | TSMC ↔ ASAP7 scripts already exist (`ASAP7_to_TSMC_GCN_validation.py`, `TSMC_to_ASAP7_GCN_validation.py`) |
| **P1 D-FF (A-Q1, C)** | None | **Paper §4.1 re-check required** — what was actually done? |

**Critical insight**: All four P0 items are derivable from already-saved predictions or from existing test datasets. The running background experiments mostly support P1 (longer-horizon evidence). **Do not block rebuttal work on these jobs.**

## 4. When each job finishes — what to harvest

| Job | When | Harvest into rebuttal? |
|---|---|---|
| GCN ASAP7 6 configs | ~2026-06-10 04:30 KST | K-scaling extension, secondary to rebuttal |
| GCN TSMC V_5-30-55 3 configs | ~2026-06-10 08:00 KST | Same — secondary |
| MLP TSMC intra/agn 300k | ~2026-06-10 mid-afternoon | If B-Q1 needs 2-D MLP, then yes |
| MLP ASAP7 intra/agn 300k | ~2026-06-11 (once launched) | Same |
| All MLP validation | Not before MLP test data + validation script exist | Post-rebuttal unless explicitly needed |

## 5. Do-not-touch list

- The 9 background python processes listed in §2. They are using GPU 0/1/2/3 + CPU. Killing one means a re-run of multi-hour work and a potential delay to the rebuttal evidence pipeline.
- Files written to `pretraining/model_test_code/gnn/data_result_npy_directory_final/V_5-30-55_300k/`, `…/ASAP7_V_0-30-60_300k/`, `…/ASAP7_V_5-30-55_300k/` — wait until the script logs "Saved results to …".
- Existing TT predictions under `data_result_npy_directory_final/V_0-30-60_300k/` are the **rebuttal P0 source data**. Do not modify or move them.

## 6. Cross-reference

- Authoritative rebuttal plan: `Projects/ICCAD2026_rebuttal_plan.md`
- Original review text: `Projects/ICCAD2026_review.txt`
- 2-D V×T results summary (pre-rebuttal context): `docs/results/2026-06-05-TSMC-2D-results-summary.md`
- 2-D V×T paper-extension plan (parked): `docs/superpowers/plans/2026-05-30-maml-2d-training.md`
- HANDOFF doc (pre-rebuttal): `docs/HANDOFF_2026-06-04.md`
