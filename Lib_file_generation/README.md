# Lib File Generation with GCN/MLP MAML Predictions

GCN/MLP MAML 모델로 cell delay / transition / 6종 constraint LUT를 예측해서 25 PT (5 corner × 5 temp) 전체에 대한 `.lib` 파일을 생성하는 파이프라인.

## 파이프라인

```
run_lib_generation.py (coordinator, 4 GPU × ~6 PT)
  per PT:
    ├─ predict_comb_lib.py        comb cell 예측  → predicted_comb_sweep/.../all_voltages_<C>_<T>.0/
    ├─ predict_seq_lib.py         seq cell 예측   → predicted_seq/<C>_<T>/
    └─ merge_libs() (in-process)  comb + seq 합침 → predicted_lropt/predicted_<C>_<T>_all_predictions_lropt/
                                                       predicted_TSMC_<C>_<T>_<V>.lib   (V = 060..120, 61 files)
```

Idempotent: 각 stage 의 출력 dir 에 61개 `.lib` 가 이미 있으면 skip.

## 사용법

### 전체 25 PT 실행

```bash
nohup python run_lib_generation.py > run_lib_generation.log 2>&1 &
```

환경변수로 범위/병렬도 조정:

| Env | 기본값 | 설명 |
|---|---|---|
| `CORNERS_FILTER` | (empty → 5 corner 전체) | 예: `FF,SF,FS` |
| `N_GPUS` | 4 | 사용할 GPU 수 |
| `WORKERS_PER_GPU` | 1 | GPU 당 동시 워커 (메모리 안전 기본 1) |

### 단일 stage (디버그/재실행용)

```bash
# 한 PT 의 seq cell 만
python predict_seq_lib.py <corner> <temp> <gpu_id>
# 예: python predict_seq_lib.py FF 0 0

# 한 PT 의 comb cell 만 (run_lib_generation 이 사용하는 호출 형태)
python predict_comb_lib.py \
    --lib_few_shot --all_cells --data_type all \
    --graph_mode stage_aware --mode interpolation --adaptation_method selective_adam \
    --all_voltages --all_corners_temps --corners FF --temps 0 \
    --dataset_dir /.../dataset_all/GNN_dataset_TSMC \
    --lib_dir     /.../dataset_all/TSMC_lib_files \
    --output_dir  /.../Lib_file_generation/predicted_comb_sweep \
    --gpu 0
```

## 핵심 모델 설정

| Stage | data_type | inner_lr | optimizer |
|---|---|---|---|
| comb | cell, transition | 3e-4 (default) | selective_adam (Adam, wd=1e-4, 40 steps) |
| seq delay | cell, transition | 3e-4 | 동일 |
| seq constraint | setup / hold / recovery / removal / non_seq_setup / non_seq_hold | **3e-3** (lr-optimized) | 동일 |

Constraint 6종 모두 동일 lr 적용 — `predict_seq_lib.py` 에서 `args.inner_lr = 3e-3` 으로 일괄 세팅 후 6 cat 루프.

Support points (interpolation 5-shot): voltage idx `[0, 13, 30, 45, 60]` → V = 0.60 / 0.73 / 0.90 / 1.05 / 1.20.

## 대상 셀

- **Comb**: TSMC_lib_files 의 모든 comb cell (`--all_cells`)
- **Seq (3 cells)**: `DFCNQD1BWP30P140`, `SDFSNQD0BWP30P140`, `SDFCSNQD1BWP30P140`

## 입력 데이터 경로

| 자원 | 경로 |
|---|---|
| Train/test PTH (cell, transition, 6 constraint × stage_aware) | `dataset_all/GNN_dataset_TSMC/` |
| Comb support용 lib | `dataset_all/TSMC_lib_files/TSMC_<C>_<T>/` |
| Seq support용 lib | `dataset_all/TSMC_lib_files/TSMC_seq_cell/TSMC_<C>seq_<T>/` |
| Loopclose topology cache | `data_processing/gnn/topology_cache/...typical_loopclose.pth` |
| Pretrained cell MAML | `pretrained_models/gnn_maml_tsmc_process_checkpoints/...cell_stage_aware_..._fc256x2.pth` |
| Pretrained transition MAML | 위 dir 의 `...transition_stage_aware_..._fc256x2_pooloutput.pth` |

## 출력 구조

```
Lib_file_generation/
├── predicted_lropt/                                    # 최종 산출물 (25 dirs)
│   ├── predicted_FF_0_all_predictions_lropt/
│   │   ├── predicted_TSMC_FF_0_060.lib  ...  predicted_TSMC_FF_0_120.lib   (61 files)
│   │   └── ...
│   ├── predicted_FF_25_all_predictions_lropt/
│   └── ...
├── predicted_comb_sweep/      # 중간: comb stage 출력 (재실행 시 cache)
└── predicted_seq/             # 중간: seq stage 출력
```

## Constraint 처리 디테일

`predict_comb_lib.py` 의 `run_predictions_with_adaptation()` 가 `data_type ∈ {setup, hold, recovery, removal, non_seq_setup, non_seq_hold}` 일 때 자동 적용:

1. **norm_stats alias**: `output_load` slot 을 `input_slew` 의 통계로 alias (constraint_template 의 양축이 모두 slew(ns) 이기 때문).
2. **per-task sign_flip**: `y_low` vs `y_high` 비교로 부호 보정 후 adapt, 결과는 원래 부호 공간으로 복원 — hold/recovery/non_seq_hold 같은 음수 LUT 와 cross-zero task 모두 자연스럽게 처리.

## 부속 도구

- `compare_synthesis_results.py` — Original (SPICE) vs Predicted lib 의 synthesis WNS/TNS QoR 비교
