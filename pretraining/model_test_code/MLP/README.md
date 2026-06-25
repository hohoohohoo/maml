# Model Test Code (MLP / MAML)

ASAP7·TSMC 의 pretrained MAML/MLP 모델을 검증하는 스크립트 모음. 4가지 워크플로 지원:

1. **Topology Validation** — MAML / MLP 단일 모델을 4 config (ASAP7·TSMC × Intra·Agnostic) 에서 평가
2. **Multi-Comparison Sweep** — 2~5개 MAML 모델 (innerdiv / meta / num_iterations / layer_length) 을 동시 비교, 선택적으로 MLP baseline 도 포함
3. **Adaptation-Method Comparison** — selective_adam vs adam 등 adaptation 방식 비교
4. **AADAM Iteration Sweep** — `MLP_Aadam` (256 hidden) 의 학습 iteration 별 성능 비교

## 디렉토리 구조

```
model_test_code/MLP/
├── run_topology_validation.py                  # 유저 친화 wrapper (interactive / CLI)
│
├── topology_validation/                        # Topology validation 우산 (4 종 워크플로)
│   ├── MAML_topology_validation.py            (코어: 단일 MAML)
│   ├── MLP_topology_validation.py             (코어: 단일 MLP)
│   ├── run_maml_topology_validation_{sweep,parallel}.sh
│   ├── run_mlp_topology_validation_{sweep,parallel}.sh
│   ├── configs/  (5 json)
│   │
│   ├── aadam_sweep/                           # 변형 1: aadam iteration 별 MLP sweep
│   │   ├── generate_aadam_sweep_commands.py
│   │   ├── run_aadam_iteration_sweep.sh      # ../MLP_topology_validation.py 사용
│   │   └── configs/
│   │
│   ├── multi_comparison/                      # 변형 2: 2~5 MAML 동시 비교 (자체 main .py)
│   │   ├── MAML_topology_validation_multi_comparison.py
│   │   ├── generate_multi_comparison_commands.py
│   │   ├── run_maml_multi_comparison_sweep.sh
│   │   └── configs/
│   │
│   └── adaptation_comparison/                 # 변형 3: adapt 방식 비교
│       ├── MAML_adaptation_method_comparison_validation.py
│       ├── analyze_adaptation_method_comparison.py
│       ├── run_adaptation_method_comparison_sweep.sh
│       ├── configs/
│       └── results/  (이전 adaptation_method_comparison_results)
│
├── data_result_npy_directory_maml/             # MAML 결과 npy (top-level 유지)
├── data_result_npy_directory_baseline/         # MLP/baseline 결과 npy (top-level 유지)
└── README.md
```

상위 `utils/` (공유):
- `test_dataset_config.py` — config ID 0~5 정의 + 모델 경로
- `data_management_utils.py`, `maml_functions.py`, `mlp_functions.py`, `gnn_functions.py` — 공유 helper
- `parse_*.py` — sweep JSON 파서 (maml_topology / sweep_config / voltage_variation)

## Config ID 매핑 (`utils/test_dataset_config.py`)

| ID | 이름 | Tech | Topology |
|---|---|---|---|
| 0 | ASAP7 Intra Topology | ASAP7 | intra |
| 1 | ASAP7 Topology Agnostic | ASAP7 | agnostic |
| 2 | TSMC Intra Topology | TSMC | intra |
| 3 | TSMC Topology Agnostic | TSMC | agnostic |

(legacy patched 변형 — 구 config 6/7 — 은 위 2/3 으로 통합됨. `combined_data/` train pool 사용.)

## 모델 경로

| 구분 | 경로 |
|---|---|
| MAML (TSMC) | `../../pretrained_models/training_loss_taskdivide_all/*_combined_519traintask_*_tsmc.pth` |
| MAML (ASAP7) | `../../pretrained_models/checkpoints/taskdivide_all_checkpoints/*_519traintask_*_upgraded.pth` |
| MLP | `../../pretrained_models/MLP_pretrained_model/training_loss_pretrained_*` |

`test_dataset_config.py` 의 `maml_model_path` / `mlp_model_path` lambda 가 자동 빌드.

## Quick Start

### 1) Topology Validation (단일 모델)

**Wrapper (interactive 권장)**:
```bash
python run_topology_validation.py
```

**Direct CLI** (워크플로 dir 안에서):
```bash
cd topology_validation

# MAML on ASAP7 Intra
python MAML_topology_validation.py --config 0 --mode extrapolation \
    --cells NAND3x2 OR2x6 --data_type cell --save_results

# MLP on TSMC Agnostic
python MLP_topology_validation.py --config 3 --mode interpolation \
    --model_type aadam --num_iterations 300000 --save_results
```

**Sweep** (각 워크플로 subdir 안에서 실행, cwd = subdir):
```bash
cd topology_validation

# Sequential sweep (dry-run 먼저)
./run_maml_topology_validation_sweep.sh configs/maml_topology_sweep.json --dry-run
./run_maml_topology_validation_sweep.sh configs/maml_topology_sweep.json

# Parallel 변형
./run_maml_topology_validation_parallel.sh configs/maml_validation_sweep.json
./run_mlp_topology_validation_parallel.sh configs/mlp_topology_validation_parallel.json
```

### 2) Multi-Comparison Sweep (2~5 모델 동시)

JSON 예시 (`multi_comparison/configs/maml_multi_comparison_sweep.json`):
```json
{
  "base_config": {"config": 0, "mode": "extrapolation", "inner": 1, "gpu_id": "0", "save_results": true},
  "sweep_params": {
    "data_type": ["transition"],
    "vary": ["innerdiv"],
    "comparison_configs": [
      {"innerdiv": [50, 100, 200], "meta": [32], "num_iterations": [300000]}
    ]
  },
  "mlp_comparison": {"enabled": true, "mlp_model_type": "aadam", "mlp_iterations": 300000}
}
```

실행:
```bash
cd topology_validation/multi_comparison
./run_maml_multi_comparison_sweep.sh configs/maml_multi_comparison_sweep.json --dry-run
./run_maml_multi_comparison_sweep.sh configs/maml_multi_comparison_sweep.json
```

검증 규칙:
- `vary` 로 선택된 param 은 **2~5개 값**, 나머지 param 은 1개씩.
- `mlp_comparison.enabled=true` 면 MLP baseline 도 비교.

### 3) Adaptation-Method Comparison

```bash
cd topology_validation/adaptation_comparison
./run_adaptation_method_comparison_sweep.sh configs/maml_optim_comparison_sweep.json
# 결과: adaptation_comparison/results/*.json
python analyze_adaptation_method_comparison.py     # 결과 분석
```

### 4) AADAM Iteration Sweep (topology_validation 의 하위)

```bash
cd topology_validation/aadam_sweep
./run_aadam_iteration_sweep.sh configs/aadam_iteration_sweep.json
```

## 주요 파라미터

**공통:**
| Parameter | 기본값 | 설명 |
|---|---|---|
| `--mode` | extrapolation | `extrapolation` 또는 `interpolation` |
| `--data_type` | (config default) | `cell` 또는 `transition` |
| `--indices` | mode-dependent | support set 인덱스. extrap: `[5,30,55]`, interp: `[13,30,45]` (endpoint 자동 추가) |
| `--gpu_id` | (config default) | GPU device ID |
| `--save_results` | False | 예측·실측을 `.npy` 로 저장 |
| `--num_test_samples` | 1000000 | 처리할 test sample 수 |

**MAML 전용:**
| Parameter | 기본값 | 설명 |
|---|---|---|
| `--inner` | 1 | inner loop steps |
| `--innerdiv` | 100 | inner lr divisor |
| `--meta` | (config default) | meta batch size |
| `--layer_length` | 40 | hidden layer size |
| `--num_iterations` | 300000 | 학습 iteration |
| `--adaptation_method` | selective_adam | `selective_adam` 또는 `adam` |

**MLP 전용:**
| Parameter | 기본값 | 설명 |
|---|---|---|
| `--model_type` | aadam | `aadam` (256 hidden) 또는 `mlp` (40 hidden) |
| `--num_iterations` | 300000 | pretrained iteration |

## Output (npy)

`--save_results` 면:

```
data_result_npy_directory_maml/{TECH}_{topology}_{cell}_{data_type}_{mode}_MAML_innerdiv{N}_meta{M}_layer40_{iter}_pred.npy
data_result_npy_directory_maml/{TECH}_{topology}_{cell}_{data_type}_{mode}_MAML_innerdiv{N}_meta{M}_layer40_{iter}_act.npy

data_result_npy_directory_baseline/{TECH}_{topology}_{cell}_{data_type}_{mode}_{model_type}_{iter}_pred.npy
data_result_npy_directory_baseline/{TECH}_{topology}_{cell}_{data_type}_{mode}_{model_type}_{iter}_act.npy
```

(`{TECH}` = `ASAP7` / `TSMC`, `{topology}` = `intra` / `agnostic`)

분석 시:
```python
import numpy as np
pred = np.load("data_result_npy_directory_maml/.../*_pred.npy")
act  = np.load("data_result_npy_directory_maml/.../*_act.npy")
```

## Metrics

각 cell 에 대해 4개 region 별:
- **NRMSE** (%) — Total / Left extrap / Right extrap / Interpolation
- **MAPE** (%) — 동일
- **MAE** — MAML 만

## Sweep 스크립트 매핑

| 워크플로 dir / sh | 호출하는 Python | JSON 설정 예시 |
|---|---|---|
| `topology_validation/run_maml_topology_validation_sweep.sh` | `MAML_topology_validation.py` | `configs/maml_topology_sweep.json` |
| `topology_validation/run_maml_topology_validation_parallel.sh` | 동일, 병렬화 | `configs/maml_validation_sweep.json` |
| `topology_validation/run_mlp_topology_validation_sweep.sh` | `MLP_topology_validation.py` | `configs/mlp_topology_validation_sweep.json` |
| `topology_validation/run_mlp_topology_validation_parallel.sh` | 동일, 병렬화 | `configs/mlp_topology_validation_parallel.json` |
| `topology_validation/multi_comparison/run_maml_multi_comparison_sweep.sh` | `MAML_topology_validation_multi_comparison.py` (via `generate_multi_comparison_commands.py`) | `configs/maml_multi_comparison_sweep.json` |
| `topology_validation/adaptation_comparison/run_adaptation_method_comparison_sweep.sh` | `MAML_adaptation_method_comparison_validation.py` | `configs/maml_optim_comparison_sweep.json` |
| `topology_validation/aadam_sweep/run_aadam_iteration_sweep.sh` | `../MLP_topology_validation.py --model_type aadam` (via `generate_aadam_sweep_commands.py`) | `configs/aadam_iteration_sweep.json` |

## Troubleshooting

**모델 파일 없음** — `test_dataset_config.py` 의 `maml_model_path` / `mlp_model_path` 가 생성하는 경로와 실제 ckpt 가 일치하는지 확인. `innerdiv`, `meta`, `inner`, `num_iterations` 가 실제 학습 시 값과 일치해야 함.

**Test data 없음** — `test_data_dir` + `test_input_pattern` 으로 빌드된 경로 확인. TSMC 의 경우 `combined_data/{cell}/` 안에 cell-specific test PTH 가 있어야 함.

**GPU 메모리 부족** — `num_test_samples` 축소, 또는 sweep 시 동시 실행 수 줄이기.

**Dry-run 권장** — sweep 스크립트 모두 `--dry-run` 으로 먼저 생성될 명령어 확인 후 본 실행.
