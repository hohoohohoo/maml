# GNN Model Pretraining

이 디렉토리는 GNN 데이터셋을 활용하여 GCN 모델을 pretraining하는 코드를 포함합니다.

## 학습 방식

두 가지 학습 방식을 지원합니다:

### 1. MAML (Model-Agnostic Meta-Learning)
- Meta-learning 기반 few-shot 학습
- Inner/Outer loop 구조
- K=5 support set 샘플링
- Task 단위 학습 (각 cell의 timing arc가 하나의 task)

### 2. Baseline (Mini-batch Training)
- 표준 mini-batch SGD 학습
- Adam optimizer with weight decay
- 전체 데이터에서 랜덤 샘플링

---

## 디렉토리 구조

```
pretraining/model_pretraining_code/gnn/
├── maml_gnn_training_asap7_process.py      # ASAP7 MAML 학습 스크립트
├── maml_gnn_training_tsmc_process.py       # TSMC MAML 학습 스크립트
├── baseline_gnn_training_asap7_process.py  # ASAP7 Baseline 학습 스크립트
├── baseline_gnn_training_tsmc_process.py   # TSMC Baseline 학습 스크립트
├── run_maml_gnn_asap7_process_sweep.sh     # ASAP7 MAML sweep 실행 스크립트
├── run_maml_gnn_tsmc_process_sweep.sh      # TSMC MAML sweep 실행 스크립트
├── run_baseline_gnn_asap7_process_sweep.sh # ASAP7 Baseline sweep 실행 스크립트
├── run_baseline_gnn_tsmc_process_sweep.sh  # TSMC Baseline sweep 실행 스크립트
├── json_configs/                            # Sweep 설정 파일
│   ├── gnn_maml_asap7_process_sweep_config.json
│   ├── gnn_maml_tsmc_process_sweep_config.json
│   ├── gnn_baseline_asap7_process_sweep_config.json
│   └── gnn_baseline_tsmc_process_sweep_config.json
├── utils/                                   # 유틸리티 모듈
└── past_codes/                              # 이전 버전 코드
```

---

## 빠른 시작

### MAML 학습 실행
```bash
cd /home/tkdgn2907/Deepsets_test/MAML/Projects/pretraining/model_pretraining_code/gnn

# TSMC MAML sweep
./run_maml_gnn_tsmc_process_sweep.sh json_configs/gnn_maml_tsmc_process_sweep_config.json

# ASAP7 MAML sweep
./run_maml_gnn_asap7_process_sweep.sh json_configs/gnn_maml_asap7_process_sweep_config.json

# Dry-run (실제 실행 없이 명령어 확인)
./run_maml_gnn_tsmc_process_sweep.sh json_configs/gnn_maml_tsmc_process_sweep_config.json --dry-run
```

### Baseline 학습 실행
```bash
# TSMC Baseline sweep
./run_baseline_gnn_tsmc_process_sweep.sh json_configs/gnn_baseline_tsmc_process_sweep_config.json

# ASAP7 Baseline sweep
./run_baseline_gnn_asap7_process_sweep.sh json_configs/gnn_baseline_asap7_process_sweep_config.json
```

---

## Node Feature 구조 (11D)

| Index | Feature | 설명 |
|-------|---------|------|
| 0 | node_type | 노드 타입 (VDD, VSS, MOS, etc.) |
| 1 | is_pmos | PMOS 여부 (0 or 1) |
| 2 | width | 트랜지스터 width |
| 3 | num_fins | Fin 개수 (ASAP7) / multiplier (TSMC) |
| 4 | voltage | 전압 feature |
| 5 | input_slew | 입력 slew rate |
| 6 | output_load | 출력 load capacitance |
| 7 | param_a | Process parameter A |
| 8 | param_b | Process parameter B |
| 9 | param_c | Process parameter C |
| 10 | temperature | 온도 |

---

## JSON Config 구조

### MAML Config 예시
```json
{
  "experiment_name": "gnn_maml_tsmc_process_sweep",
  "base_config": {
    "data_type": "transition",
    "graph_mode": "full_graph",
    "voltage_mode": "vdd_only",
    "temp_mode": "typical",
    "inputport": false,
    "related_pin_only": true,
    "meta_lr": 0.0001,
    "innerdiv": 10,
    "total_iterations": 300000,
    "K": 5,
    "tasks_per_meta_batch": 16,
    "inner_steps": 1,
    "pooling": "mean",
    "gpu": 2
  },
  "sweep_params": {
    "conv_hidden_dim": [64],
    "num_conv_layers": [2],
    "fc_hidden_dim": [256],
    "num_fc_layers": [2]
  }
}
```

### Baseline Config 예시
```json
{
  "experiment_name": "gnn_baseline_tsmc_process_sweep",
  "base_config": {
    "data_type": "transition",
    "graph_mode": "full_graph",
    "voltage_mode": "vdd_only",
    "related_pin_only": true,
    "lr": 0.0001,
    "wd": 0.005,
    "batch_size": 5,
    "total_iterations": 3000000,
    "pooling": "mean",
    "gpu": 3
  },
  "sweep_params": {
    "conv_hidden_dim": [64],
    "num_conv_layers": [2],
    "fc_hidden_dim": [256],
    "num_fc_layers": [2]
  }
}
```

---

## 주요 파라미터

### 공통 파라미터

| 파라미터 | 설명 | 기본값 |
|----------|------|--------|
| `data_type` | cell (propagation delay) 또는 transition (output slew) | cell |
| `graph_mode` | stage_aware 또는 full_graph | stage_aware |
| `voltage_mode` | all_nodes, vdd_only, vdd_mos | all_nodes |
| `pooling` | mean, max, add, output (output-node-only) | mean |
| `gpu` | 사용할 GPU 번호 | 0 |

### MAML 전용 파라미터

| 파라미터 | 설명 | 기본값 |
|----------|------|--------|
| `meta_lr` | Outer loop learning rate | 0.0001 |
| `innerdiv` | Inner loop learning rate = meta_lr / innerdiv | 10 |
| `inner_steps` | Inner loop gradient steps | 1 |
| `K` | Support set 샘플 개수 | 5 |
| `tasks_per_meta_batch` | Meta-batch 당 task 수 | 16 |
| `total_iterations` | 전체 meta-iteration 수 | 300000 |

### Baseline 전용 파라미터

| 파라미터 | 설명 | 기본값 |
|----------|------|--------|
| `lr` | Learning rate | 0.0001 |
| `wd` | Weight decay (L2 regularization) | 0.005 |
| `batch_size` | Mini-batch 크기 | 5 |
| `total_iterations` | 전체 iteration 수 | 3000000 |

### 아키텍처 파라미터 (Sweep 대상)

| 파라미터 | 설명 | 예시 값 |
|----------|------|---------|
| `conv_hidden_dim` | GCN hidden dimension | [32, 64, 128] |
| `num_conv_layers` | GCN layer 수 | [2, 3, 4] |
| `fc_hidden_dim` | FC hidden dimension | [64, 128, 256] |
| `num_fc_layers` | FC layer 수 | [2, 3] |

---

## Topology 옵션

### Graph Mode

| 옵션 | 설명 |
|------|------|
| `stage_aware` | Pull-up/Pull-down 경로만 추출 |
| `full_graph` | 전체 트랜지스터 포함 |

### Voltage Mode

| 옵션 | 설명 |
|------|------|
| `all_nodes` | 모든 노드에 전압 할당 |
| `vdd_only` | VDD 노드에만 전압, 나머지 0 |
| `vdd_mos` | VDD와 MOS 노드에만 전압 |

### Input Slew Mode

| 옵션 | 파라미터 | 설명 |
|------|----------|------|
| all (기본) | `related_pin_only: false` | 모든 입력 포트에 slew 할당 |
| related_pin_only | `related_pin_only: true` | related_pin에만 slew 할당 |

### Pooling 전략

| 옵션 | 설명 |
|------|------|
| `mean` | Global mean pooling (기본) |
| `max` | Global max pooling |
| `add` | Global sum pooling |
| `output` | Output node만 사용 (no pooling) |

---

## 출력 모델 경로

### MAML 모델
```
pretrained_models/gnn_maml_{pdk}_process_final{suffix}/
├── gnn_maml_{pdk}_process_{data_type}_{graph_mode}_innerdiv{X}_meta{Y}_iter{N}_inner{S}_{arch}.pth
└── ...
```

### Baseline 모델
```
pretrained_models/gnn_baseline_{pdk}_process_final{suffix}/
├── gnn_baseline_{pdk}_process_{data_type}_{graph_mode}_iter{N}_{arch}.pth
└── ...
```

**Suffix 예시:**
- `_vdd_only`: voltage_mode=vdd_only
- `_relpin`: related_pin_only=true
- `_inputport`: inputport topology 사용

---

## 학습 재개 (Resume)

### Auto Resume
```json
{
  "resume_config": {
    "auto_resume": true
  }
}
```
가장 최근 checkpoint를 자동으로 찾아서 학습 재개

### 특정 Checkpoint에서 재개
```json
{
  "resume_config": {
    "resume": "/path/to/checkpoint.pth",
    "additional_iterations": 100000
  }
}
```

---

## Loss Logging

```json
{
  "loss_logging": {
    "enabled": true,
    "log_every": 100,
    "save_dir": null
  }
}
```

| 옵션 | 설명 |
|------|------|
| `enabled` | Loss logging 활성화 |
| `log_every` | 저장 간격 (iterations) |
| `save_dir` | 저장 디렉토리 (null = 기본 위치) |

---

## 입력 데이터셋 경로

### TSMC
```
dataset_all/GNN_dataset_TSMC/
├── train_{data_type}_{graph_mode}{suffix}.pth
└── test_by_{data_type}_{graph_mode}{suffix}/
```

### ASAP7
```
dataset_all/GNN_dataset_ASAP7/
├── train_{data_type}_{graph_mode}{suffix}.pth
└── test_by_{data_type}_{graph_mode}{suffix}/
```

---

## Shell Script 옵션

```bash
./run_maml_gnn_tsmc_process_sweep.sh <config.json> [OPTIONS]

Options:
  --dry-run     실제 실행 없이 명령어 확인
  --no-commit   Git commit 생략
```

---

## 주의사항

1. **GPU 메모리**: Large architecture (conv_hidden_dim=128 이상)는 더 많은 GPU 메모리 필요

2. **데이터셋 필요**: 학습 전 GNN 데이터셋 생성 필요
   ```bash
   # data_processing/gnn/ 에서 실행
   ./data_preprocessing_gnn_process_cached_tsmc.sh
   ```

3. **Sweep 조합**: sweep_params의 모든 조합이 순차적으로 학습됨
   - 예: 4개 × 3개 × 4개 × 3개 = 144개 조합

4. **Checkpoint 간격**: chunk_size마다 checkpoint 저장 (기본 100000)

5. **Memory Mapping**: 대용량 데이터셋은 mmap=True로 로드하여 메모리 효율화
