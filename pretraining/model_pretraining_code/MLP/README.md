# MLP Model Pretraining

이 디렉토리는 MLP 데이터셋을 활용하여 MLP 및 MAML 모델을 pretraining하는 코드를 포함합니다.

## 학습 방식

두 가지 학습 방식을 지원합니다:

### 1. MAML (Model-Agnostic Meta-Learning)
- Meta-learning 기반 few-shot 학습
- Inner/Outer loop 구조
- K-shot support set 샘플링
- Task 단위 학습 (각 cell의 timing arc가 하나의 task)
- Model: `maml_optimized.py` → `OptimizedMAML`, `MAMLModel_3hidden`

### 2. MLP (Standard Mini-batch Training)
- 표준 mini-batch SGD 학습
- Adam optimizer with weight decay
- 전체 데이터에서 랜덤 샘플링
- Model: `networks.py` → `MLP_pretraining`

---

## 디렉토리 구조

```
pretraining/model_pretraining_code/MLP/
├── maml_mlp_training.py              # MAML 학습 스크립트
├── baseline_mlp_training.py          # Baseline MLP 학습 스크립트
├── run_mlp_training.py               # Interactive wrapper 스크립트
├── run_maml_mlp_sweep.sh             # MAML sweep 실행 스크립트
├── run_baseline_mlp_sweep.sh         # Baseline MLP sweep 실행 스크립트
├── json_configs/                      # Sweep 설정 파일
│   ├── maml_sweep_config.json
│   ├── maml_single_config.json
│   ├── mlp_sweep_config.json
│   └── mlp_single_config.json
├── past_code/                         # 이전 버전 코드
└── README.md                          # 이 파일
```

---

## 빠른 시작

### MAML 학습 실행
```bash
cd /home/tkdgn2907/Deepsets_test/MAML/Projects/pretraining/model_pretraining_code/MLP

# JSON config를 사용한 sweep 실행
./run_maml_mlp_sweep.sh json_configs/maml_sweep_config.json

# Dry-run (실제 실행 없이 명령어 확인)
./run_maml_mlp_sweep.sh json_configs/maml_sweep_config.json --dry-run

# Git commit 없이 실행
./run_maml_mlp_sweep.sh json_configs/maml_sweep_config.json --no-commit
```

### Baseline MLP 학습 실행
```bash
# JSON config를 사용한 sweep 실행
./run_baseline_mlp_sweep.sh json_configs/mlp_sweep_config.json

# Dry-run
./run_baseline_mlp_sweep.sh json_configs/mlp_sweep_config.json --dry-run
```

### Interactive Mode
```bash
# 대화형 모드로 학습 설정
python run_mlp_training.py
```

---

## Input Feature 구조 (9D)

| Index | Feature | 설명 |
|-------|---------|------|
| 0 | param_a | Process parameter A |
| 1 | param_b | Process parameter B |
| 2 | param_c | Process parameter C |
| 3 | temperature | 온도 |
| 4 | voltage | 전압 |
| 5 | additional_dim | 추가 feature |
| 6 | delay_indicator | Delay 지시자 |
| 7 | input_slew (index_1) | 입력 slew rate |
| 8 | output_load (index_2) | 출력 load capacitance |

**정규화 대상 feature:** index 3, 4, 7, 8 (temperature, voltage, slew, load_cap)

---

## JSON Config 구조

### MAML Config 예시
```json
{
  "experiment_name": "maml_layer_sweep",
  "description": "Sweep over layer lengths and inner loop configurations",
  "base_config": {
    "dataset_config": 2,
    "data_type": "cell",
    "gpu": "2",
    "num_iterations": 300000,
    "inner": 1,
    "layer_length": 40,
    "auto_resume": true
  },
  "sweep_params": {
    "meta": [32, 64],
    "innerdiv": [5, 10, 50, 100]
  },
  "loss_logging": {
    "enabled": true,
    "log_every": 1000,
    "save_dir": null
  }
}
```

### MLP Config 예시
```json
{
  "experiment_name": "mlp_model_sweep",
  "description": "MLP model type and learning rate sweep",
  "base_config": {
    "dataset_config": 3,
    "data_type": "transition",
    "gpu_id": "5",
    "num_iterations": 1200000,
    "learning_rate": 0.0001,
    "model_type": "mlp"
  },
  "sweep_params": {
    "data_type": ["cell", "transition"],
    "dataset_config": [0, 1, 2, 3]
  }
}
```

---

## 주요 파라미터

### 공통 파라미터

| 파라미터 | 설명 | 기본값 |
|----------|------|--------|
| `dataset_config` | 데이터셋 구성 (0-3) | 필수 |
| `data_type` | cell (propagation delay) 또는 transition (output slew) | cell |
| `gpu` / `gpu_id` | 사용할 GPU 번호 | 0 |
| `num_iterations` | 전체 iteration 수 | 300000 |

### MAML 전용 파라미터

| 파라미터 | 설명 | 기본값 |
|----------|------|--------|
| `inner` | Inner loop gradient steps | 1 |
| `innerdiv` | Inner LR divisor (inner_lr = 0.001 / innerdiv) | 100 |
| `meta` | Tasks per meta batch | 32 |
| `layer_length` | Hidden layer size | 40 |
| `auto_resume` | 최신 checkpoint에서 자동 재개 | false |
| `resume` | 특정 checkpoint 경로 지정 | null |

### MLP 전용 파라미터

| 파라미터 | 설명 | 기본값 |
|----------|------|--------|
| `learning_rate` | Learning rate | 0.0001 |
| `model_type` | aadam (hidden=256) 또는 mlp (hidden=40) | aadam |

---

## Dataset Configuration

| Config ID | PDK | Topology Type | 설명 |
|-----------|-----|---------------|------|
| 0 | ASAP7 | intra | ASAP7 Intra-Topology |
| 1 | ASAP7 | agnostic | ASAP7 Technology-Agnostic |
| 2 | TSMC | intra | TSMC Intra-Topology |
| 3 | TSMC | agnostic | TSMC Technology-Agnostic |

---

## 출력 모델 경로

### MAML 모델
```
pretrained_models/training_loss_taskdivide_all/
├── {data_type}_innerdiv{X}_meta{Y}_{topology}_{task_info}_3hidden_({layer})_{iter}_inner{S}_upgraded{tech_suffix}.pth
└── ...

pretrained_models/checkpoints/training_loss_taskdivide_all_checkpoints/
└── (checkpoint files)
```

### MLP 모델
```
pretrained_models/MLP_pretrained_model/
├── training_loss_pretrained_{tech}_{topology}_{data_type}_{model_type}_{iterations}.pth
└── training_loss_checkpoints_{tech}_{topology}_{data_type}_{model_type}_{iterations}/
```

**Topology 이름:**
- `intratopology` / `intra_topology`: Intra-Topology
- `topology_agnostic`: Technology-Agnostic

**Tech suffix:**
- ASAP7: (없음)
- TSMC: `_tsmc`

---

## 학습 재개 (Resume) - MAML Only

### Auto Resume
```json
{
  "base_config": {
    "auto_resume": true
  }
}
```
가장 최근 checkpoint를 자동으로 찾아서 학습 재개

### 특정 Checkpoint에서 재개
```json
{
  "base_config": {
    "resume": "/path/to/checkpoint.pth"
  }
}
```

---

## Loss Logging

```json
{
  "loss_logging": {
    "enabled": true,
    "log_every": 1000,
    "save_dir": null
  }
}
```

| 옵션 | 설명 |
|------|------|
| `enabled` | Loss logging 활성화 |
| `log_every` | 저장 간격 (iterations) |
| `save_dir` | 저장 디렉토리 (null = 기본 위치) |

**출력 위치:**
- MAML: `pretrained_models/loss_logs_maml/`
- MLP: `pretrained_models/loss_logs_mlp/`

---

## Shell Script 옵션

```bash
./run_maml_mlp_sweep.sh <config.json> [OPTIONS]
./run_baseline_mlp_sweep.sh <config.json> [OPTIONS]

Options:
  --dry-run     실제 실행 없이 명령어 확인
  --no-commit   Git commit 생략
```

---

## Model Import 관계

```
model_code/
├── mlp_maml.py           # MLP MAML 모델 (OptimizedMAML, MAMLModel_3hidden)
├── baseline_mlp.py       # Baseline MLP 모델 (MLP_pretraining, MLP_Aadam, MLP)
├── gnn_maml.py           # GNN MAML 모델
├── hetero_gnn_maml.py    # Heterogeneous GNN MAML 모델
└── past_codes/           # 이전 버전 코드 (maml.py 등)

pretraining/model_pretraining_code/utils/
├── dataset_config.py     # 데이터셋 설정 및 로더
├── maml_utils.py         # MAML 유틸리티 함수
└── mlp_utils.py          # MLP 유틸리티 함수
```

**사용 중인 모델 파일:**
- MAML: `mlp_maml.py`
- Baseline MLP: `baseline_mlp.py`

---

## 입력 데이터셋 경로

```
dataset_all/temp_dataset_{PDK}/{topology}_data/
├── train_{data_type}_input.pth
├── train_{data_type}_output.pth
├── test_{cell}_{data_type}_input.pth
└── test_{cell}_{data_type}_output.pth
```

---

## 주의사항

1. **GPU 메모리**: 학습 중 GPU 메모리 사용량 모니터링 가능

2. **데이터셋 필요**: 학습 전 MLP 데이터셋 생성 필요
   ```bash
   # data_processing/MLP/ 에서 실행
   python asap7/run_asap7_topology_preprocessing.py
   python tsmc/run_tsmc_topology_preprocessing.py
   ```

3. **Sweep 조합**: sweep_params의 모든 조합이 순차적으로 학습됨
   - 예: 2개 × 4개 = 8개 조합

4. **Checkpoint 간격**: chunk_size (기본 10000)마다 checkpoint 저장

5. **NaN 감지**: 학습 중 NaN/Inf 감지 시 자동으로 모델 재초기화

