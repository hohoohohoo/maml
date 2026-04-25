# JSON Configuration Files

이 디렉토리는 MAML 학습 및 파라미터 sweep을 위한 JSON 설정 파일들을 관리합니다.

## 파일 분류

### 1. 단일 실험 설정

#### `example_single_config.json`
단일 MAML 학습 실험을 위한 예제 설정입니다.

**사용법:**
```bash
# Python 직접 실행
python MAML_topology_pretraining.py --config json_configs/example_single_config.json

# Shell script 사용
./run_single_from_json.sh json_configs/example_single_config.json
```

**파라미터:**
- dataset_config: 0 (ASAP7 intra-topology)
- layer_length: 40
- inner: 1
- innerdiv: 100
- meta: 32
- num_iterations: 100000

---

### 2. Parameter Sweep 설정

#### `example_sweep_config.json` - 리스트 방식
직접 값들을 리스트로 지정하는 방식입니다.

**생성되는 실험 수:** 3 × 2 × 2 = **12개**

```json
"sweep_params": {
  "layer_length": [40, 64, 128],
  "inner": [1, 2],
  "innerdiv": [50, 100]
}
```

**사용법:**
```bash
python run_parameter_sweep.py --config json_configs/example_sweep_config.json --dry-run
python run_parameter_sweep.py --config json_configs/example_sweep_config.json
```

---

#### `example_sweep_config_range.json` - Range 방식 ⭐
min, max, step으로 범위를 지정하는 방식입니다.

**생성되는 실험 수:** 4 × 3 × 4 = **48개**

```json
"sweep_params": {
  "layer_length": {"min": 40, "max": 100, "step": 20},
  "inner": {"min": 1, "max": 5, "step": 2},
  "innerdiv": {"min": 10, "max": 100, "step": 30}
}
```

**사용법:**
```bash
python run_parameter_sweep.py --config json_configs/example_sweep_config_range.json --dry-run
./run_sweep_from_json.sh json_configs/example_sweep_config_range.json --dry-run
```

---

#### `example_sweep_config_mixed.json` - 혼합 방식
리스트와 range를 함께 사용하는 방식입니다.

**생성되는 실험 수:** 7 × 2 × 2 × 4 = **112개**

```json
"sweep_params": {
  "layer_length": {"min": 40, "max": 256, "step": 32},  // range
  "inner": [1, 2],                                       // list
  "innerdiv": [50, 100],                                 // list
  "meta": {"min": 16, "max": 64, "step": 16}            // range
}
```

---

#### `example_sweep_config_fixed.json` - Fixed Experiments
미리 정의한 특정 실험들만 실행합니다.

**생성되는 실험 수:** **4개** (정확히 지정한 만큼)

```json
"fixed_experiments": [
  {"layer_length": 40, "inner": 1, "innerdiv": 100, "meta": 32},
  {"layer_length": 64, "inner": 1, "innerdiv": 100, "meta": 32},
  {"layer_length": 64, "inner": 2, "innerdiv": 50, "meta": 64},
  {"layer_length": 128, "inner": 1, "innerdiv": 100, "meta": 16}
]
```

---

#### `my_custom_sweep.json` - 커스텀 예제
리스트와 range를 혼합한 커스텀 예제입니다.

**생성되는 실험 수:** 5 × 2 × 3 × 2 = **60개**

---

## 커스텀 설정 파일 만들기

### 단일 실험용

```json
{
  "dataset_config": 0,
  "data_type": "cell",
  "gpu": "0",
  "num_iterations": 100000,
  "layer_length": 64,
  "inner": 1,
  "innerdiv": 100,
  "meta": 32
}
```

저장 후:
```bash
python MAML_topology_pretraining.py --config json_configs/my_experiment.json
```

### Parameter Sweep용

```json
{
  "experiment_name": "my_sweep",
  "description": "My parameter sweep experiment",
  "base_config": {
    "dataset_config": 0,
    "data_type": "cell",
    "gpu": "0",
    "num_iterations": 100000
  },
  "sweep_params": {
    "layer_length": {"min": 40, "max": 128, "step": 20},
    "inner": [1, 2],
    "innerdiv": [50, 100],
    "meta": [32, 64]
  }
}
```

저장 후:
```bash
python run_parameter_sweep.py --config json_configs/my_sweep.json --dry-run
```

## 사용 가능한 파라미터

| 파라미터 | 설명 | 예시 값 |
|---------|------|--------|
| `dataset_config` | 데이터셋 설정 | 0, 1, 2, 3 |
| `data_type` | 데이터 타입 | "cell", "transition" |
| `gpu` | GPU ID | "0", "1" |
| `inner` | Inner loop steps | 1, 2, 3 |
| `innerdiv` | Inner LR divisor | 50, 100, 200 |
| `meta` | Meta batch size | 16, 32, 64 |
| `num_iterations` | 학습 iteration 수 | 50000, 100000, 300000 |
| `layer_length` | Hidden layer size | 40, 64, 128, 256 |
| `resume` | Resume 모델 경로 | "/path/to/model.pth" |
| `auto_resume` | 자동 resume | true, false |

## Dataset Config 옵션

- **0**: ASAP7 intra-topology
- **1**: ASAP7 topology-agnostic
- **2**: TSMC intra-topology
- **3**: TSMC topology-agnostic

---

# MLP 설정 파일

## MLP 파일 목록

### `mlp_single_config.json` - 단일 MLP 실험

MLP 단일 학습 실험을 위한 예제 설정입니다.

**사용법:**
```bash
# Python 직접 실행
python MLP_topology_pretraining.py --config json_configs/mlp_single_config.json

# Shell script 사용
./run_mlp_single.sh json_configs/mlp_single_config.json
```

**파라미터:**
- dataset_config: 0 (ASAP7 intra-topology)
- model_type: "aadam" (hidden=256)
- learning_rate: 0.0001
- num_iterations: 100000

---

### `mlp_sweep_config.json` - MLP Parameter Sweep

MLP 모델 타입과 learning rate를 sweep하는 설정입니다.

**생성되는 실험 수:** 2 × 3 = **6개**

```json
"sweep_params": {
  "model_type": ["aadam", "mlp"],
  "learning_rate": [0.0001, 0.0005, 0.001]
}
```

**사용법:**
```bash
# Dry-run
./run_mlp_sweep.sh json_configs/mlp_sweep_config.json --dry-run

# 실제 실행
./run_mlp_sweep.sh json_configs/mlp_sweep_config.json
```

---

## MLP 사용 가능한 파라미터

| 파라미터 | 설명 | 예시 값 |
|---------|------|--------|
| `dataset_config` | 데이터셋 설정 | 0, 1, 2, 3 |
| `data_type` | 데이터 타입 | "cell", "transition" |
| `gpu_id` | GPU ID | "0", "1" |
| `num_iterations` | 학습 iteration 수 | 100000, 300000 |
| `learning_rate` | Learning rate | 0.0001, 0.0005, 0.001 |
| `model_type` | 모델 타입 | "aadam" (hidden=256), "mlp" (hidden=40) |

## MLP 커스텀 설정 파일 만들기

### 단일 실험용

```json
{
  "dataset_config": 0,
  "data_type": "cell",
  "gpu_id": "0",
  "num_iterations": 200000,
  "learning_rate": 0.0001,
  "model_type": "aadam"
}
```

저장 후:
```bash
python MLP_topology_pretraining.py --config json_configs/my_mlp_experiment.json
```

### Parameter Sweep용

```json
{
  "experiment_name": "mlp_lr_sweep",
  "description": "MLP learning rate sweep",
  "base_config": {
    "dataset_config": 0,
    "data_type": "cell",
    "gpu_id": "0",
    "num_iterations": 100000,
    "model_type": "aadam"
  },
  "sweep_params": {
    "learning_rate": {
      "min": 0.0001,
      "max": 0.001,
      "step": 0.0001
    }
  }
}
```

저장 후:
```bash
./run_mlp_sweep.sh json_configs/mlp_lr_sweep.json --dry-run
```
