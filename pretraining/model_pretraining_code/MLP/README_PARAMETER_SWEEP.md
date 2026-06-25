# MAML Parameter Sweep Automation

JSON 기반 설정 파일을 통해 MAML 학습의 여러 파라미터 조합을 자동으로 실험할 수 있는 도구입니다.

## 주요 기능

- ✅ **Grid Search**: 여러 파라미터의 모든 조합을 자동으로 생성하여 실험
- ✅ **Fixed Experiments**: 특정 실험 조합을 미리 정의하여 실행
- ✅ **실험 관리**: 각 실험의 로그, 상태, 소요 시간 자동 기록
- ✅ **부분 실행**: 특정 범위의 실험만 선택적으로 실행
- ✅ **Dry Run**: 실제 실행 전 어떤 명령이 실행될지 미리 확인

## 빠른 시작

### 1. 예제 설정 파일 생성

```bash
python run_parameter_sweep.py --create-example
```

이 명령은 네 개의 예제 파일을 생성합니다:
- `example_sweep_config.json` - Grid search (리스트 방식)
- `example_sweep_config_range.json` - Grid search (range 방식, min/max/step)
- `example_sweep_config_mixed.json` - Grid search (리스트 + range 혼합)
- `example_sweep_config_fixed.json` - Fixed experiments (수동 정의)

### 2. Dry Run으로 확인

```bash
python run_parameter_sweep.py --config example_sweep_config.json --dry-run
```

### 3. 실제 실행

```bash
python run_parameter_sweep.py --config example_sweep_config.json
```

## 설정 파일 형식

### 방법 1: Grid Search (자동 조합 생성)

모든 파라미터 조합을 자동으로 생성합니다.

#### 1-A. 리스트 방식 (직접 값 지정)

```json
{
  "experiment_name": "maml_layer_sweep",
  "description": "Layer length와 inner loop 조합 실험",
  "base_config": {
    "dataset_config": 0,
    "data_type": "cell",
    "gpu": "0",
    "num_iterations": 100000,
    "meta": 32
  },
  "sweep_params": {
    "layer_length": [40, 64, 128],
    "inner": [1, 2],
    "innerdiv": [50, 100]
  }
}
```

**생성되는 실험 수**: 3 × 2 × 2 = **12개 실험**

#### 1-B. Range 방식 (min/max/step 지정)

하드코딩 없이 범위를 지정하여 자동 생성합니다.

```json
{
  "experiment_name": "maml_layer_sweep_range",
  "description": "Range 방식으로 파라미터 범위 지정",
  "base_config": {
    "dataset_config": 0,
    "data_type": "cell",
    "gpu": "0",
    "num_iterations": 100000,
    "meta": 32
  },
  "sweep_params": {
    "layer_length": {
      "min": 40,
      "max": 128,
      "step": 20
    },
    "inner": {
      "min": 1,
      "max": 3,
      "step": 1
    },
    "innerdiv": {
      "min": 50,
      "max": 100,
      "step": 25
    }
  }
}
```

**생성되는 값**:
- `layer_length`: [40, 60, 80, 100, 120] (5개)
- `inner`: [1, 2, 3] (3개)
- `innerdiv`: [50, 75, 100] (3개)

**생성되는 실험 수**: 5 × 3 × 3 = **45개 실험**

#### 1-C. 혼합 방식 (리스트 + Range)

두 방식을 함께 사용할 수 있습니다.

```json
{
  "experiment_name": "maml_mixed_sweep",
  "description": "리스트와 Range 혼합",
  "base_config": {
    "dataset_config": 0,
    "data_type": "cell",
    "gpu": "0",
    "num_iterations": 100000
  },
  "sweep_params": {
    "layer_length": {
      "min": 40,
      "max": 256,
      "step": 32
    },
    "inner": [1, 2],
    "innerdiv": [50, 100],
    "meta": {
      "min": 16,
      "max": 64,
      "step": 16
    }
  }
}
```

**생성되는 값**:
- `layer_length`: [40, 72, 104, 136, 168, 200, 232] (range)
- `inner`: [1, 2] (list)
- `innerdiv`: [50, 100] (list)
- `meta`: [16, 32, 48, 64] (range)

**생성되는 실험 수**: 7 × 2 × 2 × 4 = **112개 실험**

### 방법 2: Fixed Experiments (수동 정의)

특정 실험 조합만 선택적으로 실행합니다.

```json
{
  "experiment_name": "maml_custom_experiments",
  "description": "특별히 선택한 실험들",
  "base_config": {
    "dataset_config": 0,
    "data_type": "cell",
    "gpu": "0",
    "num_iterations": 100000
  },
  "fixed_experiments": [
    {
      "layer_length": 40,
      "inner": 1,
      "innerdiv": 100,
      "meta": 32
    },
    {
      "layer_length": 64,
      "inner": 2,
      "innerdiv": 50,
      "meta": 64
    },
    {
      "layer_length": 128,
      "inner": 1,
      "innerdiv": 100,
      "meta": 16
    }
  ]
}
```

**생성되는 실험 수**: **3개 실험** (정확히 정의한 만큼)

## 사용 가능한 파라미터

설정 파일에서 사용할 수 있는 모든 MAML 파라미터:

| 파라미터 | 설명 | 예시 값 |
|---------|------|--------|
| `dataset_config` | 데이터셋 설정 (0-3) | 0, 1, 2, 3 |
| `data_type` | 데이터 타입 | "cell", "transition" |
| `gpu` | GPU ID | "0", "1" |
| `inner` | Inner loop steps | 1, 2, 3 |
| `innerdiv` | Inner LR divisor | 50, 100, 200 |
| `meta` | Meta batch size | 16, 32, 64 |
| `num_iterations` | 학습 iteration 수 | 50000, 100000, 300000 |
| `layer_length` | Hidden layer size | 40, 64, 128, 256 |
| `auto_resume` | 자동 resume | true, false |
| `resume` | Resume 모델 경로 | "/path/to/model.pth" |

## 고급 사용법

### 부분 실행 (특정 실험만 실행)

총 12개 실험 중 5번부터 10번까지만 실행:

```bash
python run_parameter_sweep.py \
    --config sweep_config.json \
    --start-from 5 \
    --max-experiments 6
```

### 실험 중단 후 재개

실험이 중단되었을 때, 특정 번호부터 재개:

```bash
python run_parameter_sweep.py \
    --config sweep_config.json \
    --start-from 7
```

## 결과 구조

실험 실행 후 다음과 같은 구조로 결과가 저장됩니다:

```
sweep_results/
└── maml_layer_sweep_20250128_143025/
    ├── config.json                    # 사용된 설정 파일
    ├── experiment_log.csv             # 모든 실험의 요약
    └── logs/
        ├── exp001.log                 # 실험 1의 상세 로그
        ├── exp002.log                 # 실험 2의 상세 로그
        └── ...
```

### experiment_log.csv 예시

| experiment_name | index | status | duration_seconds | layer_length | inner | innerdiv | meta |
|----------------|-------|--------|-----------------|--------------|-------|----------|------|
| exp_001 | 1 | success | 1234.5 | 40 | 1 | 50 | 32 |
| exp_002 | 2 | success | 1245.2 | 40 | 1 | 100 | 32 |
| exp_003 | 3 | failed | 234.1 | 40 | 2 | 50 | 32 |

## 실전 예시

### 예시 1: Layer length 영향 분석

```json
{
  "experiment_name": "layer_length_analysis",
  "base_config": {
    "dataset_config": 0,
    "data_type": "cell",
    "gpu": "0",
    "num_iterations": 50000,
    "inner": 1,
    "innerdiv": 100,
    "meta": 32
  },
  "sweep_params": {
    "layer_length": [40, 64, 96, 128, 192, 256]
  }
}
```

→ 6개 실험 (layer_length만 변경)

### 예시 2: Learning rate와 batch size 조합

```json
{
  "experiment_name": "lr_batch_sweep",
  "base_config": {
    "dataset_config": 1,
    "data_type": "cell",
    "gpu": "0",
    "num_iterations": 100000,
    "layer_length": 40,
    "inner": 1
  },
  "sweep_params": {
    "innerdiv": [25, 50, 100, 200],
    "meta": [16, 32, 64]
  }
}
```

→ 4 × 3 = 12개 실험

### 예시 3: 데이터셋별 최적 설정 찾기

```json
{
  "experiment_name": "dataset_comparison",
  "base_config": {
    "data_type": "cell",
    "gpu": "0",
    "num_iterations": 100000,
    "layer_length": 64,
    "inner": 1,
    "innerdiv": 100,
    "meta": 32
  },
  "sweep_params": {
    "dataset_config": [0, 1, 2, 3]
  }
}
```

→ 4개 실험 (각 데이터셋별)

## 결과 분석

실험 완료 후 `experiment_log.csv`를 사용하여 분석:

```python
import pandas as pd
import matplotlib.pyplot as plt

# 결과 로드
df = pd.read_csv('sweep_results/maml_layer_sweep_20250128_143025/experiment_log.csv')

# 성공한 실험만 필터링
df_success = df[df['status'] == 'success']

# Layer length vs Duration 분석
plt.figure(figsize=(10, 6))
plt.scatter(df_success['layer_length'], df_success['duration_seconds'])
plt.xlabel('Layer Length')
plt.ylabel('Training Duration (seconds)')
plt.title('Layer Length vs Training Time')
plt.show()

# 파라미터별 성공률
success_rate = df.groupby('layer_length')['status'].apply(
    lambda x: (x == 'success').mean() * 100
)
print(success_rate)
```

## 팁과 모범 사례

1. **시작은 작게**: 먼저 작은 `num_iterations`로 빠르게 여러 조합 테스트
2. **Dry run 활용**: 실행 전 항상 `--dry-run`으로 확인
3. **GPU 관리**: 여러 GPU가 있다면 설정 파일에서 `gpu` 파라미터 활용
4. **점진적 확장**: 좋은 결과를 보인 파라미터 조합 주변을 더 세밀하게 탐색
5. **로그 보관**: `sweep_results/` 디렉토리를 백업하여 실험 이력 관리

## 문제 해결

### Q: 실험이 실패했을 때 어떻게 하나요?

A: `experiment_log.csv`에서 실패한 실험 번호를 확인하고 해당 로그 파일 확인:

```bash
cat sweep_results/[실험명]/logs/exp_003.log
```

### Q: 메모리 부족 에러가 발생하면?

A: `meta` 파라미터를 줄이거나 `layer_length`를 감소시켜보세요.

### Q: 여러 GPU로 병렬 실행하고 싶어요

A: 각 GPU별로 설정 파일을 만들고 동시에 실행:

```bash
# Terminal 1 (GPU 0)
python run_parameter_sweep.py --config sweep_gpu0.json

# Terminal 2 (GPU 1)
python run_parameter_sweep.py --config sweep_gpu1.json
```

## 관련 파일

- `run_parameter_sweep.py` - 메인 스크립트
- `MAML_topology_pretraining.py` - 실제 학습 스크립트
- `example_sweep_config.json` - Grid search (리스트 방식) 예제
- `example_sweep_config_range.json` - Grid search (range 방식) 예제
- `example_sweep_config_mixed.json` - Grid search (혼합 방식) 예제
- `example_sweep_config_fixed.json` - Fixed experiments 예제
