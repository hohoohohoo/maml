# Topology Processing Pipeline

전체 topology 데이터 처리 workflow를 단일 스크립트로 실행할 수 있는 통합 pipeline입니다.

## Pipeline 구조

```
┌─────────────────────────────────────────────────────────────────┐
│                    Topology Processing Pipeline                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┼─────────────┐
                │             │             │
                ▼             ▼             ▼
         ┌───────────┐ ┌───────────┐ ┌──────────┐
         │   ASAP7   │ │   TSMC    │ │  Other   │
         └───────────┘ └───────────┘ └──────────┘
                │             │
                └─────────────┘
                      │
        ┌─────────────┼─────────────┬─────────────┐
        │             │             │             │
        ▼             ▼             ▼             ▼
   [Step 1]      [Step 2]      [Step 3]       Done
  Preprocess   Pretraining   Validation        ✓
```

## 3단계 Workflow

### Step 1: Data Preprocessing
**스크립트**:
- ASAP7: `data_processing/run_asap7_topology_preprocessing.py`
- TSMC: `data_processing/run_tsmc_topology_preprocessing.py`

- `.lib` 파일을 읽어서 topology 데이터 추출
- Cell delay 또는 Transition delay 처리
- Tensor 형태로 저장

**출력**:
- ASAP7: `dataset_ASAP7_topology/processed/`
- TSMC: `dataset_TSMC_topology/processed/`

### Step 2: Model Pretraining
**스크립트**: `pretraining/model_pretraining_code/run_topology_pretraining.py`

- Train 데이터로 모델 학습 (MLP 또는 MAML)
- Checkpoint 저장

**출력**:
- 학습된 모델 checkpoint

### Step 3: Model Validation
**스크립트**: `pretraining/model_test_code/run_topology_validation.py`

- Test 데이터로 모델 평가
- Extrapolation 또는 Interpolation 테스트
- 성능 메트릭 계산 및 결과 저장

**출력**:
- Validation 결과

## 사용법

### Interactive Mode (권장)

```bash
cd /home/tkdgn2907/Deepsets_test/MAML/Projects
./run_topology_pipeline.sh -i
```

대화형으로 다음을 선택합니다:
1. 시작 단계 선택 (1-3)
2. 종료 단계 선택 (1-3)
3. PDK 선택 (ASAP7 또는 TSMC)
4. 각 단계별 설정은 해당 스크립트의 interactive mode 사용

**예시 1 - 전체 Pipeline**:
```
Select starting step:
────────────────────────────────────────────────────────────────
  [1] Data Preprocessing (Start from beginning)
  [2] Model Pretraining (Skip preprocessing)
  [3] Model Validation (Only validation)
  [4] Full Pipeline (All steps)

Select step [1-4] (default: 4): 4

Select PDK:
────────────────────────────────────────────────────────────────
  [0] ASAP7
  [1] TSMC (Default)

Select PDK [0/1] (default: 1): 1
```

**예시 2 - Step 3만 실행**:
```
Select starting step:
────────────────────────────────────────────────────────────────
  [1] Data Preprocessing (Start from beginning)
  [2] Model Pretraining (Skip preprocessing)
  [3] Model Validation (Only validation)
  [4] Full Pipeline (All steps)

Select step [1-4] (default: 4): 3

Select ending step:
────────────────────────────────────────────────────────────────
  [3] Model Validation

Select end step [3-3] (default: 3): 3
```

### Command-line Mode

#### 전체 Pipeline 실행

```bash
# ASAP7 intra-topology 전체 pipeline (MAML)
./run_topology_pipeline.sh \
    --pdk asap7 \
    --model maml \
    --config 0 \
    --data-dir 1 \
    --topology-type intra

# TSMC agnostic 전체 pipeline (MLP)
./run_topology_pipeline.sh \
    --pdk tsmc \
    --model mlp \
    --config 3 \
    --dataset-type original_agnostic
```

#### 부분 Pipeline 실행

```bash
# Step 1-2만 실행 (데이터 준비 및 학습)
./run_topology_pipeline.sh \
    --pdk asap7 \
    --model maml \
    --config 0 \
    --start-step 1 \
    --end-step 2 \
    --topology-type intra

# Step 3만 실행 (검증만)
./run_topology_pipeline.sh \
    --pdk tsmc \
    --model mlp \
    --config 2 \
    --start-step 3 \
    --end-step 3
```

## Command-line 옵션

### 필수 옵션 (Command-line mode)

- `--pdk PDK`: PDK 선택 (asap7 또는 tsmc)
- `--model MODEL`: 모델 타입 (mlp 또는 maml) - Step 2 이상 필요
- `--config CONFIG`: Dataset configuration (0-3) - Step 2 이상 필요
  - 0: ASAP7 intra-topology
  - 1: ASAP7 technology agnostic
  - 2: TSMC intra-topology
  - 3: TSMC technology agnostic

### Pipeline 제어

- `--start-step STEP`: 시작 단계 (1-3, default: 1)
- `--end-step STEP`: 종료 단계 (1-3, default: 3)
- `-i, --interactive`: Interactive mode 실행

### ASAP7 Preprocessing 옵션

- `--data-dir DIR`: Data directory index (0-3, default: 1)
  - 0: processed
  - 1: processed_simple (default)
  - 2: test_processed
  - 3: test_processed_simple
- `--topology-type TYPE`: Topology type (intra 또는 agnostic)

### TSMC Preprocessing 옵션

- `--dataset-type TYPE`: Dataset type
  - `original_agnostic`: Technology agnostic (default)
  - `original_intra`: Intra topology
  - `nor_nand`: Adding NOR/NAND
  - `seq`: Sequential cells

### Common Preprocessing 옵션

- `--delay-type TYPE`: Delay type (cell 또는 transition, default: cell)

### Pretraining 옵션

- `--data-type TYPE`: Data type (cell 또는 transition)
- `--gpu-id ID`: GPU device ID (default: 0)

### MLP 옵션

- `--model-type TYPE`: MLP model type (aadam 또는 mlp)
- `--num-iterations N`: Number of iterations (default: 300000)

### MAML 옵션

- `--inner N`: Inner loop steps (default: 1)
- `--innerdiv N`: Inner learning rate divisor (default: 100)
- `--meta N`: Meta batch size (default: 32)
- `--num-iterations N`: Number of iterations (default: 300000)

### Validation 옵션

- `--test-mode MODE`: Test mode (extrapolation 또는 interpolation)

## 실행 예시

### 예시 1: ASAP7 Intra-Topology 전체 workflow (MAML)

```bash
# Interactive mode
./run_topology_pipeline.sh -i

# Command-line mode
./run_topology_pipeline.sh \
    --pdk asap7 \
    --model maml \
    --config 0 \
    --data-dir 1 \
    --topology-type intra \
    --delay-type cell \
    --inner 1 \
    --innerdiv 100 \
    --meta 32
```

**실행 과정**:
1. ASAP7 processed_simple 데이터 전처리 (intra-topology)
2. MAML 모델 학습 (config 0)
3. 모델 검증

### 예시 2: TSMC Technology Agnostic 전체 workflow (MLP)

```bash
./run_topology_pipeline.sh \
    --pdk tsmc \
    --model mlp \
    --config 3 \
    --dataset-type original_agnostic \
    --delay-type cell \
    --model-type aadam \
    --num-iterations 300000
```

**실행 과정**:
1. TSMC original agnostic 데이터 전처리
2. MLP 모델 학습 (config 3)
3. 모델 검증

### 예시 3: 데이터 준비만 (Step 1)

```bash
# ASAP7 전처리만
./run_topology_pipeline.sh \
    --pdk asap7 \
    --start-step 1 \
    --end-step 1 \
    --data-dir 1 \
    --topology-type intra \
    --delay-type cell

# TSMC 전처리만
./run_topology_pipeline.sh \
    --pdk tsmc \
    --start-step 1 \
    --end-step 1 \
    --dataset-type original_intra \
    --delay-type transition
```

### 예시 4: 학습 및 검증만 (Step 2-3)

```bash
# 이미 데이터가 준비되어 있는 경우
./run_topology_pipeline.sh \
    --pdk asap7 \
    --model maml \
    --config 0 \
    --start-step 2 \
    --end-step 3 \
    --gpu-id 1
```

### 예시 5: 검증만 (Step 3)

```bash
./run_topology_pipeline.sh \
    --pdk tsmc \
    --model mlp \
    --config 2 \
    --start-step 3 \
    --end-step 3 \
    --test-mode interpolation
```

## 로그 파일

Pipeline 실행 중 모든 로그는 다음 파일에 저장됩니다:

```
/home/tkdgn2907/Deepsets_test/MAML/Projects/topology_pipeline.log
```

각 단계의 성공/실패 여부와 타임스탬프가 기록됩니다.

## 에러 처리

- 각 단계가 실패하면 pipeline이 즉시 중단됩니다
- 로그 파일에서 실패 원인을 확인할 수 있습니다
- 실패한 단계부터 다시 시작할 수 있습니다:

```bash
# Step 2에서 실패한 경우, Step 2부터 재시작
./run_topology_pipeline.sh \
    --pdk asap7 \
    --model maml \
    --config 0 \
    --start-step 2
```

## 진행 상황 표시

Pipeline 실행 중 다음과 같은 상태 표시가 나타납니다:

```
================================================================================
                  Topology Processing Pipeline
================================================================================

────────────────────────────────────────────────────────────────
Pipeline Configuration:
  PDK: TSMC
  Start Step: 1
  End Step: 3
  Log File: /home/.../topology_pipeline.log
────────────────────────────────────────────────────────────────

[STEP 1/3] Data Preprocessing
────────────────────────────────────────────────────────────────
[2025-11-26 12:00:00] Starting: Data Preprocessing
...
✓ Completed: Data Preprocessing
[2025-11-26 12:15:00] SUCCESS: Data Preprocessing

[STEP 2/3] Model Pretraining
────────────────────────────────────────────────────────────────
...
```

## Configuration 매핑

| Config ID | PDK   | Topology Type | Description |
|-----------|-------|---------------|-------------|
| 0         | ASAP7 | Intra         | ASAP7 intra-topology pretraining |
| 1         | ASAP7 | Agnostic      | ASAP7 technology agnostic pretraining |
| 2         | TSMC  | Intra         | TSMC intra-topology pretraining |
| 3         | TSMC  | Agnostic      | TSMC technology agnostic pretraining |

## 주의사항

1. **PDK와 Config 매칭**: PDK와 config가 일치해야 합니다
   - ASAP7 PDK → config 0 또는 1
   - TSMC PDK → config 2 또는 3

2. **디스크 공간**: 각 단계에서 대용량 데이터 파일이 생성되므로 충분한 디스크 공간 필요

3. **실행 시간**: 전체 pipeline은 수 시간이 소요될 수 있습니다

4. **의존성**: Step 2-3은 Step 1이 완료되어야 실행 가능

## 문제 해결

### Pipeline이 중간에 멈춤

로그 파일 확인:
```bash
tail -f /home/tkdgn2907/Deepsets_test/MAML/Projects/topology_pipeline.log
```

### 특정 단계만 재실행

```bash
# Step 2만 다시 실행
./run_topology_pipeline.sh \
    --pdk asap7 \
    --model maml \
    --config 0 \
    --start-step 2 \
    --end-step 2
```

### Help 확인

```bash
./run_topology_pipeline.sh --help
```

## 관련 문서

- [Voltage Variation Pipeline](README_VOLTAGE_VARIATION_PIPELINE.md)
- Individual script documentation in respective directories
