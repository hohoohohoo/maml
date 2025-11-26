# Voltage Variation Processing Pipeline

전체 voltage variation 데이터 처리 workflow를 단일 스크립트로 실행할 수 있는 통합 pipeline입니다.

## Pipeline 구조

```
┌─────────────────────────────────────────────────────────────────┐
│                  Voltage Variation Pipeline                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┼─────────────┐
                │             │             │
                ▼             ▼             ▼
         ┌───────────┐ ┌───────────┐ ┌──────────┐
         │   TSMC    │ │  ASAP7    │ │  Other   │
         └───────────┘ └───────────┘ └──────────┘
                │             │
                └─────────────┘
                      │
        ┌─────────────┼─────────────┬─────────────┬─────────────┐
        │             │             │             │             │
        ▼             ▼             ▼             ▼             ▼
   [Step 1]      [Step 2]      [Step 3]      [Step 4]      Done
  Preprocess  Merge & Split  Pretraining   Validation       ✓
```

## 4단계 Workflow

### Step 1: Data Preprocessing
**스크립트**: `data_processing/run_voltage_variation_preprocessing.py`

- `.lib` 파일을 읽어서 voltage variation 데이터 추출
- Cell delay 또는 Transition delay 처리
- Tensor 형태로 저장

**출력**:
- TSMC: `dataset_TSMC_dim5/processed/cell_TSMC_<CORNER>_<TEMP>_dataset_*.pth`
- ASAP7: `dataset_ASAP7_dim5/dataset_test5(dim5)_<CORNER>/processed/cell_<CELL>_<VT>_<NUM>_25_dataset_*.pth`

### Step 2: Data Merge & Train/Test Split
**스크립트**: `data_processing/run_voltage_variation_data_merge_split.py`

- Cell type별 데이터를 merge
- 80:20 비율로 train/test split
- Random seed 42로 재현성 보장

**출력**:
- TSMC: `dataset_TSMC_dim5/taskdivide_<corner>_<temp>/train*.pth, test*.pth`
- ASAP7: `dataset_ASAP7_dim5/taskdivide_<vt>_<corner>/train*.pth, test*.pth`

### Step 3: Model Pretraining
**스크립트**: `pretraining/model_pretraining_code/run_voltage_variation_pretraining.py`

- Train 데이터로 모델 학습
- Checkpoint 저장

**출력**:
- 학습된 모델 checkpoint

### Step 4: Model Validation
**스크립트**: `pretraining/model_test_code/run_voltage_variation_validation.py`

- Test 데이터로 모델 평가
- 성능 메트릭 계산 및 결과 저장

**출력**:
- Validation 결과

## 사용법

### Interactive Mode (권장)

```bash
cd /home/tkdgn2907/Deepsets_test/MAML/Projects
./run_voltage_variation_pipeline.sh -i
```

대화형으로 다음을 선택합니다:
1. 시작 단계 선택 (1-4)
2. 종료 단계 선택 (1-4)
3. 각 단계별 설정은 해당 스크립트의 interactive mode 사용

**예시 1 - 전체 Pipeline**:
```
Select starting step:
────────────────────────────────────────────────────────────────
  [1] Data Preprocessing (Start from beginning)
  [2] Data Merge & Split (Skip preprocessing)
  [3] Model Pretraining (Skip to training)
  [4] Model Validation (Only validation)
  [5] Full Pipeline (All steps)

Select step [1-5] (default: 5): 5
```

**예시 2 - Step 4만 실행**:
```
Select starting step:
────────────────────────────────────────────────────────────────
  [1] Data Preprocessing (Start from beginning)
  [2] Data Merge & Split (Skip preprocessing)
  [3] Model Pretraining (Skip to training)
  [4] Model Validation (Only validation)
  [5] Full Pipeline (All steps)

Select step [1-5] (default: 5): 4

Select ending step:
────────────────────────────────────────────────────────────────
  [4] Model Validation

Select end step [4-4] (default: 4): 4
```

**예시 3 - Step 2-4 실행**:
```
Select starting step:
────────────────────────────────────────────────────────────────
  [1] Data Preprocessing (Start from beginning)
  [2] Data Merge & Split (Skip preprocessing)
  [3] Model Pretraining (Skip to training)
  [4] Model Validation (Only validation)
  [5] Full Pipeline (All steps)

Select step [1-5] (default: 5): 2

Select ending step:
────────────────────────────────────────────────────────────────
  [2] Data Merge & Split
  [3] Model Pretraining
  [4] Model Validation

Select end step [2-4] (default: 4): 4
```

### Command-line Mode

#### 전체 Pipeline 실행

```bash
# ASAP7 LVT-FF 전체 pipeline
./run_voltage_variation_pipeline.sh \
    -t asap7 \
    --vt-type LVT \
    --corner FF

# TSMC FF_0 전체 pipeline
./run_voltage_variation_pipeline.sh \
    -t tsmc \
    --folder TSMC_FF_0

# Transition delay 포함
./run_voltage_variation_pipeline.sh \
    -t asap7 \
    --vt-type RVT \
    --corner TT \
    --delay-type transition
```

#### 부분 Pipeline 실행

```bash
# Step 1-2만 실행 (데이터 준비만)
./run_voltage_variation_pipeline.sh \
    -t asap7 \
    --vt-type LVT \
    --corner FF \
    --start-step 1 \
    --end-step 2

# Step 3-4만 실행 (학습 및 검증만)
./run_voltage_variation_pipeline.sh \
    -t tsmc \
    --folder TSMC_FF_0 \
    --start-step 3 \
    --end-step 4

# Step 1만 실행 (전처리만)
./run_voltage_variation_pipeline.sh \
    -t asap7 \
    --vt-type SLVT \
    --corner SS \
    --start-step 1 \
    --end-step 1
```

## Command-line 옵션

### 필수 옵션

- `-t, --dataset-type TYPE`: Dataset type (tsmc 또는 asap7)

### Pipeline 제어

- `--start-step STEP`: 시작 단계 (1-4, default: 1)
- `--end-step STEP`: 종료 단계 (1-4, default: 4)
- `-i, --interactive`: Interactive mode 실행

### Dataset 옵션

- `--delay-type TYPE`: Delay type (cell 또는 transition, default: cell)

### ASAP7 전용

- `--vt-type VT`: VT type (LVT, RVT, SLVT, SRAM)
- `--corner CORNER`: Corner (FF, TT, SS)
- `--folder FOLDER`: 특정 폴더 지정 (예: AO_LVT_FF)

### TSMC 전용

- `--folder FOLDER`: 특정 폴더 지정 (예: TSMC_FF_0)

## 실행 예시

### 예시 1: ASAP7 전체 workflow

```bash
# Interactive mode
./run_voltage_variation_pipeline.sh -i

# Command-line mode - LVT-FF 조합의 모든 cell type 처리
./run_voltage_variation_pipeline.sh \
    -t asap7 \
    --vt-type LVT \
    --corner FF \
    --delay-type cell
```

**실행 과정**:
1. `AO_LVT_FF`, `OA_LVT_FF`, `simple_LVT_FF`, `INVBUF_LVT_FF` 등 전처리
2. 모든 cell type merge 후 train/test split
3. 모델 학습
4. 모델 검증

### 예시 2: TSMC 특정 폴더 전체 workflow

```bash
./run_voltage_variation_pipeline.sh \
    -t tsmc \
    --folder TSMC_FF_0
```

**실행 과정**:
1. `TSMC_FF_0` 폴더 전처리
2. Train/test split
3. 모델 학습
4. 모델 검증

### 예시 3: 데이터 준비만 (Step 1-2)

```bash
# 전처리 + Split만 수행
./run_voltage_variation_pipeline.sh \
    -t asap7 \
    --vt-type RVT \
    --corner TT \
    --start-step 1 \
    --end-step 2
```

### 예시 4: 학습 및 검증만 (Step 3-4)

```bash
# 이미 데이터가 준비되어 있는 경우
./run_voltage_variation_pipeline.sh \
    -t tsmc \
    --folder TSMC_FF_0 \
    --start-step 3 \
    --end-step 4
```

## 로그 파일

Pipeline 실행 중 모든 로그는 다음 파일에 저장됩니다:

```
/home/tkdgn2907/Deepsets_test/MAML/Projects/voltage_variation_pipeline.log
```

각 단계의 성공/실패 여부와 타임스탬프가 기록됩니다.

## 에러 처리

- 각 단계가 실패하면 pipeline이 즉시 중단됩니다
- 로그 파일에서 실패 원인을 확인할 수 있습니다
- 실패한 단계부터 다시 시작할 수 있습니다:

```bash
# Step 2에서 실패한 경우, Step 2부터 재시작
./run_voltage_variation_pipeline.sh \
    -t asap7 \
    --vt-type LVT \
    --corner FF \
    --start-step 2
```

## 진행 상황 표시

Pipeline 실행 중 다음과 같은 상태 표시가 나타납니다:

```
================================================================================
              Voltage Variation Processing Pipeline
================================================================================

────────────────────────────────────────────────────────────────
Pipeline Configuration:
  Dataset Type: asap7
  Delay Type: cell
  Start Step: 1
  End Step: 4
  VT Type: LVT
  Corner: FF
  Log File: /home/.../voltage_variation_pipeline.log
────────────────────────────────────────────────────────────────

[STEP 1/4] Data Preprocessing
────────────────────────────────────────────────────────────────
[2025-11-26 12:00:00] Starting: Data Preprocessing
...
✓ Completed: Data Preprocessing
[2025-11-26 12:15:00] SUCCESS: Data Preprocessing

[STEP 2/4] Data Merge & Train/Test Split
────────────────────────────────────────────────────────────────
...
```

## 주의사항

1. **디스크 공간**: 각 단계에서 대용량 데이터 파일이 생성되므로 충분한 디스크 공간 필요
2. **실행 시간**: 전체 pipeline은 수 시간이 소요될 수 있습니다
3. **중간 파일**: 각 단계의 출력이 다음 단계의 입력이므로, 중간 파일 삭제 시 주의
4. **의존성**: Step 3-4는 Step 1-2가 완료되어야 실행 가능

## 문제 해결

### Pipeline이 중간에 멈춤

로그 파일 확인:
```bash
tail -f /home/tkdgn2907/Deepsets_test/MAML/Projects/voltage_variation_pipeline.log
```

### 특정 단계만 재실행

```bash
# Step 2만 다시 실행
./run_voltage_variation_pipeline.sh \
    -t asap7 \
    --vt-type LVT \
    --corner FF \
    --start-step 2 \
    --end-step 2
```

### Help 확인

```bash
./run_voltage_variation_pipeline.sh --help
```

## 관련 문서

- [Voltage Variation Preprocessing](data_processing/README_VOLTAGE_VARIATION.md)
- [Dataset Merge & Split](data_processing/README_DATASET_MERGE.md)
