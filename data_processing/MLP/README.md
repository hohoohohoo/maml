# MLP Dataset Preprocessing

이 디렉토리는 ASAP7 및 TSMC PDK의 Liberty 파일을 MLP 학습용 데이터셋으로 변환하는 스크립트를 포함합니다.

## 디렉토리 구조

```
data_processing/MLP/
├── asap7/run_asap7_topology_preprocessing.py   # ASAP7 데이터셋 생성 래퍼
├── tsmc/run_tsmc_topology_preprocessing.py    # TSMC 데이터셋 생성 래퍼
├── asap7/build_and_split_dataset_asap7.py      # ASAP7 데이터셋 빌더 (핵심 로직)
├── tsmc/build_and_split_dataset_tsmc.py       # TSMC 데이터셋 빌더 (핵심 로직)
├── README_DATASET_MERGE.md               # Dataset merge 관련 문서
├── utils/                                 # 유틸리티 모듈
│   ├── datasets.py                       # 데이터셋 로더
│   ├── libdata_extract_MAML_cell.py      # Cell delay 추출
│   ├── libdata_extract_MAML_transition.py # Transition delay 추출
│   ├── transform_sample_MAML_asap7.py    # ASAP7 feature 변환
│   └── transform_sample_MAML_tsmc.py     # TSMC feature 변환
└── past_codes/                           # 이전 버전 코드
```

---

## 빠른 시작

### ASAP7 데이터셋 생성
```bash
cd /home/tkdgn2907/Deepsets_test/MAML/Projects/data_processing/MLP

# Interactive mode (권장)
python asap7/run_asap7_topology_preprocessing.py

# Command-line mode
python asap7/run_asap7_topology_preprocessing.py -i 1 -d cell -t intra
```

### TSMC 데이터셋 생성
```bash
cd /home/tkdgn2907/Deepsets_test/MAML/Projects/data_processing/MLP

# Interactive mode (권장)
python tsmc/run_tsmc_topology_preprocessing.py

# Command-line mode
python tsmc/run_tsmc_topology_preprocessing.py -t original_agnostic -d transition
```

---

## ASAP7 데이터셋 옵션

### 1. Data Directory 선택

| Index | 디렉토리 | Parameter A 값 | 설명 |
|-------|----------|----------------|------|
| 0 | `processed` | 0.625, 0.875, 1.125, 1.375 | 전체 파라미터 범위 |
| 1 | `processed_simple` (기본) | 0.625, 0.875, 1.125, 1.375 | 전체 파라미터 (단순화) |
| 2 | `test_processed` | 0.75, 1.0, 1.25 | 테스트용 파라미터 |
| 3 | `test_processed_simple` | 0.75, 1.0, 1.25 | 테스트용 (단순화) |

### 2. Delay Type 선택

| 옵션 | 설명 | 사용 모듈 |
|------|------|-----------|
| cell (기본) | Cell delay (propagation delay) | `libdata_extract_MAML_cell` |
| transition | Output transition time (slew) | `libdata_extract_MAML_transition` |

### 3. Topology Type 선택

| 옵션 | 설명 | Test Cell Types |
|------|------|-----------------|
| intra (기본) | Intra-Topology | AND2x6, NAND3x2, NOR2xp67, OR2x6 |
| agnostic | Technology-Agnostic | A2O1A1O1Ixp25, AO21x1, AO32x1, O2A1O1Ixp5, OAI22x1 |

### Process Parameters (ASAP7)

**Non-test directories (processed, processed_simple):**
```
param_a: 0.625, 0.875, 1.125, 1.375
param_b: 0.089, 0.06, 0.091, 0.064, 0.093, 0.068, 0.095, 0.072
param_c: 0.35, 0.465, 0.37, 0.473, 0.39, 0.478, 0.41, 0.485
```

**Test directories (test_processed, test_processed_simple):**
```
param_a: 0.75, 1.0, 1.25
param_b: 0.09, 0.062, 0.092, 0.066, 0.094, 0.07
param_c: 0.36, 0.47, 0.38, 0.475, 0.40, 0.48
```

---

## TSMC 데이터셋 옵션

### 1. Dataset Type 선택

| 옵션 | 폴더 패턴 | 설명 | Test Cell Types |
|------|----------|------|-----------------|
| original_agnostic (기본) | `TSMC_??_*` | Technology Agnostic | 14개 (HA, FA, OA, AO 계열) |
| original_intra | `TSMC_??_*` | Intra Topology | 6개 (AN4, ND3, NR3, OR4, XNR3, XOR3) |
| nor_nand | `TSMC_*2_*` | NOR/NAND 추가 | 없음 (전체 train) |
| seq | `TSMC_*seq*` | Sequential Cells | 2개 (DFCNQD1, SDFSNQD0) |

### 2. Delay Type 선택

| 옵션 | 설명 | 사용 모듈 |
|------|------|-----------|
| transition (기본) | Output transition time | `libdata_extract_MAML_transition` |
| cell | Cell delay | `libdata_extract_MAML_cell` |

### Process Parameters (TSMC)

Corner와 Temperature에 따른 NMOS/PMOS 파라미터 쌍:

| Corner | Index | param_a (nmos, pmos) | param_b (nmos, pmos) | param_c (nmos, pmos) |
|--------|-------|----------------------|----------------------|----------------------|
| FF | 0 | 1.427, 1.457 | 0.026, 0.045 | 0.024, 2.000 |
| TT | 1 | 1.430, 1.470 | 0, 0 | 0.024, 2.000 |
| SS | 2 | 1.443, 1.483 | -0.026, -0.05 | 0.024, 2.000 |
| FS | 3 | 1.43, 1.47 | 0.0208, -0.04 | 0.024, 2.000 |
| SF | 4 | 1.43, 1.47 | 0.036, -0.0208 | 0.024, 2.000 |

---

## Node Feature 구조

### ASAP7 Feature (5D Input)

| Index | Feature | 설명 |
|-------|---------|------|
| 0 | voltage | 전압 (normalized) |
| 1 | input_slew | 입력 slew rate |
| 2 | output_load | 출력 load capacitance |
| 3 | param_a | Process parameter A |
| 4 | param_b_or_c | Process parameter B 또는 C (폴더에 따라) |

### TSMC Feature (5D Input)

| Index | Feature | 설명 |
|-------|---------|------|
| 0 | voltage | 전압 (normalized) |
| 1 | input_slew | 입력 slew rate |
| 2 | output_load | 출력 load capacitance |
| 3 | process_param | Corner 기반 process parameter |
| 4 | temperature | 온도 (normalized) |

---

## 출력 파일 구조

### Train 데이터
```
dataset_all/temp_dataset_{PDK}/{topology_type}_data/
├── train_{delay_type}_input.pth     # [N, 5] 입력 텐서
├── train_{delay_type}_output.pth    # [N, 1] 출력 텐서 (delay)
└── temp_combined_data_{delay}_{topology}_{dir_idx}/  # 임시 데이터
```

### Test 데이터
```
dataset_all/temp_dataset_{PDK}/{topology_type}_data/
├── test_{cell_type}_{delay_type}_input.pth
├── test_{cell_type}_{delay_type}_output.pth
└── ...
```

예시:
- `train_cell_input.pth`
- `test_AND2x6_cell_input.pth`
- `test_NAND3x2_transition_output.pth`

---

## Test Cell Types (상세)

### ASAP7 Intra-Topology
```
AND2x6, NAND3x2, NOR2xp67, OR2x6
```

### ASAP7 Technology-Agnostic
```
A2O1A1O1Ixp25, AO21x1, AO32x1, O2A1O1Ixp5, OAI22x1
```

### TSMC Technology-Agnostic
```
HA1D0BWP30P140, FA1D0BWP30P140,
IOA21D0BWP30P140, IOA21D1BWP30P140,
OA21D0BWP30P140, OA21D1BWP30P140,
OA211D0BWP30P140, OA211D1BWP30P140,
IAO21D0BWP30P140, IAO21D1BWP30P140,
AO21D0BWP30P140, AO21D1BWP30P140,
AO211D0BWP30P140, AO211D1BWP30P140
```

### TSMC Intra-Topology
```
AN4D0BWP30P140, ND3D0BWP30P140, NR3D1BWP30P140,
OR4D0BWP30P140, XNR3D1BWP30P140, XOR3D1BWP30P140
```

### TSMC Sequential
```
DFCNQD1BWP30P140, SDFSNQD0BWP30P140
```

---

## Command-line 옵션

### ASAP7

```bash
python asap7/run_asap7_topology_preprocessing.py [OPTIONS]

Options:
  -i, --data-dir-index {0,1,2,3}  데이터 디렉토리 인덱스
  -d, --delay-type {cell,transition}  Delay 타입
  -t, --topology-type {intra,agnostic}  Topology 타입
  --train-only  Train 데이터만 생성
  -y, --yes  확인 프롬프트 건너뛰기
```

### TSMC

```bash
python tsmc/run_tsmc_topology_preprocessing.py [OPTIONS]

Options:
  -t, --dataset-type {original_agnostic,original_intra,nor_nand,seq}
  -d, --delay-type {cell,transition}  Delay 타입
  --train-only  Train 데이터만 생성
  --test-only  Test 데이터만 생성
  -y, --yes  확인 프롬프트 건너뛰기
```

---

## 데이터 처리 흐름

```
1. Liberty 파일 (.lib)
   ↓
2. libdata_extract_MAML_{cell,transition}.py
   - Pin 데이터 추출 (timing arcs, slew, load)
   ↓
3. transform_sample_MAML_{asap7,tsmc}.py
   - Feature 변환 및 정규화
   - Process parameter 추가
   ↓
4. build_and_split_dataset_{asap7,tsmc}.py
   - Train/Test split (cell type 기반)
   - 텐서로 변환 및 저장
   ↓
5. .pth 파일 (Train/Test datasets)
```

---

## 주의사항

1. **Train에서 제외되는 셀**: Test cell types에 해당하는 셀은 자동으로 train에서 제외됨

2. **INV 셀 처리 (TSMC)**: INV 셀은 특별 처리되어 train에만 포함됨

3. **Random Seed**: 재현성을 위해 seed=42 고정

4. **메모리 사용**: 대용량 데이터셋의 경우 충분한 RAM 필요

5. **폴더 매칭**: TSMC dataset type에 따라 다른 폴더 패턴 사용
   - `TSMC_??_*`: 기본 (2글자 corner + temperature)
   - `TSMC_*2_*`: NOR/NAND 추가 셀
   - `TSMC_*seq*`: Sequential 셀

---

## 관련 문서

- [README_DATASET_MERGE.md](README_DATASET_MERGE.md) - Dataset merge 및 train/test split 상세 문서
