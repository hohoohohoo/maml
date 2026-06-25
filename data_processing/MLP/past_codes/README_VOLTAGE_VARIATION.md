# Voltage Variation Data Preprocessing

이 스크립트는 TSMC와 ASAP7의 voltage variation 데이터를 처리합니다.

## 개요

### TSMC Voltage Variation
- **입력**: `/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/TSMC_lib_files/TSMC_<CORNER>_<TEMP>/`
- **폴더 패턴**: `TSMC_<CORNER>_<TEMP>` (예: TSMC_FF_0, TSMC_TT_25)
- **Voltage 범위**: 60-120
- **출력**: `/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_TSMC_dim5/processed/`
  - 파일명: `cell_TSMC_<CORNER>_<TEMP>_dataset_input.pth`
  - 예: `cell_TSMC_FF_0_dataset_input.pth`

### ASAP7 Voltage Variation
- **입력**: `/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/ASAP7_lib_files/voltage_variation/<CELL>_<VT>_<CORNER>/`
- **폴더 패턴**: `<CELL>_<VT>_<CORNER>` (예: AO_LVT_FF, OA_RVT_TT)
- **Voltage 범위**: 40-101
- **Corner 매핑**: FF→1, TT→2, SS→3
- **출력**: `/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_ASAP7_dim5/dataset_test5(dim5)_<CORNER>/processed/`
  - 파일명: `cell_<CELL>_<VT>_<CORNER_NUM>_25_dataset_input.pth`
  - 예: `cell_AO_LVT_1_25_dataset_input.pth` (1=FF)

## 사용법

### Interactive Mode (권장)

```bash
python run_voltage_variation_preprocessing.py
```

대화형으로 다음을 선택합니다:
1. Dataset type (TSMC 또는 ASAP7)
2. Delay type (cell 또는 transition)
3. **ASAP7**: VT type과 Corner 선택 → 해당하는 모든 폴더 자동 처리
4. **TSMC**: 처리할 특정 폴더 선택

### Command-line Mode

#### TSMC - 단일 폴더 처리

```bash
python run_voltage_variation_preprocessing.py -t tsmc --folder TSMC_FF_0
python run_voltage_variation_preprocessing.py -t tsmc --folder TSMC_TT_25 --delay-type transition --yes
```

#### ASAP7 - 단일 폴더 처리

```bash
python run_voltage_variation_preprocessing.py -t asap7 --folder AO_LVT_FF
python run_voltage_variation_preprocessing.py -t asap7 --folder OA_RVT_TT --delay-type cell --yes
```

#### ASAP7 - VT/Corner별 일괄 처리 (신규!)

VT type과 Corner를 지정하면 해당하는 **모든 cell type 폴더를 자동으로 찾아서 순차 처리**합니다.

```bash
# LVT + FF 조합의 모든 폴더 처리 (AO_LVT_FF, OA_LVT_FF, simple_LVT_FF, INVBUF_LVT_FF 등)
python run_voltage_variation_preprocessing.py -t asap7 --vt-type LVT --corner FF

# RVT + TT 조합, transition delay
python run_voltage_variation_preprocessing.py -t asap7 --vt-type RVT --corner TT --delay-type transition

# Confirmation 없이 실행
python run_voltage_variation_preprocessing.py -t asap7 --vt-type SLVT --corner SS --yes
```

### Help

```bash
python run_voltage_variation_preprocessing.py --help
```

## Prefix 및 파일명 규칙

### TSMC
- **입력 폴더**: `TSMC_FF_0`
- **Prefix**: `TSMC_FF_0_`
- **입력 lib 파일**: `TSMC_FF_0_060.lib`, `TSMC_FF_0_061.lib`, ...
- **출력 파일**:
  - `cell_TSMC_FF_0_dataset_input.pth`
  - `cell_TSMC_FF_0_dataset_output.pth`
  - `transition_TSMC_FF_0_dataset_input.pth`
  - `transition_TSMC_FF_0_dataset_output.pth`

### ASAP7
- **입력 폴더**: `AO_LVT_FF`
- **Corner 매핑**: FF→1, TT→2, SS→3
- **Prefix**: `AO_LVT_1_25_` (1은 FF를 의미)
- **입력 lib 파일**: `AO_LVT_1_25_040.lib`, `AO_LVT_1_25_041.lib`, ...
- **출력 폴더**: `dataset_test5(dim5)_FF/processed/`
- **출력 파일**:
  - `cell_AO_LVT_1_25_dataset_input.pth`
  - `cell_AO_LVT_1_25_dataset_output.pth`
  - `transition_AO_LVT_1_25_dataset_input.pth`
  - `transition_AO_LVT_1_25_dataset_output.pth`

## 데이터 흐름

```
TSMC:
TSMC_lib_files/TSMC_FF_0/TSMC_FF_0_*.lib
    ↓ (run_voltage_variation_preprocessing.py)
    ↓ (build_and_split_dataset_test5dim.py with prefix=TSMC_FF_0_)
dataset_TSMC_dim5/processed/cell_TSMC_FF_0_dataset_input.pth

ASAP7:
ASAP7_lib_files/voltage_variation/AO_LVT_FF/AO_LVT_1_25_*.lib
    ↓ (run_voltage_variation_preprocessing.py)
    ↓ (build_and_split_dataset_test5dim.py with prefix=AO_LVT_1_25_)
dataset_ASAP7_dim5/dataset_test5(dim5)_FF/processed/cell_AO_LVT_1_25_dataset_input.pth
```

## ASAP7 VT/Corner별 일괄 처리 예제

### Interactive Mode 실행 예시

```
$ python run_voltage_variation_preprocessing.py

================================================================================
             Voltage Variation Data Preprocessing Wrapper
================================================================================

Available dataset types:
--------------------------------------------------------------------------------
  [1] ASAP7 Voltage Variation
      • Path: ASAP7_lib_files/voltage_variation/<CELL>_<VT>_<CORNER>
      ...

Select dataset type [0/1] (default: 0): 1

Select VT Type:
--------------------------------------------------------------------------------
  [0] LVT
  [1] RVT
  [2] SLVT
  [3] SRAM

Select VT type [0-3] (default: 0): 0

Selected VT Type: LVT

Select Corner:
--------------------------------------------------------------------------------
  [0] FF
  [1] TT
  [2] SS

Select corner [0-2] (default: 0): 0

Found 2 matching folders:
  • AO_LVT_FF
  • INVBUF_LVT_FF

================================================================================
Will process all 2 folders sequentially
================================================================================

Proceed with processing all folders? [Y/n]: y

================================================================================
Processing folder 1/2: AO_LVT_FF
================================================================================
...
✅ Successfully processed: AO_LVT_FF

================================================================================
Processing folder 2/2: INVBUF_LVT_FF
================================================================================
...
✅ Successfully processed: INVBUF_LVT_FF

================================================================================
PROCESSING SUMMARY
================================================================================
Total folders: 2
✅ Successful: 2
❌ Failed: 0
================================================================================
```

### Command-line Mode 실행 예시

```bash
$ python run_voltage_variation_preprocessing.py -t asap7 --vt-type LVT --corner FF

Found 2 matching folders for LVT - FF:
  • AO_LVT_FF
  • INVBUF_LVT_FF

================================================================================
Processing folder 1/2: AO_LVT_FF
================================================================================
...
✅ Successfully processed: AO_LVT_FF

================================================================================
Processing folder 2/2: INVBUF_LVT_FF
================================================================================
...
✅ Successfully processed: INVBUF_LVT_FF
```

## 주요 변경사항

### ASAP7
1. **폴더 구조**: `voltage_variation/` 하위의 `<CELL>_<VT>_<CORNER>` 폴더들 처리
2. **Corner 매핑**: Corner 이름(FF, TT, SS)을 숫자(1, 2, 3)로 매핑
3. **Prefix**: `<CELL>_<VT>_<CORNER_NUM>_25_` 형식
4. **출력 디렉토리**: Corner별로 분리된 디렉토리 (`dataset_test5(dim5)_FF/processed/`)
5. **일괄 처리 기능 (신규)**: VT type과 Corner 선택하면 해당하는 모든 cell type 폴더를 자동으로 찾아서 순차 처리

### TSMC
1. **Prefix 수정**: `TSMC_` 접두어 추가
   - 이전: `FF_0_` → 이후: `TSMC_FF_0_`
2. **출력**: 모든 결과가 `processed/` 디렉토리에 저장

## 예상 출력 예제

### ASAP7 (AO_LVT_FF 처리 시)
```
dataset_ASAP7_dim5/
└── dataset_test5(dim5)_FF/
    └── processed/
        ├── cell_AO_LVT_1_25_dataset_input.pth
        ├── cell_AO_LVT_1_25_dataset_output.pth
        ├── transition_AO_LVT_1_25_dataset_input.pth
        └── transition_AO_LVT_1_25_dataset_output.pth
```

### TSMC (TSMC_FF_0 처리 시)
```
dataset_TSMC_dim5/
└── processed/
    ├── cell_TSMC_FF_0_dataset_input.pth
    ├── cell_TSMC_FF_0_dataset_output.pth
    ├── transition_TSMC_FF_0_dataset_input.pth
    └── transition_TSMC_FF_0_dataset_output.pth
```

## 문제 해결

### 폴더를 찾을 수 없음
```
Error: No folders found for dataset type 'asap7'
```

**해결**: ASAP7의 경우 voltage_variation 디렉토리에 폴더가 있는지 확인
```bash
ls /home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/ASAP7_lib_files/voltage_variation/
```

### Command-line mode에서 folder 누락
```
Error: --folder argument is required
```

**해결**: `--folder` 인자를 반드시 지정
```bash
python run_voltage_variation_preprocessing.py -t asap7 --folder AO_LVT_FF
```

### 폴더명 패턴 오류
```
Error: Folder name 'XXX' doesn't match ASAP7 pattern
```

**해결**: 폴더명이 정확한 패턴을 따르는지 확인
- ASAP7: `<CELL>_<VT>_<CORNER>` (예: AO_LVT_FF)
- TSMC: `TSMC_<CORNER>_<TEMP>` (예: TSMC_FF_0)

## 관련 파일

- `run_voltage_variation_preprocessing.py`: Main wrapper script
- `build_and_split_dataset_test5dim.py`: 실제 데이터 처리 스크립트
- `libdata_extract_MAML_cell.py`: Cell delay 추출
- `libdata_extract_MAML_transition.py`: Transition delay 추출
