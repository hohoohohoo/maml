# Dataset Merge and Train/Test Split

이 디렉토리에는 ASAP7과 TSMC 데이터셋을 merge하고 train/test로 split하는 스크립트들이 포함되어 있습니다.

## 개요

### ASAP7 Dataset
- **조합 수**: 12개 (4 VT types × 3 corners)
- **VT types**: LVT, RVT, SLVT, SRAM
- **Corners**: FF(1), TT(2), SS(3)
- **Cell types**: AO, OA, simple, INVBUF
- **처리 과정**:
  1. 각 VT/corner 조합마다 cell type들을 merge
  2. Cell과 transition 데이터를 80:20으로 train/test split
  3. `taskdivide_<vt>_<corner>` 디렉토리에 저장

### TSMC Dataset
- **조합 수**: 15개 (3 corners × 5 temperatures)
- **Corners**: FF, SS, TT
- **Temperatures**: 0, 25, 50, 75, 100°C
- **처리 과정**:
  1. 각 corner/temperature 조합의 데이터를 80:20으로 train/test split
  2. `taskdivide_<corner>_<temp>` 디렉토리에 저장

## 파일 구조

```
data_processing/
├── run_dataset_merge_split.py          # Main wrapper script
├── create_asap7_merged_datasets.py     # ASAP7 processing script
├── create_tsmc_merged_datasets.py      # TSMC processing script
└── README_DATASET_MERGE.md             # This file
```

## 사용법

### Interactive Mode (권장)

```bash
python run_dataset_merge_split.py
```

대화형으로 dataset type을 선택하고 실행합니다.

### Command-line Mode

```bash
# ASAP7 dataset 처리
python run_dataset_merge_split.py --dataset-type asap7

# TSMC dataset 처리
python run_dataset_merge_split.py --dataset-type tsmc

# Confirmation 없이 실행
python run_dataset_merge_split.py -t asap7 --yes
python run_dataset_merge_split.py -t tsmc -y
```

### 개별 스크립트 직접 실행

```bash
# ASAP7만 처리
python create_asap7_merged_datasets.py

# TSMC만 처리
python create_tsmc_merged_datasets.py
```

## 입력 데이터 경로

### ASAP7
```
/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_ASAP7_dim5/
├── dataset_test5(dim5)_FF/processed/
├── dataset_test5(dim5)_TT/processed/
└── dataset_test5(dim5)_SS/processed/
```

각 processed 디렉토리에는 다음 형식의 파일들이 있어야 합니다:
- `cell_<TYPE>_<VT>_<CORNER>_25_dataset_input.pth`
- `cell_<TYPE>_<VT>_<CORNER>_25_dataset_output.pth`
- `transition_<TYPE>_<VT>_<CORNER>_25_dataset_input.pth`
- `transition_<TYPE>_<VT>_<CORNER>_25_dataset_output.pth`

### TSMC
```
/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_TSMC_dim5/processed/
```

다음 형식의 파일들이 있어야 합니다:
- `cell_TSMC_<CORNER>_<TEMP>_dataset_input.pth`
- `cell_TSMC_<CORNER>_<TEMP>_dataset_output.pth`
- `transition_TSMC_<CORNER>_<TEMP>_dataset_input.pth`
- `transition_TSMC_<CORNER>_<TEMP>_dataset_output.pth`

## 출력 데이터 구조

### ASAP7
```
/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_ASAP7_dim5/
├── taskdivide_lvt_ff/
│   ├── traindatainput/
│   │   ├── cell_train_input.pth
│   │   └── transition_train_input.pth
│   ├── traindataoutput/
│   │   ├── cell_train_output.pth
│   │   └── transition_train_output.pth
│   ├── testdatainput/
│   │   ├── cell_test_input.pth
│   │   └── transition_test_input.pth
│   └── testdataoutput/
│       ├── cell_test_output.pth
│       └── transition_test_output.pth
├── taskdivide_lvt_tt/
├── taskdivide_lvt_ss/
├── taskdivide_rvt_ff/
... (12 combinations total)
```

### TSMC
```
/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_TSMC_dim5/
├── taskdivide_ff_0/
│   ├── traindatainput/
│   │   ├── cell_train_input.pth
│   │   └── transition_train_input.pth
│   ├── traindataoutput/
│   │   ├── cell_train_output.pth
│   │   └── transition_train_output.pth
│   ├── testdatainput/
│   │   ├── cell_test_input.pth
│   │   └── transition_test_input.pth
│   └── testdataoutput/
│       ├── cell_test_output.pth
│       └── transition_test_output.pth
├── taskdivide_ff_25/
├── taskdivide_ff_50/
... (15 combinations total)
```

## 데이터 Split 비율

- **Train**: 80%
- **Test**: 20%
- **Random seed**: 42 (재현성을 위해 고정)

## 주요 기능

1. **자동 디렉토리 생성**: 필요한 모든 출력 디렉토리를 자동으로 생성
2. **에러 처리**: 파일이 없거나 로드 실패 시 해당 조합을 건너뜀
3. **진행 상황 출력**: 각 조합의 처리 과정과 데이터 shape를 출력
4. **재현성**: Random seed를 고정하여 동일한 train/test split 보장

## Workflow Integration

이 스크립트들은 다음 workflow의 일부입니다:

```
1. build_and_split_dataset_test5dim.py
   ↓ (각 조합별로 cell type별 데이터 생성)

2. create_asap7_merged_datasets.py / create_tsmc_merged_datasets.py
   ↓ (cell type들을 merge하고 train/test split)

3. Training scripts (train/test 데이터 사용)
```

## 예상 실행 시간

- **ASAP7**: ~5-10분 (12 combinations)
- **TSMC**: ~3-5분 (15 combinations)

실제 시간은 데이터 크기와 시스템 성능에 따라 달라집니다.

## 문제 해결

### 파일을 찾을 수 없음
- 입력 데이터 경로가 올바른지 확인
- `build_and_split_dataset_test5dim.py`를 먼저 실행했는지 확인

### 메모리 부족
- 큰 데이터셋의 경우 충분한 RAM 필요
- 필요시 swap 메모리 늘리기

### 권한 오류
- 출력 디렉토리에 쓰기 권한이 있는지 확인
- `chmod +x` 명령으로 스크립트 실행 권한 부여

## 연락처

문제나 질문이 있으면 프로젝트 관리자에게 문의하세요.
