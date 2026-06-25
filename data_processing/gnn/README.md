# GNN Dataset Preprocessing

이 디렉토리는 ASAP7 및 TSMC PDK의 Liberty 파일을 GNN 학습용 데이터셋으로 변환하는 스크립트를 포함합니다.

## 디렉토리 구조

```
data_processing/gnn/
├── asap7/data_preprocessing_gnn_process_cached_asap7.sh  # ASAP7 데이터셋 생성 스크립트
├── tsmc/data_preprocessing_gnn_process_cached_tsmc.sh   # TSMC 데이터셋 생성 스크립트
├── precompute/precompute_stage_aware_topology.py              # Stage-aware 토폴로지 캐시 생성
├── precompute/precompute_full_graph_topology.py               # Full-graph 토폴로지 캐시 생성
├── asap7/build_gnn_dataset_process_cached_asap7.py        # ASAP7 데이터셋 빌더
├── tsmc/build_gnn_dataset_process_cached_tsmc.py        # TSMC 데이터셋 빌더
├── topology_cache/                                  # 토폴로지 캐시 저장 위치
└── utils/                                           # 유틸리티 모듈
```

## 빠른 시작

### ASAP7 데이터셋 생성
```bash
cd /home/tkdgn2907/Deepsets_test/MAML/Projects/data_processing/gnn
./data_preprocessing_gnn_process_cached_asap7.sh
```

### TSMC 데이터셋 생성
```bash
cd /home/tkdgn2907/Deepsets_test/MAML/Projects/data_processing/gnn
./data_preprocessing_gnn_process_cached_tsmc.sh
```

---

## Graph Mode

두 가지 그래프 모드를 지원합니다:

### 1. Stage-Aware Mode (권장)
- Pull-up/Pull-down 경로만 추출하여 작은 그래프 생성
- Intermediate node (N_2, N_3 등) 포함
- Cell의 동작 방식에 맞는 구조화된 그래프

### 2. Full-Graph Mode
- 모든 트랜지스터를 포함하는 완전한 그래프
- Bidirectional edges 기본 포함
- Intermediate node 없이 MOS 직접 연결

---

## Topology Cache 옵션

토폴로지 캐시는 그래프의 구조(노드, 엣지)를 미리 계산하여 저장합니다.

### 캐시 파일 명명 규칙
```
{mode}_topology_cache_{pdk}_{netlist}{options}.pth
```
예시:
- `stage_aware_topology_cache_asap7sc7p5t_28_L.pth`
- `full_graph_topology_cache_tsmc_tcbn28hpcplusbwp30p140_110a_lpe_typical.pth`
- `stage_aware_topology_cache_asap7sc7p5t_28_L_gatectrl_bidir.pth`

---

## Step 2: Graph Adjacency Matrix Configuration

### (1) Netlist 파일 선택
| PDK | 파일 형식 | 예시 |
|-----|----------|------|
| ASAP7 | CDL | `asap7sc7p5t_28_L.cdl` |
| TSMC | SPI | `tcbn28hpcplusbwp30p140_110a_lpe_typical.spi` |

### (2) Weighted Adjacency Matrix (TSMC only)
SPI 파일의 저항값을 이용하여 가중치 인접 행렬 생성

| 옵션 | 설명 | Edge Weight |
|------|------|-------------|
| Binary (default) | 연결 여부만 표현 | 0 또는 1 |
| Weighted | 저항 기반 가중치 | `1 / (1 + resistance)` |

**Weighted 선택 시 추가 옵션:**
- **Parasitic Capacitance**: 노드별 기생 캐패시턴스 합산을 feature로 추가 (11D → 12D)

### (3) Gate Control Edges (Stage-Aware only)
Intermediate gate 노드에서 트랜지스터로 제어 관계 엣지 추가

| 옵션 | 설명 | Cache Suffix |
|------|------|--------------|
| Disabled (default) | Gate control 없음 | - |
| Enabled | weight=0.5로 엣지 추가 | `_gatectrl` |

```
예시: N_7 (intermediate gate) → PM0, PM1 (트랜지스터)
```

### (4) Input Port Edges (Stage-Aware only)
입력 포트 노드(A, B, C 등)를 그래프에 추가하고 트랜지스터와 연결

| 옵션 | 설명 | Cache Suffix |
|------|------|--------------|
| Disabled (default) | 입력 포트 노드 없음 | - |
| Enabled | weight=0.5로 엣지 추가 | `_inputport` |

```
예시: A (input port) → NM0 (connected transistor)
```

### (5) Bidirectional Edges (Stage-Aware only)
방향성 엣지에 역방향 엣지 추가

| 옵션 | 설명 | Cache Suffix |
|------|------|--------------|
| Disabled (default) | 단방향 (current flow) | - |
| Enabled | 양방향 message passing | `_bidir` |

---

## Step 3: Node Feature Configuration

### 기본 Node Feature 구조 (11D)

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

### (1) Data Type
Liberty 파일에서 추출할 타이밍 정보 선택

| 옵션 | 설명 | 출력값 |
|------|------|--------|
| cell (default) | Cell delay | Propagation delay |
| transition | Output transition | Output slew |

### (2) Voltage Feature Mode
전압 feature를 어느 노드에 할당할지 결정

| 옵션 | 설명 | Dataset Suffix |
|------|------|----------------|
| all_nodes (default) | 모든 노드에 전압 할당 | - |
| vdd_only | VDD 노드에만 전압, 나머지 0 | `_vdd_only` |
| vdd_mos | VDD와 MOS 노드에만 전압 | `_vddmos` |

**노드별 Voltage Feature:**
```
all_nodes:  VDD=0.7, MOS=0.7, Intermediate=0.7
vdd_only:   VDD=0.7, MOS=0,   Intermediate=0
vdd_mos:    VDD=0.7, MOS=0.7, Intermediate=0
```

### (3) Temperature Feature Mode (TSMC only)
온도 feature를 어느 노드에 할당할지 결정

| 옵션 | 설명 |
|------|------|
| mos_only (default) | MOS 트랜지스터 노드에만 온도 할당 |
| temp_all | 모든 노드에 온도 할당 |

### (4) Normalization Method (TSMC only)
정규화 통계 계산 시 0값 처리 방법

| 옵션 | 설명 |
|------|------|
| norm2 (default) | 0값 제외하고 mean/std 계산 |
| original | 0값 포함하여 mean/std 계산 |

### (5) Input Slew Assignment Mode
입력 slew를 어느 노드에 할당할지 결정

| 옵션 | 설명 | Dataset Suffix |
|------|------|----------------|
| all (default) | 모든 입력 포트/연결된 MOS에 할당 | - |
| related_pin_only | Liberty의 related_pin에만 할당 | `_relpin` |

**예시 (AND2 gate, related_pin=A):**
```
all mode:           A=slew, B=slew
related_pin_only:   A=slew, B=0
```

---

## 출력 파일 구조

### Train 데이터
```
GNN_dataset_{PDK}/train_{data_type}_{graph_mode}{topology_suffix}{slew_suffix}{sample_suffix}.pth
```
예시:
- `train_cell_stage_aware.pth`
- `train_cell_full_graph_vdd_only_relpin.pth`
- `train_cell_stage_aware_10pct.pth`

### Test 데이터
```
GNN_dataset_{PDK}/test_by_{data_type}_{graph_mode}{topology_suffix}{slew_suffix}/
├── {cell_name}_1.pth
├── {cell_name}_2.pth
└── ...
```
예시:
- `test_by_cell_stage_aware/`
- `test_by_cell_full_graph_vdd_only_relpin/`

---

## Default Mode 프리셋

스크립트 실행 시 Default mode 선택으로 빠른 설정 가능:

### Stage Aware (SA) 프리셋
| 설정 | 값 |
|------|-----|
| Graph Mode | stage_aware |
| Voltage Mode | all_nodes |
| Slew Mode | all |
| 출력 예시 | `test_by_cell_stage_aware/` |

### Full Graph (FG) 프리셋
| 설정 | 값 |
|------|-----|
| Graph Mode | full_graph |
| Voltage Mode | vdd_only |
| Slew Mode | related_pin_only |
| 출력 예시 | `test_by_cell_full_graph_vdd_only_relpin/` |

---

## Topology Cache 예시

### Stage-Aware Cache 구조
```python
{
    'cell_name': {
        'rise': {  # Pull-up path
            'nodes': ['VDD', 'PM0', 'PM1', 'N_7', 'Y'],
            'node_types': [1, 3, 3, 4, 2],  # VDD, MOS, MOS, intermediate, output
            'transistor_info': {...},
            'edges': [[0,1], [1,2], ...],
            'edge_attrs': [[1.0, 0.0, 0.0], ...]
        },
        'fall': {  # Pull-down path
            ...
        }
    },
    ...
}
```

### Full-Graph Cache 구조
```python
{
    'cell_name': {
        'nodes': ['VDD', 'VSS', 'A', 'B', 'Y', 'PM0', 'NM0', ...],
        'node_types': [1, 0, 5, 5, 2, 3, 3, ...],
        'transistor_info': {...},
        'edges': [[0,5], [5,6], ...],
        'edge_attrs': [[1.0, 0.0, 0.0], ...]
    },
    ...
}
```

---

## Node Type 정의

| Type | Value | 설명 |
|------|-------|------|
| VSS | 0 | Ground |
| VDD | 1 | Power supply |
| OUTPUT | 2 | Output port (Y, ZN, etc.) |
| MOS | 3 | MOS transistor |
| INTERMEDIATE | 4 | Internal net (N_2, N_7, etc.) |
| INPUT | 5 | Input port (A, B, C, etc.) |

---

## 주의사항

1. **Train 데이터에서 제외되는 셀:**
   - ASAP7: AND2x6, NAND3x2, NOR2xp67, OR2x6
   - TSMC: AN4D0BWP30P140, ND3D0BWP30P140, NR3D1BWP30P140, OR4D0BWP30P140, XNR3D1BWP30P140, XOR3D1BWP30P140

2. **캐시 재사용:** 같은 topology 옵션의 캐시가 있으면 재사용 가능

3. **메모리 최적화:** Test 데이터 생성 시 parallel 모드 권장 (기본 8 workers)

4. **Sampling 옵션:** Train 데이터 크기 조절 가능 (5%, 10%, 20%, full)
