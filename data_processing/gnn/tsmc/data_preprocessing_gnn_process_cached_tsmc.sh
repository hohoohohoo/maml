#!/bin/bash

# TSMC GNN 데이터셋 Train/Test Split 처리 스크립트 (Unified 3D Format)
# build_gnn_dataset_tsmc_unified.py를 사용하여 한번에 처리
#
# 특징:
# - Temperature 기반 자동 Train/Test 분리
#   - Train: 5 corners (FF, FS, TT, SF, SS) × 6 temps (-25, 12.5, 37.5, 62.5, 87.5, 125) = 30 conditions
#   - Test: temps (0, 25, 50, 75, 100) + variant folders (FF2, TTseq 등)
# - Unified 3D tensor format: [61 libs, total_nodes, 11 features]
# - Global normalization (train 데이터에서 통계 계산 → train/test 모두 적용)
# - Node features: 11D (7 base + 4 process parameters)
#
# 출력 형식:
# - Train: train_{data_type}_{graph_mode}.pth (단일 파일, 3D tensor)
# - Test: test_by_{data_type}_{graph_mode}/ 폴더 내 cell별 .pth 파일

OUTPUT_DIR="/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/GNN_dataset_TSMC/"
TSMC_LIB_BASE="/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/TSMC_lib_files"
SPI_DIR="/home/tkdgn2907/Deepsets_test/MAML/Projects/cdl_files"
CACHE_DIR="/home/tkdgn2907/Deepsets_test/MAML/Projects/data_processing/gnn/topology_cache"

# Fixed process parameters (FF_n, FF_p, TT_n, TT_p, SS_n, SS_p, FS_n, FS_p, SF_n, SF_p)
PARAM_A="1.427,1.457,1.430,1.470,1.443,1.483,1.43,1.47,1.43,1.47"
PARAM_B="0.026,0.045,0,0,-0.026,-0.05,0.0208,-0.04,0.036,-0.0208"
PARAM_C="0.024,2.000,0.024,2.000,0.024,2.000,0.024,2.000,0.024,2.000"

echo "=============================================================================="
echo "🚀 TSMC GNN Dataset Processing - Unified 3D Format"
echo "=============================================================================="
echo ""
echo "📋 Train/Test Split 기준:"
echo "   Train: 5 corners × 6 temps (-25, 12.5, 37.5, 62.5, 87.5, 125) = 30 conditions"
echo "   Test: temps (0, 25, 50, 75, 100) + variants (FF2, TTseq, etc.)"
echo ""
echo "⚠️  Train에서 제외되는 INTRA_TOPOLOGY_CELLS:"
echo "   - AN4D0BWP30P140, ND3D0BWP30P140, NR3D1BWP30P140"
echo "   - OR4D0BWP30P140, XNR3D1BWP30P140, XOR3D1BWP30P140"
echo ""
echo "📊 Output Format:"
echo "   - Train: [61, total_nodes, 11] 3D tensor (single .pth file)"
echo "   - Test: cell별 .pth 파일 (test_by_{data_type}_{graph_mode}/)"
echo "   - Node features: 11D (7 base + 4 process)"
echo "   - Global normalization (train 통계 기반)"
echo ""
echo "📂 Paths:"
echo "   Output: $OUTPUT_DIR"
echo "   TSMC lib base: $TSMC_LIB_BASE"
echo "   Cache dir: $CACHE_DIR"
echo ""
echo "📊 Fixed Process Parameters:"
echo "   A: $PARAM_A"
echo "   B: $PARAM_B"
echo "   C: $PARAM_C"
echo "=============================================================================="

# 디렉토리 생성
mkdir -p "$OUTPUT_DIR"
mkdir -p "$CACHE_DIR"

# ============================================================================
# Step 0: Configuration Mode (Default vs Custom)
# ============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Step 0: Configuration Mode"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "   [1] Default - Quick setup with preset configurations (recommended)"
echo "   [2] Custom - Manual configuration of all options"
read -p "Enter choice (1/2) [default: 1]: " config_mode_choice
config_mode_choice=${config_mode_choice:-1}

USE_DEFAULT_MODE="no"
if [ "$config_mode_choice" == "1" ]; then
    USE_DEFAULT_MODE="yes"
    echo ""
    echo "📊 Select Default Configuration:"
    echo "   [1] Stage Aware (SA) - test_by_{data_type}_stage_aware"
    echo "   [2] Full Graph (FG) - test_by_{data_type}_full_graph_vdd_only_relpin"
    read -p "Enter choice (1/2) [default: 1]: " default_arch_choice
    default_arch_choice=${default_arch_choice:-1}

    # Default SPI file for TSMC
    DEFAULT_SPI_BASENAME="tcbn28hpcplusbwp30p140_110a_lpe_typical"

    if [ "$default_arch_choice" == "2" ]; then
        # Full Graph defaults
        GRAPH_MODE="full_graph"
        CACHE_SCRIPT="../precompute/precompute_full_graph_topology.py"
        CACHE_PREFIX="full_graph_topology_cache"
        CACHE_FILE="$CACHE_DIR/${CACHE_PREFIX}_tsmc_${DEFAULT_SPI_BASENAME}.pth"

        # FG default settings
        VOLTAGE_MODE="vdd_only"
        VOLTAGE_MODE_FLAG="--voltage_mode vdd_only"
        SLEW_MODE="related_pin_only"
        SLEW_MODE_FLAG="--slew_mode related_pin_only"
        SLEW_SUFFIX="_relpin"

        # No topology options for FG
        WEIGHTED_FLAG=""
        WEIGHTED_SUFFIX=""
        INCLUDE_CAP_FLAG=""
        GATE_CONTROL_FLAG=""
        GATE_CONTROL_SUFFIX=""
        INPUT_PORTS_FLAG=""
        INPUT_PORTS_SUFFIX=""
        BIDIRECTION_FLAG=""
        BIDIRECTION_SUFFIX=""
        TOPOLOGY_SUFFIX="_vdd_only"

        # Normalization defaults
        INCLUDE_ZEROS_FLAG=""
        NORM_METHOD="norm2"
        TEMP_MODE="mos_only"
        TEMP_MODE_FLAG=""

        echo "   ✓ Selected: Full Graph (FG) with vdd_only + related_pin_only"
    else
        # Stage Aware defaults
        GRAPH_MODE="stage_aware"
        CACHE_SCRIPT="../precompute/precompute_stage_aware_topology.py"
        CACHE_PREFIX="stage_aware_topology_cache"
        CACHE_FILE="$CACHE_DIR/${CACHE_PREFIX}_tsmc_${DEFAULT_SPI_BASENAME}.pth"

        echo "   Cache file: ${CACHE_PREFIX}_tsmc_${DEFAULT_SPI_BASENAME}.pth"

        # SA default settings
        VOLTAGE_MODE="all_nodes"
        VOLTAGE_MODE_FLAG=""
        SLEW_MODE="all"
        SLEW_MODE_FLAG=""
        SLEW_SUFFIX=""

        # No topology options for SA
        WEIGHTED_FLAG=""
        WEIGHTED_SUFFIX=""
        INCLUDE_CAP_FLAG=""
        GATE_CONTROL_FLAG=""
        GATE_CONTROL_SUFFIX=""
        INPUT_PORTS_FLAG=""
        INPUT_PORTS_SUFFIX=""
        BIDIRECTION_FLAG=""
        BIDIRECTION_SUFFIX=""
        TOPOLOGY_SUFFIX=""

        # Normalization defaults
        INCLUDE_ZEROS_FLAG=""
        NORM_METHOD="norm2"
        TEMP_MODE="mos_only"
        TEMP_MODE_FLAG=""

        echo "   ✓ Selected: Stage Aware (SA) with default settings"
    fi

    # Check if default cache exists
    if [ ! -f "$CACHE_FILE" ]; then
        echo ""
        echo "⚠️  Default cache file not found: $(basename $CACHE_FILE)"
        echo "   Please run with Custom mode first to generate the cache."
        exit 1
    fi
    echo "   ✓ Using cache: $(basename $CACHE_FILE)"

    # Skip to data type selection (Step 3)
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
fi

# ============================================================================
# Custom Mode: Full configuration steps
# ============================================================================
if [ "$USE_DEFAULT_MODE" == "no" ]; then

# Step 1: Graph mode 선택 (stage-aware vs full-graph)
echo ""
echo "📊 Step 1: Select Graph Mode"
echo "   [1] Stage-aware (pull-up/pull-down paths only - recommended)"
echo "   [2] Full-graph (all transistors - larger cache)"
read -p "Enter choice (1-2) [default: 1]: " graph_mode_choice

case $graph_mode_choice in
    2)
        GRAPH_MODE="full_graph"
        CACHE_SCRIPT="../precompute/precompute_full_graph_topology.py"
        CACHE_PREFIX="full_graph_topology_cache"
        echo "   ✓ Selected: Full-graph (baseline)"
        ;;
    *)
        GRAPH_MODE="stage_aware"
        CACHE_SCRIPT="../precompute/precompute_stage_aware_topology.py"
        CACHE_PREFIX="stage_aware_topology_cache"
        echo "   ✓ Selected: Stage-aware (pull-up/pull-down)"
        ;;
esac

# Step 2: Graph Adjacency Matrix Configuration
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Step 2: Graph Adjacency Matrix Configuration"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🔍 (1) Select SPI file (TSMC netlist)"
SPI_FILES=($(find $SPI_DIR -name "*.spi" | sort))

if [ ${#SPI_FILES[@]} -eq 0 ]; then
    echo "❌ No SPI files found in $SPI_DIR"
    exit 1
fi

echo "📂 Found ${#SPI_FILES[@]} SPI files:"
for i in "${!SPI_FILES[@]}"; do
    echo "   [$i] $(basename ${SPI_FILES[$i]})"
done

read -p "Select SPI file number [default: 0]: " spi_choice
spi_choice=${spi_choice:-0}

if [ $spi_choice -ge 0 ] && [ $spi_choice -lt ${#SPI_FILES[@]} ]; then
    SELECTED_SPI="${SPI_FILES[$spi_choice]}"
else
    echo "❌ Invalid SPI number!"
    exit 1
fi

SPI_BASENAME=$(basename $SELECTED_SPI .spi)
CACHE_FILE="$CACHE_DIR/${CACHE_PREFIX}_tsmc_${SPI_BASENAME}.pth"
echo "   ✓ Selected: $(basename $SELECTED_SPI)"

# (2) Weighted adjacency matrix 옵션
echo ""
echo "📊 (2) Weighted Adjacency Matrix Option"
echo "   [1] Binary adjacency (default) - 0/1 for edges"
echo "   [2] Weighted adjacency - use resistance values from SPI file"
echo "      (smaller resistance = stronger connection)"
read -p "Enter choice (1/2) [default: 1]: " weighted_choice

WEIGHTED_FLAG=""
WEIGHTED_SUFFIX=""
INCLUDE_CAP_FLAG=""
if [ "$weighted_choice" == "2" ]; then
    WEIGHTED_FLAG="--weighted"
    WEIGHTED_SUFFIX="_weighted"
    echo "   ✓ Selected: Weighted adjacency (resistance-based)"

    # Ask if user wants to include parasitic capacitance feature
    echo ""
    echo "   📊 Include parasitic capacitance as node feature?"
    echo "      [1] No (default) - 11D node features"
    echo "      [2] Yes - 12D node features (adds parasitic cap sum per node)"
    read -p "   Enter choice (1/2) [default: 1]: " cap_choice
    if [ "$cap_choice" == "2" ]; then
        INCLUDE_CAP_FLAG="--include_parasitic_cap"
        echo "   ✓ Include parasitic cap: YES (12D features)"
    else
        echo "   ✓ Include parasitic cap: NO (11D features)"
    fi
else
    echo "   ✓ Selected: Binary adjacency (0/1)"
fi

# Initialize topology option variables
GATE_CONTROL_FLAG=""
GATE_CONTROL_SUFFIX=""
GATE_CONTROL_WEIGHT=0.0
INPUT_PORTS_FLAG=""
INPUT_PORTS_SUFFIX=""
INPUT_PORTS_WEIGHT=0.0
BIDIRECTION_FLAG=""
BIDIRECTION_SUFFIX=""

# (3)~(6): Stage-aware 전용 옵션 (full_graph는 이미 고정된 구조 사용)
if [ "$GRAPH_MODE" == "stage_aware" ]; then
    # (3) Gate control edges 옵션
    echo ""
    echo "📊 (3) Gate Control Edges Option"
    echo "   [1] Disabled (default) - No gate control edges"
    echo "   [2] Enabled (weight=0.5) - Add edges from intermediate gates to transistors"
    echo "      (Allows GNN to learn gate-transistor control relationships)"
    read -p "Enter choice (1/2) [default: 1]: " gate_control_choice

    if [ "$gate_control_choice" == "2" ]; then
        GATE_CONTROL_WEIGHT=0.5
        GATE_CONTROL_FLAG="--gate_control $GATE_CONTROL_WEIGHT"
        GATE_CONTROL_SUFFIX="_gatectrl"
        echo "   ✓ Gate control edges: ENABLED (weight=$GATE_CONTROL_WEIGHT)"
    else
        echo "   ✓ Gate control edges: DISABLED"
    fi

    # (4) Input port edges 옵션
    echo ""
    echo "📊 (4) Input Port Edges Option"
    echo "   [1] Disabled (default) - No input port nodes"
    echo "   [2] Enabled (weight=0.5) - Add input port nodes (A, B, C, etc.) with edges to transistors"
    echo "      (Similar to full_graph topology, enables GNN to learn input-transistor relationships)"
    read -p "Enter choice (1/2) [default: 1]: " input_ports_choice

    if [ "$input_ports_choice" == "2" ]; then
        INPUT_PORTS_WEIGHT=0.5
        INPUT_PORTS_FLAG="--input_ports $INPUT_PORTS_WEIGHT"
        INPUT_PORTS_SUFFIX="_inputport"
        echo "   ✓ Input port edges: ENABLED (weight=$INPUT_PORTS_WEIGHT)"
    else
        echo "   ✓ Input port edges: DISABLED"
    fi

    # (5) Bidirectional edges 옵션
    echo ""
    echo "📊 (5) Bidirectional Edges Option"
    echo "   [1] Disabled (default) - Directed edges only (source → drain)"
    echo "   [2] Enabled - Add reverse edges for bidirectional message passing"
    echo "      (Allows GNN to aggregate information in both directions)"
    read -p "Enter choice (1/2) [default: 1]: " bidirection_choice

    if [ "$bidirection_choice" == "2" ]; then
        BIDIRECTION_FLAG="--bidirection"
        BIDIRECTION_SUFFIX="_bidir"
        echo "   ✓ Bidirectional edges: ENABLED"
    else
        echo "   ✓ Bidirectional edges: DISABLED"
    fi

else
    # full_graph 모드: 이미 bidirectional, input ports 포함된 고정 구조
    echo ""
    echo "📊 (3)~(5): Skipped (full_graph already includes bidirectional edges & input ports)"
fi

# Update cache file name with weighted, gate_control, input_ports, and bidirection suffix
CACHE_FILE="$CACHE_DIR/${CACHE_PREFIX}_tsmc_${SPI_BASENAME}${WEIGHTED_SUFFIX}${GATE_CONTROL_SUFFIX}${INPUT_PORTS_SUFFIX}${BIDIRECTION_SUFFIX}.pth"

# Combined topology suffix for output filenames (to distinguish datasets with different topology options)
TOPOLOGY_SUFFIX="${WEIGHTED_SUFFIX}${GATE_CONTROL_SUFFIX}${INPUT_PORTS_SUFFIX}${BIDIRECTION_SUFFIX}"

# Topology cache 생성 또는 재사용
echo ""
if [ -f "$CACHE_FILE" ]; then
    echo "📦 Found existing topology cache: $(basename $CACHE_FILE)"
    read -p "Reuse existing cache? (y/n) [default: y]: " reuse_cache
    reuse_cache=${reuse_cache:-y}

    if [ "$reuse_cache" != "y" ]; then
        echo "🔄 Regenerating topology cache..."
        python3 $CACHE_SCRIPT \
            --spi_path "$SELECTED_SPI" \
            --output "$CACHE_FILE" \
            $WEIGHTED_FLAG \
            $GATE_CONTROL_FLAG \
            $INPUT_PORTS_FLAG \
            $BIDIRECTION_FLAG \

        if [ $? -ne 0 ]; then
            echo "❌ Failed to generate topology cache!"
            exit 1
        fi
        echo "   ✅ Topology cache generated"
    else
        echo "   ✓ Reusing existing cache"
    fi
else
    echo "🔄 Generating topology cache for the first time..."
    python3 $CACHE_SCRIPT \
        --spi_path "$SELECTED_SPI" \
        --output "$CACHE_FILE" \
        $WEIGHTED_FLAG \
        $GATE_CONTROL_FLAG \
        $INPUT_PORTS_FLAG \
        $BIDIRECTION_FLAG

    if [ $? -ne 0 ]; then
        echo "❌ Failed to generate topology cache!"
        exit 1
    fi
    echo "   ✅ Topology cache generated"
fi

fi  # End of Custom Mode block

# Step 3: Node Feature Configuration (applies to both Default and Custom modes)
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Step 3: Node Feature Configuration"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📊 (1) Select Data Type"
echo "   [1] cell (default) - Cell delay (propagation delay)"
echo "   [2] transition - Output transition time (slew)"
read -p "Enter choice (1/2) [default: 1]: " data_type_choice

if [ "$data_type_choice" == "2" ]; then
    DATA_TYPE="transition"
    echo "   ✓ Data type: transition (output slew)"
else
    DATA_TYPE="cell"
    echo "   ✓ Data type: cell (propagation delay)"
fi

# Steps 3.5-3.8: Only for Custom mode (Default mode already has these set)
if [ "$USE_DEFAULT_MODE" == "no" ]; then

# (2) Voltage mode 선택
echo ""
echo "📊 (2) Voltage Feature Mode"
echo "   [1] all_nodes (default) - Voltage applied to all nodes"
echo "   [2] vdd_only - Voltage only on VDD node, 0 for others"
echo "   [3] vdd_mos - Voltage on VDD and MOS transistor nodes only"
read -p "Enter choice (1/2/3) [default: 1]: " voltage_mode_choice

VOLTAGE_MODE_FLAG=""
if [ "$voltage_mode_choice" == "2" ]; then
    VOLTAGE_MODE="vdd_only"
    VOLTAGE_MODE_FLAG="--voltage_mode vdd_only"
    echo "   ✓ Voltage mode: vdd_only (V on VDD, 0 elsewhere)"
elif [ "$voltage_mode_choice" == "3" ]; then
    VOLTAGE_MODE="vdd_mos"
    VOLTAGE_MODE_FLAG="--voltage_mode vdd_mos"
    echo "   ✓ Voltage mode: vdd_mos (V on VDD + MOS transistors)"
else
    VOLTAGE_MODE="all_nodes"
    echo "   ✓ Voltage mode: all_nodes (V on all nodes)"
fi

# (3) Temperature mode 선택
echo ""
echo "📊 (3) Temperature Feature Mode"
echo "   [1] mos_only (default) - Temperature applied to MOS transistor nodes only"
echo "   [2] temp_all - Temperature applied to all nodes"
read -p "Enter choice (1/2) [default: 1]: " temp_mode_choice

TEMP_MODE_FLAG=""
if [ "$temp_mode_choice" == "2" ]; then
    TEMP_MODE="temp_all"
    TEMP_MODE_FLAG="--temperature_mode temp_all"
    echo "   ✓ Temperature mode: temp_all (T on all nodes)"
else
    TEMP_MODE="mos_only"
    echo "   ✓ Temperature mode: mos_only (T on MOS transistors only)"
fi

# (4) Normalization method 선택
echo ""
echo "📊 (4) Normalization Statistics Method"
echo "   [1] norm2 (default) - Exclude zeros from stats calculation"
echo "      (Non-zero values only used for mean/std)"
echo "   [2] original - Include zeros in stats calculation"
echo "      (All values including zeros used for mean/std)"
read -p "Enter choice (1/2) [default: 1]: " norm_method_choice

INCLUDE_ZEROS_FLAG=""
NORM_METHOD="norm2"
if [ "$norm_method_choice" == "2" ]; then
    INCLUDE_ZEROS_FLAG="--include_zeros_in_norm"
    NORM_METHOD="original"
    echo "   ✓ Normalization: original (include zeros in stats)"
else
    echo "   ✓ Normalization: norm2 (exclude zeros from stats)"
fi

# (5) Slew mode 선택 (input_slew assignment)
echo ""
echo "📊 (5) Input Slew Assignment Mode"
echo "   [1] all (default) - Apply input_slew to all input ports/connected MOS"
echo "   [2] related_pin_only - Apply input_slew only to the related_pin node/connected MOS"
echo "      (Uses related_pin info from Liberty file to target specific input)"
read -p "Enter choice (1/2) [default: 1]: " slew_mode_choice

SLEW_MODE_FLAG=""
SLEW_MODE="all"
SLEW_SUFFIX=""
if [ "$slew_mode_choice" == "2" ]; then
    SLEW_MODE="related_pin_only"
    SLEW_MODE_FLAG="--slew_mode related_pin_only"
    SLEW_SUFFIX="_relpin"
    echo "   ✓ Slew mode: related_pin_only (input_slew only to related_pin)"
else
    echo "   ✓ Slew mode: all (input_slew to all input ports)"
fi

fi  # End of Steps 3.5-3.8 (Custom mode only)

# Update OUTPUT_DIR
OUTPUT_DIR="/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/GNN_dataset_TSMC/"
mkdir -p "$OUTPUT_DIR"

# Step 4: TSMC 폴더 확인
echo ""
echo "🔍 Step 4: Checking TSMC folders..."
TSMC_FOLDERS=($(find $TSMC_LIB_BASE -maxdepth 1 -type d -name "TSMC_*" | sort))

if [ ${#TSMC_FOLDERS[@]} -eq 0 ]; then
    echo "❌ No TSMC folders found in $TSMC_LIB_BASE"
    exit 1
fi

echo "📂 Found ${#TSMC_FOLDERS[@]} TSMC folders"

# Train/Test 폴더 미리보기
echo ""
echo "📋 Expected Train folders (5 corners × 6 temps = 30):"
TRAIN_TEMPS=("-25" "12p5" "37p5" "62p5" "87p5" "125")
CORNERS=("FF" "FS" "TT" "SF" "SS")
train_count=0
missing_train=()

for corner in "${CORNERS[@]}"; do
    for temp in "${TRAIN_TEMPS[@]}"; do
        folder_name="TSMC_${corner}_${temp}"
        folder_path="$TSMC_LIB_BASE/$folder_name"
        if [ -d "$folder_path" ]; then
            train_count=$((train_count + 1))
        else
            missing_train+=("$folder_name")
        fi
    done
done

echo "   Found: $train_count / 30 train folders"

if [ ${#missing_train[@]} -gt 0 ]; then
    echo ""
    echo "   ❌ ERROR: Missing ${#missing_train[@]} required train folders!"
    echo "   All 30 train folders must exist to proceed."
    echo ""
    echo "   Missing folders:"
    for folder in "${missing_train[@]}"; do
        echo "      - $folder"
    done
    echo ""
    exit 1
fi

echo "   ✅ All 30 train folders found!"

echo ""
echo "📋 Expected Test folders (5 temps + variants):"
TEST_TEMPS=("0" "25" "50" "75" "100")
test_count=0

for corner in "${CORNERS[@]}"; do
    for temp in "${TEST_TEMPS[@]}"; do
        folder_name="TSMC_${corner}_${temp}"
        folder_path="$TSMC_LIB_BASE/$folder_name"
        if [ -d "$folder_path" ]; then
            test_count=$((test_count + 1))
        fi
    done
done

# Variant folders
variant_count=$(find $TSMC_LIB_BASE -maxdepth 1 -type d \( -name "TSMC_*2_*" -o -name "TSMC_*seq_*" -o -name "TSMC_Seq_*" \) | wc -l)
test_count=$((test_count + variant_count))

echo "   Found: ~$test_count test folders (including variants)"

# Step 5: Skip train 옵션
echo ""
echo "📊 Step 5: Skip Train Data Processing?"
echo "   [1] Process both train and test (default)"
echo "   [2] Skip train, only process test (requires existing train file)"
read -p "Enter choice (1/2) [default: 1]: " skip_train_choice

SKIP_TRAIN_FLAG=""
if [ "$skip_train_choice" == "2" ]; then
    # Check if train file exists (first try with topology+slew suffix, then without)
    TRAIN_FILE="${OUTPUT_DIR}train_${DATA_TYPE}_${GRAPH_MODE}${TOPOLOGY_SUFFIX}${SLEW_SUFFIX}.pth"
    TRAIN_FILE_BASE="${OUTPUT_DIR}train_${DATA_TYPE}_${GRAPH_MODE}.pth"

    if [ -f "$TRAIN_FILE" ]; then
        echo "   ✓ Skip train: Yes (train file found: $(basename $TRAIN_FILE))"
        SKIP_TRAIN_FLAG="--skip_train"
    elif [ -f "$TRAIN_FILE_BASE" ] && [ -n "$TOPOLOGY_SUFFIX" ]; then
        # Topology suffix only changes cache, not train data structure
        echo "   ✓ Skip train: Yes (using base train file without topology suffix)"
        echo "   Note: gatectrl/inputport only affects topology cache, train data structure is the same"
        SKIP_TRAIN_FLAG="--skip_train"
        TRAIN_FILE="$TRAIN_FILE_BASE"
    else
        echo "   ❌ ERROR: Train file not found:"
        echo "      - $TRAIN_FILE"
        if [ -n "$TOPOLOGY_SUFFIX" ]; then
            echo "      - $TRAIN_FILE_BASE"
        fi
        echo "   Please run without --skip_train first to generate train file"
        exit 1
    fi
else
    echo "   ✓ Skip train: No (process both train and test)"
fi

# Step 6: 실행 확인
echo ""
echo "=============================================================================="
echo "📋 Summary"
echo "=============================================================================="
echo "   Graph mode: $GRAPH_MODE"
echo "   Data type: $DATA_TYPE"
echo "   Voltage mode: $VOLTAGE_MODE"
echo "   Temperature mode: $TEMP_MODE"
echo "   Slew mode: $SLEW_MODE"
echo "   Normalization: $NORM_METHOD"
if [ -n "$WEIGHTED_FLAG" ]; then
    echo "   Weighted adjacency: YES (resistance-based)"
    if [ -n "$INCLUDE_CAP_FLAG" ]; then
        echo "   Parasitic cap feature: YES (12D node features)"
    else
        echo "   Parasitic cap feature: NO (11D node features)"
    fi
else
    echo "   Weighted adjacency: NO (binary 0/1)"
fi
if [ -n "$GATE_CONTROL_FLAG" ]; then
    echo "   Gate control edges: YES (weight=$GATE_CONTROL_WEIGHT)"
else
    echo "   Gate control edges: NO"
fi
if [ -n "$INPUT_PORTS_FLAG" ]; then
    echo "   Input port edges: YES (weight=$INPUT_PORTS_WEIGHT)"
else
    echo "   Input port edges: NO"
fi
if [ -n "$BIDIRECTION_FLAG" ]; then
    echo "   Bidirectional edges: YES"
else
    echo "   Bidirectional edges: NO"
fi
echo "   Topology suffix: '$TOPOLOGY_SUFFIX'"
echo "   Cache file: $(basename $CACHE_FILE)"
echo "   Output dir: $OUTPUT_DIR"
if [ -n "$SKIP_TRAIN_FLAG" ]; then
    echo "   Skip train: YES (only test data will be processed)"
else
    echo "   Train: 30 conditions → [61, total_nodes, 11] 3D tensor"
    echo "   Excluded from train: INTRA_TOPOLOGY_CELLS (6 cells)"
fi
echo "   Test: Cell별 .pth 파일 (test_by_${DATA_TYPE}_${GRAPH_MODE}${TOPOLOGY_SUFFIX}${SLEW_SUFFIX}/)"
echo "   Normalization: Global (train stats → train/test)"
echo "=============================================================================="
echo ""
read -p "🚀 Start processing? (y/n) [default: y]: " start_choice
start_choice=${start_choice:-y}

if [ "$start_choice" != "y" ]; then
    echo "❌ Aborted by user"
    exit 0
fi

# Step 6: 실행
echo ""
echo "=============================================================================="
echo "🔄 Starting Unified 3D Format Dataset Generation..."
echo "=============================================================================="

python3 build_gnn_dataset_process_cached_tsmc.py \
    --cache_path "$CACHE_FILE" \
    --cache_type "$GRAPH_MODE" \
    --lib_base_path "$TSMC_LIB_BASE" \
    --output_dir "$OUTPUT_DIR" \
    --data_type "$DATA_TYPE" \
    --topology_suffix "$TOPOLOGY_SUFFIX" \
    $SKIP_TRAIN_FLAG \
    $INCLUDE_CAP_FLAG \
    $VOLTAGE_MODE_FLAG \
    $TEMP_MODE_FLAG \
    $INCLUDE_ZEROS_FLAG \
    $SLEW_MODE_FLAG

if [ $? -eq 0 ]; then
    echo ""
    echo "=============================================================================="
    echo "🎉 Dataset Generation Complete!"
    echo "=============================================================================="
    echo ""
    echo "📂 Output files:"
    echo ""

    # Train 파일 확인 (skip_train이 아닌 경우에만)
    TRAIN_FILE="${OUTPUT_DIR}train_${DATA_TYPE}_${GRAPH_MODE}${TOPOLOGY_SUFFIX}${SLEW_SUFFIX}.pth"
    if [ -z "$SKIP_TRAIN_FLAG" ]; then
        echo "📦 Train dataset (Unified 3D format):"
        if [ -f "$TRAIN_FILE" ]; then
            SIZE=$(du -h "$TRAIN_FILE" | cut -f1)
            echo "   ✓ train_${DATA_TYPE}_${GRAPH_MODE}${TOPOLOGY_SUFFIX}${SLEW_SUFFIX}.pth ($SIZE)"
            echo "   Format: node_features [61, total_nodes, 11], outputs [61, num_tasks]"
        fi
    else
        echo "📦 Train dataset: SKIPPED"
    fi

    # Test 파일 확인
    TEST_DIR="${OUTPUT_DIR}test_by_${DATA_TYPE}_${GRAPH_MODE}${TOPOLOGY_SUFFIX}${SLEW_SUFFIX}/"
    echo ""
    echo "📦 Test dataset (Cell별 .pth 파일):"
    if [ -d "$TEST_DIR" ]; then
        CELL_COUNT=$(ls -1 "$TEST_DIR"*.pth 2>/dev/null | wc -l)
        TOTAL_SIZE=$(du -sh "$TEST_DIR" 2>/dev/null | cut -f1)
        echo "   ✓ Directory: test_by_${DATA_TYPE}_${GRAPH_MODE}${TOPOLOGY_SUFFIX}${SLEW_SUFFIX}/"
        echo "   ✓ Cell files: $CELL_COUNT cells"
        echo "   ✓ Total size: $TOTAL_SIZE"
        echo ""
        echo "   Sample files:"
        ls -1 "$TEST_DIR"*.pth 2>/dev/null | head -5 | while read f; do
            echo "      - $(basename $f)"
        done
        if [ $CELL_COUNT -gt 5 ]; then
            echo "      ... and $((CELL_COUNT - 5)) more"
        fi
    else
        echo "   ⚠️  Test directory not found"
    fi

    echo ""
    echo "=============================================================================="
    echo "📊 Usage Examples"
    echo "=============================================================================="
    echo ""
    echo "# MAML Training:"
    echo "python maml_gnn_training_tsmc_process.py \\"
    echo "    --dataset_dir $OUTPUT_DIR \\"
    echo "    --graph_mode $GRAPH_MODE"
    echo ""
    echo "# Baseline Training:"
    echo "python baseline_gnn_training_tsmc_process.py \\"
    echo "    --dataset_dir $OUTPUT_DIR \\"
    echo "    --graph_mode $GRAPH_MODE"
    echo ""
    echo "=============================================================================="
    echo "✅ Done!"
    echo "=============================================================================="
else
    echo ""
    echo "=============================================================================="
    echo "❌ Dataset generation failed!"
    echo "=============================================================================="
    exit 1
fi
