#!/bin/bash

# GNN 데이터셋 배치 처리 스크립트 (Process Condition + Cached Topology)
# processed, processed_simple 폴더의 모든 파일들을 GNN으로 변환 (11D node features)
# 기존 7D features + 4D process parameters (param_a, param_b, param_c, temperature)
# Cached topology를 사용하여 빠른 처리

OUTPUT_DIR="/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/GNN_dataset_ASAP7/"
PROCESSED_BASE_DIR="/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/ASAP7_lib_files/processed"
PROCESSED_SIMPLE_BASE_DIR="/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/ASAP7_lib_files/processed_simple"
CDL_DIR="/home/tkdgn2907/Deepsets_test/MAML/Projects/cdl_files"
CACHE_DIR="/home/tkdgn2907/Deepsets_test/MAML/Projects/data_processing/gnn/topology_cache"

echo "🚀 Starting Batch GNN Dataset Processing with Process Conditions (CACHED)"
echo "=========================================================================="
echo "Output directory: $OUTPUT_DIR"
echo "Processed base directory: $PROCESSED_BASE_DIR"
echo "Processed simple base directory: $PROCESSED_SIMPLE_BASE_DIR"
echo "Cache directory: $CACHE_DIR"
echo "Node features: 11D (7 base + 4 process parameters)"
echo ""
echo "⚠️  Train에서 제외되는 INTRA_TOPOLOGY_CELLS:"
echo "   - AND2x6, NAND3x2, NOR2xp67, OR2x6"

# Cache 및 output 디렉토리 생성
mkdir -p $OUTPUT_DIR
mkdir -p $CACHE_DIR

# Initialize optional suffixes
XOR_SUFFIX=""

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

    # Default CDL file basename for ASAP7
    DEFAULT_CDL_BASENAME="asap7sc7p5t_28_L"

    if [ "$default_arch_choice" == "2" ]; then
        # Full Graph defaults
        GRAPH_MODE="full_graph"
        CACHE_SCRIPT="precompute_full_graph_topology.py"
        CACHE_PREFIX="full_graph_topology_cache"
        CACHE_FILE="$CACHE_DIR/${CACHE_PREFIX}_${DEFAULT_CDL_BASENAME}.pth"

        # FG default settings
        VOLTAGE_MODE="vdd_only"
        VOLTAGE_MODE_FLAG="--voltage_mode vdd_only"
        VOLTAGE_MODE_SUFFIX="_vdd_only"
        SLEW_MODE="related_pin_only"
        SLEW_MODE_FLAG="--slew_mode related_pin_only"
        SLEW_SUFFIX="_relpin"

        # No topology options for FG
        GATE_CONTROL_FLAG=""
        GATE_CONTROL_SUFFIX=""
        INPUT_PORTS_FLAG=""
        INPUT_PORTS_SUFFIX=""
        BIDIRECTION_FLAG=""
        BIDIRECTION_SUFFIX=""

        echo "   ✓ Selected: Full Graph (FG) with vdd_only + related_pin_only"
    else
        # Stage Aware defaults
        GRAPH_MODE="stage_aware"
        CACHE_SCRIPT="precompute_stage_aware_topology.py"
        CACHE_PREFIX="stage_aware_topology_cache"
        CACHE_FILE="$CACHE_DIR/${CACHE_PREFIX}_${DEFAULT_CDL_BASENAME}.pth"

        # SA default settings
        VOLTAGE_MODE="all_nodes"
        VOLTAGE_MODE_FLAG=""
        VOLTAGE_MODE_SUFFIX=""
        SLEW_MODE="all"
        SLEW_MODE_FLAG=""
        SLEW_SUFFIX=""

        # No topology options for SA
        GATE_CONTROL_FLAG=""
        GATE_CONTROL_SUFFIX=""
        INPUT_PORTS_FLAG=""
        INPUT_PORTS_SUFFIX=""
        BIDIRECTION_FLAG=""
        BIDIRECTION_SUFFIX=""

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

    # Skip to data type selection
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
fi

# ============================================================================
# Custom Mode: Full configuration steps
# ============================================================================
if [ "$USE_DEFAULT_MODE" == "no" ]; then

# Step 0.5: Graph mode 선택 (stage-aware vs full-graph)
echo ""
echo "📊 Select Graph Mode:"
echo "   [1] Stage-aware (pull-up/pull-down paths only - smaller cache)"
echo "   [2] Full-graph (all transistors - larger cache)"
read -p "Enter choice (1-2) [default: 1]: " graph_mode_choice

case $graph_mode_choice in
    2)
        GRAPH_MODE="full_graph"
        CACHE_SCRIPT="precompute_full_graph_topology.py"
        CACHE_PREFIX="full_graph_topology_cache"
        echo "✓ Selected mode: Full-graph (baseline)"
        ;;
    *)
        GRAPH_MODE="stage_aware"
        CACHE_SCRIPT="precompute_stage_aware_topology.py"
        CACHE_PREFIX="stage_aware_topology_cache"
        echo "✓ Selected mode: Stage-aware (pull-up/pull-down)"
        ;;
esac

fi  # End of graph mode selection for Custom mode

# Initialize SELECTED_BASE_DIRS (will be set in test/train mode selection)
SELECTED_BASE_DIRS=()

# Step 2: Graph Adjacency Matrix Configuration (Custom mode only)
if [ "$USE_DEFAULT_MODE" == "no" ]; then
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Step 2: Graph Adjacency Matrix Configuration"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🔍 (1) Select CDL file"
CDL_FILES=($(find $CDL_DIR -name "*.cdl" | sort))

if [ ${#CDL_FILES[@]} -eq 0 ]; then
    echo "❌ No CDL files found in $CDL_DIR"
    exit 1
fi

echo "📂 Found ${#CDL_FILES[@]} CDL files:"
for i in "${!CDL_FILES[@]}"; do
    echo "   [$i] $(basename ${CDL_FILES[$i]})"
done

read -p "Select CDL file number [default: 0]: " cdl_choice
cdl_choice=${cdl_choice:-0}

if [ $cdl_choice -ge 0 ] && [ $cdl_choice -lt ${#CDL_FILES[@]} ]; then
    SELECTED_CDL="${CDL_FILES[$cdl_choice]}"
else
    echo "❌ Invalid CDL number!"
    exit 1
fi

CDL_BASENAME=$(basename $SELECTED_CDL .cdl)
echo "✓ Selected CDL: $(basename $SELECTED_CDL)"

# (2) Gate control edges 옵션
echo ""
echo "📊 (2) Gate Control Edges Option"
echo "   [1] Disabled (default) - No gate control edges"
echo "   [2] Enabled (weight=0.5) - Add edges from intermediate gates to transistors"
echo "      (Allows GNN to learn gate-transistor control relationships)"
read -p "Enter choice (1/2) [default: 1]: " gate_control_choice

GATE_CONTROL_FLAG=""
GATE_CONTROL_SUFFIX=""
GATE_CONTROL_WEIGHT=0.0
if [ "$gate_control_choice" == "2" ]; then
    GATE_CONTROL_WEIGHT=0.5
    GATE_CONTROL_FLAG="--gate_control $GATE_CONTROL_WEIGHT"
    GATE_CONTROL_SUFFIX="_gatectrl"
    echo "   ✓ Gate control edges: ENABLED (weight=$GATE_CONTROL_WEIGHT)"
else
    echo "   ✓ Gate control edges: DISABLED"
fi

# (3) Input port edges 옵션
echo ""
echo "📊 (3) Input Port Edges Option"
echo "   [1] Disabled (default) - No input port nodes"
echo "   [2] Enabled (weight=0.5) - Add input port nodes (A, B, C, etc.) with edges to transistors"
echo "      (Similar to full_graph topology, enables GNN to learn input-transistor relationships)"
read -p "Enter choice (1/2) [default: 1]: " input_ports_choice

INPUT_PORTS_FLAG=""
INPUT_PORTS_SUFFIX=""
INPUT_PORTS_WEIGHT=0.0
if [ "$input_ports_choice" == "2" ]; then
    INPUT_PORTS_WEIGHT=0.5
    INPUT_PORTS_FLAG="--input_ports $INPUT_PORTS_WEIGHT"
    INPUT_PORTS_SUFFIX="_inputport"
    echo "   ✓ Input port edges: ENABLED (weight=$INPUT_PORTS_WEIGHT)"
else
    echo "   ✓ Input port edges: DISABLED"
fi

# (4) Bidirectional edges 옵션
echo ""
echo "📊 (4) Bidirectional Edges Option"
echo "   [1] Disabled (default) - Unidirectional edges (current flow direction)"
echo "   [2] Enabled - Bidirectional edges (adds reverse edges for all connections)"
echo "      (Allows GNN to aggregate information in both directions)"
read -p "Enter choice (1/2) [default: 1]: " bidirection_choice

BIDIRECTION_FLAG=""
BIDIRECTION_SUFFIX=""
if [ "$bidirection_choice" == "2" ]; then
    BIDIRECTION_FLAG="--bidirection"
    BIDIRECTION_SUFFIX="_bidir"
    echo "   ✓ Bidirectional edges: ENABLED"
else
    echo "   ✓ Bidirectional edges: DISABLED"
fi

# (5) Slew mode 선택 (input_slew assignment)
echo ""
echo "📊 (5) Input Slew Assignment Mode"
echo "   [1] all (default) - Apply input_slew to all input ports/connected MOS"
echo "   [2] related_pin_only - Apply input_slew only to the related_pin node/connected MOS"
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

# Update cache file name with gate_control, input_ports, and bidirection suffix
CACHE_FILE="$CACHE_DIR/${CACHE_PREFIX}_${CDL_BASENAME}${GATE_CONTROL_SUFFIX}${INPUT_PORTS_SUFFIX}${BIDIRECTION_SUFFIX}.pth"

# Topology cache 생성 또는 재사용
echo ""
if [ -f "$CACHE_FILE" ]; then
    echo "📦 Found existing topology cache: $(basename $CACHE_FILE)"
    read -p "Reuse existing cache? (y/n) [default: y]: " reuse_cache
    reuse_cache=${reuse_cache:-y}

    if [ "$reuse_cache" == "y" ]; then
        echo "✓ Reusing existing cache"
    else
        echo "🔄 Regenerating topology cache..."
        python3 $CACHE_SCRIPT \
            --cdl_path "$SELECTED_CDL" \
            --output "$CACHE_FILE" \
            $GATE_CONTROL_FLAG \
            $INPUT_PORTS_FLAG \
            $BIDIRECTION_FLAG \

        if [ $? -ne 0 ]; then
            echo "❌ Failed to generate topology cache!"
            exit 1
        fi
        echo "✅ Topology cache generated successfully"
    fi
else
    echo "🔄 Generating topology cache for the first time..."
    python3 $CACHE_SCRIPT \
        --cdl_path "$SELECTED_CDL" \
        --output "$CACHE_FILE" \
        $GATE_CONTROL_FLAG \
        $INPUT_PORTS_FLAG \
        $BIDIRECTION_FLAG

    if [ $? -ne 0 ]; then
        echo "❌ Failed to generate topology cache!"
        exit 1
    fi
    echo "✅ Topology cache generated successfully"
fi

fi  # End of Custom Mode CDL/topology block

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
    echo "✓ Data type: transition (output slew)"
else
    DATA_TYPE="cell"
    echo "✓ Data type: cell (propagation delay)"
fi

# (2) Voltage mode 선택 (Custom mode only - Default mode already has this set)
if [ "$USE_DEFAULT_MODE" == "no" ]; then
echo ""
echo "📊 (2) Voltage Feature Mode"
echo "   [1] all_nodes (default) - Voltage applied to all nodes"
echo "   [2] vdd_only - Voltage only on VDD node, 0 for others"
echo "   [3] vdd_mos - Voltage on VDD and MOS transistor nodes only"
read -p "Enter choice (1/2/3) [default: 1]: " voltage_mode_choice

VOLTAGE_MODE_FLAG=""
VOLTAGE_MODE_SUFFIX=""
if [ "$voltage_mode_choice" == "2" ]; then
    VOLTAGE_MODE="vdd_only"
    VOLTAGE_MODE_FLAG="--voltage_mode vdd_only"
    VOLTAGE_MODE_SUFFIX="_vdd_only"
    echo "✓ Voltage mode: vdd_only (V on VDD, 0 elsewhere)"
elif [ "$voltage_mode_choice" == "3" ]; then
    VOLTAGE_MODE="vdd_mos"
    VOLTAGE_MODE_FLAG="--voltage_mode vdd_mos"
    VOLTAGE_MODE_SUFFIX="_vddmos"
    echo "✓ Voltage mode: vdd_mos (V on VDD + MOS transistors)"
else
    VOLTAGE_MODE="all_nodes"
    VOLTAGE_MODE_SUFFIX=""
    echo "✓ Voltage mode: all_nodes (V on all nodes)"
fi
fi  # End of Voltage mode selection (Custom mode only)
echo

# Test dataset 여부 선택
echo "Is this a test dataset?"
echo "   [1] No (default) - Training dataset"
echo "   [2] Yes - Test dataset (batch mode: aggregate all folders, then sample per cell)"
read -p "Enter choice (1/2) [default: 1]: " test_choice

MAX_SAMPLES_PER_CELL_OPTION=""
MAX_TRAIN_TASKS_OPTION=""
BATCH_TEST_MODE=""
if [ "$test_choice" == "2" ]; then
    IS_TEST="--is_test"
    BATCH_TEST_MODE="yes"
    echo "✓ Test dataset: Yes (Batch mode enabled)"

    # ============== AUTO-VALIDATE TEST DIRECTORIES ==============
    # Fixed test directories for ASAP7
    TEST_PROCESSED_DIR="/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/ASAP7_lib_files/test_processed"
    TEST_PROCESSED_SIMPLE_DIR="/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/ASAP7_lib_files/test_processed_simple"

    echo ""
    echo "🔍 Validating test directories..."
    echo "   test_processed: $TEST_PROCESSED_DIR"
    echo "   test_processed_simple: $TEST_PROCESSED_SIMPLE_DIR"

    # Expected combinations: 3x3x3x5 = 135 folders each
    # param_a: 0, 1, 2 (3 values)
    # param_b: 0, 1, 2 (3 values)
    # param_c: 0, 1, 2 (3 values)
    # temperature: 0, 25, 50, 75, 100 (5 values)
    EXPECTED_COUNT=135
    PARAM_A_VALUES=(0 1 2)
    PARAM_B_VALUES=(0 1 2)
    PARAM_C_VALUES=(0 1 2)
    TEMP_VALUES=(0 25 50 75 100)

    # Validate invbuf folders in test_processed
    echo ""
    echo "📁 Checking invbuf folders in test_processed..."
    INVBUF_MISSING=()
    INVBUF_FOUND=0
    for a in "${PARAM_A_VALUES[@]}"; do
        for b in "${PARAM_B_VALUES[@]}"; do
            for c in "${PARAM_C_VALUES[@]}"; do
                for t in "${TEMP_VALUES[@]}"; do
                    folder_name="invbuf_${a}_${b}_${c}_${t}"
                    folder_path="$TEST_PROCESSED_DIR/$folder_name"
                    if [ -d "$folder_path" ]; then
                        INVBUF_FOUND=$((INVBUF_FOUND + 1))
                    else
                        INVBUF_MISSING+=("$folder_name")
                    fi
                done
            done
        done
    done
    echo "   Found: $INVBUF_FOUND / $EXPECTED_COUNT"

    # Validate simple folders in test_processed_simple
    echo ""
    echo "📁 Checking simple folders in test_processed_simple..."
    SIMPLE_MISSING=()
    SIMPLE_FOUND=0
    for a in "${PARAM_A_VALUES[@]}"; do
        for b in "${PARAM_B_VALUES[@]}"; do
            for c in "${PARAM_C_VALUES[@]}"; do
                for t in "${TEMP_VALUES[@]}"; do
                    folder_name="simple_${a}_${b}_${c}_${t}"
                    folder_path="$TEST_PROCESSED_SIMPLE_DIR/$folder_name"
                    if [ -d "$folder_path" ]; then
                        SIMPLE_FOUND=$((SIMPLE_FOUND + 1))
                    else
                        SIMPLE_MISSING+=("$folder_name")
                    fi
                done
            done
        done
    done
    echo "   Found: $SIMPLE_FOUND / $EXPECTED_COUNT"

    # Check if all folders exist
    if [ ${#INVBUF_MISSING[@]} -gt 0 ] || [ ${#SIMPLE_MISSING[@]} -gt 0 ]; then
        echo ""
        echo "❌ ERROR: Missing folders detected!"

        if [ ${#INVBUF_MISSING[@]} -gt 0 ]; then
            echo ""
            echo "   Missing invbuf folders (${#INVBUF_MISSING[@]}):"
            for folder in "${INVBUF_MISSING[@]:0:10}"; do
                echo "      - $folder"
            done
            if [ ${#INVBUF_MISSING[@]} -gt 10 ]; then
                echo "      ... and $((${#INVBUF_MISSING[@]} - 10)) more"
            fi
        fi

        if [ ${#SIMPLE_MISSING[@]} -gt 0 ]; then
            echo ""
            echo "   Missing simple folders (${#SIMPLE_MISSING[@]}):"
            for folder in "${SIMPLE_MISSING[@]:0:10}"; do
                echo "      - $folder"
            done
            if [ ${#SIMPLE_MISSING[@]} -gt 10 ]; then
                echo "      ... and $((${#SIMPLE_MISSING[@]} - 10)) more"
            fi
        fi

        echo ""
        echo "Please ensure all 3x3x3x5 = 135 combinations exist for both invbuf and simple."
        exit 1
    fi

    echo ""
    echo "✅ All $EXPECTED_COUNT invbuf + $EXPECTED_COUNT simple folders found!"

    # Ask which directories to process
    echo ""
    echo "Which test directories to process?"
    echo "   [1] Both invbuf + simple (default)"
    echo "   [2] Simple only (faster - for validation)"
    read -p "Enter choice (1/2) [default: 1]: " dir_choice

    case $dir_choice in
        2)
            SELECTED_BASE_DIRS=("$TEST_PROCESSED_SIMPLE_DIR")
            echo "✓ Processing: simple only (faster)"
            ;;
        *)
            SELECTED_BASE_DIRS=("$TEST_PROCESSED_DIR" "$TEST_PROCESSED_SIMPLE_DIR")
            echo "✓ Processing: both invbuf + simple"
            ;;
    esac

    # Test dataset의 경우 max_samples_per_cell 옵션 제공 (cell 당 샘플링)
    echo ""
    echo "Limit samples per cell? (after aggregating all folders)"
    echo "   [1] No limit (default) - Use all samples"
    echo "   [2] 10000 samples per cell (recommended)"
    echo "   [3] Custom limit"
    read -p "Enter choice (1/2/3) [default: 1]: " sample_limit_choice

    case $sample_limit_choice in
        2)
            MAX_SAMPLES_PER_CELL_OPTION="--max_samples_per_cell 10000"
            echo "✓ Max samples per cell: 10000"
            ;;
        3)
            read -p "Enter max samples per cell: " custom_limit
            if [[ "$custom_limit" =~ ^[0-9]+$ ]] && [ "$custom_limit" -gt 0 ]; then
                MAX_SAMPLES_PER_CELL_OPTION="--max_samples_per_cell $custom_limit"
                echo "✓ Max samples per cell: $custom_limit"
            else
                echo "⚠️  Invalid input, using no limit"
            fi
            ;;
        *)
            echo "✓ Max samples per cell: No limit"
            ;;
    esac

    # Processing mode option
    echo ""
    echo "Select processing mode:"
    echo "   [1] Standard - Load all folders into memory before saving"
    echo "   [2] Streaming - Save to disk after each folder (memory-efficient)"
    echo "   [3] Parallel (recommended) - Process folders in parallel (fastest)"
    read -p "Enter choice (1/2/3) [default: 3]: " processing_mode_choice

    STREAMING_OPTION=""
    PARALLEL_OPTION=""
    NUM_WORKERS_OPTION=""
    case $processing_mode_choice in
        1)
            echo "✓ Processing mode: Standard"
            ;;
        2)
            STREAMING_OPTION="--streaming"
            echo "✓ Processing mode: Streaming (memory-efficient)"
            ;;
        *)
            # Default: parallel mode
            read -p "Number of workers [default: 8]: " num_workers
            num_workers=${num_workers:-8}
            PARALLEL_OPTION="--parallel"
            NUM_WORKERS_OPTION="--num_workers $num_workers"
            echo "✓ Processing mode: Parallel ($num_workers workers)"
            ;;
    esac

    # Filter cells option (for validation - only process specific cells)
    echo ""
    echo "Filter specific cells? (for faster validation data generation)"
    echo "   [1] All cells (default)"
    echo "   [2] Validation cells only (21 cells - MUCH faster)"
    read -p "Enter choice (1/2) [default: 1]: " filter_choice

    FILTER_CELLS_OPTION=""
    case $filter_choice in
        2)
            # Validation cells: INTRA_TOPOLOGY + TOPOLOGY_AGNOSTIC cells
            # Include both short names (TSMC) and full ASAP7 names for compatibility
            #VALIDATION_CELLS_SHORT="FAx1,HAxp5,MAJIxp5,MAJx2,MAJx3,XNOR2x1,XNOR2x2,XNOR2xp5,XOR2x1,XOR2x2,XOR2xp5,AND2x6,OR2x6,NAND3x2,NOR2xp67"
            VALIDATION_CELLS_SHORT="NOR2xp67"
            #VALIDATION_CELLS_ASAP7="FAx1_ASAP7_75t_L"
            VALIDATION_CELLS_ASAP7="NOR2xp67_ASAP7_75t_L"
            VALIDATION_CELLS="${VALIDATION_CELLS_SHORT},${VALIDATION_CELLS_ASAP7}"
            FILTER_CELLS_OPTION="--filter_cells $VALIDATION_CELLS"
            echo "✓ Filter: Validation cells only (${VALIDATION_CELLS})"
            ;;
        *)
            echo "✓ Filter: All cells"
            ;;
    esac
else
    IS_TEST=""
    BATCH_TEST_MODE=""
    BATCH_TRAIN_MODE="yes"
    echo "✓ Test dataset: No (Training dataset - BATCH MODE)"

    # ============== AUTO-VALIDATE TRAIN DIRECTORIES ==============
    # Fixed train directories for ASAP7
    TRAIN_PROCESSED_DIR="/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/ASAP7_lib_files/processed"
    TRAIN_PROCESSED_SIMPLE_DIR="/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/ASAP7_lib_files/processed_simple"

    echo ""
    echo "🔍 Validating train directories..."
    echo "   processed: $TRAIN_PROCESSED_DIR"
    echo "   processed_simple: $TRAIN_PROCESSED_SIMPLE_DIR"

    # Expected combinations: 4x4x4x6 = 384 folders each
    # param_a: 0, 1, 2, 3 (4 values)
    # param_b: 0, 1, 2, 3 (4 values)
    # param_c: 0, 1, 2, 3 (4 values)
    # temperature: -25, 12p5, 37p5, 62p5, 87p5, 125 (6 values)
    EXPECTED_TRAIN_COUNT=384
    TRAIN_PARAM_A_VALUES=(0 1 2 3)
    TRAIN_PARAM_B_VALUES=(0 1 2 3)
    TRAIN_PARAM_C_VALUES=(0 1 2 3)
    TRAIN_TEMP_VALUES=("-25" "12p5" "37p5" "62p5" "87p5" "125")

    # Validate invbuf folders in processed
    echo ""
    echo "📁 Checking invbuf folders in processed..."
    TRAIN_INVBUF_MISSING=()
    TRAIN_INVBUF_FOUND=0
    for a in "${TRAIN_PARAM_A_VALUES[@]}"; do
        for b in "${TRAIN_PARAM_B_VALUES[@]}"; do
            for c in "${TRAIN_PARAM_C_VALUES[@]}"; do
                for t in "${TRAIN_TEMP_VALUES[@]}"; do
                    folder_name="invbuf_${a}_${b}_${c}_${t}"
                    folder_path="$TRAIN_PROCESSED_DIR/$folder_name"
                    if [ -d "$folder_path" ]; then
                        TRAIN_INVBUF_FOUND=$((TRAIN_INVBUF_FOUND + 1))
                    else
                        TRAIN_INVBUF_MISSING+=("$folder_name")
                    fi
                done
            done
        done
    done
    echo "   Found: $TRAIN_INVBUF_FOUND / $EXPECTED_TRAIN_COUNT"

    # Validate simple folders in processed_simple
    echo ""
    echo "📁 Checking simple folders in processed_simple..."
    TRAIN_SIMPLE_MISSING=()
    TRAIN_SIMPLE_FOUND=0
    for a in "${TRAIN_PARAM_A_VALUES[@]}"; do
        for b in "${TRAIN_PARAM_B_VALUES[@]}"; do
            for c in "${TRAIN_PARAM_C_VALUES[@]}"; do
                for t in "${TRAIN_TEMP_VALUES[@]}"; do
                    folder_name="simple_${a}_${b}_${c}_${t}"
                    folder_path="$TRAIN_PROCESSED_SIMPLE_DIR/$folder_name"
                    if [ -d "$folder_path" ]; then
                        TRAIN_SIMPLE_FOUND=$((TRAIN_SIMPLE_FOUND + 1))
                    else
                        TRAIN_SIMPLE_MISSING+=("$folder_name")
                    fi
                done
            done
        done
    done
    echo "   Found: $TRAIN_SIMPLE_FOUND / $EXPECTED_TRAIN_COUNT"

    # Check if all folders exist
    if [ ${#TRAIN_INVBUF_MISSING[@]} -gt 0 ] || [ ${#TRAIN_SIMPLE_MISSING[@]} -gt 0 ]; then
        echo ""
        echo "❌ ERROR: Missing train folders detected!"

        if [ ${#TRAIN_INVBUF_MISSING[@]} -gt 0 ]; then
            echo ""
            echo "   Missing invbuf folders (${#TRAIN_INVBUF_MISSING[@]}):"
            for folder in "${TRAIN_INVBUF_MISSING[@]:0:10}"; do
                echo "      - $folder"
            done
            if [ ${#TRAIN_INVBUF_MISSING[@]} -gt 10 ]; then
                echo "      ... and $((${#TRAIN_INVBUF_MISSING[@]} - 10)) more"
            fi
        fi

        if [ ${#TRAIN_SIMPLE_MISSING[@]} -gt 0 ]; then
            echo ""
            echo "   Missing simple folders (${#TRAIN_SIMPLE_MISSING[@]}):"
            for folder in "${TRAIN_SIMPLE_MISSING[@]:0:10}"; do
                echo "      - $folder"
            done
            if [ ${#TRAIN_SIMPLE_MISSING[@]} -gt 10 ]; then
                echo "      ... and $((${#TRAIN_SIMPLE_MISSING[@]} - 10)) more"
            fi
        fi

        echo ""
        echo "Please ensure all 4x4x4x6 = 384 combinations exist for both invbuf and simple."
        exit 1
    fi

    echo ""
    echo "✅ All $EXPECTED_TRAIN_COUNT invbuf + $EXPECTED_TRAIN_COUNT simple folders found!"

    # Override SELECTED_BASE_DIRS with train directories
    SELECTED_BASE_DIRS=("$TRAIN_PROCESSED_DIR" "$TRAIN_PROCESSED_SIMPLE_DIR")

    # Step: Include XOR cells option
    echo ""
    echo "📊 Include XOR cells in training data?"
    echo "   [1] No (default) - Simple cells only (invbuf + simple)"
    echo "   [2] Yes - Include XOR cells (invbuf + simple + XOR)"
    read -p "Enter choice (1/2) [default: 1]: " xor_choice

    XOR_SUFFIX=""
    if [ "$xor_choice" == "2" ]; then
        echo ""
        echo "🔍 Validating XOR directories in processed_simple..."

        # XOR directories follow the same pattern: XOR_a_b_c_t
        XOR_MISSING=()
        XOR_FOUND=0
        for a in "${TRAIN_PARAM_A_VALUES[@]}"; do
            for b in "${TRAIN_PARAM_B_VALUES[@]}"; do
                for c in "${TRAIN_PARAM_C_VALUES[@]}"; do
                    for t in "${TRAIN_TEMP_VALUES[@]}"; do
                        folder_name="XOR_${a}_${b}_${c}_${t}"
                        folder_path="$TRAIN_PROCESSED_SIMPLE_DIR/$folder_name"
                        if [ -d "$folder_path" ]; then
                            XOR_FOUND=$((XOR_FOUND + 1))
                        else
                            XOR_MISSING+=("$folder_name")
                        fi
                    done
                done
            done
        done
        echo "   Found: $XOR_FOUND / $EXPECTED_TRAIN_COUNT"

        if [ ${#XOR_MISSING[@]} -gt 0 ]; then
            echo ""
            echo "⚠️  WARNING: Missing XOR folders (${#XOR_MISSING[@]}):"
            for folder in "${XOR_MISSING[@]:0:10}"; do
                echo "      - $folder"
            done
            if [ ${#XOR_MISSING[@]} -gt 10 ]; then
                echo "      ... and $((${#XOR_MISSING[@]} - 10)) more"
            fi
            echo ""
            read -p "Continue without missing XOR folders? (y/n) [default: n]: " continue_xor
            if [ "$continue_xor" != "y" ]; then
                echo "❌ Aborted. Please ensure all XOR folders exist."
                exit 1
            fi
            echo "⚠️  Continuing with $XOR_FOUND available XOR folders..."
        else
            echo "✅ All $EXPECTED_TRAIN_COUNT XOR folders found!"
        fi

        XOR_SUFFIX="_xor"
        echo "✓ XOR cells: ENABLED (will be included in training data)"
    else
        echo "✓ XOR cells: DISABLED (simple cells only)"
    fi

    # Sampling option for train data
    echo ""
    echo "📊 Train data sampling options:"
    echo "   [1] No limit (default) - Use all tasks from all directories"
    echo "   [2] 10% per directory (recommended) - Sample 10% from each directory"
    echo "   [3] 20% per directory"
    echo "   [4] 5% per directory"
    echo "   [5] Custom ratio per directory"
    echo "   [6] Fixed total task limit (e.g., 100000 tasks)"
    read -p "Enter choice (1-6) [default: 1]: " sampling_choice

    SAMPLE_RATIO_OPTION=""
    MAX_TRAIN_TASKS_OPTION=""
    SAMPLE_SUFFIX="_full"  # Default: full dataset

    case $sampling_choice in
        2)
            SAMPLE_RATIO_OPTION="--sample_ratio_per_dir 0.1"
            SAMPLE_SUFFIX="_10pct"
            echo "✓ Sampling: 10% per directory"
            ;;
        3)
            SAMPLE_RATIO_OPTION="--sample_ratio_per_dir 0.2"
            SAMPLE_SUFFIX="_20pct"
            echo "✓ Sampling: 20% per directory"
            ;;
        4)
            SAMPLE_RATIO_OPTION="--sample_ratio_per_dir 0.05"
            SAMPLE_SUFFIX="_5pct"
            echo "✓ Sampling: 5% per directory"
            ;;
        5)
            read -p "Enter sampling ratio (0.01-0.99): " custom_ratio
            if [[ "$custom_ratio" =~ ^0\.[0-9]+$ ]]; then
                SAMPLE_RATIO_OPTION="--sample_ratio_per_dir $custom_ratio"
                # Convert to percentage for suffix (e.g., 0.15 -> 15pct)
                pct_value=$(echo "$custom_ratio * 100" | bc | cut -d'.' -f1)
                SAMPLE_SUFFIX="_${pct_value}pct"
                echo "✓ Sampling: ${custom_ratio} (${pct_value}%) per directory"
            else
                echo "⚠️  Invalid input, using no limit"
            fi
            ;;
        6)
            read -p "Enter max train tasks: " custom_tasks
            if [[ "$custom_tasks" =~ ^[0-9]+$ ]] && [ "$custom_tasks" -gt 0 ]; then
                MAX_TRAIN_TASKS_OPTION="--max_train_tasks $custom_tasks"
                SAMPLE_SUFFIX="_${custom_tasks}tasks"
                echo "✓ Max train tasks: $custom_tasks"
            else
                echo "⚠️  Invalid input, using no limit"
            fi
            ;;
        *)
            SAMPLE_SUFFIX="_full"
            echo "✓ Sampling: No limit (all tasks)"
            ;;
    esac

    # Processing mode option for train data (parallel support)
    echo ""
    echo "📊 Train processing mode:"
    echo "   [1] Standard - Sequential processing (slower, lower memory)"
    echo "   [2] Parallel (recommended) - Process folders in parallel (faster)"
    read -p "Enter choice (1/2) [default: 2]: " train_processing_choice

    TRAIN_PARALLEL_OPTION=""
    TRAIN_NUM_WORKERS_OPTION=""
    case $train_processing_choice in
        1)
            echo "✓ Processing mode: Standard (sequential)"
            ;;
        *)
            # Default: parallel mode
            read -p "Number of workers [default: 8]: " train_num_workers
            train_num_workers=${train_num_workers:-8}
            TRAIN_PARALLEL_OPTION="--parallel"
            TRAIN_NUM_WORKERS_OPTION="--num_workers $train_num_workers"
            echo "✓ Processing mode: Parallel ($train_num_workers workers)"
            ;;
    esac
fi
echo

# 처리 통계 변수
TOTAL_PROCESSED=0
TOTAL_FAILED=0

# Collect all folders to process
ALL_FOLDERS_INFO=()

# 각 base directory에 대해 처리
for base_dir in "${SELECTED_BASE_DIRS[@]}"; do
    base_name=$(basename $base_dir)

    # 하위 디렉토리 찾기
    FOLDERS=($(find $base_dir -maxdepth 1 -type d -not -path $base_dir | sort))

    if [ ${#FOLDERS[@]} -eq 0 ]; then
        echo "   ⚠️  No subdirectories found in $base_name, skipping..."
        continue
    fi

    # Batch modes (test/train): silently use all validated folders
    if [ -n "$BATCH_TEST_MODE" ] || [ -n "$BATCH_TRAIN_MODE" ]; then
        # Filter folders based on XOR option (train mode only)
        if [ -n "$BATCH_TRAIN_MODE" ] && [ -z "$XOR_SUFFIX" ]; then
            # XOR not enabled: exclude XOR_* folders
            FILTERED_FOLDERS=()
            for folder in "${FOLDERS[@]}"; do
                folder_name=$(basename "$folder")
                if [[ ! "$folder_name" =~ ^XOR_ ]]; then
                    FILTERED_FOLDERS+=("$folder")
                fi
            done
            SELECTED_FOLDERS=("${FILTERED_FOLDERS[@]}")
        else
            SELECTED_FOLDERS=("${FOLDERS[@]}")
        fi
    else
        # Legacy single folder mode: show selection UI
        echo "🔄 Processing base directory: $base_name"
        echo "📁 Found ${#FOLDERS[@]} folders in $base_name"

        for i in "${!FOLDERS[@]}"; do
            echo "   [$i] $(basename ${FOLDERS[$i]})"
        done
        echo

        echo "Choose processing mode:"
        echo "   [a] Process ALL folders"
        echo "   [s] Process a SINGLE folder"
        echo "   [r] Process a RANGE of folders"
        read -p "Enter choice (a/s/r): " choice

        if [ "$choice" == "s" ]; then
            read -p "Enter folder number to process: " folder_num
            if [ $folder_num -ge 0 ] && [ $folder_num -lt ${#FOLDERS[@]} ]; then
                SELECTED_FOLDERS=("${FOLDERS[$folder_num]}")
                echo "✓ Selected: $(basename ${SELECTED_FOLDERS[0]})"
            else
                echo "❌ Invalid folder number!"
                exit 1
            fi
        elif [ "$choice" == "r" ]; then
            read -p "Enter start folder number: " start_num
            read -p "Enter end folder number (inclusive): " end_num

            if [ $start_num -ge 0 ] && [ $end_num -lt ${#FOLDERS[@]} ] && [ $start_num -le $end_num ]; then
                SELECTED_FOLDERS=("${FOLDERS[@]:$start_num:$((end_num - start_num + 1))}")
                echo "✓ Selected range: [$start_num-$end_num] (${#SELECTED_FOLDERS[@]} folders)"
                echo "   First: $(basename ${SELECTED_FOLDERS[0]})"
                echo "   Last: $(basename ${SELECTED_FOLDERS[-1]})"
            else
                echo "❌ Invalid folder range!"
                exit 1
            fi
        else
            SELECTED_FOLDERS=("${FOLDERS[@]}")
            echo "✓ Processing all folders"
        fi
        echo
    fi

    # Collect folder info for each selected folder
    for folder in "${SELECTED_FOLDERS[@]}"; do
        folder_name=$(basename $folder)

        # 폴더에서 .lib 파일들 찾기
        lib_files=($(find $folder -maxdepth 1 -name "*.lib" | sort))

        if [ ${#lib_files[@]} -eq 0 ]; then
            # Only show warning for non-batch modes
            if [ -z "$BATCH_TEST_MODE" ] && [ -z "$BATCH_TRAIN_MODE" ]; then
                echo "   ⚠️  No .lib files found in $folder_name, skipping..."
            fi
            continue
        fi

        # 첫 번째 .lib 파일에서 prefix 추출
        first_lib=$(basename ${lib_files[0]})
        lib_prefix=$(echo "$first_lib" | sed -E 's/[0-9]+\.lib$//')

        # Voltage range 추출
        start_voltage=$(basename ${lib_files[0]} .lib | grep -oE '[0-9]+$')
        end_voltage=$(basename ${lib_files[-1]} .lib | grep -oE '[0-9]+$')

        if [ -z "$start_voltage" ] || [ -z "$end_voltage" ]; then
            start_num=40
            end_num=101
        else
            start_num=$((10#${start_voltage}))
            end_num=$((10#${end_voltage}))
            end_num=$((end_num + 1))
        fi

        # Store folder info: folder_name, base_dir, prefix, start, end
        ALL_FOLDERS_INFO+=("$folder_name,$base_dir,$lib_prefix,$start_num,$end_num")

        # Only show verbose output for non-batch modes
        if [ -z "$BATCH_TEST_MODE" ] && [ -z "$BATCH_TRAIN_MODE" ]; then
            echo "   📁 Added: $folder_name (prefix: $lib_prefix, voltage: $start_num-$end_num)"
        fi
    done
done

echo ""
echo "📊 Collected ${#ALL_FOLDERS_INFO[@]} folders to process"
echo ""

# Process based on mode (batch test vs training)
if [ -n "$BATCH_TEST_MODE" ]; then
    # ============== BATCH TEST MODE ==============
    echo "🧪 BATCH TEST MODE: Aggregating all folders, then sampling per cell"
    echo "=========================================================================="

    # Create temp file for folder info
    TEMP_FOLDERS_FILE=$(mktemp)
    echo "# folder_path,lib_base_path,prefix,start,end" > "$TEMP_FOLDERS_FILE"
    for info in "${ALL_FOLDERS_INFO[@]}"; do
        echo "$info" >> "$TEMP_FOLDERS_FILE"
    done

    echo "📄 Temp folders file: $TEMP_FOLDERS_FILE"
    cat "$TEMP_FOLDERS_FILE"
    echo ""

    # Output directory for cell-based test files (includes data_type to separate cell vs transition)
    # Include INPUT_PORTS_SUFFIX, VOLTAGE_MODE_SUFFIX and SLEW_SUFFIX since they affect node features
    TEST_OUTPUT_DIR="${OUTPUT_DIR}test_by_${DATA_TYPE}_${GRAPH_MODE}${INPUT_PORTS_SUFFIX}${VOLTAGE_MODE_SUFFIX}${SLEW_SUFFIX}/"
    mkdir -p "$TEST_OUTPUT_DIR"

    echo "📂 Test output directory: $TEST_OUTPUT_DIR"
    echo ""

    # Call Python (mode auto-detected from test_folders_file + output_dir)
    echo "⚙️  Running batch test data generation..."
    python3 build_gnn_dataset_cached_with_process.py \
        --cache_path "$CACHE_FILE" \
        --cache_type "$GRAPH_MODE" \
        --data_type "$DATA_TYPE" \
        --test_folders_file "$TEMP_FOLDERS_FILE" \
        --output_dir "$TEST_OUTPUT_DIR" \
        $MAX_SAMPLES_PER_CELL_OPTION \
        $STREAMING_OPTION \
        $PARALLEL_OPTION \
        $NUM_WORKERS_OPTION \
        $FILTER_CELLS_OPTION \
        $VOLTAGE_MODE_FLAG \
        $SLEW_MODE_FLAG

    if [ $? -eq 0 ]; then
        TOTAL_PROCESSED=${#ALL_FOLDERS_INFO[@]}
        echo "✅ Batch test processing complete!"
    else
        TOTAL_FAILED=${#ALL_FOLDERS_INFO[@]}
        echo "❌ Batch test processing failed!"
    fi

    # Clean up temp file
    rm -f "$TEMP_FOLDERS_FILE"

elif [ -n "$BATCH_TRAIN_MODE" ]; then
    # ============== BATCH TRAIN MODE (UNIFIED) ==============
    echo "🏋️ BATCH TRAIN MODE: Aggregating ALL train folders into unified dataset"
    echo "=========================================================================="

    # Create temp file for folder info
    TEMP_TRAIN_FOLDERS_FILE=$(mktemp)
    echo "# folder_path,lib_base_path,prefix,start,end" > "$TEMP_TRAIN_FOLDERS_FILE"
    for info in "${ALL_FOLDERS_INFO[@]}"; do
        echo "$info" >> "$TEMP_TRAIN_FOLDERS_FILE"
    done

    echo "📄 Train folders file: $TEMP_TRAIN_FOLDERS_FILE"
    echo "   Total folders: ${#ALL_FOLDERS_INFO[@]}"
    echo ""

    # Output file for unified train dataset (includes sampling info)
    # Include INPUT_PORTS_SUFFIX and VOLTAGE_MODE_SUFFIX since they affect node features
    # SAMPLE_SUFFIX goes at the end to indicate what % of data was used
    TRAIN_OUTPUT_FILE="${OUTPUT_DIR}train_${DATA_TYPE}_${GRAPH_MODE}${INPUT_PORTS_SUFFIX}${VOLTAGE_MODE_SUFFIX}${SLEW_SUFFIX}${XOR_SUFFIX}${SAMPLE_SUFFIX}.pth"
    mkdir -p "$(dirname $TRAIN_OUTPUT_FILE)"

    echo "📂 Train output file: $TRAIN_OUTPUT_FILE"
    echo ""

    # Call Python (mode auto-detected from train_folders_file + train_output)
    echo "⚙️  Running unified train data generation..."
    python3 build_gnn_dataset_cached_with_process.py \
        --cache_path "$CACHE_FILE" \
        --cache_type "$GRAPH_MODE" \
        --data_type "$DATA_TYPE" \
        --train_folders_file "$TEMP_TRAIN_FOLDERS_FILE" \
        --train_output "$TRAIN_OUTPUT_FILE" \
        $SAMPLE_RATIO_OPTION \
        $MAX_TRAIN_TASKS_OPTION \
        $TRAIN_PARALLEL_OPTION \
        $TRAIN_NUM_WORKERS_OPTION \
        $VOLTAGE_MODE_FLAG \
        $SLEW_MODE_FLAG

    if [ $? -eq 0 ]; then
        TOTAL_PROCESSED=${#ALL_FOLDERS_INFO[@]}
        echo "✅ Batch train processing complete!"
        if [ -f "$TRAIN_OUTPUT_FILE" ]; then
            FILE_SIZE=$(du -h "$TRAIN_OUTPUT_FILE" | cut -f1)
            echo "📦 Output: $FILE_SIZE - $(basename $TRAIN_OUTPUT_FILE)"
        fi
    else
        TOTAL_FAILED=${#ALL_FOLDERS_INFO[@]}
        echo "❌ Batch train processing failed!"
    fi

    # Clean up temp file
    rm -f "$TEMP_TRAIN_FOLDERS_FILE"

else
    # ============== SINGLE FOLDER MODE (legacy) ==============
    echo "🏋️ SINGLE FOLDER MODE: Processing each folder separately"
    echo "=========================================================================="

    for info in "${ALL_FOLDERS_INFO[@]}"; do
        IFS=',' read -r folder_name base_dir lib_prefix start_num end_num <<< "$info"

        # 출력 폴더명 생성
        base_name=$(basename $base_dir)
        if [ "$base_name" == "processed_simple" ]; then
            suffix="${folder_name#simple_}"
            output_folder_name="${base_name}_${suffix}"
        else
            output_folder_name="${base_name}_${folder_name}"
        fi

        echo "🔄 Processing folder: $folder_name -> $output_folder_name"
        echo "   🏷️  Library prefix: $lib_prefix"
        echo "   📈 Voltage range: $start_num to $end_num (exclusive)"

        # Output 파일명
        OUTPUT_SUBDIR="${OUTPUT_DIR}${output_folder_name}/graph_data/"
        mkdir -p "$OUTPUT_SUBDIR"
        ALL_DATA_FILE="${OUTPUT_SUBDIR}${DATA_TYPE}_all_graph_data_${GRAPH_MODE}.pth"

        # GNN 데이터셋 생성
        echo "   ⚙️  Generating GNN dataset..."

        python3 build_gnn_dataset_cached_with_process.py \
            --cache_path "$CACHE_FILE" \
            --cache_type "$GRAPH_MODE" \
            --start $start_num \
            --end $end_num \
            --prefix "$lib_prefix" \
            --data_dir "$folder_name" \
            --lib_base_path "$base_dir" \
            --save_input "$ALL_DATA_FILE" \
            --data_type "$DATA_TYPE" \
            $MAX_TRAIN_TASKS_OPTION \
            $SLEW_MODE_FLAG

        if [ $? -eq 0 ]; then
            echo "   ✅ Successfully processed $folder_name"
            TOTAL_PROCESSED=$((TOTAL_PROCESSED + 1))

            if [ -f "$ALL_DATA_FILE" ]; then
                FILE_SIZE=$(du -h "$ALL_DATA_FILE" | cut -f1)
                echo "   📦 Output: $FILE_SIZE - $(basename $ALL_DATA_FILE)"
            fi
        else
            echo "   ❌ Failed to process $folder_name"
            TOTAL_FAILED=$((TOTAL_FAILED + 1))
        fi
        echo
    done
fi

echo ""
echo "🎉 GNN Dataset Processing Complete!"
echo "=========================================================================="
echo "   Graph mode: $GRAPH_MODE"
echo "   Data type: $DATA_TYPE"
echo "   Voltage mode: $VOLTAGE_MODE"
echo "   Node features: 11D (7 base + 4 process parameters)"
echo "   Topology cache: $(basename $CACHE_FILE)"
echo "   Slew mode: $SLEW_MODE"
if [ -n "$BATCH_TEST_MODE" ]; then
    echo "   Mode: BATCH TEST (aggregate all folders, sample per cell)"
    if [ -n "$PARALLEL_OPTION" ]; then
        echo "   Processing: Parallel ($(echo $NUM_WORKERS_OPTION | grep -oE '[0-9]+') workers)"
    elif [ -n "$STREAMING_OPTION" ]; then
        echo "   Processing: Streaming (memory-efficient)"
    else
        echo "   Processing: Standard"
    fi
    if [ -n "$MAX_SAMPLES_PER_CELL_OPTION" ]; then
        echo "   Max samples per cell: $(echo $MAX_SAMPLES_PER_CELL_OPTION | grep -oE '[0-9]+')"
    fi
    echo "   Test output: ${OUTPUT_DIR}test_by_${DATA_TYPE}_${GRAPH_MODE}${INPUT_PORTS_SUFFIX}${VOLTAGE_MODE_SUFFIX}${SLEW_SUFFIX}/"
elif [ -n "$BATCH_TRAIN_MODE" ]; then
    echo "   Mode: BATCH TRAIN (unified dataset from all folders)"
    if [ -n "$TRAIN_PARALLEL_OPTION" ]; then
        echo "   Processing: Parallel ($(echo $TRAIN_NUM_WORKERS_OPTION | grep -oE '[0-9]+') workers)"
    else
        echo "   Processing: Standard (sequential)"
    fi
    if [ -n "$SAMPLE_RATIO_OPTION" ]; then
        ratio=$(echo $SAMPLE_RATIO_OPTION | grep -oE '[0-9]+\.[0-9]+')
        echo "   Sampling: ${ratio} ($(echo "$ratio * 100" | bc)%) per directory"
    elif [ -n "$MAX_TRAIN_TASKS_OPTION" ]; then
        echo "   Max train tasks: $(echo $MAX_TRAIN_TASKS_OPTION | grep -oE '[0-9]+')"
    else
        echo "   Sampling: No limit (all tasks)"
    fi
    echo "   Excluded from train: INTRA_TOPOLOGY_CELLS (4 cells)"
    if [ -n "$XOR_SUFFIX" ]; then
        echo "   XOR cells: INCLUDED"
    else
        echo "   XOR cells: NOT included"
    fi
    echo "   Train output: ${OUTPUT_DIR}train_${DATA_TYPE}_${GRAPH_MODE}${INPUT_PORTS_SUFFIX}${VOLTAGE_MODE_SUFFIX}${SLEW_SUFFIX}${XOR_SUFFIX}${SAMPLE_SUFFIX}.pth"
else
    echo "   Mode: SINGLE FOLDER (per-folder processing)"
    if [ -n "$MAX_TRAIN_TASKS_OPTION" ]; then
        echo "   Max train tasks: $(echo $MAX_TRAIN_TASKS_OPTION | grep -oE '[0-9]+')"
    fi
fi
echo "   Total folders processed: $TOTAL_PROCESSED"
echo "   Total failed: $TOTAL_FAILED"
echo "   Results in: $OUTPUT_DIR"
echo ""
echo "📊 Cache file location:"
echo "   $CACHE_FILE"
echo ""
echo "✅ Done!"
