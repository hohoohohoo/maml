#!/bin/bash

# TSMC data processing script with a,b,c parameter mapping based on corner/temperature
# Maps corner and temperature combinations to specific a,b,c parameters

OUTPUT_DIR="/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_tsmc_processed/topology_agnostic_data_reduced"
DATA_DIR="/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_TSMC"  # TSMC data directory

# Parameter mappings for a,b,c
# All parameters now use nmos,pmos pairs format
PARAM_A="1.427,1.457,1.430,1.470,1.443,1.483,1.43,1.47,1.43,1.47"     # A parameters (nmos,pmos pairs)
PARAM_B="0.026,0.045,0,0,-0.026,-0.05,0.0208,-0.04,0.036,-0.0208"                  # B parameters (nmos,pmos pairs)  
PARAM_C="0.024,2.000,0.024,2.000,0.024,2.000,0.024,2.000,0.024,2.000"     # C parameters (nmos,pmos pairs)

# Corner and temperature mappings
# Define how corners and temperatures map to a,b,c indices
# Corner to A parameter index mapping (for nmos,pmos pairs)
declare -A CORNER_TO_A_INDEX=(
    ["FF"]=0  # FF -> a[0,1] = 0.625, 0.625
    ["TT"]=1  # TT -> a[2,3] = 0.875, 0.875  
    ["SS"]=2  # SS -> a[4,5] = 1.125, 1.125
    ["FS"]=3  # FS -> a[6,7] = 1.375, 1.375
    ["SF"]=4  # SF -> a[8,9] = 1.375, 1.375
)

declare -A TEMP_TO_BC_INDEX=(
    ["0"]=0    # 0°C -> b[0,1], c[0,1] (first nmos/pmos pair)
    ["25"]=1   # 25°C -> b[2,3], c[2,3] (second pair)
    ["50"]=2   # 50°C -> b[4,5], c[4,5] (third pair)  
    ["75"]=3   # 75°C -> b[6,7], c[6,7] (fourth pair)
    ["100"]=4  # 100°C -> b[6,7], c[6,7] (same as 75°C)
)

# Test cell types (empty means all data goes to train)
TEST_CELL_TYPES=("HA1D0BWP30P140" "FA1D0BWP30P140" "IOA21D0BWP30P140" "IOA21D1BWP30P140" "OA21D0BWP30P140" "OA21D1BWP30P140" "OA211D0BWP30P140" "OA211D1BWP30P140" "IAO21D0BWP30P140" "IAO21D1BWP30P140" "AO21D0BWP30P140" "AO21D1BWP30P140" "AO211D0BWP30P140" "AO211D1BWP30P140" )  # Can be modified to specific cell types if needed

echo "🚀 Processing TSMC data with a,b,c parameter mapping..."
echo "📁 Data directory: $DATA_DIR"
echo "📊 Output directory: $OUTPUT_DIR"
echo "🎯 Cell types for separate test sets: ${TEST_CELL_TYPES[@]}"
echo ""
echo "🔧 Parameter A mapping: $PARAM_A"
echo "🔧 Parameter B mapping (nmos,pmos pairs): $PARAM_B"
echo "🔧 Parameter C mapping (nmos,pmos pairs): $PARAM_C"
echo ""

# Check if DATA_DIR exists
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Data directory does not exist: $DATA_DIR"
    exit 1
fi

# Find all TSMC folders
TSMC_FOLDERS=$(find "$DATA_DIR" -maxdepth 1 -type d -name "TSMC_*" | sort)
FOLDER_COUNT=$(echo "$TSMC_FOLDERS" | wc -l)

if [ $FOLDER_COUNT -eq 0 ]; then
    echo "❌ No TSMC folders found in $DATA_DIR"
    exit 1
fi

echo "📂 Found $FOLDER_COUNT TSMC folders:"

# Count total .lib files
TOTAL_LIB_FILES=0
for FOLDER in $TSMC_FOLDERS; do
    FOLDER_NAME=$(basename "$FOLDER")
    LIB_COUNT=$(find "$FOLDER" -maxdepth 1 -name "*.lib" 2>/dev/null | wc -l)
    TOTAL_LIB_FILES=$((TOTAL_LIB_FILES + LIB_COUNT))
    echo "   $FOLDER_NAME: $LIB_COUNT .lib files"
done

echo ""
echo "✅ Will process $FOLDER_COUNT folders with $TOTAL_LIB_FILES total .lib files"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Process each cell type separately (if specified)
if [[ "${TEST_CELL_TYPES[@]}" != " " ]] && [[ -n "${TEST_CELL_TYPES[@]// }" ]]; then
    SUCCESSFUL_CELL_TYPES=()
    FAILED_CELL_TYPES=()
    
    for CELL_TYPE in "${TEST_CELL_TYPES[@]}"; do
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "🔄 Processing cell type: $CELL_TYPE"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        # Create cell-specific output directory
        CELL_OUTPUT_DIR="$OUTPUT_DIR/$CELL_TYPE"
        mkdir -p "$CELL_OUTPUT_DIR"
        
        # Run the processing script for this specific cell type
        python3 build_and_split_dataset_tsmc.py \
            --data-dirs "$DATA_DIR" \
            --output-dir "$CELL_OUTPUT_DIR" \
            --test-cell-types "$CELL_TYPE" \
            --param-a "$PARAM_A" \
            --param-b "$PARAM_B" \
            --param-c "$PARAM_C"
        
        PYTHON_EXIT_CODE=$?
        
        if [ $PYTHON_EXIT_CODE -eq 0 ]; then
            echo "✅ Successfully processed cell type: $CELL_TYPE"
            SUCCESSFUL_CELL_TYPES+=("$CELL_TYPE")
        else
            echo "❌ Failed to process cell type: $CELL_TYPE"
            FAILED_CELL_TYPES+=("$CELL_TYPE")
        fi
    done
fi

# Create train dataset (all data or excluding test cell types)
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔄 Creating TRAIN dataset"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check if TEST_CELL_TYPES is empty or contains only spaces
if [[ "${TEST_CELL_TYPES[@]}" == " " ]] || [[ -z "${TEST_CELL_TYPES[@]// }" ]]; then
    echo "📝 No test cell types specified - creating train dataset with ALL cell types"
    python3 build_and_split_dataset_tsmc_simple.py \
        --data-dirs "$DATA_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --test-cell-types "NONE" \
        --param-a "$PARAM_A" \
        --param-b "$PARAM_B" \
        --param-c "$PARAM_C" \
        --train-only
else
    echo "📝 Creating train dataset excluding: ${TEST_CELL_TYPES[@]}"
    python3 build_and_split_dataset_tsmc_simple.py \
        --data-dirs "$DATA_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --test-cell-types ${TEST_CELL_TYPES[@]} \
        --param-a "$PARAM_A" \
        --param-b "$PARAM_B" \
        --param-c "$PARAM_C" \
        --train-only
fi

TRAIN_EXIT_CODE=$?

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✅ Successfully created train dataset"
else
    echo "❌ Failed to create train dataset"
fi

# Summary
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 FINAL SUMMARY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [[ "${TEST_CELL_TYPES[@]}" != " " ]] && [[ -n "${TEST_CELL_TYPES[@]// }" ]]; then
    if [ ${#SUCCESSFUL_CELL_TYPES[@]} -gt 0 ]; then
        echo "✅ Successfully processed cell types (${#SUCCESSFUL_CELL_TYPES[@]}/${#TEST_CELL_TYPES[@]}):"
        for CELL_TYPE in "${SUCCESSFUL_CELL_TYPES[@]}"; do
            echo "   📁 $CELL_TYPE:"
            echo "      🎯 TEST:  $OUTPUT_DIR/$CELL_TYPE/tsmc_merged_test_input.pth"
            echo "      🎯 TEST:  $OUTPUT_DIR/$CELL_TYPE/tsmc_merged_test_output.pth"
            echo "      📝 INFO:  $OUTPUT_DIR/$CELL_TYPE/tsmc_merged_test_cell_info.txt"
        done
    fi
    
    if [ ${#FAILED_CELL_TYPES[@]} -gt 0 ]; then
        echo ""
        echo "❌ Failed to process cell types (${#FAILED_CELL_TYPES[@]}/${#TEST_CELL_TYPES[@]}):"
        for CELL_TYPE in "${FAILED_CELL_TYPES[@]}"; do
            echo "   ❌ $CELL_TYPE"
        done
    fi
fi

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ Train dataset:"
    echo "   🏋️ TRAIN: $OUTPUT_DIR/tsmc_topology_agnostic_train_input.pth"
    echo "   🏋️ TRAIN: $OUTPUT_DIR/tsmc_topology_agnostic_train_output.pth"
    if [[ "${TEST_CELL_TYPES[@]}" != " " ]] && [[ -n "${TEST_CELL_TYPES[@]// }" ]]; then
        echo "   📝 Excluded: ${TEST_CELL_TYPES[@]}"
    else
        echo "   📝 Included: ALL cell types"
    fi
fi

echo ""
echo "🎯 Dataset Structure:"
echo "   📁 Output directory: $OUTPUT_DIR"
echo "   📂 Total folders processed: $FOLDER_COUNT"
echo "   📄 Total .lib files: $TOTAL_LIB_FILES"
echo ""
echo "💡 Dataset Features:"
echo "   🔢 A parameters: $PARAM_A"
echo "   🔢 B parameters: $PARAM_B"
echo "   🔢 C parameters: $PARAM_C"
echo "   🌡️ Temperature values: 0, 25, 50, 75, 100"
echo ""
echo "💻 Usage example:"
echo "   import torch"
echo "   # Load train data"
echo "   train_input = torch.load('$OUTPUT_DIR/tsmc_topology_agnostic_train_input.pth')"
echo "   train_output = torch.load('$OUTPUT_DIR/tsmc_topology_agnostic_train_output.pth')"
echo "   # Features: [a, b, c, ...]"

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "🎉 Processing completed successfully!"
    exit 0
else
    echo ""
    echo "⚠️ Some errors occurred during processing."
    exit 1
fi