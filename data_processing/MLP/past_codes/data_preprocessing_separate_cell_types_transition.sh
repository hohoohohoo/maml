#!/bin/bash

# Separate cell type test sets processing script
# Creates separate test set files for each specified cell type
# Train set remains the same (all other cell types)

OUTPUT_DIR="../../../dataset_ex2_upgrade/intra_topology_data"
#DATA_DIRS=("../../dataset_ex2/test_processed_simple" "../../dataset_ex2/test_processed")  # Multiple directories
DATA_DIRS=("../../dataset_ex2/processed_simple" )  # Multiple directories

# Parameter mappings for a,b,c 
#PARAM_A="0.75,1.0,1.25"                          # A parameters
PARAM_A="0.625,0.875,1.125,1.375"                          # A parameters
#PARAM_B="0.09,0.062,0.092,0.066,0.094,0.07"     # B parameters (nmos,pmos pairs) 
PARAM_B="0.089,0.06,0.091,0.064,0.093,0.068,0.095,0.072"     # B parameters (nmos,pmos pairs)  
#PARAM_C="0.36,0.47,0.38,0.475,0.40,0.48"        # C parameters (nmos,pmos pairs)
PARAM_C="0.35,0.465,0.37,0.473,0.39,0.478,0.41,0.485"        # C parameters (nmos,pmos pairs)

# Each cell type will get its own test set, train set contains all other cell types
# Using actual cell names from TSMC library
#TEST_CELL_TYPES=("MAJIxp5" "MAJx2" "MAJx3" "HAxp5" "FAx1" "XOR2xp5" "XOR2x2" "XOR2x1" "XNOR2xp5" "XNOR2x2" "XNOR2x1")  # Each will create separate test files
TEST_CELL_TYPES=("AND2x6" "NAND3x2" "NOR2xp67" "OR2x6")  # Each will create separate test files
# Each cell type will get its own test set, train set contains all other cell types
#TEST_CELL_TYPES=("XOR2" "MAJ" "MAJI" "HA" "FA" "XNOR2")  # Each will create separate test files
#TEST_CELL_TYPES=("MAJIxp5" "MAJx2" "MAJx3" "HAxp5" "FAx1" "XOR2xp5" "XOR2x2" "XOR2x1" "XNOR2xp5" "XNOR2x2" "XNOR2x1")  # Each will create separate test files
echo "🚀 Creating separate test sets for each cell type..."
echo "📁 Base directories: ${DATA_DIRS[@]}"
echo "📊 Output directory: $OUTPUT_DIR"
echo "🎯 Cell types for separate test sets: ${TEST_CELL_TYPES[@]}"
echo "🔧 Parameter A mapping: $PARAM_A"
echo "🔧 Parameter B mapping (nmos,pmos pairs): $PARAM_B"  
echo "🔧 Parameter C mapping (nmos,pmos pairs): $PARAM_C"

# Check if all DATA_DIRS exist and collect all folders
ALL_FOLDERS=()
TOTAL_DATA_DIRS=0

for DATA_DIR in "${DATA_DIRS[@]}"; do
    echo ""
    echo "🔍 Checking directory: $DATA_DIR"
    
    if [ ! -d "$DATA_DIR" ]; then
        echo "   ⚠️ Directory does not exist: $DATA_DIR (skipping)"
        continue
    fi
    
    TOTAL_DATA_DIRS=$((TOTAL_DATA_DIRS + 1))
    
    # Find all subdirectories (ex2_* or others)
    DIR_FOLDERS=$(find $DATA_DIR -maxdepth 1 -type d ! -path $DATA_DIR | sort)
    DIR_FOLDER_COUNT=$(echo "$DIR_FOLDERS" | wc -l)
    
    if [ $DIR_FOLDER_COUNT -eq 0 ]; then
        echo "   ⚠️ No subdirectories found in $DATA_DIR"
        continue
    fi
    
    echo "   📂 Found $DIR_FOLDER_COUNT folders in $(basename $DATA_DIR):"
    
    # Add folders from this directory to the global list
    for FOLDER in $DIR_FOLDERS; do
        FOLDER_NAME=$(basename $FOLDER)
        LIB_COUNT=$(find "$FOLDER" -maxdepth 1 -name "*.lib" | wc -l)
        
        echo "      $FOLDER_NAME: $LIB_COUNT .lib files"
        
        if [ $LIB_COUNT -gt 0 ]; then
            ALL_FOLDERS+=("$FOLDER")
        else
            echo "         ❌ No .lib files found, skipping..."
        fi
    done
done

if [ $TOTAL_DATA_DIRS -eq 0 ]; then
    echo "❌ No valid data directories found"
    exit 1
fi

if [ ${#ALL_FOLDERS[@]} -eq 0 ]; then
    echo "❌ No valid folders with .lib files found in any directory"
    exit 1
fi

echo ""
echo "📂 Summary: Found ${#ALL_FOLDERS[@]} valid folders across $TOTAL_DATA_DIRS directories"

# Calculate total lib files
TOTAL_LIB_FILES=0
for FOLDER in "${ALL_FOLDERS[@]}"; do
    FOLDER_NAME=$(basename $FOLDER)
    LIB_COUNT=$(find "$FOLDER" -maxdepth 1 -name "*.lib" | wc -l)
    TOTAL_LIB_FILES=$((TOTAL_LIB_FILES + LIB_COUNT))
done

echo "✅ Will process ${#ALL_FOLDERS[@]} valid folders with $TOTAL_LIB_FILES total .lib files"
echo "🎯 Creating separate test sets for: ${TEST_CELL_TYPES[@]}"
echo "🏋️ Train set will contain all other cell types (same for all test sets)"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Create temporary directory with symbolic links to all folders
TEMP_DATA_DIR="$OUTPUT_DIR/temp_combined_data"

# Clean up any existing temp directory to avoid conflicts
if [ -d "$TEMP_DATA_DIR" ]; then
    echo "🧹 Removing existing temporary directory to avoid conflicts..."
    rm -rf "$TEMP_DATA_DIR"
fi

mkdir -p "$TEMP_DATA_DIR"

echo "🔗 Creating temporary combined directory with symbolic links..."
for FOLDER in "${ALL_FOLDERS[@]}"; do
    FOLDER_NAME=$(basename $FOLDER)
    PARENT_DIR=$(basename $(dirname $FOLDER))
    
    # Keep original folder name, add suffix only if conflict occurs
    LINK_NAME="$FOLDER_NAME"
    
    if [ -e "$TEMP_DATA_DIR/$LINK_NAME" ]; then
        echo "   ⚠️ Link name conflict: $LINK_NAME, adding parent directory prefix"
        # Add parent directory prefix to resolve conflict
        LINK_NAME="${PARENT_DIR}_${FOLDER_NAME}"
        
        # If still conflicts, use hash
        if [ -e "$TEMP_DATA_DIR/$LINK_NAME" ]; then
            echo "   ⚠️ Still conflicting: $LINK_NAME, using absolute path hash"
            HASH=$(echo "$FOLDER" | md5sum | cut -d' ' -f1 | cut -c1-8)
            LINK_NAME="${FOLDER_NAME}_${HASH}"
        fi
    fi
    
    ln -sf "$(realpath $FOLDER)" "$TEMP_DATA_DIR/$LINK_NAME"
    echo "   ✅ Linked: $(basename $FOLDER) -> $LINK_NAME"
done

# Check for --train-only flag
if [[ "$1" == "--train-only" ]]; then
    echo "🏋️ TRAIN ONLY MODE: Skipping individual test set creation"
    SUCCESSFUL_CELL_TYPES=("${TEST_CELL_TYPES[@]}")  # Assume all successful for train-only mode
    FAILED_CELL_TYPES=()
else
    # Process each cell type separately
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
        
        # Run the processing script for this specific cell type (test only mode)
        python3 build_and_split_dataset_separate_cell_types_transition.py \
            --data-dirs "$TEMP_DATA_DIR" \
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

# Create shared train dataset excluding all TEST_CELL_TYPES
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔄 Creating SHARED TRAIN dataset (excluding all test cell types)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check if TEST_CELL_TYPES is empty or contains only spaces
if [[ "${TEST_CELL_TYPES[@]}" == " " ]] || [[ -z "${TEST_CELL_TYPES[@]// }" ]]; then
    echo "📝 No test cell types specified - creating train dataset with ALL cell types"
    python3 build_and_split_dataset_separate_cell_types_transition.py \
        --data-dirs "$TEMP_DATA_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --test-cell-types "NONE" \
        --param-a "$PARAM_A" \
        --param-b "$PARAM_B" \
        --param-c "$PARAM_C" \
        --train-only
else
    python3 build_and_split_dataset_separate_cell_types_transition.py \
        --data-dirs "$TEMP_DATA_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --test-cell-types ${TEST_CELL_TYPES[@]} \
        --param-a "$PARAM_A" \
        --param-b "$PARAM_B" \
        --param-c "$PARAM_C" \
        --train-only
fi

TRAIN_EXIT_CODE=$?

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✅ Successfully created shared train dataset"
else
    echo "❌ Failed to create shared train dataset"
fi

# Clean up temporary directory
echo ""
echo "🧹 Cleaning up temporary directory..."
rm -rf "$TEMP_DATA_DIR"

# Summary
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 FINAL SUMMARY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ ${#SUCCESSFUL_CELL_TYPES[@]} -gt 0 ]; then
    echo "✅ Successfully processed cell types (${#SUCCESSFUL_CELL_TYPES[@]}/${#TEST_CELL_TYPES[@]}):"
    for CELL_TYPE in "${SUCCESSFUL_CELL_TYPES[@]}"; do
        echo "   📁 $CELL_TYPE:"
        echo "      🎯 TEST:  $OUTPUT_DIR/$CELL_TYPE/cell_merged_test_input.pth"
        echo "      🎯 TEST:  $OUTPUT_DIR/$CELL_TYPE/cell_merged_test_output.pth"
        echo "      📝 INFO:  $OUTPUT_DIR/$CELL_TYPE/cell_merged_test_cell_info.txt"
    done
    
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✅ Shared train dataset (excluding all test cell types):"
        echo "   🏋️ TRAIN: $OUTPUT_DIR/cell_topology_agnostic_train_input.pth"
        echo "   🏋️ TRAIN: $OUTPUT_DIR/cell_topology_agnostic_train_output.pth"
        echo "   📝 Excluded: ${TEST_CELL_TYPES[@]}"
    fi
fi

if [ ${#FAILED_CELL_TYPES[@]} -gt 0 ]; then
    echo ""
    echo "❌ Failed to process cell types (${#FAILED_CELL_TYPES[@]}/${#TEST_CELL_TYPES[@]}):"
    for CELL_TYPE in "${FAILED_CELL_TYPES[@]}"; do
        echo "   ❌ $CELL_TYPE"
    done
fi

echo ""
echo "🎯 Dataset Structure:"
echo "   📁 Output directory: $OUTPUT_DIR"
echo "   📂 Total folders processed: ${#ALL_FOLDERS[@]}"
echo "   📄 Total .lib files: $TOTAL_LIB_FILES"
echo ""
echo "💡 Dataset Structure:"
echo "   🎯 Each cell type has its own test directory"
echo "   🏋️ One shared train set excluding all test cell types"
echo ""
echo "💻 Usage example:"
echo "   import torch"
echo "   # Load shared train data (excludes all test cell types)"
echo "   train_input = torch.load('$OUTPUT_DIR/shared_train_input.pth')"
echo "   train_output = torch.load('$OUTPUT_DIR/shared_train_output.pth')"
echo "   "
echo "   # Load AND2x6-specific test data"
echo "   test_input = torch.load('$OUTPUT_DIR/AND2x6/cell_merged_test_input.pth')"
echo "   test_output = torch.load('$OUTPUT_DIR/AND2x6/cell_merged_test_output.pth')"
echo "   "
echo "   # Or load NAND2xp67-specific test data"
echo "   test_input = torch.load('$OUTPUT_DIR/NAND2xp67/cell_merged_test_input.pth')"
echo "   test_output = torch.load('$OUTPUT_DIR/NAND2xp67/cell_merged_test_output.pth')"

if [ ${#SUCCESSFUL_CELL_TYPES[@]} -eq ${#TEST_CELL_TYPES[@]} ]; then
    echo ""
    echo "🎉 All cell types processed successfully!"
    exit 0
else
    echo ""
    echo "⚠️ Some cell types failed to process. Check individual cell type outputs."
    exit 1
fi