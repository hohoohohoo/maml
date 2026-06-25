#!/bin/bash

# Cell type-based train/test split dataset processing script
# Separates specific cell types for test set, others for train set
# Based on cell names from timing data in lib files

OUTPUT_DIR="../../dataset_ex2/cell_type_split"
DATA_DIRS=("../../dataset_ex2/test_processed_simple" "../../dataset_ex2/test_processed")  # Multiple directories

# Parameter mappings for a,b,c 
PARAM_A="0.75,1.0,1.25"                          # A parameters
PARAM_A="0.625,0.875,1.125,1.375"                          # A parameters
PARAM_B="0.09,0.062,0.092,0.066,0.094,0.07"     # B parameters (nmos,pmos pairs) 
PARAM_B="0.089,0.06,0.091,0.064,0.093,0.068,0.095,0.072"     # B parameters (nmos,pmos pairs)  
PARAM_C="0.36,0.47,0.38,0.475,0.40,0.48"        # C parameters (nmos,pmos pairs)
PARAM_C="0.35,0.465,0.37,0.473,0.39,0.478,0.41,0.485"        # C parameters (nmos,pmos pairs)

# Cell types to include in TEST set (others go to TRAIN set)
# You can modify this list to specify which cell types should be in test set
TEST_CELL_TYPES="XOR2,MAJ,MAJI,HA,FA,XNOR2"  # Examples: INV, NAND2, NOR2, AND2, OR2, XOR2, etc.
TEST_CELL_TYPES=""  # Examples: INV, NAND2, NOR2, AND2, OR2, XOR2, etc.

echo "🚀 Creating cell type-based train/test split dataset..."
echo "📁 Base directories: ${DATA_DIRS[@]}"
echo "📊 Output directory: $OUTPUT_DIR"
echo "🎯 Test cell types: $TEST_CELL_TYPES"
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
echo "📂 Summary: Found ${#ALL_FOLDERS[@]} valid folders across $TOTAL_DATA_DIRS directories:"

# Calculate total lib files
TOTAL_LIB_FILES=0
for FOLDER in "${ALL_FOLDERS[@]}"; do
    FOLDER_NAME=$(basename $FOLDER)
    LIB_COUNT=$(find "$FOLDER" -maxdepth 1 -name "*.lib" | wc -l)
    echo "   $FOLDER_NAME: $LIB_COUNT .lib files"
    TOTAL_LIB_FILES=$((TOTAL_LIB_FILES + LIB_COUNT))
done

echo ""
echo "✅ Will process ${#ALL_FOLDERS[@]} valid folders with $TOTAL_LIB_FILES total .lib files"
echo "🎯 Cell types for TEST set: $TEST_CELL_TYPES"
echo "🏋️ Remaining cell types will go to TRAIN set"
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

# Run the cell type-based processing script
echo ""
echo "🔄 Building cell type-based train/test split dataset..."
python3 build_and_split_dataset_by_cell_type.py --merge \
    "$OUTPUT_DIR" \
    "$TEMP_DATA_DIR" \
    "$PARAM_A" \
    "$PARAM_B" \
    "$PARAM_C" \
    "$TEST_CELL_TYPES"

PYTHON_EXIT_CODE=$?

# Clean up temporary directory
echo ""
echo "🧹 Cleaning up temporary directory..."
rm -rf "$TEMP_DATA_DIR"

if [ $PYTHON_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "✅ Cell type-based train/test split dataset creation completed!"
    echo "📊 Results:"
    echo "   📁 Output directory: $OUTPUT_DIR"
    echo ""
    echo "   🎯 TEST SET (target cell types: $TEST_CELL_TYPES):"
    echo "      📄 Input: $OUTPUT_DIR/merged_test_input.pth"
    echo "      📄 Output: $OUTPUT_DIR/merged_test_output.pth"
    echo ""
    echo "   🏋️ TRAIN SET (remaining cell types):"
    echo "      📄 Input: $OUTPUT_DIR/merged_train_input.pth"
    echo "      📄 Output: $OUTPUT_DIR/merged_train_output.pth"
    echo ""
    echo "   📋 Folder mapping: $OUTPUT_DIR/merged_train_folder_mapping.txt"
    echo ""
    echo "💡 Usage:"
    echo "   import torch"
    echo "   # Load test data (target cell types)"
    echo "   test_input = torch.load('$OUTPUT_DIR/merged_test_input.pth')"
    echo "   test_output = torch.load('$OUTPUT_DIR/merged_test_output.pth')"
    echo "   # Load train data (remaining cell types)" 
    echo "   train_input = torch.load('$OUTPUT_DIR/merged_train_input.pth')"
    echo "   train_output = torch.load('$OUTPUT_DIR/merged_train_output.pth')"
    echo ""
    echo "🔍 To modify test cell types, edit TEST_CELL_TYPES variable in this script"
    echo "   Examples: 'INV', 'INV,NAND2', 'NAND2,NOR2,AND2', etc."
else
    echo "❌ Failed to create cell type-based dataset"
    exit 1
fi

echo ""
echo "🎯 Train/Test Split Summary:"
echo "   📁 Processed directories: ${DATA_DIRS[@]}"
echo "   📂 Total folders processed: ${#ALL_FOLDERS[@]}"
echo "   📄 Total .lib files: $TOTAL_LIB_FILES"
echo "   🎯 TEST SET: Contains only specified cell types ($TEST_CELL_TYPES)"
echo "   🏋️ TRAIN SET: Contains all other cell types"
echo "   💡 This allows training on most cell types while testing generalization"
echo "      to specific target cell types across both simple and complex cells."