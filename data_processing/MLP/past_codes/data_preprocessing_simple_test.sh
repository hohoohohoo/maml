#!/bin/bash

# Unified simple TEST dataset processing script
# Creates only one merged test dataset from all simple cell folders
# Processes all cell types (NAND, NOR, INV, AND, OR, XOR, etc.)

OUTPUT_DIR="../../dataset_ex2/unified_simple_test"
DATA_DIR="../../dataset_ex2/test_processed_simple"  # Directory containing simple cell test data

# Parameter mappings for a,b,c (DIFFERENT VALUES FOR TEST)
PARAM_A="0.75,1.0,1.25"                          # A parameters - DIFFERENT from train
PARAM_B="0.09,0.062,0.092,0.066,0.094,0.07"     # B parameters (nmos,pmos pairs) - DIFFERENT from train  
PARAM_C="0.36,0.47,0.38,0.475,0.40,0.48"        # C parameters (nmos,pmos pairs) - DIFFERENT from train

echo "🚀 Creating unified simple cell TEST dataset..."
echo "📁 Base directory: $DATA_DIR"
echo "📊 Output directory: $OUTPUT_DIR"
echo "🔧 Parameter A mapping (TEST): $PARAM_A"
echo "🔧 Parameter B mapping (TEST, nmos,pmos pairs): $PARAM_B"  
echo "🔧 Parameter C mapping (TEST, nmos,pmos pairs): $PARAM_C"

# Check if DATA_DIR exists
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Test simple data directory does not exist: $DATA_DIR"
    exit 1
fi

# Find all subdirectories (ex2_* or others)
FOLDERS=$(find $DATA_DIR -maxdepth 1 -type d ! -path $DATA_DIR | sort)
FOLDER_COUNT=$(echo "$FOLDERS" | wc -l)

if [ $FOLDER_COUNT -eq 0 ]; then
    echo "❌ No subdirectories found in $DATA_DIR"
    exit 1
fi

echo "📂 Found $FOLDER_COUNT simple test folders to process:"

# Validate folders and count total files
VALID_FOLDERS=()
TOTAL_LIB_FILES=0

for FOLDER in $FOLDERS; do
    FOLDER_NAME=$(basename $FOLDER)
    LIB_COUNT=$(find "$FOLDER" -maxdepth 1 -name "*.lib" | wc -l)
    
    echo "   $FOLDER_NAME: $LIB_COUNT .lib files"
    
    if [ $LIB_COUNT -eq 0 ]; then
        echo "     ❌ No .lib files found, skipping..."
        continue
    fi
    
    VALID_FOLDERS+=("$FOLDER")
    TOTAL_LIB_FILES=$((TOTAL_LIB_FILES + LIB_COUNT))
done

if [ ${#VALID_FOLDERS[@]} -eq 0 ]; then
    echo "❌ No valid simple test folders found to process"
    exit 1
fi

echo ""
echo "✅ Will process ${#VALID_FOLDERS[@]} valid simple test folders with $TOTAL_LIB_FILES total .lib files"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the unified processing script for SIMPLE TEST
echo "🔄 Building unified SIMPLE TEST dataset using merge mode..."
python3 build_and_split_dataset_simple_test.py --merge \
    "$OUTPUT_DIR" \
    "$DATA_DIR" \
    "$PARAM_A" \
    "$PARAM_B" \
    "$PARAM_C"

if [ $? -eq 0 ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "✅ Unified SIMPLE TEST dataset creation completed!"
    echo "📊 Results:"
    echo "   📁 Output directory: $OUTPUT_DIR"
    echo "   📄 Unified SIMPLE TEST input: $OUTPUT_DIR/merged_simple_test_input.pth"
    echo "   📄 Unified SIMPLE TEST output: $OUTPUT_DIR/merged_simple_test_output.pth"
    echo "   📋 Simple test folder mapping: $OUTPUT_DIR/merged_simple_test_folder_mapping.txt"
    echo ""
    echo "💡 Usage:"
    echo "   import torch"
    echo "   simple_test_input_data = torch.load('$OUTPUT_DIR/merged_simple_test_input.pth')"
    echo "   simple_test_output_data = torch.load('$OUTPUT_DIR/merged_simple_test_output.pth')"
    echo ""
    echo "📌 Cell types included: NAND, NOR, INV, AND, OR, XOR, FA, HA, MAJ, etc."
else
    echo "❌ Failed to create unified SIMPLE TEST dataset"
    exit 1
fi