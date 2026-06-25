#!/bin/bash

# Unified invbuf TEST dataset processing script
# Creates only one merged test dataset from all invbuf folders
# File naming pattern: invbuf_a_b_c_d_e.lib where:
# - a,b,c: 0-4 values mapped to parameter lists
# - d,e: temperature, voltage (read from lib file)

OUTPUT_DIR="../../dataset_ex2/unified_invbuf_test"
DATA_DIR="../../dataset_ex2/test_processed"  # Directory containing invbuf_* subdirectories for TEST

# Parameter mappings for a,b,c (DIFFERENT VALUES FOR TEST)
PARAM_A="0,0.75,1.0,1.25"                          # A parameters (4 values for indices 0-3) - DIFFERENT from train
PARAM_B="0,0,0.09,0.062,0.092,0.066,0.094,0.07"    # B parameters (nmos,pmos pairs) - DIFFERENT from train  
PARAM_C="0.36,0.47,0.38,0.475,0.40,0.48"       # C parameters (nmos,pmos pairs) - DIFFERENT from train

echo "🚀 Creating unified invbuf TEST dataset..."
echo "📁 Base directory: $DATA_DIR"
echo "📊 Output directory: $OUTPUT_DIR"
echo "🔧 Parameter A mapping (TEST): $PARAM_A"
echo "🔧 Parameter B mapping (TEST, nmos,pmos pairs): $PARAM_B"  
echo "🔧 Parameter C mapping (TEST, nmos,pmos pairs): $PARAM_C"

# Check if DATA_DIR exists
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Test data directory does not exist: $DATA_DIR"
    exit 1
fi

# Find all invbuf_* subdirectories
FOLDERS=$(find $DATA_DIR -maxdepth 1 -type d -name "invbuf_*" | sort)
FOLDER_COUNT=$(echo "$FOLDERS" | wc -l)

if [ $FOLDER_COUNT -eq 0 ]; then
    echo "❌ No invbuf_* folders found in $DATA_DIR"
    exit 1
fi

echo "📂 Found $FOLDER_COUNT invbuf test folders to process:"

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
    elif [ $LIB_COUNT -lt 61 ]; then
        echo "     ⚠️ Insufficient .lib files (expected 61), but will include..."
    fi
    
    VALID_FOLDERS+=("$FOLDER")
    TOTAL_LIB_FILES=$((TOTAL_LIB_FILES + LIB_COUNT))
done

if [ ${#VALID_FOLDERS[@]} -eq 0 ]; then
    echo "❌ No valid test folders found to process"
    exit 1
fi

echo ""
echo "✅ Will process ${#VALID_FOLDERS[@]} valid test folders with $TOTAL_LIB_FILES total .lib files"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the unified processing script for TEST
echo "🔄 Building unified TEST dataset using merge mode..."
python3 build_and_split_dataset_invbuf_test.py --merge \
    "$OUTPUT_DIR" \
    "$DATA_DIR" \
    "$PARAM_A" \
    "$PARAM_B" \
    "$PARAM_C"

if [ $? -eq 0 ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "✅ Unified TEST dataset creation completed!"
    echo "📊 Results:"
    echo "   📁 Output directory: $OUTPUT_DIR"
    echo "   📄 Unified TEST input: $OUTPUT_DIR/merged_invbuf_test_input.pth"
    echo "   📄 Unified TEST output: $OUTPUT_DIR/merged_invbuf_test_output.pth"
    echo "   📋 Test folder mapping: $OUTPUT_DIR/test_folder_mapping.txt"
    echo ""
    echo "💡 Usage:"
    echo "   import torch"
    echo "   test_input_data = torch.load('$OUTPUT_DIR/merged_invbuf_test_input.pth')"
    echo "   test_output_data = torch.load('$OUTPUT_DIR/merged_invbuf_test_output.pth')"
else
    echo "❌ Failed to create unified TEST dataset"
    exit 1
fi