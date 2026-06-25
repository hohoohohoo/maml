#!/bin/bash

# Unified invbuf dataset processing script
# Creates only one merged dataset from all invbuf folders
# File naming pattern: invbuf_a_b_c_d_e.lib where:
# - a,b,c: 0-4 values mapped to parameter lists
# - d,e: temperature, voltage (read from lib file)

OUTPUT_DIR="../../dataset_ex2/unified_invbuf"
DATA_DIR="../../dataset_ex2/processed"  # Directory containing invbuf_* subdirectories

# Parameter mappings for a,b,c (you can customize these values)
PARAM_A="0.625,0.875,1.125,1.375"                        # A parameters (4 values for indices 0-3)
PARAM_B="0.089,0.06,0.091,0.064,0.093,0.068,0.095,0.072"    # B parameters (nmos,pmos pairs: 0n,0p,1n,1p,2n,2p,3n,3p)
PARAM_C="0.35,0.465,0.37,0.473,0.39,0.478,0.41,0.485"              # C parameters (nmos,pmos pairs: 0n,0p,1n,1p,2n,2p,3n,3p)

echo "🚀 Creating unified invbuf dataset..."
echo "📁 Base directory: $DATA_DIR"
echo "📊 Output directory: $OUTPUT_DIR"
echo "🔧 Parameter A mapping: $PARAM_A"
echo "🔧 Parameter B mapping (nmos,pmos pairs): $PARAM_B"  
echo "🔧 Parameter C mapping (nmos,pmos pairs): $PARAM_C"

# Check if DATA_DIR exists
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Data directory does not exist: $DATA_DIR"
    exit 1
fi

# Find all invbuf_* subdirectories
FOLDERS=$(find $DATA_DIR -maxdepth 1 -type d -name "invbuf_*" | sort)
FOLDER_COUNT=$(echo "$FOLDERS" | wc -l)

if [ $FOLDER_COUNT -eq 0 ]; then
    echo "❌ No invbuf_* folders found in $DATA_DIR"
    exit 1
fi

echo "📂 Found $FOLDER_COUNT invbuf folders to process:"

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
    echo "❌ No valid folders found to process"
    exit 1
fi

echo ""
echo "✅ Will process ${#VALID_FOLDERS[@]} valid folders with $TOTAL_LIB_FILES total .lib files"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the unified processing script
echo "🔄 Building unified dataset using merge mode..."
python3 build_and_split_dataset_invbuf.py --merge \
    "$OUTPUT_DIR" \
    "$DATA_DIR" \
    "$PARAM_A" \
    "$PARAM_B" \
    "$PARAM_C"

if [ $? -eq 0 ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "✅ Unified dataset creation completed!"
    echo "📊 Results:"
    echo "   📁 Output directory: $OUTPUT_DIR"
    echo "   📄 Unified input: $OUTPUT_DIR/merged_invbuf_input.pth"
    echo "   📄 Unified output: $OUTPUT_DIR/merged_invbuf_output.pth"
    echo "   📋 Folder mapping: $OUTPUT_DIR/folder_mapping.txt"
    echo ""
    echo "💡 Usage:"
    echo "   import torch"
    echo "   input_data = torch.load('$OUTPUT_DIR/merged_invbuf_input.pth')"
    echo "   output_data = torch.load('$OUTPUT_DIR/merged_invbuf_output.pth')"
else
    echo "❌ Failed to create unified dataset"
    exit 1
fi