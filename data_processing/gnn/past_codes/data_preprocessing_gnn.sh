#!/bin/bash

# GNN 데이터셋 배치 처리 스크립트
# INVBUF, simple, AO, OA 폴더의 모든 파일들을 GNN으로 변환하고 task별로 8:2 분할

OUTPUT_DIR="/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_temp/"
SPICE_BASE_DIR="/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/ASAP7_lib_files/voltage_variation"

echo "🚀 Starting Batch GNN Dataset Processing"
echo "=================================================="
echo "Output directory: $OUTPUT_DIR"
echo "Spice base directory: $SPICE_BASE_DIR"

# Output 디렉토리 생성
mkdir -p $OUTPUT_DIR

# INVBUF, simple, AO, OA 폴더들 찾기
FOLDERS=($(find $SPICE_BASE_DIR -maxdepth 1 -type d \( -name "INVBUF*" -o -name "simple*" -o -name "AO*" -o -name "OA*" \) | sort))

echo "📁 Found ${#FOLDERS[@]} folders:"
for i in "${!FOLDERS[@]}"; do
    echo "   [$i] $(basename ${FOLDERS[$i]})"
done
echo

# 사용자에게 폴더 선택 옵션 제공
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
    read -p "Enter start folder index (e.g., 0): " start_idx
    read -p "Enter end folder index (e.g., 5, exclusive): " end_idx
    if [ $start_idx -ge 0 ] && [ $end_idx -le ${#FOLDERS[@]} ] && [ $start_idx -lt $end_idx ]; then
        SELECTED_FOLDERS=("${FOLDERS[@]:$start_idx:$((end_idx-start_idx))}")
        echo "✓ Processing folders $start_idx to $((end_idx-1)):"
        for folder in "${SELECTED_FOLDERS[@]}"; do
            echo "     - $(basename $folder)"
        done
    else
        echo "❌ Invalid range! Must have 0 <= start < end <= ${#FOLDERS[@]}"
        exit 1
    fi
else
    SELECTED_FOLDERS=("${FOLDERS[@]}")
    echo "✓ Processing all folders"
fi
echo

# Data type 선택
echo "Choose data type:"
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
echo

# Graph mode 선택
echo "Choose graph mode:"
echo "   [1] stage_aware (default) - Current path only (stage-aware extraction)"
echo "   [2] full_graph - All transistors in the cell (baseline)"
read -p "Enter choice (1/2) [default: 1]: " graph_mode_choice

if [ "$graph_mode_choice" == "2" ]; then
    GRAPH_MODE="full_graph"
    echo "✓ Graph mode: full_graph (baseline - all transistors)"
else
    GRAPH_MODE="stage_aware"
    echo "✓ Graph mode: stage_aware (current path only)"
fi
echo

# 각 폴더별로 처리
for folder in "${SELECTED_FOLDERS[@]}"; do
    folder_name=$(basename $folder)
    echo "🔄 Processing folder: $folder_name"
    
    # 폴더에서 .lib 파일들 찾기
    lib_files=($(find $folder -name "*.lib" | sort))
    
    if [ ${#lib_files[@]} -eq 0 ]; then
        echo "   ⚠️  No .lib files found in $folder_name, skipping..."
        continue
    fi
    
    echo "   📚 Found ${#lib_files[@]} .lib files"

    # 첫 번째 .lib 파일에서 prefix 추출
    first_lib=$(basename ${lib_files[0]})
    # 예: AO_RVT_2_25_040.lib -> AO_RVT_2_25_
    # 파일명에서 마지막 밑줄과 숫자 제거하여 prefix 생성
    lib_prefix=$(echo "$first_lib" | sed -E 's/_[0-9]+\.lib$//')
    lib_prefix="${lib_prefix}_"

    echo "   🏷️  Library prefix: $lib_prefix"
    echo "   📄 Sample file: $first_lib"
    echo "   📊 Data type: $DATA_TYPE"
    echo "   🎯 Graph mode: $GRAPH_MODE"

    # GNN 데이터셋 생성 (분할 없이)
    echo "   ⚙️  Generating GNN dataset (no split)..."
    python3 build_gnn_dataset_no_split.py "${OUTPUT_DIR}${folder_name}/" "$lib_prefix" "$folder_name" "$SPICE_BASE_DIR" "$DATA_TYPE" "$GRAPH_MODE"

    if [ $? -eq 0 ]; then
        echo "   ✅ Successfully processed $folder_name"
    else
        echo "   ❌ Failed to process $folder_name"
    fi
    echo
done

echo "🎉 Batch GNN Dataset Processing Complete!"
echo "   Check results in: $OUTPUT_DIR"