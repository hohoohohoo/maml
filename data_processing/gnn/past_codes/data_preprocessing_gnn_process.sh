#!/bin/bash

# GNN 데이터셋 배치 처리 스크립트 (Process Condition Parameters)
# processed, processed_simple 폴더의 모든 파일들을 GNN으로 변환 (11D node features)
# 기존 7D features + 4D process parameters (param_a, param_b, param_c, temperature)

OUTPUT_DIR="/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_temp_process/"
PROCESSED_BASE_DIR="/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/ASAP7_lib_files/processed"
PROCESSED_SIMPLE_BASE_DIR="/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/ASAP7_lib_files/processed_simple"

echo "🚀 Starting Batch GNN Dataset Processing with Process Conditions"
echo "=================================================================="
echo "Output directory: $OUTPUT_DIR"
echo "Processed base directory: $PROCESSED_BASE_DIR"
echo "Processed simple base directory: $PROCESSED_SIMPLE_BASE_DIR"
echo "Node features: 11D (7 base + 4 process parameters)"

# Output 디렉토리 생성
mkdir -p $OUTPUT_DIR

# 처리할 base directory 선택
echo ""
echo "Choose base directory:"
echo "   [1] processed (default)"
echo "   [2] processed_simple"
echo "   [3] Both"
read -p "Enter choice (1/2/3) [default: 1]: " base_dir_choice

SELECTED_BASE_DIRS=()
if [ "$base_dir_choice" == "2" ]; then
    SELECTED_BASE_DIRS=("$PROCESSED_SIMPLE_BASE_DIR")
    echo "✓ Selected: processed_simple"
elif [ "$base_dir_choice" == "3" ]; then
    SELECTED_BASE_DIRS=("$PROCESSED_BASE_DIR" "$PROCESSED_SIMPLE_BASE_DIR")
    echo "✓ Selected: Both (processed + processed_simple)"
else
    SELECTED_BASE_DIRS=("$PROCESSED_BASE_DIR")
    echo "✓ Selected: processed"
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

# Test dataset 여부 선택
echo "Is this a test dataset?"
echo "   [1] No (default) - Training dataset"
echo "   [2] Yes - Test dataset"
read -p "Enter choice (1/2) [default: 1]: " test_choice

if [ "$test_choice" == "2" ]; then
    IS_TEST="true"
    echo "✓ Test dataset: Yes"
else
    IS_TEST="false"
    echo "✓ Test dataset: No"
fi
echo

# 각 base directory에 대해 처리
for base_dir in "${SELECTED_BASE_DIRS[@]}"; do
    base_name=$(basename $base_dir)
    echo "🔄 Processing base directory: $base_name"

    # 하위 디렉토리 찾기 (simple, INVBUF, AO, OA 등)
    FOLDERS=($(find $base_dir -maxdepth 1 -type d -not -path $base_dir | sort))

    if [ ${#FOLDERS[@]} -eq 0 ]; then
        echo "   ⚠️  No subdirectories found in $base_name, skipping..."
        continue
    fi

    echo "📁 Found ${#FOLDERS[@]} folders in $base_name:"
    for i in "${!FOLDERS[@]}"; do
        echo "   [$i] $(basename ${FOLDERS[$i]})"
    done
    echo

    # 사용자에게 폴더 선택 옵션 제공
    echo "Choose processing mode for $base_name:"
    echo "   [a] Process ALL folders"
    echo "   [s] Process a SINGLE folder"
    echo "   [r] Process a RANGE of folders"
    read -p "Enter choice (a/s/r): " choice

    if [ "$choice" == "s" ]; then
        read -p "Enter folder number to process: " folder_num
        if [ $folder_num -ge 0 ] && [ $folder_num -lt ${#FOLDERS[@]} ]; then
            SELECTED_FOLDERS=(${FOLDERS[$folder_num]})
            echo "✓ Selected: $(basename ${SELECTED_FOLDERS[0]})"
        else
            echo "❌ Invalid folder number!"
            continue
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
            continue
        fi
    else
        SELECTED_FOLDERS=("${FOLDERS[@]}")
        echo "✓ Processing all folders"
    fi
    echo

    # 각 폴더별로 처리
    for folder in "${SELECTED_FOLDERS[@]}"; do
        folder_name=$(basename $folder)
        echo "🔄 Processing folder: $folder_name (from $base_name)"

        # 폴더에서 .lib 파일들 찾기 (최상위 디렉토리만, subdirectory 제외)
        lib_files=($(find $folder -maxdepth 1 -name "*.lib" -type f | sort))

        if [ ${#lib_files[@]} -eq 0 ]; then
            echo "   ⚠️  No .lib files found in $folder_name, skipping..."
            continue
        fi

        echo "   📚 Found ${#lib_files[@]} .lib files"

        # 첫 번째 .lib 파일에서 prefix 추출
        first_lib=$(basename ${lib_files[0]})
        # 예1: invbuf_0_0_0_040.lib -> invbuf_0_0_0_
        # 예2: simple_0_0_0_12.5_040.lib -> simple_0_0_0_12.5_
        # 파일명에서 마지막 밑줄과 voltage index (3자리 숫자) 제거하여 prefix 생성
        lib_prefix=$(echo "$first_lib" | sed -E 's/_0[0-9]{2}\.lib$//')
        lib_prefix="${lib_prefix}_"

        echo "   🏷️  Library prefix: $lib_prefix"
        echo "   📄 Sample file: $first_lib"
        echo "   📊 Data type: $DATA_TYPE"
        echo "   🎯 Graph mode: $GRAPH_MODE"
        echo "   🔬 Process parameters: Enabled (11D node features)"
        echo "   🧪 Test dataset: $IS_TEST"

        # GNN 데이터셋 생성 (process conditions 포함)
        # Output directory name: base_name과 folder_name에서 중복되는 prefix 제거
        # 예: processed_simple + simple_0_0_0_12p5 -> processed_0_0_0_12p5 (simple 중복 제거)
        clean_base_name=$(echo "$base_name" | sed -E 's/_simple$//')  # processed_simple -> processed
        output_folder_name="${clean_base_name}_${folder_name}"

        echo "   ⚙️  Generating GNN dataset with process conditions (no split)..."
        python3 build_gnn_dataset_with_process.py "${OUTPUT_DIR}${output_folder_name}/" "$lib_prefix" "$folder_name" "$base_dir" "$DATA_TYPE" "$GRAPH_MODE" "$IS_TEST"

        if [ $? -eq 0 ]; then
            echo "   ✅ Successfully processed $folder_name (11D features)"
        else
            echo "   ❌ Failed to process $folder_name"
        fi
        echo
    done
done

echo "🎉 Batch GNN Dataset Processing with Process Conditions Complete!"
echo "   Check results in: $OUTPUT_DIR"
echo "   Node features: 11D (7 base + 4 process parameters)"
