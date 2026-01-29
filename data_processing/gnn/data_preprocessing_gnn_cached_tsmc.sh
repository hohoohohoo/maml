#!/bin/bash

# TSMC GNN 데이터셋 배치 처리 스크립트 (Cached Topology Version)
# Supports both stage-aware (pull-up/pull-down) and full-graph topology caching
# TSMC SPI 파일에서 topology cache를 생성하고 GNN 데이터셋을 빌드합니다.

OUTPUT_DIR="/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_TSMC_GNN/"
SPICE_BASE_DIR="/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/TSMC_lib_files"
SPI_DIR="/home/tkdgn2907/Deepsets_test/MAML/Projects/cdl_files"
CACHE_DIR="/home/tkdgn2907/Deepsets_test/MAML/Projects/data_processing/gnn/topology_cache"

echo "Starting TSMC Batch GNN Dataset Processing (CACHED TOPOLOGY)"
echo "=================================================="
echo "Output directory: $OUTPUT_DIR"
echo "TSMC lib directory: $SPICE_BASE_DIR"
echo "Cache directory: $CACHE_DIR"

# Cache 및 output 디렉토리 생성
mkdir -p $OUTPUT_DIR
mkdir -p $CACHE_DIR

# Step 0: Graph mode 선택 (stage-aware vs full-graph)
echo ""
echo "Select Graph Mode:"
echo "   [1] Stage-aware (pull-up/pull-down paths only - smaller cache)"
echo "   [2] Full-graph (all transistors - larger cache)"
read -p "Enter choice (1-2) [default: 1]: " graph_mode_choice

case $graph_mode_choice in
    2)
        GRAPH_MODE="full_graph"
        CACHE_SCRIPT="precompute_full_graph_topology.py"
        CACHE_PREFIX="cell_topology_cache_tsmc"
        BUILDER_SCRIPT="build_gnn_dataset_full_graph_cached_tsmc.py"
        echo "Selected mode: Full-graph (baseline)"
        ;;
    *)
        GRAPH_MODE="stage_aware"
        CACHE_SCRIPT="precompute_stage_aware_topology.py"
        CACHE_PREFIX="stage_aware_topology_cache_tsmc"
        BUILDER_SCRIPT="build_gnn_dataset_stage_aware_cached_tsmc.py"
        echo "Selected mode: Stage-aware (pull-up/pull-down)"
        ;;
esac

# Step 1: SPI 파일 선택
echo ""
echo "Select TSMC SPI file:"
SPI_FILES=($(find $SPI_DIR -name "*.spi" | sort))

if [ ${#SPI_FILES[@]} -eq 0 ]; then
    echo "No SPI files found in $SPI_DIR"
    exit 1
fi

echo "Found ${#SPI_FILES[@]} SPI files:"
for i in "${!SPI_FILES[@]}"; do
    echo "   [$i] $(basename ${SPI_FILES[$i]})"
done

read -p "Select SPI file number [default: 0]: " spi_choice
spi_choice=${spi_choice:-0}

if [ $spi_choice -ge 0 ] && [ $spi_choice -lt ${#SPI_FILES[@]} ]; then
    SELECTED_SPI="${SPI_FILES[$spi_choice]}"
else
    echo "Invalid SPI number!"
    exit 1
fi

SPI_BASENAME=$(basename $SELECTED_SPI .spi)
CACHE_FILE="$CACHE_DIR/${CACHE_PREFIX}_${SPI_BASENAME}.pth"
echo "Selected SPI: $(basename $SELECTED_SPI)"

# Topology cache 생성 또는 재사용
echo ""
if [ -f "$CACHE_FILE" ]; then
    echo "Found existing topology cache: $(basename $CACHE_FILE)"
    read -p "Reuse existing cache? (y/n) [default: y]: " reuse_cache
    reuse_cache=${reuse_cache:-y}

    if [ "$reuse_cache" == "y" ]; then
        echo "Reusing existing cache"
    else
        echo "Regenerating topology cache..."
        python3 $CACHE_SCRIPT \
            --spi_path "$SELECTED_SPI" \
            --output "$CACHE_FILE"

        if [ $? -ne 0 ]; then
            echo "Failed to generate topology cache!"
            exit 1
        fi
        echo "Topology cache generated successfully"
    fi
else
    echo "Generating topology cache for the first time..."
    python3 $CACHE_SCRIPT \
        --spi_path "$SELECTED_SPI" \
        --output "$CACHE_FILE"

    if [ $? -ne 0 ]; then
        echo "Failed to generate topology cache!"
        exit 1
    fi
    echo "Topology cache generated successfully"
fi

# Step 2: Process corner 선택
echo ""
echo "Select Process Corner:"
echo "   [1] FF (Fast-Fast)"
echo "   [2] TT (Typical-Typical)"
echo "   [3] SS (Slow-Slow)"
echo "   [4] SF (Slow-Fast)"
echo "   [5] FS (Fast-Slow)"
echo "   [6] FF2, TT2, SS2, etc. (alternate versions)"
echo "   [7] ALL corners"
read -p "Enter choice (1-7): " corner_choice

case $corner_choice in
    1)
        CORNER_PATTERN="TSMC_FF_"
        ;;
    2)
        CORNER_PATTERN="TSMC_TT_"
        ;;
    3)
        CORNER_PATTERN="TSMC_SS_"
        ;;
    4)
        CORNER_PATTERN="TSMC_SF_"
        ;;
    5)
        CORNER_PATTERN="TSMC_FS_"
        ;;
    6)
        CORNER_PATTERN="TSMC_*2_"
        ;;
    7)
        CORNER_PATTERN="TSMC_"
        ;;
    *)
        echo "Invalid choice!"
        exit 1
        ;;
esac

echo "Selected corner pattern: $CORNER_PATTERN"

# Step 3: 선택한 corner에 맞는 폴더만 찾기
echo ""
echo "Finding folders matching pattern: ${CORNER_PATTERN}*"

FOLDERS=($(find $SPICE_BASE_DIR -maxdepth 1 -type d -name "${CORNER_PATTERN}*" | sort))

if [ ${#FOLDERS[@]} -eq 0 ]; then
    echo "No folders found matching pattern: ${CORNER_PATTERN}*"
    exit 1
fi

echo "Found ${#FOLDERS[@]} folders:"
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
        echo "Selected: $(basename ${SELECTED_FOLDERS[0]})"
    else
        echo "Invalid folder number!"
        exit 1
    fi
elif [ "$choice" == "r" ]; then
    read -p "Enter start folder number: " start_num
    read -p "Enter end folder number (inclusive): " end_num

    if [ $start_num -ge 0 ] && [ $end_num -lt ${#FOLDERS[@]} ] && [ $start_num -le $end_num ]; then
        SELECTED_FOLDERS=("${FOLDERS[@]:$start_num:$((end_num - start_num + 1))}")
        echo "Selected range: [$start_num-$end_num] (${#SELECTED_FOLDERS[@]} folders)"
    else
        echo "Invalid folder range!"
        exit 1
    fi
else
    SELECTED_FOLDERS=("${FOLDERS[@]}")
    echo "Processing all folders"
fi
echo

# Data type 선택
echo "Choose data type:"
echo "   [1] cell (default)"
echo "   [2] transition"
read -p "Enter choice (1/2) [default: 1]: " data_type_choice

if [ "$data_type_choice" == "2" ]; then
    DATA_TYPE="transition"
    echo "Data type: transition"
else
    DATA_TYPE="cell"
    echo "Data type: cell"
fi
echo

# Step 4: 각 폴더별로 처리
TOTAL_PROCESSED=0
TOTAL_FAILED=0

for folder in "${SELECTED_FOLDERS[@]}"; do
    folder_name=$(basename $folder)
    echo "Processing folder: $folder_name"

    # 폴더에서 .lib 파일들 찾기
    lib_files=($(find $folder -name "*.lib" | sort))

    if [ ${#lib_files[@]} -eq 0 ]; then
        echo "   No .lib files found in $folder_name, skipping..."
        continue
    fi

    echo "   Found ${#lib_files[@]} .lib files"

    # 첫 번째 .lib 파일에서 prefix 추출
    first_lib=$(basename ${lib_files[0]})
    # 예: TSMC_FF_25_060.lib -> TSMC_FF_25_
    lib_prefix=$(echo "$first_lib" | sed -E 's/_[0-9]+\.lib$//')
    lib_prefix="${lib_prefix}_"

    echo "   Library prefix: $lib_prefix"
    echo "   Sample file: $first_lib"
    echo "   Using cache: $(basename $CACHE_FILE)"
    echo "   Graph mode: $GRAPH_MODE"
    echo "   Data type: $DATA_TYPE"

    # Voltage range 추출 (파일명에서)
    start_voltage=$(basename ${lib_files[0]} .lib | grep -oE '[0-9]+$')
    end_voltage=$(basename ${lib_files[-1]} .lib | grep -oE '[0-9]+$')

    # 숫자로 변환
    start_num=$((10#$start_voltage))
    end_num=$((10#$end_voltage))
    end_num=$((end_num + 1))  # Exclusive end

    echo "   Voltage range: $start_num to $end_num (exclusive)"

    # Output 파일명
    OUTPUT_SUBDIR="${OUTPUT_DIR}${folder_name}/graph_data/"
    mkdir -p "$OUTPUT_SUBDIR"

    ALL_DATA_FILE="${OUTPUT_SUBDIR}${DATA_TYPE}_all_graph_data_${GRAPH_MODE}.pth"

    # GNN 데이터셋 생성 (TSMC cached version)
    echo "   Generating GNN dataset using TSMC cached topology..."

    python3 $BUILDER_SCRIPT \
        --cache_path "$CACHE_FILE" \
        --start $start_num \
        --end $end_num \
        --prefix "$lib_prefix" \
        --data_dir "$folder_name" \
        --lib_base_path "$SPICE_BASE_DIR" \
        --save_input "$ALL_DATA_FILE" \
        --data_type "$DATA_TYPE"

    if [ $? -eq 0 ]; then
        echo "   Successfully processed $folder_name"
        TOTAL_PROCESSED=$((TOTAL_PROCESSED + 1))

        # 파일 크기 확인
        if [ -f "$ALL_DATA_FILE" ]; then
            FILE_SIZE=$(du -h "$ALL_DATA_FILE" | cut -f1)
            echo "   Output file:"
            echo "      All data: $FILE_SIZE - $(basename $ALL_DATA_FILE)"
        fi
    else
        echo "   Failed to process $folder_name"
        TOTAL_FAILED=$((TOTAL_FAILED + 1))
    fi
    echo
done

echo "TSMC Batch GNN Dataset Processing Complete!"
echo "=================================================="
echo "   Graph mode: $GRAPH_MODE"
echo "   Topology cache: $(basename $CACHE_FILE)"
echo "   Total processed: $TOTAL_PROCESSED"
echo "   Total failed: $TOTAL_FAILED"
echo "   Results in: $OUTPUT_DIR"
echo ""
echo "Cache file location:"
echo "   $CACHE_FILE"
echo ""
echo "Done!"
