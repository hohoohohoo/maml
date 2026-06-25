#!/bin/bash
# GNN Topology Validation - TSMC Dataset - Parallel Execution by Cell
# Runs each cell as independent background process on different GPUs

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <config.json> [--dry-run] [--gpus 0,1,2,3] [--jobs-per-gpu 2]"
    echo ""
    echo "Example:"
    echo "  $0 json_configs/validation_gnn_topology_sweep_config.json --gpus 4,5,6,7"
    echo "  $0 json_configs/validation_gnn_topology_sweep_config.json --gpus 4,5,6,7 --jobs-per-gpu 2"
    echo "  $0 json_configs/validation_gnn_topology_sweep_config.json --dry-run"
    exit 1
fi

CONFIG_FILE="$1"
DRY_RUN=false
GPU_LIST="5,6,7"
JOBS_PER_GPU=2

# Parse optional flags
shift
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --gpus)
            GPU_LIST="$2"
            shift 2
            ;;
        --jobs-per-gpu)
            JOBS_PER_GPU="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Convert GPU list to array
IFS=',' read -ra GPUS <<< "$GPU_LIST"
NUM_GPUS=${#GPUS[@]}

# Calculate max parallel jobs
MAX_PARALLEL=$((NUM_GPUS * JOBS_PER_GPU))

echo "=========================================="
echo "GNN Topology Validation - TSMC PARALLEL MODE"
echo "=========================================="
echo "Config file: $CONFIG_FILE"
echo "Available GPUs: ${GPUS[*]} (total: $NUM_GPUS)"
echo "Jobs per GPU: $JOBS_PER_GPU"
echo "Max parallel jobs: $MAX_PARALLEL"
echo ""

# Parse base config from JSON
MODEL_TYPE=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config']['model_type'])")
DATA_TYPE=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config']['data_type'])")
GRAPH_MODE=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config']['graph_mode'])")
VOLTAGE_MODE=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config'].get('voltage_mode', 'all_nodes'))")
NORMALIZATION=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config'].get('normalization', 'zscore'))")
TEMP_MODE=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config'].get('temp_mode', 'typical'))")
CACHE_PATH=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config'].get('cache_path', ''))")
INPUTPORT=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE'))['base_config']; print('--inputport' if c.get('inputport', False) else '')")
RELATED_PIN_ONLY=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE'))['base_config']; print('--related_pin_only' if c.get('related_pin_only', False) else '')")
MODE=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config']['mode'])")
TOTAL_POINTS=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config']['total_points'])")
NUM_TEST_SAMPLES=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config']['num_test_samples'])")
NUM_ITERATIONS=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config']['num_iterations'])")
SAVE_RESULTS=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE'))['base_config']; print('--save_results' if c.get('save_results', False) else '')")
FILTER_CONTINUOUS=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE'))['base_config']; print('--filter_continuous' if c.get('filter_continuous', False) else '')")
CONTINUITY_THRESHOLD=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config'].get('continuity_threshold', 0.18))")
POOLING=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config'].get('pooling', 'mean'))")
ADAPTATION_METHOD=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config'].get('adaptation_method', 'selective_adam'))")
DATASET_DIR=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config'].get('dataset_dir', ''))")
OUTPUT_PREFIX=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config'].get('output_prefix', 'TSMC_GCN'))")
OUTPUT_DIR=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config'].get('output_dir', 'final'))")

# MAML-specific params
INNERDIV=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE')).get('maml_config', {}).get('innerdiv', 10))")
TASKS_PER_META_BATCH=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE')).get('maml_config', {}).get('tasks_per_meta_batch', 16))")
INNER_STEPS=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE')).get('maml_config', {}).get('inner_steps', 1))")

# Get sweep parameters (use first architecture config)
CONV_DIM=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['sweep_params']['conv_hidden_dim'][0])")
CONV_LAYERS=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['sweep_params']['num_conv_layers'][0])")
FC_DIM=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['sweep_params']['fc_hidden_dim'][0])")
FC_LAYERS=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['sweep_params']['num_fc_layers'][0])")
EXPERIMENTS=$(python3 -c "import json; print(' '.join(json.load(open('$CONFIG_FILE'))['sweep_params']['experiment']))")

# Define TSMC cell lists (matching TSMC_GCN_topology_validation.py)
INTRA_TOPOLOGY_CELLS="AN4D0BWP30P140 ND3D0BWP30P140 NR3D1BWP30P140 OR4D0BWP30P140 XNR3D1BWP30P140 XOR3D1BWP30P140"
TOPOLOGY_AGNOSTIC_CELLS="OA21D0BWP30P140 OA21D1BWP30P140 OA211D0BWP30P140 OA211D1BWP30P140 IOA21D0BWP30P140 IOA21D1BWP30P140 HA1D0BWP30P140 FA1D0BWP30P140 IAO21D0BWP30P140 IAO21D1BWP30P140 AO21D0BWP30P140 AO21D1BWP30P140 AO211D0BWP30P140 AO211D1BWP30P140 SDFSNQD0BWP30P140 DFCNQD1BWP30P140"

# Check if cells are defined in config (override defaults)
INTRA_CELLS_CONFIG=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); cells=c.get('cell_lists', {}).get('intra_topology', []); print(' '.join(cells) if cells else '')" 2>/dev/null)
AGNOSTIC_CELLS_CONFIG=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); cells=c.get('cell_lists', {}).get('topology_agnostic', []); print(' '.join(cells) if cells else '')" 2>/dev/null)

if [ -n "$INTRA_CELLS_CONFIG" ]; then
    INTRA_TOPOLOGY_CELLS="$INTRA_CELLS_CONFIG"
fi
if [ -n "$AGNOSTIC_CELLS_CONFIG" ]; then
    TOPOLOGY_AGNOSTIC_CELLS="$AGNOSTIC_CELLS_CONFIG"
fi

# Create log directory
LOG_DIR="$SCRIPT_DIR/parallel_logs/tsmc_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "Model type: $MODEL_TYPE"
echo "Data type: $DATA_TYPE"
echo "Graph mode: $GRAPH_MODE"
echo "Voltage mode: $VOLTAGE_MODE"
echo "Pooling: $POOLING"
echo "Adaptation method: $ADAPTATION_METHOD"
echo "Output prefix: $OUTPUT_PREFIX"
echo "Output dir: $OUTPUT_DIR"
echo ""
echo "Architecture: conv${CONV_DIM}x${CONV_LAYERS}_fc${FC_DIM}x${FC_LAYERS}"
echo "Experiments: $EXPERIMENTS"
echo ""
echo "Cell lists:"
echo "  Intra topology: $INTRA_TOPOLOGY_CELLS"
echo "  Topology agnostic: $TOPOLOGY_AGNOSTIC_CELLS"
echo ""
echo "Log directory: $LOG_DIR"
echo ""

# Function to build base command
build_base_cmd() {
    local EXP=$1
    local CELL=$2
    local GPU=$3

    CMD="python -u TSMC_GCN_topology_validation.py"
    CMD="$CMD --experiment $EXP"
    CMD="$CMD --cells $CELL"
    CMD="$CMD --model_type $MODEL_TYPE"
    CMD="$CMD --data_type $DATA_TYPE"
    CMD="$CMD --graph_mode $GRAPH_MODE"
    CMD="$CMD --voltage_mode $VOLTAGE_MODE"
    CMD="$CMD --normalization $NORMALIZATION"
    CMD="$CMD --temp_mode $TEMP_MODE"
    CMD="$CMD --mode $MODE"
    CMD="$CMD --total_points $TOTAL_POINTS"
    CMD="$CMD --num_test_samples $NUM_TEST_SAMPLES"
    CMD="$CMD --num_iterations $NUM_ITERATIONS"
    CMD="$CMD --conv_hidden_dim $CONV_DIM"
    CMD="$CMD --num_conv_layers $CONV_LAYERS"
    CMD="$CMD --fc_hidden_dim $FC_DIM"
    CMD="$CMD --num_fc_layers $FC_LAYERS"
    CMD="$CMD --pooling $POOLING"
    CMD="$CMD --adaptation_method $ADAPTATION_METHOD"
    CMD="$CMD --output_prefix $OUTPUT_PREFIX"
    CMD="$CMD --output_dir $OUTPUT_DIR"
    CMD="$CMD --gpu $GPU"

    if [ -n "$CACHE_PATH" ]; then
        CMD="$CMD --cache_path $CACHE_PATH"
    fi
    if [ -n "$INPUTPORT" ]; then
        CMD="$CMD $INPUTPORT"
    fi
    if [ -n "$RELATED_PIN_ONLY" ]; then
        CMD="$CMD $RELATED_PIN_ONLY"
    fi
    if [ -n "$DATASET_DIR" ]; then
        CMD="$CMD --dataset_dir $DATASET_DIR"
    fi

    if [ "$MODEL_TYPE" = "maml" ]; then
        CMD="$CMD --innerdiv $INNERDIV"
        CMD="$CMD --tasks_per_meta_batch $TASKS_PER_META_BATCH"
        CMD="$CMD --inner_steps $INNER_STEPS"
    fi
    if [ -n "$SAVE_RESULTS" ]; then
        CMD="$CMD $SAVE_RESULTS"
    fi
    if [ -n "$FILTER_CONTINUOUS" ]; then
        CMD="$CMD $FILTER_CONTINUOUS"
        CMD="$CMD --continuity_threshold $CONTINUITY_THRESHOLD"
    fi

    echo "$CMD"
}

# Collect all jobs
declare -a JOBS
JOB_IDX=0

for EXP in $EXPERIMENTS; do
    if [ "$EXP" = "intra_topology" ]; then
        CELLS="$INTRA_TOPOLOGY_CELLS"
    else
        CELLS="$TOPOLOGY_AGNOSTIC_CELLS"
    fi

    for CELL in $CELLS; do
        JOBS[$JOB_IDX]="$EXP|$CELL"
        JOB_IDX=$((JOB_IDX + 1))
    done
done

TOTAL_JOBS=${#JOBS[@]}
echo "Total jobs: $TOTAL_JOBS"
echo "Jobs will be distributed across $NUM_GPUS GPUs ($JOBS_PER_GPU jobs per GPU)"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN MODE - Commands to be executed:"
    echo ""
    for i in "${!JOBS[@]}"; do
        IFS='|' read -r EXP CELL <<< "${JOBS[$i]}"
        GPU_IDX=$((i % NUM_GPUS))
        GPU=${GPUS[$GPU_IDX]}
        echo "[$((i+1))/$TOTAL_JOBS] GPU $GPU: $EXP - $CELL"
        CMD=$(build_base_cmd "$EXP" "$CELL" "$GPU")
        echo "  $CMD"
        echo ""
    done
    exit 0
fi

# Run jobs in parallel, limited by MAX_PARALLEL (NUM_GPUS * JOBS_PER_GPU)
echo "Starting parallel execution (max $MAX_PARALLEL concurrent jobs)..."
echo ""

PIDS=()
RUNNING=0

for i in "${!JOBS[@]}"; do
    IFS='|' read -r EXP CELL <<< "${JOBS[$i]}"
    GPU_IDX=$((i % NUM_GPUS))
    GPU=${GPUS[$GPU_IDX]}

    # Wait if max parallel jobs reached
    while [ $RUNNING -ge $MAX_PARALLEL ]; do
        for pid_idx in "${!PIDS[@]}"; do
            if ! kill -0 "${PIDS[$pid_idx]}" 2>/dev/null; then
                wait "${PIDS[$pid_idx]}"
                unset 'PIDS[pid_idx]'
                RUNNING=$((RUNNING - 1))
            fi
        done
        sleep 1
    done

    LOG_FILE="$LOG_DIR/${EXP}_${CELL}_gpu${GPU}.log"
    CMD=$(build_base_cmd "$EXP" "$CELL" "$GPU")

    echo "[$((i+1))/$TOTAL_JOBS] Starting: $EXP - $CELL on GPU $GPU"
    echo "  Log: $LOG_FILE"

    # Run in background
    eval "$CMD" > "$LOG_FILE" 2>&1 &
    PIDS+=($!)
    RUNNING=$((RUNNING + 1))
done

# Wait for all remaining jobs
echo ""
echo "Waiting for remaining jobs to complete..."
for pid in "${PIDS[@]}"; do
    wait "$pid"
done

echo ""
echo "=========================================="
echo "All jobs completed!"
echo "=========================================="
echo "Logs saved in: $LOG_DIR"
echo ""

# Summary
echo "Results summary:"
PASS_COUNT=0
FAIL_COUNT=0
for log in "$LOG_DIR"/*.log; do
    if [ -f "$log" ]; then
        BASENAME=$(basename "$log" .log)
        if grep -q "Error\|Exception\|Traceback" "$log"; then
            echo "  FAILED: $BASENAME"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        else
            echo "  OK: $BASENAME"
            PASS_COUNT=$((PASS_COUNT + 1))
        fi
    fi
done

echo ""
echo "Summary: $PASS_COUNT passed, $FAIL_COUNT failed"

# Clean up PyTorch Geometric JIT cache files
echo ""
echo "Cleaning up JIT cache files..."
rm -f "$SCRIPT_DIR"/torch_geometric.*.py 2>/dev/null
echo "Done."

exit 0
