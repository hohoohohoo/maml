#!/bin/bash
# MAML Topology Validation - Parallel Execution
# Runs experiments as independent background processes on different GPUs

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Check arguments
if [ $# -lt 1 ]; then
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║     MAML Topology Validation - PARALLEL MODE                   ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Usage: $0 <config.json> [--dry-run] [--gpus 0,1,2,3] [--jobs-per-gpu 2]"
    echo ""
    echo "Options:"
    echo "  --dry-run         Preview experiments without running"
    echo "  --gpus            Comma-separated list of GPU IDs (default: 0,1,2)"
    echo "  --jobs-per-gpu    Number of concurrent jobs per GPU (default: 1)"
    echo ""
    echo "Example:"
    echo "  $0 configs/maml_validation_sweep.json --gpus 0,1,2,3"
    echo "  $0 configs/maml_validation_sweep.json --gpus 0,1 --jobs-per-gpu 4"
    echo "  $0 configs/maml_validation_sweep.json --dry-run"
    exit 1
fi

CONFIG_FILE="$1"
DRY_RUN=false
GPU_LIST="3,4"
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
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Config file not found: $CONFIG_FILE${NC}"
    exit 1
fi

# Convert GPU list to array
IFS=',' read -ra GPUS <<< "$GPU_LIST"
NUM_GPUS=${#GPUS[@]}

# Calculate max parallel jobs
MAX_PARALLEL=$((NUM_GPUS * JOBS_PER_GPU))

echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     MAML Topology Validation - PARALLEL MODE                   ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Config file:${NC} $CONFIG_FILE"
echo -e "${YELLOW}Available GPUs:${NC} ${GPUS[*]} (total: $NUM_GPUS)"
echo -e "${YELLOW}Jobs per GPU:${NC} $JOBS_PER_GPU"
echo -e "${YELLOW}Max parallel jobs:${NC} $MAX_PARALLEL"
echo ""

# Parse experiments using Python utility
EXPERIMENTS=$(python ../../utils/parse_maml_topology_sweep.py "$CONFIG_FILE")

if [ $? -ne 0 ]; then
    echo -e "${RED}Error parsing sweep configuration${NC}"
    exit 1
fi

# Count and collect experiments into array
declare -a JOBS
declare -a JOB_SUMMARIES
JOB_IDX=0

while IFS= read -r line; do
    if [[ $line == EXPERIMENT* ]]; then
        # Extract parameters (format: "EXPERIMENT N: config=X mode=Y data_type=Z: <params>")
        PARAMS=$(echo "$line" | cut -d':' -f3-)
        SUMMARY=$(echo "$line" | cut -d':' -f1-2)

        JOBS[$JOB_IDX]="$PARAMS"
        JOB_SUMMARIES[$JOB_IDX]="$SUMMARY"
        JOB_IDX=$((JOB_IDX + 1))
    fi
done <<< "$EXPERIMENTS"

TOTAL_JOBS=${#JOBS[@]}

if [ $TOTAL_JOBS -eq 0 ]; then
    echo -e "${RED}Error: No experiments found in config file${NC}"
    exit 1
fi

# Create log directory
LOG_DIR="$SCRIPT_DIR/parallel_logs/maml_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo -e "${BLUE}Total experiments:${NC} $TOTAL_JOBS"
echo -e "${BLUE}Log directory:${NC} $LOG_DIR"
echo ""

# Dry run mode - show all commands
if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}DRY RUN MODE - Commands to be executed:${NC}"
    echo ""
    for i in "${!JOBS[@]}"; do
        GPU_IDX=$((i % NUM_GPUS))
        GPU=${GPUS[$GPU_IDX]}

        # Remove any existing gpu_id from params for display
        CLEAN_PARAMS=$(echo "${JOBS[$i]}" | sed 's/--gpu_id [0-9]*//g')

        echo -e "${BLUE}[$((i+1))/$TOTAL_JOBS] GPU $GPU: ${JOB_SUMMARIES[$i]}${NC}"
        echo "  python MAML_topology_validation.py $CLEAN_PARAMS --gpu_id $GPU"
        echo ""
    done
    exit 0
fi

# Run jobs in parallel, limited by MAX_PARALLEL
echo -e "${GREEN}Starting parallel execution (max $MAX_PARALLEL concurrent jobs)...${NC}"
echo ""

PIDS=()
PID_TO_JOB=()
RUNNING=0

for i in "${!JOBS[@]}"; do
    GPU_IDX=$((i % NUM_GPUS))
    GPU=${GPUS[$GPU_IDX]}
    PARAMS="${JOBS[$i]}"

    # Remove any existing gpu_id from params (we'll use distributed GPU)
    PARAMS=$(echo "$PARAMS" | sed 's/--gpu_id [0-9]*//g')

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

    # Create log filename from parameters
    CONFIG_ID=$(echo "$PARAMS" | grep -o '\-\-config [0-9]*' | awk '{print $2}')
    MODE=$(echo "$PARAMS" | grep -o '\-\-mode [a-z]*' | awk '{print $2}')
    DATA_TYPE=$(echo "$PARAMS" | grep -o '\-\-data_type [a-z]*' | awk '{print $2}')

    LOG_FILE="$LOG_DIR/exp${i}_config${CONFIG_ID}_${MODE}_${DATA_TYPE}_gpu${GPU}.log"

    CMD="python -u MAML_topology_validation.py $PARAMS --gpu_id $GPU"

    echo -e "${BLUE}[$((i+1))/$TOTAL_JOBS]${NC} Starting on GPU $GPU: config=$CONFIG_ID mode=$MODE data_type=$DATA_TYPE"
    echo "  Log: $LOG_FILE"

    # Run in background
    eval "$CMD" > "$LOG_FILE" 2>&1 &
    PIDS+=($!)
    RUNNING=$((RUNNING + 1))
done

# Wait for all remaining jobs
echo ""
echo -e "${YELLOW}Waiting for remaining jobs to complete...${NC}"
for pid in "${PIDS[@]}"; do
    wait "$pid"
done

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    Parallel Execution Complete                 ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Logs saved in:${NC} $LOG_DIR"
echo ""

# Summary
echo -e "${YELLOW}Results summary:${NC}"
PASS_COUNT=0
FAIL_COUNT=0

for log in "$LOG_DIR"/*.log; do
    if [ -f "$log" ]; then
        BASENAME=$(basename "$log" .log)
        if grep -q "Error\|Exception\|Traceback" "$log"; then
            echo -e "  ${RED}FAILED:${NC} $BASENAME"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        else
            echo -e "  ${GREEN}OK:${NC} $BASENAME"
            PASS_COUNT=$((PASS_COUNT + 1))
        fi
    fi
done

echo ""
echo -e "${BLUE}Summary:${NC} ${GREEN}$PASS_COUNT passed${NC}, ${RED}$FAIL_COUNT failed${NC}"

if [ $FAIL_COUNT -eq 0 ]; then
    echo -e "\n${GREEN}All experiments completed successfully!${NC}"
    exit 0
else
    echo -e "\n${YELLOW}Some experiments failed. Check logs for details.${NC}"
    exit 1
fi
