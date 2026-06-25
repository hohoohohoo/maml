#!/bin/bash
# Run MAML topology validation sweep experiments

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default values
DRY_RUN=false
NO_COMMIT=false

# Parse command line arguments
CONFIG_FILE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --no-commit)
            NO_COMMIT=true
            shift
            ;;
        -*)
            echo -e "${RED}Error: Unknown option: $1${NC}"
            exit 1
            ;;
        *)
            if [ -z "$CONFIG_FILE" ]; then
                CONFIG_FILE="$1"
            else
                echo -e "${RED}Error: Multiple config files specified${NC}"
                exit 1
            fi
            shift
            ;;
    esac
done

# Check if config file is provided
if [ -z "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: No configuration file specified${NC}"
    echo ""
    echo "Usage: $0 <sweep_config.json> [--dry-run] [--no-commit]"
    echo ""
    echo "Options:"
    echo "  --dry-run     Preview experiments without running"
    echo "  --no-commit   Skip git commit of config file"
    echo ""
    echo "Supported adaptation methods (set in base_config):"
    echo "  - selective_adam: Grad/Move scaling + conditional Adam (default)"
    echo "  - adam: Direct Adam optimization (no grad/move)"
    echo ""
    echo "Example:"
    echo "  $0 configs/maml_topology_sweep.json"
    exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Configuration file not found: $CONFIG_FILE${NC}"
    exit 1
fi

echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║       MAML Topology Validation - Parameter Sweep              ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Sweep configuration:${NC} $CONFIG_FILE"
echo ""

# Parse JSON and generate experiments using Python
EXPERIMENTS=$(python ../../utils/parse_maml_topology_sweep.py "$CONFIG_FILE")

if [ $? -ne 0 ]; then
    echo -e "${RED}Error parsing sweep configuration${NC}"
    exit 1
fi

# Count number of experiments
NUM_EXPERIMENTS=$(echo "$EXPERIMENTS" | grep -c "^EXPERIMENT")

echo -e "${BLUE}Total experiments to run: $NUM_EXPERIMENTS${NC}"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}DRY RUN MODE - No experiments will be executed${NC}"
    echo ""
fi

# Git commit for experiment tracking
if [ "$NO_COMMIT" = false ] && [ "$DRY_RUN" = false ]; then
    if git rev-parse --git-dir > /dev/null 2>&1; then
        echo "📝 Creating git commit for experiment tracking..."
        git add "$CONFIG_FILE" 2>/dev/null || true

        if git diff --cached --quiet; then
            echo "ℹ️  No changes to commit (config file unchanged)"
        else
            TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
            COMMIT_MSG="Start MAML topology validation sweep: $(basename $CONFIG_FILE)

Experiment Configuration:
- Config file: $CONFIG_FILE
- Timestamp: $TIMESTAMP
- Number of experiments: $NUM_EXPERIMENTS
- Script: run_maml_topology_validation_sweep.sh

🤖 Generated with Claude Code
"
            git commit -m "$COMMIT_MSG" --no-verify
            if [ $? -eq 0 ]; then
                echo "✅ Experiment config committed to git"
                COMMIT_HASH=$(git rev-parse --short HEAD)
                echo "   Commit: $COMMIT_HASH"
            fi
        fi
        echo ""
    fi
fi

# Track success/failure
TOTAL=0
SUCCESS=0
FAILED=0

# Process each experiment
while IFS= read -r line; do
    if [[ $line == EXPERIMENT* ]]; then
        TOTAL=$((TOTAL + 1))

        # Extract experiment number
        EXP_NUM=$(echo "$line" | cut -d':' -f1 | sed 's/EXPERIMENT //')

        # Parse parameters
        CONFIG_ID=$(echo "$line" | grep -o 'config=[0-9]*' | cut -d'=' -f2)
        MODE=$(echo "$line" | grep -o 'mode=[a-z]*' | cut -d'=' -f2)
        DATA_TYPE=$(echo "$line" | grep -o 'data_type=[a-z]*' | cut -d'=' -f2)

        echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${BLUE}Experiment $TOTAL/$NUM_EXPERIMENTS${NC}"
        echo -e "${YELLOW}  Config: $CONFIG_ID | Mode: $MODE | Data Type: $DATA_TYPE${NC}"
        echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

        if [ "$DRY_RUN" = true ]; then
            echo -e "${YELLOW}  [DRY RUN] Would execute: python MAML_topology_validation.py with above parameters${NC}"
            echo ""
            continue
        fi

        # Get the full parameter string (skip "EXPERIMENT N:" and "summary:" parts)
        PARAMS=$(echo "$line" | cut -d':' -f3-)

        # Build command
        CMD="python MAML_topology_validation.py $PARAMS"

        # Execute
        echo "Executing: $CMD"
        echo ""

        if eval $CMD; then
            SUCCESS=$((SUCCESS + 1))
            echo -e "${GREEN}✅ Experiment $TOTAL completed successfully${NC}"
        else
            FAILED=$((FAILED + 1))
            echo -e "${RED}❌ Experiment $TOTAL failed${NC}"
        fi
        echo ""
    fi
done <<< "$EXPERIMENTS"

# Print summary
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    Sweep Summary                               ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "Total experiments: ${BLUE}$TOTAL${NC}"

if [ "$DRY_RUN" = false ]; then
    echo -e "Successful: ${GREEN}$SUCCESS${NC}"
    echo -e "Failed: ${RED}$FAILED${NC}"

    if [ $FAILED -eq 0 ]; then
        echo -e "\n${GREEN}✅ All experiments completed successfully!${NC}"
        exit 0
    else
        echo -e "\n${YELLOW}⚠️  Some experiments failed. Check logs above.${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}DRY RUN completed - no experiments were executed${NC}"
fi
