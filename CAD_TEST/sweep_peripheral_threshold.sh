#!/bin/bash
# Sweep peripheral_loss_threshold for LUT table validation

# Parameter grid
THRESHOLDS=(1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2 5e-2 0.1 0.2 0.5)

# Configuration
SCRIPT="validate_lut_table_sweep_full_cells.py"
NUM_TABLES=50
GPU=0

# Output file
OUTPUT_FILE="sweep_peripheral_threshold_results_$(date +%Y%m%d_%H%M%S).txt"

echo "Peripheral Threshold Sweep Results" > $OUTPUT_FILE
echo "===================================" >> $OUTPUT_FILE
echo "Script: $SCRIPT" >> $OUTPUT_FILE
echo "Tables: $NUM_TABLES" >> $OUTPUT_FILE
echo "GPU: $GPU" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

for threshold in "${THRESHOLDS[@]}"; do
    echo "Running: threshold=$threshold"
    echo "----------------------------------------" >> $OUTPUT_FILE
    echo "peripheral_loss_threshold=$threshold" >> $OUTPUT_FILE
    echo "----------------------------------------" >> $OUTPUT_FILE

    python $SCRIPT \
        --unit_convert \
        --num_tables $NUM_TABLES \
        --gpu $GPU \
        --peripheral_loss_threshold $threshold \
        2>&1 | grep -E "(OVERALL|PERIPHERAL SUPPORT LOSS|Percentiles|PERIPHERAL ADAM USAGE)" >> $OUTPUT_FILE

    echo "" >> $OUTPUT_FILE
done

echo "Results saved to: $OUTPUT_FILE"
