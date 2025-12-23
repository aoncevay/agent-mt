#!/bin/bash
# Run workflows in parallel for the full dataset
# Limits concurrent jobs to avoid overwhelming the system and API rate limits
# Note: set -e is not used here because we handle errors per-job in parallel execution

MODEL="qwen3-32b"
# MODEL options: qwen3-32b, qwen3-235b, gpt-oss-20b, gpt-oss-120b, gpt-4-1-mini, gpt-4-1, o4-mini
# For ARNs: claude-sonnet-4, claude-sonnet-4-5
DATASET="wmt25"
# DATASET options: wmt25, dolfin
MAX_PARALLEL=3  # Number of workflows to run in parallel (adjust based on your system/API limits)

# Define all workflows
WORKFLOWS=(
    "zero_shot"
    "MaMT_translate_postedit_proofread"
    "IRB_refine"
    "MAATS_multi_agents"
    "SbS_chat_step_by_step"
    "DeLTA_multi_agents"
    "ADT_multi_agents"
)

# Function to run a single workflow
run_workflow() {
    local workflow=$1
    local use_term=$2
    local term_flag=""
    
    if [ "$use_term" == "true" ]; then
        term_flag="--use_terminology"
    fi
    
    echo "[$(date +'%H:%M:%S')] Starting: $workflow $term_flag"
    python src/run.py --dataset $DATASET --workflow $workflow --model $MODEL $term_flag
    
    if [ $? -eq 0 ]; then
        echo "[$(date +'%H:%M:%S')] ✓ Completed: $workflow $term_flag"
    else
        echo "[$(date +'%H:%M:%S')] ✗ Failed: $workflow $term_flag"
        return 1
    fi
}

# Export function for parallel execution
export -f run_workflow
export MODEL DATASET

echo "=========================================="
echo "Running workflows in parallel (max $MAX_PARALLEL at a time)"
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Workflows: ${#WORKFLOWS[@]} (without terminology) + ${#WORKFLOWS[@]} (with terminology if wmt25)"
echo "=========================================="
echo ""

# Run workflows without terminology in parallel batches
echo "=========================================="
echo "Phase 1: Running workflows WITHOUT terminology"
echo "=========================================="
echo ""

for workflow in "${WORKFLOWS[@]}"; do
    # Wait if we've reached the parallel limit
    while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do
        sleep 1
    done
    
    # Run workflow in background
    run_workflow "$workflow" "false" &
done

# Wait for all background jobs to complete and check for failures
FAILED=0
for job in $(jobs -p); do
    wait $job || FAILED=$((FAILED + 1))
done

if [ $FAILED -gt 0 ]; then
    echo "⚠ Warning: $FAILED workflow(s) failed in Phase 1"
else
    echo "✓ All workflows (without terminology) completed"
fi
echo ""

# Run workflows with terminology if dataset is wmt25
if [ "$DATASET" == "wmt25" ]; then
    echo "=========================================="
    echo "Phase 2: Running workflows WITH terminology"
    echo "=========================================="
    echo ""
    
    for workflow in "${WORKFLOWS[@]}"; do
        # Wait if we've reached the parallel limit
        while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do
            sleep 1
        done
        
        # Run workflow in background
        run_workflow "$workflow" "true" &
    done
    
    # Wait for all background jobs to complete and check for failures
    FAILED=0
    for job in $(jobs -p); do
        wait $job || FAILED=$((FAILED + 1))
    done
    
    if [ $FAILED -gt 0 ]; then
        echo "⚠ Warning: $FAILED workflow(s) failed in Phase 2"
    else
        echo "✓ All workflows (with terminology) completed"
    fi
    echo ""
fi

echo "=========================================="
echo "All workflows finished!"
echo "=========================================="
