#!/bin/bash
# Run workflows sequentially for testing with a few samples
# Stops on first error to catch issues early

set -e  # Exit on error

MODEL="qwen3-32b"
# MODEL options: qwen3-32b, qwen3-235b, gpt-oss-20b, gpt-oss-120b, gpt-4-1-mini, gpt-4-1, o4-mini
# For ARNs: claude-sonnet-4, claude-sonnet-4-5
MAX_SAMPLES=2
DATASET="wmt25"

echo "=========================================="
echo "Running workflows sequentially"
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Max samples: $MAX_SAMPLES"
echo "=========================================="
echo ""

# Without terminology
echo "[1/7] Running zero_shot..."
python src/run.py --dataset $DATASET --workflow zero_shot --model $MODEL --max_samples $MAX_SAMPLES
echo "✓ zero_shot completed"
echo ""

echo "[2/7] Running MaMT_translate_postedit_proofread..."
python src/run.py --dataset $DATASET --workflow MaMT_translate_postedit_proofread --model $MODEL --max_samples $MAX_SAMPLES
echo "✓ MaMT_translate_postedit_proofread completed"
echo ""

echo "[3/7] Running IRB_refine..."
python src/run.py --dataset $DATASET --workflow IRB_refine --model $MODEL --max_samples $MAX_SAMPLES
echo "✓ IRB_refine completed"
echo ""

echo "[4/7] Running MAATS_multi_agents..."
python src/run.py --dataset $DATASET --workflow MAATS_multi_agents --model $MODEL --max_samples $MAX_SAMPLES
echo "✓ MAATS_multi_agents completed"
echo ""

echo "[5/7] Running SbS_chat_step_by_step..."
python src/run.py --dataset $DATASET --workflow SbS_chat_step_by_step --model $MODEL --max_samples $MAX_SAMPLES
echo "✓ SbS_chat_step_by_step completed"
echo ""

echo "[6/7] Running DeLTA_multi_agents..."
python src/run.py --dataset $DATASET --workflow DeLTA_multi_agents --model $MODEL --max_samples $MAX_SAMPLES
echo "✓ DeLTA_multi_agents completed"
echo ""

echo "[7/7] Running ADT_multi_agents..."
python src/run.py --dataset $DATASET --workflow ADT_multi_agents --model $MODEL --max_samples $MAX_SAMPLES
echo "✓ ADT_multi_agents completed"
echo ""

# With terminology (only available for wmt25)
if [ "$DATASET" == "wmt25" ]; then
    echo "=========================================="
    echo "Running workflows WITH terminology"
    echo "=========================================="
    echo ""
    
    echo "[1/7] Running zero_shot (with terminology)..."
    python src/run.py --dataset $DATASET --workflow zero_shot --use_terminology --model $MODEL --max_samples $MAX_SAMPLES
    echo "✓ zero_shot (with terminology) completed"
    echo ""
    
    echo "[2/7] Running MaMT_translate_postedit_proofread (with terminology)..."
    python src/run.py --dataset $DATASET --workflow MaMT_translate_postedit_proofread --use_terminology --model $MODEL --max_samples $MAX_SAMPLES
    echo "✓ MaMT_translate_postedit_proofread (with terminology) completed"
    echo ""
    
    echo "[3/7] Running IRB_refine (with terminology)..."
    python src/run.py --dataset $DATASET --workflow IRB_refine --use_terminology --model $MODEL --max_samples $MAX_SAMPLES
    echo "✓ IRB_refine (with terminology) completed"
    echo ""
    
    echo "[4/7] Running MAATS_multi_agents (with terminology)..."
    python src/run.py --dataset $DATASET --workflow MAATS_multi_agents --use_terminology --model $MODEL --max_samples $MAX_SAMPLES
    echo "✓ MAATS_multi_agents (with terminology) completed"
    echo ""
    
    echo "[5/7] Running SbS_chat_step_by_step (with terminology)..."
    python src/run.py --dataset $DATASET --workflow SbS_chat_step_by_step --use_terminology --model $MODEL --max_samples $MAX_SAMPLES
    echo "✓ SbS_chat_step_by_step (with terminology) completed"
    echo ""
    
    echo "[6/7] Running DeLTA_multi_agents (with terminology)..."
    python src/run.py --dataset $DATASET --workflow DeLTA_multi_agents --use_terminology --model $MODEL --max_samples $MAX_SAMPLES
    echo "✓ DeLTA_multi_agents (with terminology) completed"
    echo ""
    
    echo "[7/7] Running ADT_multi_agents (with terminology)..."
    python src/run.py --dataset $DATASET --workflow ADT_multi_agents --use_terminology --model $MODEL --max_samples $MAX_SAMPLES
    echo "✓ ADT_multi_agents (with terminology) completed"
    echo ""
fi

echo "=========================================="
echo "All workflows completed successfully!"
echo "=========================================="
