# Example usage:
# bash run/translate_vs_postedit.sh wmt25
# bash run/translate_vs_postedit.sh dolfin

MODEL1="qwen3-32b"
MODEL2="qwen3-235b"

DATASET=$1 # wmt25 or dolfin

WORKFLOWS=(
    "MaMT_translate_postedit_proofread"
    "IRB_refine"
    "MAATS_multi_agents"
)

for WORKFLOW in "${WORKFLOWS[@]}"; do
    python src/run.py --base_model $MODEL1 --model $MODEL2 --workflow $WORKFLOW --dataset $DATASET --use_terminology
    python src/run.py --base_model $MODEL2 --model $MODEL1 --workflow $WORKFLOW --dataset $DATASET --use_terminology
done