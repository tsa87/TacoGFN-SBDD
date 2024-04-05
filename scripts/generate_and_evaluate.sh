#!/bin/bash

declare -A MODEL_PATHS
MODEL_PATHS["model_weights/tacogfn/crossdocked/model_state.pt"]="crossdocked"
MODEL_PATHS["model_weights/tacogfn/zinc/model_state.pt"]="zinc"
MODEL_PATHS["model_weights/tacogfn/zinc-no-pocket/model_state.pt"]="zinc-no-pocket"

for MODEL_PATH in "${!MODEL_PATHS[@]}"; do
    COMMENT_PREFIX=${MODEL_PATHS[$MODEL_PATH]}

    python3 src/tasks/generate_molecules.py \
        --model_path "$MODEL_PATH" \
        --num_per_pocket 100 \
        --comment "${COMMENT_PREFIX}"

    # Evaluate molecules
    python3 src/tasks/evaluate_molecules.py \
        --molecules_path "misc/generated_molecules/1.0_1.0_100_${COMMENT_PREFIX}.json"
done


python3 src/tasks/generate_molecules.py \
    --model_path "model_weights/tacogfn/crossdocked/model_state.pt" \
    --num_per_pocket 500 \
    --comment "crossdocked-ranked"

# Evaluate molecules
python3 src/tasks/evaluate_molecules.py \
    --molecules_path "misc/generated_molecules/1.0_1.0_500_crossdocked-ranked.json"