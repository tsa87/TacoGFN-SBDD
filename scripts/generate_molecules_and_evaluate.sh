#!/bin/bash

# Define the model path and comment as constants
MODEL_PATH="/home/tsa87/refactor-tacogfn/logs/20240121-zinc-mo-256/model_state_25000.pt"
COMMENT="zinc-mo-256"

# # An array of beta_temp values
# BETA_TEMPS=(0 0.5 0.75 1.0)

# # Loop through each beta_temp value
# for BETA_TEMP in "${BETA_TEMPS[@]}"; do
#     # Generate molecules
#     python3 src/tasks/generate_molecules.py \
#         --model_path "$MODEL_PATH" \
#         --beta_temp "$BETA_TEMP" \
#         --comment "$COMMENT"

#     # Evaluate molecules
#     python3 src/tasks/evaluate_molecules.py \
#         --molecules_path "/home/tsa87/refactor-tacogfn/misc/generated_molecules/20240124_${BETA_TEMP}_${COMMENT}.json"
# done


# SAMPLE_TEMPS=( 1.25 1.5 1.75 2.0 )
# for SAMPLE_TEMP in "${SAMPLE_TEMPS[@]}"; do
#     # Generate molecules
#     python3 src/tasks/generate_molecules.py \
#         --model_path "$MODEL_PATH" \
#         --sample_temp "$SAMPLE_TEMP" \
#         --comment "$COMMENT"

#     # Evaluate molecules
#     python3 src/tasks/evaluate_molecules.py \
#         --molecules_path "/home/tsa87/refactor-tacogfn/misc/generated_molecules/20240124_1.0_${SAMPLE_TEMP}_${COMMENT}.json"
# done

# An array of num per pocket 
NUM_PER_POCKET=( 200 500 1000 )
for NUM in "${NUM_PER_POCKET[@]}"; do
    # Generate molecules
    python3 src/tasks/generate_molecules.py \
        --model_path "$MODEL_PATH" \
        --num_per_pocket "$NUM" \
        --beta_temp 0.75 \
        --comment "$COMMENT"

    # Evaluate molecules
    python3 src/tasks/evaluate_molecules.py \
        --molecules_path "/home/tsa87/refactor-tacogfn/misc/generated_molecules/20240124_0.75_1.0_${NUM}_${COMMENT}.json"
done
