#!/bin/bash

# Define the model path and comment as constants
# MODEL_PATH="/home/tsa87/refactor-tacogfn/logs/20240121-zinc-mo-256/model_state_25000.pt"
# COMMENT="zinc-mo-256"

# This experiments defines an array of beta temps
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

# This experiments defines an array of sample temps
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

# This experiments defines an array of num per pocket 
# NUM_PER_POCKET=( 200 500 1000 )
# for NUM in "${NUM_PER_POCKET[@]}"; do
#     # Generate molecules
#     python3 src/tasks/generate_molecules.py \
#         --model_path "$MODEL_PATH" \
#         --num_per_pocket "$NUM" \
#         --beta_temp 0.75 \
#         --comment "$COMMENT"

#     # Evaluate molecules
#     python3 src/tasks/evaluate_molecules.py \
#         --molecules_path "/home/tsa87/refactor-tacogfn/misc/generated_molecules/20240124_0.75_1.0_${NUM}_${COMMENT}.json"
# done

# We will take the replicate with a median docking score for docking 
declare -A MODEL_PATHS
DATE=20240128
# MODEL_PATHS["/home/tsa87/refactor-tacogfn/logs/20240121-new-crossdocked-mo-256-256/model_state_31000.pt"]="new-crossdocked-mo-256-256"
# MODEL_PATHS["/home/tsa87/refactor-tacogfn/logs/20240121-new-zinc-mo-256-128/model_state_39000.pt"]="new-zinc-mo-256-128-39000"
# MODEL_PATHS["/home/tsa87/refactor-tacogfn/logs/20240121-zinc-mo-256-pocket-graph/model_state_39000.pt"]="zinc-mo-256-pocket-graph-39000"
MODEL_PATHS["/home/tsa87/refactor-tacogfn/logs/20240121-zinc-mo-256-no-pharmaco/model_state_25000.pt"]="zinc-mo-256-no-pharmaco"
REPLATES=(1)
for MODEL_PATH in "${!MODEL_PATHS[@]}"; do
    COMMENT_PREFIX=${MODEL_PATHS[$MODEL_PATH]}
    for REPLATE in "${REPLATES[@]}"; do
        # Generate molecules
        python3 src/tasks/generate_molecules.py \
            --model_path "$MODEL_PATH" \
            --comment "${COMMENT_PREFIX}-${REPLATE}"

        # Evaluate molecules
        python3 src/tasks/evaluate_molecules.py \
            --molecules_path "/home/tsa87/refactor-tacogfn/misc/generated_molecules/${DATE}_1.0_100_${COMMENT_PREFIX}-${REPLATE}.json"
    done
done

# REPLATES=(1)
# for MODEL_PATH in "${!MODEL_PATHS[@]}"; do
#     COMMENT_PREFIX=${MODEL_PATHS[$MODEL_PATH]}
#     for REPLATE in "${REPLATES[@]}"; do
#         # Generate molecules
#         python3 src/tasks/generate_molecules.py \
#             --model_path "$MODEL_PATH" \
#             --num_per_pocket 500 \
#             --comment "${COMMENT_PREFIX}-${REPLATE}"

#         # Evaluate molecules
#         python3 src/tasks/evaluate_molecules.py \
#             --molecules_path "/home/tsa87/refactor-tacogfn/misc/generated_molecules/${DATE}_1.0_500_${COMMENT_PREFIX}-${REPLATE}.json"
#     done
# done




