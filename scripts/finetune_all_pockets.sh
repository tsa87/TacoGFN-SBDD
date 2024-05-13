#!/bin/bash

file_path="scripts/test_idx_splits/idx_0.txt"

# Read the single line from the file
read -r line < "$file_path"

# Set the Internal Field Separator to comma for the array assignment
IFS=',' read -ra numbers <<< "$line"

# Loop through the array of numbers
for pocket_index in "${numbers[@]}"; do
    echo "Finetuning pocket $pocket_index"
    python src/tacogfn/tasks/finetune_pocket_frag_one_pocket.py \
        --model_path logs/20240504-crossdocked-mo-256-pocket_graph-adj_ds/model_state_9000.pt \
        --pocket_index "$pocket_index"
done