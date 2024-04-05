#!/bin/bash

# Array of file paths
FILES=(
    "misc/evaluations/archive/20240117_alpha_default_evaluated.json"
    "misc/evaluations/targetdiff_agg_data_evaluated.json"
    "misc/evaluations/pocket2mol_agg_data_evaluated.json"
    "misc/evaluations/decompdiff_subpocket_evaluated.json"
    "misc/evaluations/decompdiff_beta_evaluated.json"
    "misc/evaluations/decompdiff_agg_data_evaluated.json"
    "misc/evaluations/A_0_100_crossdocked-mo-256-pocket-graph-1_evaluated.json" # Crossdocked
    "misc/evaluations/C_0_500_crossdocked-mo-256-pocket-graph-1_evaluated.json" # Crossdocked Ranked
    "misc/evaluations/0_100_zinc-mo-256-pocket-graph-39000-3_evaluated.json" # ZINC
    "misc/evaluations/D_0_500_zinc-mo-256-pocket-graph-39000-1_evaluated.json" # ZINC Ranked
    "misc/evaluations/0_100_zinc-mo-256-no-pharmaco-1_evaluated.json" # ZINC No Pharmaco    
    "misc/evaluations/E_0_100_crossdocked-mo-256-pocket_graph-adj_ds_evaluated.json" # Adj DS
    "misc/evaluations/F_0_100_crossdocked-mo-256-pocket_graph-bigger-batch_evaluated.json" # Bigger Batch Crossdocked
    # "misc/evaluations/0_500_new-zinc-mo-256-128-39000-1_evaluated.json"
    # "misc/evaluations/0_100_zinc-mo-256-pocket-graph-3_evaluated.json"
    # "misc/evaluations/B_0_100_new-zinc-mo-256-128-39000-1_evaluated.json" 
    # "misc/evaluations/0_500_new-crossdocked-mo-256-256-1_evaluated.json"
    # "misc/evaluations/0_100_new-crossdocked-mo-256-256-3_evaluated.json" 
    # "misc/evaluations/0_100_new-zinc-mo-256-128-39000-3_evaluated.json"
)

# Loop over files and run the command
for file in "${FILES[@]}"
do
    python3 src/tasks/aggergate_evals.py --eval_path "$file" --normalize_docking_score
done