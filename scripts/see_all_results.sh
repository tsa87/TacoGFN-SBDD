#!/bin/bash

# Array of file paths
declare -A FILES=(
    ["misc/evaluations/targetdiff_agg_data_evaluated.json"]="TargetDiff"
    ["misc/evaluations/pocket2mol_agg_data_evaluated.json"]="Pocket2Mol"
    ["misc/evaluations/decompdiff_subpocket_evaluated.json"]="DecompDiff Subpocket"
    ["misc/evaluations/decompdiff_beta_evaluated.json"]="DecompDiff Beta"
    ["misc/evaluations/decompdiff_agg_data_evaluated.json"]="DecompDiff"
    ["misc/evaluations/A_0_100_crossdocked-mo-256-pocket-graph-1_evaluated.json"]="TacoGFN Crossdocked"
    ["misc/evaluations/C_0_500_crossdocked-mo-256-pocket-graph-1_evaluated.json"]="TacoGFN Crossdocked Ranked"
    ["misc/evaluations/0_100_zinc-mo-256-pocket-graph-39000-3_evaluated.json"]="TacoGFN ZINC"
    ["misc/evaluations/D_0_500_zinc-mo-256-pocket-graph-39000-1_evaluated.json"]="TacoGFN ZINC Ranked"
    ["misc/evaluations/0_100_zinc-mo-256-no-pharmaco-1_evaluated.json"]="TacoGFN ZINC No Pharmaco"
)

# Loop over files and run the command
for file in "${!FILES[@]}"
do
    echo "Experiment: ${FILES[$file]}"
    python3 src/tasks/aggergate_evals.py --eval_path "$file"
done