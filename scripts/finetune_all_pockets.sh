while IFS=',' read -r pocket_index; do
    echo "$pocket_index"
    # python src/tacogfn/tasks/finetune_pocket_frag_one_pocket.py \
    #     --model_path logs/20240504-crossdocked-mo-256-pocket_graph-adj_ds/model_state_9000.pt \
    #     --pocket_index "$pocket_index"
done < scripts/test.txt

for pocket_index in 1,2,3; do
    echo "$pocket_index"
done