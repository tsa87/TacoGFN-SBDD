import os

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from src.pharmaconet.modeling import ModelingModule

score_threshold = {
    "Hydrophobic": 0.12035781145095825,
    "PiStacking_P": 0.0017433513421565294,
    "PiStacking_T": 0.027523474767804146,
    "PiCation_lring": 0.009127387776970863,
    "PiCation_pring": 0.001319622853770852,
    "HBond_ldon": 0.0827205702662468,
    "HBond_pdon": 0.12345150858163834,
    "SaltBridge_lneg": 0.12117615342140198,
    "SaltBridge_pneg": 0.06078973412513733,
    "XBond": 0.0020935165230184793,
}


def run_inference_on_gpu(gpu_id, data_subset, failed_queue):
    # Set the current GPU
    torch.cuda.set_device(gpu_id)

    # Initialize your model on the specified GPU
    module = ModelingModule(
        "model_weights/model.tar", f"cuda:{gpu_id}", score_threshold=score_threshold
    )

    for rec_id, lig_id in tqdm(data_subset):
        try:
            rec_path = os.path.join("dataset/crossdock", rec_id + "_rec.pdb")
            lig_path = os.path.join("dataset/crossdocked_pocket10", lig_id)
            out_path = os.path.join("dataset/new_pharmacophores", rec_id + ".pt")

            p_model = module.run(rec_path, ref_ligand_path=lig_path)
            p_model.save(out_path)
        except Exception as e:
            failed_queue.put((rec_id, lig_id, str(e)))


def main():
    num_gpus = 8
    split_file = torch.load("dataset/pocket_to_ligands.pt")
    dataset = list(split_file.items())
    chunk_size = len(dataset) // num_gpus
    data_subsets = [
        dataset[i : i + chunk_size] for i in range(0, len(dataset), chunk_size)
    ]

    # Queue for collecting failed cases
    failed_queue = mp.Queue()

    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=run_inference_on_gpu,
            args=(gpu_id, data_subsets[gpu_id], failed_queue),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Collect failed cases
    failed = []
    while not failed_queue.empty():
        failed.append(failed_queue.get())

    print("Failed cases:", failed)


if __name__ == "__main__":
    main()
