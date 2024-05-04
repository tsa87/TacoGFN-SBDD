import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

# Third-party imports
from absl import flags
from rdkit import Chem
from tqdm import tqdm

from src.tacogfn.tasks import pocket_frag

_MODEL_PATH = flags.DEFINE_string(
    "model_path",
    "logs/2024_01_11_run_pharmaco_frag_beta_qed/model_state.pt",
    "Path to the model state file.",
)

_NUM_PER_POCKET = flags.DEFINE_integer(
    "num_per_pocket",
    100,
    "Number of molecules to generate per pocket.",
)

_RETAIN_PER_POCKET = flags.DEFINE_integer(
    "retain_per_pocket",
    100,
    "Number of molecules to retain per pocket.",
)

_BATCH_SIZE = flags.DEFINE_integer(
    "batch_size",
    50,
    "Batch size for generating molecules.",
)

_SAVE_FOLDER = flags.DEFINE_string(
    "save_folder",
    "misc/generated_molecules",
    "Path to save the generated molecules.",
)

_SAMPLE_TEMP = flags.DEFINE_float(
    "sample_temp",
    1.0,
    "Temperature for sampling.",
)

_BETA_TEMP = flags.DEFINE_float(
    "beta_temp",
    1.0,
    "Temperature for beta.",
)

_COMMENT = flags.DEFINE_string(
    "comment",
    "",
    "Comment for the experiment.",
)


def main() -> None:
    flags.FLAGS(sys.argv)

    model_state = torch.load(_MODEL_PATH.value)
    trail = pocket_frag.PharmacophoreTrainer(model_state["cfg"])

    trail.model.load_state_dict(model_state["models_state_dict"][0])
    trail.model.eval()
    trail.model.to("cuda")
    trail.ctx.device = "cuda"

    test_idxs = trail.pharmaco_db.get_partition_idxs("test")

    results = {}

    for idx in tqdm(test_idxs):
        # Generate molecules in batches
        batch_sizes = [_BATCH_SIZE.value] * (_NUM_PER_POCKET.value // _BATCH_SIZE.value)
        if _NUM_PER_POCKET.value % _BATCH_SIZE.value != 0:
            batch_sizes.append(_NUM_PER_POCKET.value % _BATCH_SIZE.value)

        start = time.time()
        all_mols = []
        all_preds = []
        for size in batch_sizes:
            mols = trail.sample_molecules(
                [idx] * size, sample_temp=_SAMPLE_TEMP.value, beta_temp=_BETA_TEMP.value
            )
            preds = trail.task.predict_docking_score(
                mols, torch.tensor([idx] * size), info_only=True
            ).tolist()

            all_mols.extend(mols)
            all_preds.extend(preds)

        end = time.time()
        smiles = [Chem.MolToSmiles(mol) for mol in all_mols]

        pdb_id = trail.pharmaco_db.idx_to_id[idx]
        results[pdb_id] = {
            "smiles": smiles,
            "preds": all_preds,
            "time": end - start,
        }

        # Retain the top n molecules with lowest preds
        top_n_indices = np.argsort(all_preds)[: _RETAIN_PER_POCKET.value]
        top_n_smiles = [smiles[i] for i in top_n_indices]
        top_n_preds = [all_preds[i] for i in top_n_indices]

        results[pdb_id]["smiles"] = top_n_smiles
        results[pdb_id]["preds"] = top_n_preds

    today_date = pd.Timestamp.today().strftime("%Y%m%d")
    exp_name = trail.cfg.log_dir.split("/")[-1]

    save_path = os.path.join(
        _SAVE_FOLDER.value,
        f"{today_date}_{_BETA_TEMP.value}_{_SAMPLE_TEMP.value}_{_NUM_PER_POCKET.value}_{_COMMENT.value}.json",
    )

    with open(save_path, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
