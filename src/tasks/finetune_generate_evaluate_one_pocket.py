"""
python src/tasks/generate_evaluate_one_pocket.py \
    --model_path "logs/finetune-1.5docking_score2/model_state_200.pt" \
    --pocket_index 2 \
    --comment "finetune-1.5docking_score2-pocket2-generation" 

"""

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
from rdkit.Chem import Descriptors
from tqdm import tqdm

from src.tacogfn.tasks import finetune_pocket_frag_one_pocket
from src.tacogfn.utils import misc, molecules, sascore
from src.tacogfn.utils.unidock import unidock_scores

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

_POCKET_INDEX = flags.DEFINE_integer(
    "pocket_index",
    0,
    "Index of the pocket to generate molecules for.",
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
    trail = finetune_pocket_frag_one_pocket.UniDockFinetuneTrainer(model_state["cfg"])

    trail.model.load_state_dict(model_state["models_state_dict"][0])
    trail.model.eval()
    trail.model.to("cuda")
    trail.ctx.device = "cuda"

    results = {}

    idx = _POCKET_INDEX.value

    # Generate molecules in batches
    batch_sizes = [_BATCH_SIZE.value] * (_NUM_PER_POCKET.value // _BATCH_SIZE.value)
    if _NUM_PER_POCKET.value % _BATCH_SIZE.value != 0:
        batch_sizes.append(_NUM_PER_POCKET.value % _BATCH_SIZE.value)

    start = time.time()
    all_mols = []
    all_preds = []
    for size in tqdm(batch_sizes):
        mols = trail.sample_molecules(
            [idx] * size, sample_temp=_SAMPLE_TEMP.value, beta_temp=_BETA_TEMP.value
        )
        all_mols.extend(mols)

    end = time.time()
    smiles = [Chem.MolToSmiles(mol) for mol in all_mols]

    qeds = [Descriptors.qed(mol) for mol in mols]
    sas = [(10.0 - sascore.calculateScore(mol)) / 9 for mol in mols]
    diversity = molecules.compute_diversity(mols)

    docking_scores = unidock_scores(
        smiles,
        trail.task.rec_path,
        trail.task.pocket_x,
        trail.task.pocket_y,
        trail.task.pocket_z,
    )

    pdb_id = trail.pharmaco_db.idx_to_id[idx]
    results[pdb_id] = {
        "smiles": smiles,
        "docking_scores": [
            float(s) for s in docking_scores
        ],  # Convert docking scores to float type
        "qeds": qeds,
        "sas": sas,
        "diversity": diversity,
        "time": end - start,
    }

    print(f"Mean docking score: {np.mean(docking_scores)}")
    print(f"Mean QED: {np.mean(qeds)}")
    print(f"Mean SAS: {np.mean(sas)}")
    print(f"Mean diversity: {np.mean(diversity)}")

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
