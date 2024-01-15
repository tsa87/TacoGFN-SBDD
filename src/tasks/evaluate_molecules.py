import concurrent.futures
import json
import os
import sys

import torch
from absl import flags
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm

from src.tacogfn.eval import docking
from src.tacogfn.utils import molecules, sascore

_REC_FOLDER = flags.DEFINE_string(
    "rec_folder",
    "dataset/crossdock_pdbqt",
    "Path to the folder containing the receptor files.",
)


_MOLECULES_PATH = flags.DEFINE_string(
    "molecules_path",
    "misc/generated_molecules/20240112_2024_01_11_run_pharmaco_frag_beta_qed_100_per_pocket.json",
    "Path to the generated molecules.",
)

_POCKET_TO_SCORE_PATH = flags.DEFINE_string(
    "pocket_to_score_path",
    "dataset/pocket_to_score.pt",
    "Path to the pocket to native ligand docking score file.",
)

_POCKET_TO_CENTRIOD_PATH = flags.DEFINE_string(
    "pocket_to_centroid_path",
    "dataset/pocket_to_centroid.pt",
    "Path to the pocket to centroid file.",
)

_RESULTS_FOLDER = flags.DEFINE_string(
    "results_save_path",
    "misc/evaluations",
    "Path to save the evaluated molecules.",
)


def compute_docking_scores(
    pocket_id: str,
    smiles_list: list[str],
    centroid: tuple[float, float, float],
):
    docking_res = []
    rec_path = os.path.join(
        _REC_FOLDER.value,
        f"{pocket_id}_rec.pdbqt",
    )

    n = len(smiles_list)
    with tqdm(total=n, desc="Docking Progress") as pbar:
        # Use ProcessPoolExecutor to execute the tasks in parallel
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for result in executor.map(
                docking.default_compute_docking_score_from_smiles,
                [rec_path] * n,
                smiles_list,
                [centroid] * n,
            ):
                docking_res.append(result)
                pbar.update(1)

    return docking_res


def main() -> None:
    flags.FLAGS(sys.argv)

    pocket_to_centroid = torch.load(_POCKET_TO_CENTRIOD_PATH.value)
    # pocket_to_score = torch.load(_POCKET_TO_SCORE_PATH.value)
    generated_results = json.load(open(_MOLECULES_PATH.value))

    evaluated_results = {}

    for pocket, val in tqdm(generated_results.items()):
        centroid = pocket_to_centroid[pocket]
        # native_docking_score = pocket_to_score[pocket]

        time = val["time"]
        smiles = val["smiles"]
        mols = [Chem.MolFromSmiles(smi) for smi in smiles]

        qeds = [Descriptors.qed(mol) for mol in mols]
        sas = [(10.0 - sascore.calculateScore(mol)) / 9 for mol in mols]
        docking_scores = compute_docking_scores(pocket, smiles, centroid)
        diversity = molecules.compute_diversity(mols)

        evaluated_results[pocket] = {
            "time": time,
            "smiles": smiles,
            "qeds": qeds,
            "sas": sas,
            "docking_scores": docking_scores,
            "diversity": diversity,
            "centroid": centroid,
            # "native_docking_score": native_docking_score,
        }

    filename = _MOLECULES_PATH.value.split("/")[-1].split(".")[0]
    save_path = os.path.join(_RESULTS_FOLDER.value, f"{filename}_evaluated.json")

    with open(save_path, "w") as f:
        json.dump(evaluated_results, f)


if __name__ == "__main__":
    main()
