import concurrent.futures
import json
import os
import sys
from pathlib import Path

import torch
from absl import flags
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm

from src.tacogfn.eval import docking
from src.tacogfn.utils import misc, molecules, sascore

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

_MOLS_PER_POCKET = flags.DEFINE_integer(
    "mols_per_pocket",
    100,
    "Number of molecules to evaluate per pocket.",
)

_NUM_OF_POCKETS = flags.DEFINE_integer(
    "num_of_pockets",
    100,
    "Number of pockets to evaluate.",
)

_DOCK = flags.DEFINE_boolean(
    "dock",
    False,
    "Whether to compute docking scores.",
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
    pocket_to_score = torch.load(_POCKET_TO_SCORE_PATH.value)
    generated_results = json.load(open(_MOLECULES_PATH.value))

    # Only evaluate the first _NUM_OF_POCKETS.value pockets
    generated_results = dict(list(generated_results.items())[: _NUM_OF_POCKETS.value])

    evaluated_results = {}

    ref_fps = misc.get_reference_fps()

    for pocket, val in tqdm(generated_results.items()):
        centroid = pocket_to_centroid[pocket]
        native_docking_score = pocket_to_score[pocket]

        smiles = val["smiles"][: _MOLS_PER_POCKET.value]
        mols = [Chem.MolFromSmiles(smi) for smi in smiles]

        qeds = [Descriptors.qed(mol) for mol in mols]
        sas = [(10.0 - sascore.calculateScore(mol)) / 9 for mol in mols]
        diversity = molecules.compute_diversity(mols)
        novelty = molecules.compute_novelty(mols, ref_fps)

        evaluated_results[pocket] = {
            # "time": time,
            "smiles": smiles,
            "qeds": qeds,
            "sas": sas,
            # "preds": preds,
            "diversity": diversity,
            "novelty": novelty,
            "centroid": centroid,
            "native_docking_score": native_docking_score,
        }

        if "time" in val.keys():
            evaluated_results[pocket]["time"] = val["time"]
        if "preds" in val.keys():
            evaluated_results[pocket]["preds"] = val["preds"]

        if "docking_scores" in val.keys():
            evaluated_results[pocket]["docking_scores"] = val["docking_scores"]
        else:
            if _DOCK.value:
                docking_scores = compute_docking_scores(pocket, smiles, centroid)
                evaluated_results[pocket]["docking_scores"] = docking_scores

    filename = Path(_MOLECULES_PATH.value.split("/")[-1]).stem
    save_path = os.path.join(_RESULTS_FOLDER.value, f"{filename}_evaluated.json")
    print(f"Saving evaluated results to {save_path}")

    with open(save_path, "w") as f:
        json.dump(evaluated_results, f)


if __name__ == "__main__":
    main()
