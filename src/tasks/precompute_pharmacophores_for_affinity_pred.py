import os
import sys

import torch
from absl import flags
from tqdm import tqdm

from src.scoring import PrecalculationModule

_OUT_PATH = flags.DEFINE_string(
    "out_path",
    "dataset/affinity_prediction_pharmacophores/crossdocked_dim_256",
    "Path to save the pharmacophores.",
)

_HEAD_PATH = flags.DEFINE_string(
    "head_path",
    "model_weights/crossdocked_dim_256.pth",
    "Path to PharmacoNet head path.",
)

# Typically don't need to change these
_MODEL_PATH = flags.DEFINE_string(
    "model_path", "model_weights/model.tar", "Path to PharmacoNet model base path."
)

_POCKET_TO_LIGANDS_PATH = flags.DEFINE_string(
    "pocket_to_ligands_path",
    "dataset/pocket_to_ligands.pt",
    "Mapping from pocket to reference ligand paths.",
)

_POCKET_FOLDER = flags.DEFINE_string(
    "pocket_folder", "dataset/crossdock", "Path to the pocket folder."
)

_LIGAND_FOLDER = flags.DEFINE_string(
    "ligand_folder", "dataset/crossdocked_pocket10", "Path to the ligand folder."
)

_DEVICE = flags.DEFINE_string("device", "cuda", "Device to run the model on.")


def main() -> None:
    flags.FLAGS(sys.argv)

    os.makedirs(_OUT_PATH.value, exist_ok=True)

    split_file = torch.load(_POCKET_TO_LIGANDS_PATH.value)
    predictor = PrecalculationModule(
        _MODEL_PATH.value, _HEAD_PATH.value, device=_DEVICE.value
    )

    failed_pairs = []
    for pdb_id, lig_path in tqdm(split_file.items()):
        lig_path = os.path.join(_LIGAND_FOLDER.value, lig_path)
        rec_path = os.path.join(_POCKET_FOLDER.value, pdb_id + "_rec.pdb")
        out_path = os.path.join(_OUT_PATH.value, pdb_id + "_rec.pt")

        try:
            cache = predictor.run(rec_path, ref_ligand_path=lig_path)
            torch.save(cache, out_path)
        except Exception as e:
            failed_pairs.append((rec_path, lig_path))
            # print a detailed error message
            print(f"Failed to compute pharmacophore for {pdb_id}.")
            print(e)
        except KeyboardInterrupt:
            break

    print(f"Failed to compute pharmacophores for {len(failed_pairs)} pairs.")
    torch.save(failed_pairs, os.path.join(_OUT_PATH.value, "failed_pairs.pt"))


if __name__ == "__main__":
    main()
