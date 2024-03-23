""" Example Usage
python3 src/tasks/aggergate_evals.py  --eval_paths 
    /home/tsa87/refactor-tacogfn/misc/generated_molecules/20240124_0.5_zinc-mo-256.json
"""

import json
import sys
from collections import defaultdict

import numpy as np
from absl import flags
from tabulate import tabulate

_EVAL_PATH = flags.DEFINE_string(
    "eval_path",
    "misc/evaluations/20240112_2024_01_11_run_pharmaco_frag_beta_qed_100_per_pocket.json",
    "Path to the generated molecules.",
)

_NORMALIZE_DOCKING_SCORE = flags.DEFINE_boolean(
    "normalize_docking_score",
    False,
    "Flag to indicate whether to compute normalized docking scores.",
)


def is_okay(qed, sas):
    if qed > 0.5 and sas > 0.5:
        return True
    return False


def is_novel_okay(novelty, qed, sas):
    if novelty > 0.5 and qed > 0.5 and sas > 0.5:
        return True
    return False


def is_hit(qed, sas, docking_score):
    if qed > 0.5 and sas > 0.5 and docking_score < -8:
        return True
    return False


def is_novel_hit(novelty, qed, sas, docking_score):
    if novelty > 0.5 and qed > 0.5 and sas > 0.5 and docking_score < -8:
        return True
    return False


def main():
    flags.FLAGS(sys.argv)

    with open(_EVAL_PATH.value, "r") as f:
        eval_data = json.load(f)

    all_vals = defaultdict(list)

    dock = True if "docking_scores" in eval_data[next(iter(eval_data))] else False

    not_enough_novel_okay_molecules = []
    not_enough_okay_molecules = []

    for key, val in eval_data.items():
        if len(val["qeds"]) == 0:
            continue

        if "time" in val:
            all_vals["time"].append(val["time"])
        if "preds" in val:
            all_vals["preds"].append(np.mean(val["preds"]))

        all_vals["qeds"].append(np.mean(val["qeds"]))
        all_vals["sas"].append(np.mean(val["sas"]))
        all_vals["diversity"].append(np.mean(val["diversity"]))
        all_vals["novelty"].append(np.mean(val["novelty"]))

        if dock:
            val["docking_scores"] = [
                v if (v is not None and v < 0) else 0 for v in val["docking_scores"]
            ]
            all_vals["docking_scores"].append(np.mean(val["docking_scores"]))

            if _NORMALIZE_DOCKING_SCORE.value:
                from rdkit import Chem

                mols = [Chem.MolFromSmiles(smi) for smi in val["smiles"]]
                mol_atoms = [mol.GetNumHeavyAtoms() for mol in mols]
                val["normalized_docking_scores"] = [
                    v / (a ** (1 / 3)) for v, a in zip(val["docking_scores"], mol_atoms)
                ]
                all_vals["normalized_docking_scores"].append(
                    np.mean(val["normalized_docking_scores"])
                )

            # COMPUTE HIT %
            hit_list = []
            novel_hit_list = []
            is_okay_list = []
            is_novel_okay_list = []
            for i in range(len(val["qeds"])):
                hit = is_hit(
                    val["qeds"][i],
                    val["sas"][i],
                    val["docking_scores"][i],
                )
                hit_list.append(hit)

                novel_hit = is_novel_hit(
                    val["novelty"][i],
                    val["qeds"][i],
                    val["sas"][i],
                    val["docking_scores"][i],
                )
                novel_hit_list.append(novel_hit)

                is_okay_list.append(is_okay(val["qeds"][i], val["sas"][i]))
                is_novel_okay_list.append(
                    is_novel_okay(val["novelty"][i], val["qeds"][i], val["sas"][i])
                )

            all_vals["hit"].append(
                np.mean(hit_list)
            )  # proportion of molecules that are hits
            all_vals["novel_hit"].append(
                np.mean(novel_hit_list)
            )  # proportion of molecules that are novel hits

            # Compute top 10
            docking_scores = np.array(val["docking_scores"])
            sorted_indices = np.argsort(docking_scores)

            all_vals["top_10_docking_scores"].append(
                np.mean(docking_scores[sorted_indices[:10]])
            )

            # Compute top 10 hit docking scores
            top_10_okay_docking_scores = []
            for i in range(len(sorted_indices)):
                if is_okay_list[sorted_indices[i]]:
                    top_10_okay_docking_scores.append(docking_scores[sorted_indices[i]])
                if len(top_10_okay_docking_scores) == 10:
                    break
            if len(top_10_okay_docking_scores) != 10:
                not_enough_okay_molecules.append(key)
                # Fill in the rest with 0s
                for i in range(10 - len(top_10_okay_docking_scores)):
                    top_10_okay_docking_scores.append(0)

            all_vals["top_10_okay_docking_scores"].append(
                np.mean(top_10_okay_docking_scores)
            )

            top_10_novel_okay_docking_scores = []
            for i in range(len(sorted_indices)):
                if is_novel_okay_list[sorted_indices[i]]:
                    top_10_novel_okay_docking_scores.append(
                        docking_scores[sorted_indices[i]]
                    )
                if len(top_10_novel_okay_docking_scores) == 10:
                    break
            if len(top_10_novel_okay_docking_scores) != 10:
                not_enough_novel_okay_molecules.append(key)
                for i in range(10 - len(top_10_novel_okay_docking_scores)):
                    top_10_novel_okay_docking_scores.append(0)

            all_vals["top_10_novel_okay_docking_scores"].append(
                np.mean(top_10_novel_okay_docking_scores)
            )

    table_data = []
    for key, val in all_vals.items():
        table_data.append([key, np.mean(val), np.median(val)])

    print("---------------------------------------------------------------------")
    print("---------------------------------------------------------------------")
    print(_EVAL_PATH.value)

    print(
        f"Not enough novel and okay molecules: {len(not_enough_novel_okay_molecules)} / {len(eval_data) } {len(not_enough_novel_okay_molecules) / len(eval_data)}"
    )
    print(
        f"Not enough okay molecules: {len(not_enough_okay_molecules)} / {len(eval_data) } {len(not_enough_okay_molecules) / len(eval_data)}"
    )

    table_headers = ["Key", "Mean", "Median"]
    table = tabulate(table_data, headers=table_headers, tablefmt="grid")
    print(table)

    # for _, val in all_vals.items():
    #     print(np.mean(val))


if __name__ == "__main__":
    main()
