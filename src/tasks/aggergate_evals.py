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


def is_okay(qed, sas):
    if qed > 0.5 and sas > 0.5:
        return True
    return False


def is_novel_okay(novelty, qed, sas):
    if novelty > 0.6 and qed > 0.5 and sas > 0.5:
        return True
    return False


def is_hit(qed, sas, docking_score):
    if qed > 0.5 and sas > 0.5 and docking_score < -8:
        return True
    return False


def is_novel_hit(novelty, qed, sas, docking_score):
    if novelty > 0.6 and qed > 0.5 and sas > 0.5 and docking_score < -8:
        return True
    return False


def main():
    flags.FLAGS(sys.argv)

    with open(_EVAL_PATH.value, "r") as f:
        eval_data = json.load(f)

    all_vals = defaultdict(list)

    dock = True if "docking_scores" in eval_data[next(iter(eval_data))] else False

    not_enough_novel_molecules = []
    not_enough_hit_molecules = []

    for key, val in eval_data.items():
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

            all_vals["hit"].append(np.mean(hit_list))
            all_vals["novel_hit"].append(np.mean(novel_hit_list))

            # Compute top 5
            docking_scores = np.array(val["docking_scores"])
            sorted_indices = np.argsort(docking_scores)

            # Compute top 5 docking scores
            all_vals["top_10_docking_scores"].append(
                np.mean(docking_scores[sorted_indices[:5]])
            )

            # Compute top 5 hit docking scores
            top_10_hit_docking_scores = []
            for i in range(len(sorted_indices)):
                if is_okay_list[sorted_indices[i]]:
                    top_10_hit_docking_scores.append(docking_scores[sorted_indices[i]])
                if len(top_10_hit_docking_scores) == 10:
                    break
            if len(top_10_hit_docking_scores) != 10:
                not_enough_hit_molecules.append(key)

            if len(top_10_hit_docking_scores) > 0:
                all_vals["top_10_hit_docking_scores"].append(
                    np.mean(top_10_hit_docking_scores)
                )
            else:
                all_vals["top_10_hit_docking_scores"].append(0)
                # print(f"WARNING: Skipping {key} as it has no okay molecules.")

            # Compute top 5 novel hit docking scores
            top_10_novel_docking_scores = []
            for i in range(len(sorted_indices)):
                if is_novel_okay_list[sorted_indices[i]]:
                    top_10_novel_docking_scores.append(
                        docking_scores[sorted_indices[i]]
                    )
                if len(top_10_novel_docking_scores) == 10:
                    break
            if len(top_10_novel_docking_scores) != 10:
                not_enough_novel_molecules.append(key)

            if len(top_10_novel_docking_scores) > 0:
                all_vals["top_10_novel_docking_scores"].append(
                    np.mean(top_10_novel_docking_scores)
                )
            else:
                all_vals["top_10_novel_docking_scores"].append(0)
                # print(f"WARNING: Skipping {key} as it has no novel okay molecules.")

    table_data = []
    for key, val in all_vals.items():
        table_data.append([key, np.mean(val), np.median(val)])

    print(f"Not enough novel and okay molecules: {not_enough_novel_molecules}")
    print(f"Not enough okay molecules: {not_enough_hit_molecules}")

    table_headers = ["Key", "Mean", "Median"]
    table = tabulate(table_data, headers=table_headers, tablefmt="grid")
    print(table)

    # for _, val in all_vals.items():
    #     print(np.mean(val))


if __name__ == "__main__":
    main()
