""" Example Usage
python3 src/tasks/aggergate_evals.py  --eval_paths 
    /home/tsa87/refactor-tacogfn/misc/generated_molecules/20240124_0.5_zinc-mo-256.json
"""

import json
import sys

import numpy as np
from absl import flags

_EVAL_PATH = flags.DEFINE_string(
    "eval_path",
    "misc/evaluations/20240112_2024_01_11_run_pharmaco_frag_beta_qed_100_per_pocket.json",
    "Path to the generated molecules.",
)


def is_novel_hit(novelty, qed, sas, docking_score, native_docking_score):
    if novelty > 0.6 and qed > 0.5 and sas > 0.5 and docking_score < -8:
        return True
    return False


def main():
    flags.FLAGS(sys.argv)

    with open(_EVAL_PATH.value, "r") as f:
        eval_data = json.load(f)

    dock = True if "docking_scores" in eval_data[next(iter(eval_data))] else False

    all_vals = {
        "time": [],
        "qeds": [],
        "sas": [],
        "preds": [],
        "diversity": [],
        "novelty": [],
    }
    if dock:
        all_vals["docking_scores"] = []
        all_vals["is_hit"] = []

    for key, val in eval_data.items():
        all_vals["time"].append(val["time"])
        all_vals["qeds"].extend(val["qeds"])
        all_vals["sas"].extend(val["sas"])
        all_vals["preds"].extend(val["preds"])
        all_vals["diversity"].extend(val["diversity"])
        all_vals["novelty"].extend(val["novelty"])
        if dock:
            all_vals["docking_scores"].extend(
                [v for v in val["docking_scores"] if v < 0]
            )

            hit_list = []
            for i in range(len(val["qeds"])):
                hit = is_novel_hit(
                    val["novelty"][i],
                    val["qeds"][i],
                    val["sas"][i],
                    val["docking_scores"][i],
                    native_docking_score,
                )
                hit_list.append(hit)
            all_vals["is_hit"].extend(hit_list)

    for key, val in all_vals.items():
        print(key, np.mean(val))


if __name__ == "__main__":
    main()
