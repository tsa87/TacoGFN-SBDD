"""
Seh Fragment based task from
https://github.com/recursionpharma/gflownet
"""

import os
import sys
import shutil
import torch
from absl import flags

from src.tacogfn.tasks.pretrain_qed_sa import UniDockSinglePocketTrainer

def main():
    """Example of how this trainer can be run"""
    _MODEL_PATH = flags.DEFINE_string(
        "model_path",
        "logs/qed_sa_pretrain/model_state_2000.pt",
        "Path to the model state file.",
    )
    
    hps = {
        "log_dir": "./logs/qed_sa_pretrain_finetune",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "overwrite_existing_exp": True,
        "num_training_steps": 10_000,
        "num_workers": 0,
        "opt": {
            "lr_decay": 20000,
        },
        "algo": {"sampling_tau": 0.99, "offline_ratio": 0.0},
        "cond": {
            "temperature": {
                "sample_dist": "uniform",
                "dist_params": [0, 64.0],
            }
        },
        "task": {
            "pharmaco_frag": {
                "reward_multiplier": 5.0,
                "max_qed_reward": 0.5,
                "max_sa_reward": 0.7,
                "objectives": ["docking", "qed", "sa"],
                "mol_adj": 0.33,
            }
        },
    }
    
    flags.FLAGS(sys.argv)
    model_state = torch.load(_MODEL_PATH.value)
    
    if os.path.exists(hps["log_dir"]):
        if hps["overwrite_existing_exp"]:
            shutil.rmtree(hps["log_dir"])
        else:
            raise ValueError(
                f"Log dir {hps['log_dir']} already exists. Set overwrite_existing_exp=True to delete it."
            )
    os.makedirs(hps["log_dir"])

    trial = UniDockSinglePocketTrainer(hps)
    
    trial.model.load_state_dict(model_state["models_state_dict"][0])
    trial.sampling_model.load_state_dict(model_state["models_state_dict"][0])
    
    trial.print_every = 1
    trial.run()


if __name__ == "__main__":
    main()
