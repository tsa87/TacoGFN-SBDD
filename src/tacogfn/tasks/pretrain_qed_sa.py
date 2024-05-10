"""
Seh Fragment based task from
https://github.com/recursionpharma/gflownet
"""

import os
import shutil
import socket
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch_geometric.data as gd
from rdkit import Chem
from rdkit.Chem.rdchem import Mol as RDMol
from torch import Tensor
from torch.utils.data import Dataset

from rdkit.Chem import Descriptors
from src.tacogfn.config import Config
from src.tacogfn.envs.frag_mol_env import FragMolBuildingEnvContext, Graph
from src.tacogfn.models import bengio2021flow
from src.tacogfn.online_trainer import StandardOnlineTrainer
from src.tacogfn.trainer import FlatRewards, GFNTask, RewardScalar
from src.tacogfn.utils.conditioning import TemperatureConditional
from src.tacogfn.utils.unidock import unidock_scores
from src.tacogfn.utils import molecules, sascore
from src.tacogfn.tasks.utils import GeneratedStore



class UniDockSinglePocketTask(GFNTask):
    """Sets up a task where the reward is computed using a proxy for the binding energy of a molecule to
    Soluble Epoxide Hydrolases.

    The proxy is pretrained, and obtained from the original GFlowNet paper, see `gflownet.models.bengio2021flow`.

    This setup essentially reproduces the results of the Trajectory Balance paper when using the TB
    objective, or of the original paper when using Flow Matching.
    """

    def __init__(
        self,
        dataset: Dataset,
        cfg: Config,
        rng: np.random.Generator = None,
        wrap_model: Callable[[nn.Module], nn.Module] = None,
    ):
        self._wrap_model = wrap_model
        self.rng = rng
        self.cfg = cfg
        self.dataset = dataset
        self.temperature_conditional = TemperatureConditional(cfg, rng)
        self.num_cond_dim = self.temperature_conditional.encoding_size()

        self.pdb_id = "14gs_A"
        self.pocket_x, self.pocket_y, self.pocket_z = (
            24.261727272727274,
            6.685454545454546,
            27.79195454545455,
        )
        self.rec_path = os.path.join(
            "dataset/crossdocktest_pdbqt", f"{self.pdb_id}_rec.pdbqt"
        )
        self.generated_store = GeneratedStore(cfg, self.pdb_id)

    def sample_conditional_information(
        self, n: int, train_it: int
    ) -> Dict[str, Tensor]:
        pharmacophore_idxs = [0 for _ in range(n)]
        cond_info = self.temperature_conditional.sample(n)
        cond_info["pharmacophore"] = torch.as_tensor(pharmacophore_idxs)
        return cond_info

    def cond_info_to_logreward(
        self, cond_info: Dict[str, Tensor], flat_reward: FlatRewards
    ) -> RewardScalar:
        return RewardScalar(
            self.temperature_conditional.transform(cond_info, flat_reward)
        )

    def compute_docking_score(
        self,
        mols: List[RDMol],
    ) -> Tensor:
        smiles_list = [Chem.MolToSmiles(mol) for mol in mols]
        docking_scores = unidock_scores(
            smiles_list, self.rec_path, self.pocket_x, self.pocket_y, self.pocket_z
        )
        return torch.as_tensor(docking_scores)

    def compute_flat_rewards(
        self,
        mols: List[RDMol],
        pharmacophore_ids: Tensor,
    ) -> Tuple[FlatRewards, Tensor]:
        is_valid = torch.tensor([i is not None for i in mols]).bool()
        if not is_valid.any():
            return FlatRewards(torch.zeros((0, 1))), is_valid

        mols = [m for m, v in zip(mols, is_valid) if v]

        qeds = torch.as_tensor([Descriptors.qed(mol) for mol in mols])
        qed_reward = (
            qeds.clip(0.0, self.cfg.task.pharmaco_frag.max_qed_reward)
            / self.cfg.task.pharmaco_frag.max_qed_reward
        )
        sas = torch.as_tensor([(10 - sascore.calculateScore(mol)) / 9 for mol in mols])
        sa_reward = (
            sas.clip(0.0, self.cfg.task.pharmaco_frag.max_sa_reward)
            / self.cfg.task.pharmaco_frag.max_sa_reward
        )

        # 1 until 300 then linear decay to 0 until 1000
        mw = torch.as_tensor(
            [((300 - Descriptors.MolWt(mol)) / 700 + 1) for mol in mols]
        ).clip(0, 1)

        d = float(np.mean(molecules.compute_diversity(mols)))
        diversity = torch.as_tensor([d for _ in range(len(mols))])

        reward = torch.ones(len(mols))
        preds = torch.zeros(len(mols))
        
        if "docking" in self.cfg.task.pharmaco_frag.objectives:
            preds = self.compute_docking_score(mols)

            preds[preds.isnan()] = 0
            affinity_reward = (preds / -8.0 / 2).clip(0, 1)

            if self.cfg.task.pharmaco_frag.mol_adj != 0:
                mol_atom_count = [m.GetNumHeavyAtoms() for m in mols]
                mol_adj = torch.tensor(
                    [1 / c ** (self.cfg.task.pharmaco_frag.mol_adj) for c in mol_atom_count]
                )
                affinity_reward *= mol_adj
            self.generated_store.push(mols, preds)
            
            reward *= affinity_reward
        if "qed" in self.cfg.task.pharmaco_frag.objectives:
            reward *= qed_reward
        if "sa" in self.cfg.task.pharmaco_frag.objectives:
            reward *= sa_reward
        if "mw" in self.cfg.task.pharmaco_frag.objectives:
            reward *= mw
        
        reward = affinity_reward * self.cfg.task.pharmaco_frag.reward_multiplier
        reward = reward.clip(1e-4, 100).reshape((-1, 1))

        infos = {
            "docking_score": preds,
            "qed": qeds,
            "sa": sas,
            "mw": mw,
            "diversity": diversity,
        }

        return (FlatRewards(reward), is_valid, infos)


class UniDockSinglePocketTrainer(StandardOnlineTrainer):

    def set_default_hps(self, cfg: Config):
        cfg.hostname = socket.gethostname()
        cfg.pickle_mp_messages = False
        cfg.num_workers = 1
        cfg.opt.learning_rate = 1e-4
        cfg.opt.weight_decay = 1e-8
        cfg.opt.momentum = 0.9
        cfg.opt.adam_eps = 1e-8
        cfg.opt.lr_decay = 20_000
        cfg.opt.clip_grad_type = "norm"
        cfg.opt.clip_grad_param = 10
        cfg.algo.global_batch_size = 64
        cfg.algo.offline_ratio = 0
        cfg.model.num_emb = 128
        cfg.model.num_layers = 4

        cfg.algo.method = "TB"
        cfg.algo.max_nodes = 9
        cfg.algo.sampling_tau = 0.9
        cfg.algo.illegal_action_logreward = -75
        cfg.algo.train_random_action_prob = 0.01
        cfg.algo.valid_random_action_prob = 0.0
        cfg.algo.valid_offline_ratio = 0
        cfg.algo.tb.epsilon = None
        cfg.algo.tb.bootstrap_own_reward = False
        cfg.algo.tb.Z_learning_rate = 1e-3
        cfg.algo.tb.Z_lr_decay = 50_000
        cfg.algo.tb.do_parameterize_p_b = False
        cfg.algo.tb.do_sample_p_b = True

        cfg.replay.use = False
        cfg.replay.capacity = 10_000
        cfg.replay.warmup = 1_000

    def setup_task(self):
        self.task = UniDockSinglePocketTask(
            dataset=self.training_data,
            cfg=self.cfg,
            rng=self.rng,
            wrap_model=self._wrap_for_mp,
        )

    def setup_env_context(self):
        self.ctx = FragMolBuildingEnvContext(
            max_frags=self.cfg.algo.max_nodes, num_cond_dim=self.task.num_cond_dim
        )


def main():
    """Example of how this trainer can be run"""
    hps = {
        "log_dir": "./logs/qed_sa_pretrain",
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
                "reward_multiplier": 1.0,
                "max_qed_reward": 0.5,
                "max_sa_reward": 0.7,
                "objectives": ["qed", "sa"],
                "mol_adj": 0.33,
            }
        },
    }
    if os.path.exists(hps["log_dir"]):
        if hps["overwrite_existing_exp"]:
            shutil.rmtree(hps["log_dir"])
        else:
            raise ValueError(
                f"Log dir {hps['log_dir']} already exists. Set overwrite_existing_exp=True to delete it."
            )
    os.makedirs(hps["log_dir"])

    trial = UniDockSinglePocketTrainer(hps)
    trial.print_every = 1
    trial.run()


if __name__ == "__main__":
    main()
