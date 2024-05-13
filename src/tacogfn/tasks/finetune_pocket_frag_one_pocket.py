import json
import os
import shutil
import sys
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from absl import flags
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdchem import Mol as RDMol
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from src.tacogfn.config import Config
from src.tacogfn.models import pharmaco_cond_graph_transformer
from src.tacogfn.tasks.pocket_frag import PharmacophoreTrainer
from src.tacogfn.tasks.utils import GeneratedStore
from src.tacogfn.trainer import FlatRewards, GFNTask, RewardScalar
from src.tacogfn.utils import molecules, sascore
from src.tacogfn.utils.conditioning import TemperatureConditional
from src.tacogfn.utils.unidock import unidock_scores


class UniDockFinetuneTask(GFNTask):
    def __init__(
        self,
        dataset: Dataset,
        pharmaco_dataset,
        cfg: Config,
        rng: np.random.Generator = None,
        wrap_model: Callable[[nn.Module], nn.Module] = None,
    ):
        self._wrap_model = wrap_model
        self.rng = rng
        self.cfg = cfg
        self.dataset = dataset
        self.pharmaco_dataset = pharmaco_dataset

        self.pocket_idx = self.cfg.task.finetune.pocket_index
        self.pdb_id = self.pharmaco_dataset.get_keys_from_idxs([self.pocket_idx])[0]
        print(f"Finetuning on pocket {self.pdb_id}")

        self.pocket_x, self.pocket_y, self.pocket_z = torch.load(
            self.cfg.pocket_to_centroid
        )[self.pdb_id]
        self.rec_path = os.path.join(self.cfg.pdbqt_folder, f"{self.pdb_id}_rec.pdbqt")

        self.temperature_conditional = TemperatureConditional(cfg, rng)
        self.num_cond_dim = self.temperature_conditional.encoding_size()

        self.generated_store = GeneratedStore(cfg, self.pdb_id)

    def flat_reward_transform(self, y: Union[float, Tensor]) -> FlatRewards:
        return FlatRewards(torch.as_tensor(y))

    def sample_conditional_information(
        self,
        n: int,
        train_it: int,
        partition: str,
    ) -> Dict[str, Tensor]:
        pharmacophore_idxs = [self.pocket_idx for _ in range(n)]

        if partition == "train":
            cond_info = self.temperature_conditional.sample(n)
        else:
            beta = (
                np.array(self.temperature_conditional.cfg.dist_params[1])
                .repeat(n)
                .astype(np.float32)
            )
            beta_enc = torch.zeros(
                (n, self.temperature_conditional.cfg.num_thermometer_dim)
            )
            conf_info = {"beta": torch.tensor(beta), "encoding": beta_enc}

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
        add_to_store: bool = True,
    ) -> Tuple[FlatRewards, Tensor, Dict[str, Tensor]]:
        is_valid = torch.tensor([i is not None for i in mols]).bool()
        if not is_valid.any():
            return FlatRewards(torch.zeros((0, 1))), is_valid

        mols = [m for m, v in zip(mols, is_valid) if v]

        pharmacophore_ids = pharmacophore_ids[is_valid]
        preds = self.compute_docking_score(mols)

        preds[preds.isnan()] = 0
        affinity_reward = (preds / -8.0 / 2).clip(
            0, 1
        ) ** self.cfg.task.pharmaco_frag.docking_score_exp

        if self.cfg.task.pharmaco_frag.mol_adj != 0:
            mol_atom_count = [m.GetNumHeavyAtoms() for m in mols]
            mol_adj = torch.tensor(
                [1 / c ** (self.cfg.task.pharmaco_frag.mol_adj) for c in mol_atom_count]
            )
            affinity_reward *= mol_adj

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

        reward = affinity_reward * self.cfg.task.pharmaco_frag.reward_multiplier
        if "qed" in self.cfg.task.pharmaco_frag.objectives:
            reward *= qed_reward
        if "sa" in self.cfg.task.pharmaco_frag.objectives:
            reward *= sa_reward
        if "mw" in self.cfg.task.pharmaco_frag.objectives:
            reward *= mw

        reward = self.flat_reward_transform(reward).clip(1e-4, 100).reshape((-1, 1))

        infos = {
            "docking_score": preds,
            "top_100": torch.as_tensor(
                [self.generated_store.get_top_avg(100)] * len(mols)
            ),
            "qed": qeds,
            "sa": sas,
            "mw": mw,
            "diversity": diversity,
        }

        self.generated_store.push(mols, preds)
        return (FlatRewards(reward), is_valid, infos)


""" 
python src/tacogfn/tasks/finetune_pocket_frag_one_pocket.py \
    --model_path logs/20240504-crossdocked-mo-256-pocket_graph-adj_ds/model_state_9000.pt \
    --pocket_index 2 
"""


class UniDockFinetuneTrainer(PharmacophoreTrainer):

    def setup_task(self):
        self.task = UniDockFinetuneTask(
            dataset=self.training_data,
            pharmaco_dataset=self.pharmaco_db,
            cfg=self.cfg,
            rng=self.rng,
            wrap_model=self._wrap_for_mp,
        )

    def setup_model(self):
        self.model = (
            pharmaco_cond_graph_transformer.SinglePocketConditionalGraphTransformerGFN(
                self.ctx,
                self.cfg,
                do_bck=self.cfg.algo.tb.do_parameterize_p_b,
            )
        )

    def setup(self):
        super().setup()

    def _save_state(self, it):
        super()._save_state(it)


def main():
    """Example of how this model can be run."""
    _MODEL_PATH = flags.DEFINE_string(
        "model_path", None, "Path to the model state file.", required=True
    )

    _POCKET_INDEX = flags.DEFINE_integer(
        "pocket_index",
        None,
        "Index of the pocket to finetune the model on.",
        required=True,
    )
    _BATCH_SIZE = flags.DEFINE_integer(
        "batch_size",
        64,
        "Batch size for the model.",
    )
    _REC_FOLDER = flags.DEFINE_string(
        "rec_folder",
        "dataset/crossdocktest_pdbqt",
        "Path to the folder containing the receptor files.",
    )
    _POCKET_TO_CENTRIOD_PATH = flags.DEFINE_string(
        "pocket_to_centroid_path",
        "dataset/pocket_to_centroid.pt",
        "Path to the pocket to centroid file.",
    )
    _LOG_DIR = flags.DEFINE_string(
        "log_dir",
        "logs/finetune",
        "Path to the log directory.",
    )

    flags.FLAGS(sys.argv)
    model_state = torch.load(_MODEL_PATH.value)
    hps = dict(model_state["cfg"])

    hps.update(
        {
            "log_dir": f"{_LOG_DIR.value}/{_POCKET_INDEX.value}",
            "num_workers": 0,
            "validate_every": 0,
            "checkpoint_every": 100,
            "num_training_steps": 400,
            "pdbqt_folder": _REC_FOLDER.value,
            "pocket_to_centroid": _POCKET_TO_CENTRIOD_PATH.value,
            "task": {
                **hps["task"],
                "finetune": {
                    "pocket_index": _POCKET_INDEX.value,
                },
                "pharmaco_frag": {
                    **hps["task"]["pharmaco_frag"],
                    "docking_score_exp": 1.5,
                    "reward_multiplier": 5,
                },
            },
            # "replay": {
            #     "use": False,
            #     "keep_top": True,
            #     "capacity": 2500,
            #     "warmup": 1000,
            # },
            "algo": {**hps["algo"], "global_batch_size": _BATCH_SIZE.value},
            "cond": {
                "temperature": {"sample_dist": "uniform", "dist_params": [32.0, 64.0]}
            },
        }
    )

    if os.path.exists(hps["log_dir"]):
        if hps["overwrite_existing_exp"]:
            shutil.rmtree(hps["log_dir"])
        else:
            raise ValueError(
                f"Log dir {hps['log_dir']} already exists. Set overwrite_existing_exp=True to delete it."
            )
    os.makedirs(hps["log_dir"])

    trial = UniDockFinetuneTrainer(hps)

    trial.model.load_state_dict(model_state["models_state_dict"][0])
    trial.sampling_model.load_state_dict(model_state["models_state_dict"][0])
    # trail.opt.load_state_dict(model_state["optimizer_state_dict"][0])
    # trail.opt_Z.load_state_dict(model_state["optimizer_state_dict"][1])
    # trail.lr_sched.load_state_dict(model_state["scheduler_state_dict"][0])
    # trail.lr_sched_Z.load_state_dict(model_state["scheduler_state_dict"][1])

    trial.print_every = 1
    trial.run()


if __name__ == "__main__":
    main()
