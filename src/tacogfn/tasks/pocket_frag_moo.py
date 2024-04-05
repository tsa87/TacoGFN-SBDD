import json
import os
import pathlib
import shutil
import sys
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch_geometric.data as gd
from absl import flags
from rdkit.Chem import QED, Descriptors
from rdkit.Chem.rdchem import Mol as RDMol
from torch import Tensor
from torch.utils.data import Dataset

from src.tacogfn.config import Config
from src.tacogfn.data.pharmacophore import (
    PharmacoDB,
    PharmacophoreGraphDataset,
    PharmacophoreModel,
)
from src.tacogfn.envs.frag_mol_env import FragMolBuildingEnvContext
from src.tacogfn.models import bengio2021flow
from src.tacogfn.tasks.pharmaco_frag import PharmacophoreTask, PharmacophoreTrainer
from src.tacogfn.tasks.seh_frag import SEHFragTrainer, SEHTask
from src.tacogfn.trainer import FlatRewards, GFNTask, RewardScalar
from src.tacogfn.utils import metrics, molecules, sascore
from src.tacogfn.utils.conditioning import (
    FocusRegionConditional,
    MultiObjectiveWeightedPreferences,
)


class PocketMOOTask(PharmacophoreTask):

    def __init__(
        self,
        dataset: Dataset,
        pharmaco_dataset: PharmacophoreGraphDataset,
        cfg: Config,
        rng: np.random.Generator = None,
        wrap_model: Callable[[nn.Module], nn.Module] = None,
    ):
        super().__init__(dataset, pharmaco_dataset, cfg, rng, wrap_model)

        mcfg = self.cfg.task.pocket_moo
        self.objectives = ["ds", "qed", "sa"]

        self.cfg.cond.focus_region.focus_type = None
        self.focus_cond = None

        self.pref_cond = MultiObjectiveWeightedPreferences(self.cfg)
        self.temperature_sample_dist = cfg.cond.temperature.sample_dist
        self.temperature_dist_params = cfg.cond.temperature.dist_params
        self.num_thermometer_dim = cfg.cond.temperature.num_thermometer_dim
        self.num_cond_dim = (
            self.temperature_conditional.encoding_size()
            + self.pref_cond.encoding_size()
            + (self.focus_cond.encoding_size() if self.focus_cond is not None else 0)
        )
        assert len(self.objectives) == len(set(self.objectives))

    def sample_conditional_information(
        self,
        n: int,
        train_it: int,
        partition: str,
    ) -> Dict[str, Tensor]:
        cond_info = super().sample_conditional_information(n, train_it, partition)
        pref_ci = self.pref_cond.sample(n)
        focus_ci = (
            self.focus_cond.sample(n, train_it)
            if self.focus_cond is not None
            else {"encoding": torch.zeros(n, 0)}
        )
        cond_info = {
            **cond_info,
            **pref_ci,
            **focus_ci,
            "encoding": torch.cat(
                [cond_info["encoding"], pref_ci["encoding"], focus_ci["encoding"]],
                dim=1,
            ),
        }
        return cond_info

    def encode_conditional_information(self, steer_info: Tensor) -> Dict[str, Tensor]:
        """
        Encode conditional information at validation-time
        We use the maximum temperature beta for inference
        Args:
            steer_info: Tensor of shape (Batch, 2 * n_objectives) containing the preferences and focus_dirs
            in that order
        Returns:
            Dict[str, Tensor]: Dictionary containing the encoded conditional information
        """
        n = len(steer_info)
        if self.temperature_sample_dist == "constant":
            beta = torch.ones(n) * self.temperature_dist_params[0]
            beta_enc = torch.zeros((n, self.num_thermometer_dim))
        else:
            beta = torch.ones(n) * self.temperature_dist_params[-1]
            beta_enc = torch.ones((n, self.num_thermometer_dim))

        assert (
            len(beta.shape) == 1
        ), f"beta should be of shape (Batch,), got: {beta.shape}"

        # TODO: positional assumption here, should have something cleaner
        preferences = steer_info[:, : len(self.objectives)].float()
        focus_dir = steer_info[:, len(self.objectives) :].float()

        preferences_enc = self.pref_cond.encode(preferences)
        if self.focus_cond is not None:
            focus_enc = self.focus_cond.encode(focus_dir)
            encoding = torch.cat([beta_enc, preferences_enc, focus_enc], 1).float()
        else:
            encoding = torch.cat([beta_enc, preferences_enc], 1).float()
        return {
            "beta": beta,
            "encoding": encoding,
            "preferences": preferences,
            "focus_dir": focus_dir,
        }

    def relabel_condinfo_and_logrewards(
        self,
        cond_info: Dict[str, Tensor],
        log_rewards: Tensor,
        flat_rewards: FlatRewards,
        hindsight_idxs: Tensor,
    ):
        # TODO: we seem to be relabeling tensors in place, could that cause a problem?
        if self.focus_cond is None:
            raise NotImplementedError(
                "Hindsight relabeling only implemented for focus conditioning"
            )
        if self.focus_cond.cfg.focus_type is None:
            return cond_info, log_rewards
        # only keep hindsight_idxs that actually correspond to a violated constraint
        _, in_focus_mask = metrics.compute_focus_coef(
            flat_rewards, cond_info["focus_dir"], self.focus_cond.cfg.focus_cosim
        )
        out_focus_mask = torch.logical_not(in_focus_mask)
        hindsight_idxs = hindsight_idxs[out_focus_mask[hindsight_idxs]]

        # relabels the focus_dirs and log_rewards
        cond_info["focus_dir"][hindsight_idxs] = nn.functional.normalize(
            flat_rewards[hindsight_idxs], dim=1
        )

        preferences_enc = self.pref_cond.encode(cond_info["preferences"])
        focus_enc = self.focus_cond.encode(cond_info["focus_dir"])
        cond_info["encoding"] = torch.cat(
            [
                cond_info["encoding"][:, : self.num_thermometer_dim],
                preferences_enc,
                focus_enc,
            ],
            1,
        )

        log_rewards = self.cond_info_to_logreward(cond_info, flat_rewards)
        return cond_info, log_rewards

    def cond_info_to_logreward(
        self, cond_info: Dict[str, Tensor], flat_reward: FlatRewards
    ) -> RewardScalar:
        if isinstance(flat_reward, list):
            if isinstance(flat_reward[0], Tensor):
                flat_reward = torch.stack(flat_reward)
            else:
                flat_reward = torch.tensor(flat_reward)

        scalarized_reward = self.pref_cond.transform(cond_info, flat_reward)
        focused_reward = (
            self.focus_cond.transform(cond_info, flat_reward, scalarized_reward)
            if self.focus_cond is not None
            else scalarized_reward
        )
        tempered_reward = self.temperature_conditional.transform(
            cond_info, focused_reward
        )
        return RewardScalar(tempered_reward)

    def compute_flat_rewards(
        self,
        mols: List[RDMol],
        pharmacophore_ids: Tensor,
    ) -> Tuple[FlatRewards, Tensor, Dict[str, Tensor]]:
        is_valid = torch.tensor([i is not None for i in mols]).bool()
        if not is_valid.any():
            return FlatRewards(torch.zeros((0, len(self.objectives)))), is_valid

        flat_r: List[Tensor] = []

        mols = [m for m, v in zip(mols, is_valid) if v]
        pharmacophore_ids = pharmacophore_ids[is_valid]
        preds = self.predict_docking_score(mols, pharmacophore_ids)

        pdb_ids = self.pharmaco_dataset.get_keys_from_idxs(pharmacophore_ids.tolist())
        avg_preds = torch.as_tensor(
            [
                (
                    min(0, self.avg_prediction_for_pocket[pdb_id])
                    if pdb_id in self.avg_prediction_for_pocket
                    else -7.0
                )
                for pdb_id in pdb_ids
            ],
            dtype=torch.float,
        )
        preds[preds.isnan()] = 0

        affinity_reward = (preds - avg_preds).clip(
            self.cfg.task.pharmaco_frag.max_dock_reward, 0
        ) + torch.max(
            preds, avg_preds
        ) * self.cfg.task.pharmaco_frag.leaky_coefficient  # leaky reward up to avg

        affinity_reward /= (
            self.cfg.task.pharmaco_frag.max_dock_reward
        )  # still normalize reward to be in range [0, 1]
        if self.cfg.task.pharmaco_frag.mol_adj != 0:
            mol_atom_count = [m.GetNumHeavyAtoms() for m in mols]
            mol_adj = torch.tensor(
                [1 / c ** (self.cfg.task.pharmaco_frag.mol_adj) for c in mol_atom_count]
            )
            affinity_reward = affinity_reward * mol_adj * 3

        affinity_reward = affinity_reward.clip(0, 1)
        flat_r.append(affinity_reward)

        # 1 for qed above 0.7, linear decay to 0 from 0.7 to 0.0
        qeds = torch.as_tensor([Descriptors.qed(mol) for mol in mols])

        qed_reward = torch.pow(
            (
                qeds.clip(0.0, self.cfg.task.pharmaco_frag.max_qed_reward)
                / self.cfg.task.pharmaco_frag.max_qed_reward
            ),
            self.cfg.task.pharmaco_frag.qed_exponent,
        )
        flat_r.append(qed_reward)

        sas = torch.as_tensor([(10 - sascore.calculateScore(mol)) / 9 for mol in mols])
        sa_reward = torch.pow(
            (
                sas.clip(0.0, self.cfg.task.pharmaco_frag.max_sa_reward)
                / self.cfg.task.pharmaco_frag.max_sa_reward
            ),
            self.cfg.task.pharmaco_frag.sa_exponent,
        )
        flat_r.append(sa_reward)

        # 1 until 300 then linear decay to 0 until 1000
        mw = torch.as_tensor(
            [((300 - Descriptors.MolWt(mol)) / 700 + 1) for mol in mols]
        ).clip(0, 1)

        d = float(np.mean(molecules.compute_diversity(mols)))
        diversity = torch.as_tensor([d for _ in range(len(mols))])

        reward = torch.stack(flat_r, dim=1)

        infos = {
            "docking_score": preds,
            "qed": qeds,
            "sa": sas,
            "mw": mw,
            "diversity": diversity,
        }
        if self.cfg.info_only_dock_proxy:
            info_preds = self.predict_docking_score(
                mols, pharmacophore_ids, info_only=True
            )
            info_preds[info_preds.isnan()] = 0
            infos["info_only_docking_score"] = info_preds
        else:
            infos["info_only_docking_score"] = preds

        return (FlatRewards(reward), is_valid, infos)


class PocketMOOTrainer(PharmacophoreTrainer):
    task: PocketMOOTask
    ctx: FragMolBuildingEnvContext

    def set_default_hps(self, cfg: Config):
        super().set_default_hps(cfg)
        cfg.algo.sampling_tau = 0.95
        cfg.algo.valid_sample_cond_info = True

    def setup_task(self):
        self.task = PocketMOOTask(
            dataset=self.training_data,
            pharmaco_dataset=self.pharmaco_db,
            cfg=self.cfg,
            rng=self.rng,
            wrap_model=self._wrap_for_mp,
        )

    def setup(self):
        super().setup()

        tcfg = self.cfg.task.pocket_moo
        n_obj = len(tcfg.objectives)

        # making sure hyperparameters for preferences and focus regions are consistent
        if not (
            tcfg.focus_type is None
            or tcfg.focus_type == "centered"
            or (isinstance(tcfg.focus_type, list) and len(tcfg.focus_type) == 1)
        ):
            assert tcfg.preference_type is None, (
                f"Cannot use preferences with multiple focus regions, here focus_type={tcfg.focus_type} "
                f"and preference_type={tcfg.preference_type}"
            )

        if isinstance(tcfg.focus_type, list) and len(tcfg.focus_type) > 1:
            n_valid = len(tcfg.focus_type)
        else:
            n_valid = tcfg.n_valid

        # preference vectors
        if tcfg.preference_type is None:
            valid_preferences = np.ones((n_valid, n_obj))
        elif tcfg.preference_type == "dirichlet":
            valid_preferences = metrics.partition_hypersphere(
                d=n_obj, k=n_valid, normalisation="l1"
            )
        elif tcfg.preference_type == "seeded_single":
            seeded_prefs = np.random.default_rng(142857 + int(self.cfg.seed)).dirichlet(
                [1] * n_obj, n_valid
            )
            valid_preferences = seeded_prefs[0].reshape((1, n_obj))
            self.task.seeded_preference = valid_preferences[0]
        elif tcfg.preference_type == "seeded_many":
            valid_preferences = np.random.default_rng(
                142857 + int(self.cfg.seed)
            ).dirichlet([1] * n_obj, n_valid)
        else:
            raise NotImplementedError(
                f"Unknown preference type {self.cfg.task.seh_moo.preference_type}"
            )

        # TODO: this was previously reported, would be nice to serialize it
        # hps["fixed_focus_dirs"] = (
        #    np.unique(self.task.fixed_focus_dirs, axis=0).tolist() if self.task.fixed_focus_dirs is not None else None
        # )
        if self.task.focus_cond is not None:
            assert self.task.focus_cond.valid_focus_dirs.shape == (
                n_valid,
                n_obj,
            ), (
                "Invalid shape for valid_preferences, "
                f"{self.task.focus_cond.valid_focus_dirs.shape} != ({n_valid}, {n_obj})"
            )

            # combine preferences and focus directions (fixed focus cosim) since they could be used together
            # (not either/or). TODO: this relies on positional assumptions, should have something cleaner
            valid_cond_vector = np.concatenate(
                [valid_preferences, self.task.focus_cond.valid_focus_dirs], axis=1
            )
        else:
            valid_cond_vector = valid_preferences

        # self._top_k_hook = TopKHook(10, tcfg.n_valid_repeats, n_valid)
        self.test_data = RepeatedCondInfoDataset(
            valid_cond_vector, repeat=tcfg.n_valid_repeats
        )
        # self.valid_sampling_hooks.append(self._top_k_hook)

        self.algo.task = self.task

    def build_callbacks(self):
        # We use this class-based setup to be compatible with the DeterminedAI API, but no direct
        # dependency is required.
        parent = self

        class TopKMetricCB:
            def on_validation_end(self, metrics: Dict[str, Any]):
                top_k = parent._top_k_hook.finalize()
                for i in range(len(top_k)):
                    metrics[f"topk_rewards_{i}"] = top_k[i]
                print("validation end", metrics)

        return {"topk": TopKMetricCB()}

    def train_batch(
        self, batch: gd.Batch, epoch_idx: int, batch_idx: int, train_it: int
    ) -> Dict[str, Any]:
        if self.task.focus_cond is not None:
            self.task.focus_cond.step_focus_model(batch, train_it)
        return super().train_batch(batch, epoch_idx, batch_idx, train_it)

    def _save_state(self, it):
        if (
            self.task.focus_cond is not None
            and self.task.focus_cond.focus_model is not None
        ):
            self.task.focus_cond.focus_model.save(pathlib.Path(self.cfg.log_dir))
        return super()._save_state(it)


class RepeatedCondInfoDataset:
    def __init__(self, cond_info_vectors, repeat):
        self.cond_info_vectors = cond_info_vectors
        self.repeat = repeat

    def __len__(self):
        return len(self.cond_info_vectors) * self.repeat

    def __getitem__(self, idx):
        assert 0 <= idx < len(self)
        return torch.tensor(self.cond_info_vectors[int(idx // self.repeat)])


def main():
    """Example of how this model can be run."""
    _HPS_PATH = flags.DEFINE_string(
        "hps_path",
        "hps/crossdocked_mol_256.json",
        "Path to the hyperparameter file.",
    )
    flags.FLAGS(sys.argv)

    with open(_HPS_PATH.value, "r") as f:
        hps = json.load(f)

    if os.path.exists(hps["log_dir"]):
        if hps["overwrite_existing_exp"]:
            shutil.rmtree(hps["log_dir"])
        else:
            raise ValueError(
                f"Log dir {hps['log_dir']} already exists. Set overwrite_existing_exp=True to delete it."
            )
    os.makedirs(hps["log_dir"])

    trial = PocketMOOTrainer(hps)
    trial.print_every = 1
    trial.run()


if __name__ == "__main__":
    main()
