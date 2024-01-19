import os
import pathlib
import shutil
import socket
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch_geometric.loader as gl
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdchem import Mol as RDMol
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from src.tacogfn.algo.trajectory_balance import PharmacophoreTrajectoryBalance
from src.tacogfn.config import Config
from src.tacogfn.data.pharmacophore import (
    PharmacoDB,
    PharmacophoreGraphDataset,
    PharmacophoreModel,
)
from src.tacogfn.data.sampling_iterator import PharmacoCondSamplingIterator
from src.tacogfn.envs import frag_mol_env
from src.tacogfn.eval.models.baseline import BaseAffinityPrediction
from src.tacogfn.models import bengio2021flow, pharmaco_cond_graph_transformer
from src.tacogfn.online_trainer import StandardOnlineTrainer
from src.tacogfn.trainer import FlatRewards, GFNTask, RewardScalar
from src.tacogfn.utils import molecules, sascore
from src.tacogfn.utils.conditioning import TemperatureConditional


class PharmacophoreTask(GFNTask):
    """
    Sets up a task where the reward is computed using a proxy for binding energy
    based on the molecular graph and the pharmacophore model of the target.

    - Non Multi-Objective
    - Pharmacophore represented as an embedding vector
    - Vanilla fragment based molecular construction environment
    """

    def __init__(
        self,
        dataset: Dataset,
        pharmacophore_dataset: PharmacophoreGraphDataset,
        cfg: Config,
        rng: np.random.Generator = None,
        wrap_model: Callable[[nn.Module], nn.Module] = None,
    ):
        self._wrap_model = wrap_model
        self.rng = rng
        self.cfg = cfg
        self.dataset = dataset
        self.pharmacophore_dataset = pharmacophore_dataset

        self.temperature_conditional = TemperatureConditional(cfg, rng)
        self.num_cond_dim = self.temperature_conditional.encoding_size()

        self.models = self._load_task_models()

    def _load_task_models(self):
        if self.cfg.task.pharmaco_frag.affinity_predictor == "alpha":
            from src.scoring.scoring_module import AffinityPredictor

            self.affinity_model = AffinityPredictor(
                self.cfg.affinity_predictor_path, "cpu"
            )

            return {}

        elif self.cfg.task.pharmaco_frag.affinity_predictor == "beta":
            from molfeat.trans.pretrained import PretrainedDGLTransformer

            from src.tacogfn.models.beta_docking_score_predictor import (
                DockingScorePredictionModel,
            )

            self.molecule_featurizer = PretrainedDGLTransformer(
                kind="gin_supervised_contextpred", dtype=float
            )
            model_state = torch.load(self.cfg.affinity_predictor_path)
            model = DockingScorePredictionModel(
                hidden_dim=model_state["hps"]["hidden_dim"]
            )
            model.load_state_dict(model_state["models_state_dict"][0])
            model.eval()

            model, self.device = self._wrap_model(model, send_to_device=True)
            return {"affinity": model}

    def flat_reward_transform(self, y: Union[float, Tensor]) -> FlatRewards:
        return FlatRewards(torch.as_tensor(y))

    def sample_conditional_information(
        self,
        n: int,
        train_it: int,
        partition: str,
    ) -> Dict[str, Tensor]:
        pharmacophore_idxs = self.pharmacophore_dataset.sample_pharmacophore_idx(
            n, partition=partition
        )
        cond_info = self.temperature_conditional.sample(n)
        cond_info["pharmacophore"] = torch.as_tensor(pharmacophore_idxs)
        return cond_info

    def cond_info_to_logreward(
        self, cond_info: Dict[str, Tensor], flat_reward: FlatRewards
    ) -> RewardScalar:
        return RewardScalar(
            self.temperature_conditional.transform(cond_info, flat_reward)
        )

    def prepare_beta_batch(
        self,
        smiles_list: list[str],
        pharmacophore_list: list[PharmacophoreGraphDataset],
    ):
        if self.cfg.task.pharmaco_frag.affinity_predictor == "beta":
            # Featurize ligands and batch
            ligand_features = torch.tensor(self.molecule_featurizer(smiles_list))
            data_loader = gl.DataLoader(
                [
                    {
                        "pharmacophore": p,
                        "ligand_features": l,
                    }
                    for p, l in zip(pharmacophore_list, ligand_features)
                ],
                batch_size=self.cfg.algo.global_batch_size,
                shuffle=False,
            )

            return iter(data_loader)
        else:
            raise NotImplementedError()

    def predict_docking_score(
        self,
        mols: List[RDMol],
        pharmacophore_ids: Tensor,
    ) -> Tensor:
        smi_list = [Chem.MolToSmiles(mol) for mol in mols]

        if self.cfg.task.pharmaco_frag.affinity_predictor == "alpha":
            pdb_ids = self.pharmacophore_dataset.get_keys_from_idxs(
                pharmacophore_ids.tolist()
            )
            preds = []
            for smi, pdb_id in zip(smi_list, pdb_ids):
                try:
                    cache = torch.load(
                        f"dataset/docking_pharmacophores/{pdb_id}_rec.pt",
                        map_location="cpu",
                    )
                    preds.append(self.affinity_model.scoring(cache, smi))
                except FileNotFoundError:
                    print(f"Could not find pharmacophore for {pdb_id}")
                    preds.append(np.nan)
                except Exception as e:
                    print(e)
                    print(smi)
                    preds.append(np.nan)

            return torch.as_tensor(preds)

        elif self.cfg.task.pharmaco_frag.affinity_predictor == "beta":
            pharmacophore_list = [
                self.pharmacophore_dataset.get_pharmacophore_data_from_idx(p_id)
                for i, p_id in enumerate(pharmacophore_ids.tolist())
            ]

            all_preds = []
            for batch in self.prepare_beta_batch(smi_list, pharmacophore_list):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                preds = self.models["affinity"](batch).reshape((-1,)).cpu().detach()
                all_preds.append(preds)

            preds = torch.cat(all_preds)
            return preds

    def compute_flat_rewards(
        self,
        mols: List[RDMol],
        pharmacophore_ids: Tensor,
    ) -> Tuple[FlatRewards, Tensor, Dict[str, Tensor]]:
        is_valid = torch.tensor([i is not None for i in mols]).bool()
        if not is_valid.any():
            return FlatRewards(torch.zeros((0, 1))), is_valid

        mols = [m for m, v in zip(mols, is_valid) if v]
        pharmacophore_ids = pharmacophore_ids[is_valid]

        preds = self.predict_docking_score(mols, pharmacophore_ids)

        preds[preds.isnan()] = 0
        affinity_reward = (preds - self.cfg.task.pharmaco_frag.min_docking_score).clip(
            -15, 0
        ) + torch.max(
            preds, self.cfg.task.pharmaco_frag.min_docking_score
        ) * self.cfg.task.pharmaco_frag.leaky_coefficient  # leaky reward up to min_docking_score
        affinity_reward *= -1 / 10.0  # normalize reward to be in range [0, 1]
        affinity_reward = affinity_reward.clip(0, 1)

        # 1 for qed above 0.7, linear decay to 0 from 0.7 to 0.0
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
        return (
            FlatRewards(reward),
            is_valid,
            {
                "docking_score": preds,
                "qed": qeds,
                "sa": sas,
                "mw": mw,
                "diversity": diversity,
            },
        )


class PharmacophoreTrainer(StandardOnlineTrainer):
    task: PharmacophoreTask

    def sample_molecules(
        self,
        pharmacophore_idxs: list[int],
        temperatures: Optional[list[float]] = None,
        sample_temp: float = 1.0,
    ) -> List[RDMol]:
        n = len(pharmacophore_idxs)

        if temperatures is None:
            temperatures = (
                torch.rand(n) * self.cfg.cond.temperature.dist_params[1] * 1 / 4
                + torch.ones(n) * self.cfg.cond.temperature.dist_params[1] * 3 / 4
            )  # default to sampling from the upper 3/4 of the temperature range

        cond_info = {
            "encoding": self.task.temperature_conditional.encode(temperatures),
            "pharmacophore": torch.as_tensor(pharmacophore_idxs),
        }

        # HACK: change the sample temp for inference
        self.algo.graph_sampler.sample_temp = sample_temp
        with torch.no_grad():
            trajs = self.algo.create_training_data_from_own_samples(
                model=self.model, n=n, cond_info=cond_info, random_action_prob=0.0
            )
        # HACK: reset the sample temp
        self.algo.graph_sampler.sample_temp = 1.0

        mols = [self.ctx.graph_to_mol(traj["result"]) for traj in trajs]
        return mols

    def setup_model(self):
        self.model = (
            pharmaco_cond_graph_transformer.PharmacophoreConditionalGraphTransformerGFN(
                self.ctx,
                self.cfg,
                do_bck=self.cfg.algo.tb.do_parameterize_p_b,
            )
        )

    def setup_algo(self):
        self.algo = PharmacophoreTrajectoryBalance(
            self.env,
            self.ctx,
            self.pharmaco_db,
            self.rng,
            self.cfg,
        )

    def set_default_hps(self, cfg: Config):
        cfg.hostname = socket.gethostname()
        cfg.pickle_mp_messages = False
        cfg.num_workers = 8
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
        cfg.model.pharmaco_cond.pharmaco_dim = 64

        cfg.algo.method = "TB"
        cfg.algo.max_nodes = 9
        cfg.algo.sampling_tau = 0.9
        cfg.algo.illegal_action_logreward = -75
        cfg.algo.train_random_action_prob = 0.01  # 0.0
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

    def setup_env_context(self):
        self.ctx = frag_mol_env.FragMolBuildingEnvContext(
            max_frags=self.cfg.algo.max_nodes, num_cond_dim=self.task.num_cond_dim
        )

    def setup_task(self):
        self.task = PharmacophoreTask(
            dataset=self.training_data,
            pharmacophore_dataset=self.pharmaco_db,
            cfg=self.cfg,
            rng=self.rng,
            wrap_model=self._wrap_for_mp,
        )

    def build_training_data_loader(self) -> DataLoader:
        model, dev = self._wrap_for_mp(self.sampling_model, send_to_device=True)
        replay_buffer, _ = self._wrap_for_mp(self.replay_buffer, send_to_device=False)
        iterator = PharmacoCondSamplingIterator(
            self.training_data,
            model,
            self.ctx,
            self.algo,
            self.task,
            dev,
            batch_size=self.cfg.algo.global_batch_size,
            illegal_action_logreward=self.cfg.algo.illegal_action_logreward,
            replay_buffer=replay_buffer,
            ratio=self.cfg.algo.offline_ratio,
            log_dir=str(pathlib.Path(self.cfg.log_dir) / "train"),
            random_action_prob=self.cfg.algo.train_random_action_prob,
            hindsight_ratio=self.cfg.replay.hindsight_ratio,
            partition="train",
        )
        for hook in self.sampling_hooks:
            iterator.add_log_hook(hook)
        return torch.utils.data.DataLoader(
            iterator,
            batch_size=None,
            num_workers=self.cfg.num_workers,
            persistent_workers=self.cfg.num_workers > 0,
            # The 2 here is an odd quirk of torch 1.10, it is fixed and
            # replaced by None in torch 2.
            prefetch_factor=1 if self.cfg.num_workers else 2,
        )

    def build_validation_data_loader(self) -> DataLoader:
        model, dev = self._wrap_for_mp(self.model, send_to_device=True)
        iterator = PharmacoCondSamplingIterator(
            self.test_data,
            model,
            self.ctx,
            self.algo,
            self.task,
            dev,
            batch_size=self.cfg.algo.global_batch_size,
            illegal_action_logreward=self.cfg.algo.illegal_action_logreward,
            ratio=self.cfg.algo.valid_offline_ratio,
            log_dir=str(pathlib.Path(self.cfg.log_dir) / "valid"),
            sample_cond_info=self.cfg.algo.valid_sample_cond_info,
            stream=False,
            random_action_prob=self.cfg.algo.valid_random_action_prob,
            partition="test",
        )
        for hook in self.valid_sampling_hooks:
            iterator.add_log_hook(hook)
        return torch.utils.data.DataLoader(
            iterator,
            batch_size=None,
            num_workers=self.cfg.num_workers,
            persistent_workers=self.cfg.num_workers > 0,
            prefetch_factor=1 if self.cfg.num_workers else 2,
        )

    def build_final_data_loader(self) -> DataLoader:
        model, dev = self._wrap_for_mp(self.sampling_model, send_to_device=True)
        iterator = PharmacoCondSamplingIterator(
            self.training_data,
            model,
            self.ctx,
            self.algo,
            self.task,
            dev,
            batch_size=self.cfg.algo.global_batch_size,
            illegal_action_logreward=self.cfg.algo.illegal_action_logreward,
            replay_buffer=None,
            ratio=0.0,
            log_dir=os.path.join(self.cfg.log_dir, "final"),
            random_action_prob=0.0,
            hindsight_ratio=0.0,
            init_train_iter=self.cfg.num_training_steps,
            partition="train",
        )
        for hook in self.sampling_hooks:
            iterator.add_log_hook(hook)
        return torch.utils.data.DataLoader(
            iterator,
            batch_size=None,
            num_workers=self.cfg.num_workers,
            persistent_workers=self.cfg.num_workers > 0,
            prefetch_factor=1 if self.cfg.num_workers else 2,
        )

    def setup_pharamaco_dataset(self):
        split_file = torch.load(self.cfg.split_file)
        tuple_to_pharmaco_id = lambda t: t[0].split("/")[-1].split("_rec")[0]

        train_ids = [tuple_to_pharmaco_id(t) for t in split_file["train"]]
        test_ids = [tuple_to_pharmaco_id(t) for t in split_file["test"]]

        return PharmacoDB(
            self.cfg.pharmacophore_db_path,
            {"train": train_ids, "test": test_ids},
            rng=np.random.default_rng(142857),
            verbose=True,
        )

    def setup(self):
        self.pharmaco_db = self.setup_pharamaco_dataset()
        super().setup()


def main():
    """Example of how this model can be run."""
    hps = {
        "log_dir": "./logs/20240118-alpha-default-0.6-qed-sa-limit-5.0-docking-cutoff-0.2-leaky",
        "split_file": "dataset/split_by_name.pt",
        "affinity_predictor_path": "model_weights/base_100_per_pocket.pth",
        "pharmacophore_db_path": "misc/pharmacophores_db.lmdb",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "overwrite_existing_exp": True,
        "num_training_steps": 50_000,
        "num_workers": 0,
        "opt": {
            "lr_decay": 20000,
        },
        "algo": {
            "sampling_tau": 0.99,
            "offline_ratio": 0.0,
            "max_nodes": 9,
            "train_random_action_prob": 0.05,
        },
        "cond": {
            "temperature": {
                "sample_dist": "uniform",
                "dist_params": [0, 64.0],
            }
        },
        "task": {
            "pharmaco_frag": {
                "fragment_type": "zinc250k_50cutoff_brics",
                "affinity_predictor": "alpha",
                "min_docking_score": -5.0,  # no reward below this
                "leaky_coefficient": 0.2,
                "reward_multiplier": 2.0,
                "max_qed_reward": 0.6,  # no extra reward for qed above this
                "max_sa_reward": 0.6,  # no extra reward for sa above this
                "objectives": ["docking", "qed", "sa"],
            },
        },
        "model": {
            "pharmaco_cond": {
                "pharmaco_dim": 128,
            },
            "num_emb": 256,
            "num_layers": 2,
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

    trial = PharmacophoreTrainer(hps)
    trial.print_every = 1
    trial.run()


if __name__ == "__main__":
    main()
