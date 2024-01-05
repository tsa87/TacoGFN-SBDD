import os
import pathlib
import shutil
import socket
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import torch
import torch_geometric.data as gd
from rdkit.Chem.rdchem import Mol as RDMol
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from src.tacogfn.algo.trajectory_balance import PharmacophoreTrajectoryBalance
from src.tacogfn.config import Config
from src.tacogfn.data.pharmacophore import PharmacoDB, PharmacophoreGraphDataset
from src.tacogfn.data.sampling_iterator import PharmacoCondSamplingIterator
from src.tacogfn.envs import frag_mol_env
from src.tacogfn.eval.models.baseline import BaseAffinityPrediction
from src.tacogfn.models import bengio2021flow, pharmaco_cond_graph_transformer
from src.tacogfn.online_trainer import StandardOnlineTrainer
from src.tacogfn.trainer import FlatRewards, GFNTask, RewardScalar
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
        self.models = self._load_task_models()
        self.dataset = dataset
        self.pharmacophore_dataset = pharmacophore_dataset
        self.temperature_conditional = TemperatureConditional(cfg, rng)
        self.num_cond_dim = self.temperature_conditional.encoding_size()

    def _load_task_models(self):
        # TODO: change this to affinity model KAIST is developing
        model = BaseAffinityPrediction(
            self.cfg.model.pharmaco_cond.pharmaco_dim, 71, 4)
        model, self.device = self._wrap_model(model, send_to_device=True)
        return {"affinity": model}

    def flat_reward_transform(self, y: Union[float, Tensor]) -> FlatRewards:
        return FlatRewards(torch.as_tensor(y) / 8)

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

    def compute_flat_rewards(
        self,
        mols: List[RDMol],
        pharmacophore_ids: Tensor,
    ) -> Tuple[FlatRewards, Tensor]:
        graphs = [bengio2021flow.mol2graph(i) for i in mols]
        is_valid = torch.tensor([i is not None for i in graphs]).bool()
        if not is_valid.any():
            return FlatRewards(torch.zeros((0, 1))), is_valid

        mol_datalist = [g for i, g in enumerate(graphs) if is_valid[i]]
        pharmacophore_batch = [
            p
            for i, p in enumerate(
                self.pharmacophore_dataset.get_pharmacophore_datalist_from_idxs(
                    pharmacophore_ids
                )
            )
            if is_valid[i]
        ]

        mol_batch = gd.Batch.from_data_list(mol_datalist)
        pharmacophore_batch = gd.Batch.from_data_list(pharmacophore_batch)

        preds = self.models["affinity"](mol_batch, pharmacophore_batch)
        preds[preds.isnan()] = 0
        preds = self.flat_reward_transform(preds).clip(1e-4, 100).reshape((-1, 1))
        return FlatRewards(preds), is_valid




class PharmacophoreTrainer(StandardOnlineTrainer):
    task: PharmacophoreTask

    def setup_model(self):
        self.model = pharmaco_cond_graph_transformer.PharmacophoreConditionalGraphTransformerGFN(
            self.ctx,
            self.cfg,
            do_bck=self.cfg.algo.tb.do_parameterize_p_b,
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
        cfg.algo.train_random_action_prob = 0.01 #0.0
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
            partition='train',
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
            partition='test',
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
        iterator =  PharmacoCondSamplingIterator(
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
            partition='train',
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
        tuple_to_pharmaco_id = lambda t: t[0].split('/')[-1].split('_')[0]

        train_ids = [tuple_to_pharmaco_id(t) for t in split_file['train']]
        test_ids = [tuple_to_pharmaco_id(t) for t in split_file['test']]
                
        return PharmacoDB(self.cfg.pharmacophore_db_path, {
            'train': train_ids,
            'test': test_ids
        }, rng=np.random.default_rng(142857), verbose=True)
    
    
    def setup(self):
        self.pharmaco_db = self.setup_pharamaco_dataset()
        super().setup()    
    
def main():
    """Example of how this model can be run."""
    hps = {
        "log_dir": "./logs/debug_run_pharmaco_frag_pb",
        "split_file": 'dataset/split_by_name.pt',
        "pharmacophore_db_path": "misc/pharmacophores.lmdb",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "overwrite_existing_exp": True,
        "num_training_steps": 10_000,
        "num_workers": 1,
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
