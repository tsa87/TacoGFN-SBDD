import os
import pathlib
import shutil
import traceback
from typing import Any, Dict, Tuple

import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader, Dataset
from torchmetrics import PearsonCorrCoef

from src.tacogfn.data.docking_score import CrossDockDockingScoreDataset
from src.tacogfn.data.pharmacophore import PharmacoDB
from src.tacogfn.models.beta_docking_score_predictor import DockingScorePredictionModel
from src.tacogfn.utils.misc import create_logger


class DockingScoreTrainer:
    def __init__(
        self,
        hps: Dict[str, Any],
        train_datalist: list[Tuple[str, str, float]],
        test_datalist: list[Tuple[str, str, float]],
        device: str = "cuda",
        verbose=False,
    ):
        self.train_data: Dataset
        self.test_data: Dataset
        self.model: nn.Module
        self.pharmacophore_db: PharmacoDB

        self.device = device
        self.verbose = verbose
        self.hps = {**self.default_hps(), **hps}
        self.cfg = OmegaConf.create(hps)

        self.train_datalist = train_datalist
        self.test_datalist = test_datalist

        self.setup()

    def default_hps(self) -> Dict[str, Any]:
        return {
            "epochs": 1000,
            "learning_rate": 1e-4,
            "momentum": 0.9,
            "weight_decay": 1e-8,
            "adam_eps": 1e-8,
            "var_eps": 1e-6,
            "batch_size": 64,
            "clip_grad_param": 10.0,
            "seed": 0,
        }

    def setup_dataset(self):
        self.train_data = CrossDockDockingScoreDataset(
            self.train_datalist,
            self.pharmacophore_db,
            root_folder=self.hps["dataset_save_dir"],
            dataset_type="train",
        )

        self.test_data = CrossDockDockingScoreDataset(
            self.test_datalist,
            self.pharmacophore_db,
            root_folder=self.hps["dataset_save_dir"],
            dataset_type="test",
        )

    def setup_model(self):
        self.model = DockingScorePredictionModel(hidden_dim=self.hps["hidden_dim"])

    def setup(self):
        self.batch_size = self.hps["batch_size"]
        self.pharmacophore_db = PharmacoDB(self.hps["pharmacodb_path"])

        self.setup_dataset()
        self.setup_model()
        self.model.to(self.device)

        params = self.model.parameters()
        self.opt = torch.optim.Adam(
            params,
            self.hps["learning_rate"],
            (self.hps["momentum"], 0.999),
            weight_decay=self.hps["weight_decay"],
            eps=self.hps["adam_eps"],
        )
        self.clip_grad_param = self.hps["clip_grad_param"]
        self.clip_grad_callback = lambda x: nn.utils.clip_grad_norm_(
            x, self.clip_grad_param
        )

        print("\n\nHyperparameters:\n")
        yaml = OmegaConf.to_yaml(self.cfg)
        print(yaml)
        with open(pathlib.Path(self.cfg.log_dir) / "hps.yaml", "w") as f:
            f.write(yaml)

    def _build_train_loader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )

    def _build_test_loader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

    def build_callbacks(self):
        return {
            "pearson_r": PearsonCorrCoef().to("cpu"),
            "epoch_mae": lambda x, y: torch.nn.functional.l1_loss(x, y),
            "epoch_mse": lambda x, y: torch.nn.functional.mse_loss(x, y),
        }

    def step(self, loss):
        loss.backward()
        for i in self.model.parameters():
            self.clip_grad_callback(i)
        self.opt.step()
        self.opt.zero_grad()

    def on_epoch_end(self, callbacks, preds, labels):
        epoch_info = {}
        for k, c in callbacks.items():
            epoch_info[k] = c(preds, labels)
        return epoch_info

    def train_batch(self, batch, labels):
        preds = self.model(batch)

        loss = torch.nn.functional.mse_loss(preds, labels)
        mae = torch.nn.functional.l1_loss(preds, labels)

        self.step(loss)
        return preds, {
            "loss": loss.item(),
            "mae": mae.item(),
        }

    def run(self):
        self.logger = create_logger(
            name="docking_logger", logfile=self.hps["log_dir"] + "/train.log"
        )
        self.model.to(self.device)

        train_dl = self._build_train_loader()
        test_dl = self._build_test_loader()
        callbacks = self.build_callbacks()

        self.logger.info("Start training")
        it = 0
        for epoch in range(self.hps["epochs"]):
            # Training
            self.model.train()

            all_preds = []
            all_labels = []

            for batch, affinity in train_dl:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                affinity = affinity.to(self.device)

                try:
                    preds, info = self.train_batch(batch, affinity)
                    self.log(info, it, "train")

                    if self.verbose:
                        self.logger.info(
                            f"iteration {it} : "
                            + " ".join(f"{k}:{v:.2f}" for k, v in info.items())
                        )

                    all_preds.append(preds.to("cpu"))
                    all_labels.append(affinity.to("cpu"))
                    it += 1

                except RuntimeError as e:
                    traceback.print_exc()

            epoch_info = self.on_epoch_end(
                callbacks,
                torch.cat(all_preds).squeeze(1),
                torch.cat(all_labels).squeeze(1),
            )

            self.log(epoch_info, epoch, "train")
            self._save_state(epoch)

            # Evaluation
            self.model.eval()

            all_preds = []
            all_labels = []

            with torch.no_grad():
                for batch, affinity in test_dl:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    affinity = affinity.to(self.device)

                    preds = self.model(batch)

                    all_preds.append(preds.to("cpu"))
                    all_labels.append(affinity.to("cpu"))

            epoch_info = self.on_epoch_end(
                callbacks,
                torch.cat(all_preds).squeeze(1),
                torch.cat(all_labels).squeeze(1),
            )
            self.log(epoch_info, epoch, "test")

    def _save_state(self, epoch):
        path = pathlib.Path(self.hps["log_dir"]) / f"model_state_{epoch}.pt"
        torch.save(
            {
                "models_state_dict": [self.model.state_dict()],
                "optimizers_state_dict": self.opt.state_dict(),
                "hps": self.hps,
                "epoch": epoch,
            },
            open(path, "wb"),
        )

    def log(self, info, index, key):
        if not hasattr(self, "writer"):
            self.writer = SummaryWriter(self.hps["log_dir"])
        for k, v in info.items():
            self.writer.add_scalar(f"{key}/{k}", v, index)


def main():
    """Example of how this trainer can be run"""
    hps = {
        "log_dir": "logs/docking_score_prediction_beta_full_dataset",
        "dataset_save_dir": "misc/crossdock_docking_score_dataset",
        "pharmacodb_path": "misc/pharmacophores_db.lmdb",
        "overwrite_existing_exp": True,
        "hidden_dim": 364,
    }
    if os.path.exists(hps["log_dir"]):
        if hps["overwrite_existing_exp"]:
            shutil.rmtree(hps["log_dir"])
        else:
            raise ValueError(
                f"Log dir {hps['log_dir']} already exists. Set overwrite_existing_exp=True to delete it."
            )
    os.makedirs(hps["log_dir"])

    train_datalist = torch.load("dataset/crossdock_docking_scores/train.pt")
    test_datalist = torch.load("dataset/crossdock_docking_scores/test.pt")

    trial = DockingScoreTrainer(hps, train_datalist, test_datalist)
    trial.run()


if __name__ == "__main__":
    main()
