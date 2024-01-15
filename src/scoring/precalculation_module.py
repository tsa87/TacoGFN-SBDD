import math
import os
import tempfile
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from omegaconf import OmegaConf
from torch import Tensor

from molvoxel import BaseVoxelizer, create_voxelizer
from src.pharmaconet.data import pointcloud, token_inference
from src.pharmaconet.data.extract_pocket import extract_pocket
from src.pharmaconet.data.objects import Protein
from src.pharmaconet.network import build_model
from src.pharmaconet.utils import load_ligand

from .model import AffinityModule


class PrecalculationModule:
    def __init__(
        self,
        pharmaconet_path: str,
        head_path: str,
        device: str = "cuda",
    ):
        backbone_checkpoint = torch.load(pharmaconet_path, map_location="cpu")
        head_checkpoint = torch.load(head_path, map_location="cpu")

        config = OmegaConf.create(head_checkpoint["config"])

        backbone = build_model(config.MODEL.BACKBONE)
        head = build_model(config.MODEL.HEAD)

        backbone.load_state_dict(backbone_checkpoint["model"])
        head.load_state_dict(head_checkpoint["model"])
        model = AffinityModule(
            backbone,
            head,
            config.THRESHOLD.FOCUS,
            config.THRESHOLD.BOX,
            head_checkpoint["absolute_score_threshold"],
        )
        del backbone_checkpoint
        del head_checkpoint
        model.eval()
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.focus_threshold = 0.5

        in_resolution = config.VOXEL.IN.RESOLUTION
        in_size = config.VOXEL.IN.SIZE
        self.in_voxelizer: BaseVoxelizer = create_voxelizer(
            in_resolution, in_size, sigma=(1 / 3)
        )
        self.pocket_cutoff = (in_resolution * in_size * math.sqrt(3) / 2) + 5.0
        self.protein_radii = config.VOXEL.RADII.PROTEIN
        self.out_resolution = config.VOXEL.OUT.RESOLUTION
        self.out_size = config.VOXEL.OUT.SIZE

    @torch.no_grad()
    def run(
        self,
        protein_pdb_path: str,
        ref_ligand_path: Optional[str] = None,
        center: Optional[ArrayLike] = None,
    ) -> Dict[str, Tuple]:
        assert (ref_ligand_path is not None) or (center is not None)
        if ref_ligand_path is not None:
            ref_ligand = load_ligand(ref_ligand_path)
            center_array = np.mean(
                [atom.coords for atom in ref_ligand.atoms], axis=0, dtype=np.float32
            )
        else:
            center_array = np.array(center, dtype=np.float32)

        return self._run(protein_pdb_path, center_array)

    @torch.no_grad()
    def _run(
        self,
        protein_pdb_path: str,
        center: NDArray[np.float32],
    ) -> Dict[str, Tuple]:
        pocket_image, tokens = self.__parse_protein(protein_pdb_path, center)
        out = self.__ready_to_calculate(pocket_image, tokens)
        return out

    def __parse_protein(
        self,
        protein_pdb_path: str,
        center: NDArray[np.float32],
    ) -> Tuple[Tensor, Tensor]:
        with tempfile.TemporaryDirectory() as dirname:
            pocket_path = os.path.join(dirname, "pocket.pdb")
            extract_pocket(
                protein_pdb_path, pocket_path, center, self.pocket_cutoff
            )  # root(3)
            pocket_obj: Protein = Protein.from_pdbfile(pocket_path)

        pocket_positions, pocket_features = pointcloud.get_protein_pointcloud(
            pocket_obj
        )
        token_positions, token_classes = token_inference.get_token_informations(
            pocket_obj
        )
        tokens, filter = token_inference.get_token_and_filter(
            token_positions, token_classes, center, self.out_resolution, self.out_size
        )

        pocket_image = np.asarray(
            self.in_voxelizer.forward_features(
                pocket_positions, center, pocket_features, radii=self.protein_radii
            ),
            np.float32,
        )
        return torch.from_numpy(pocket_image), torch.from_numpy(tokens)

    def __ready_to_calculate(
        self,
        pocket_image: Tensor,
        tokens: Tensor,
    ):
        pocket_image = pocket_image.to(device=self.device, dtype=torch.float)
        tokens = tokens.to(device=self.device, dtype=torch.long)

        with torch.amp.autocast(self.device, enabled=self.config.AMP_ENABLE):
            pocket_image = pocket_image.unsqueeze(0)

            # NOTE: Feature Embedding
            multi_scale_features = self.model.backbone.forward_feature(pocket_image)
            bottom_features = multi_scale_features[-1]
            (
                cavities_narrow,
                cavities_wide,
            ) = self.model.backbone.forward_cavity_extraction(bottom_features)
            (
                token_scores_list,
                token_features_list,
            ) = self.model.backbone.forward_token_prediction(bottom_features, [tokens])

            token_scores = token_scores_list[0].sigmoid()
            token_features = token_features_list[0]
            cavity_narrow = (
                cavities_narrow.squeeze(0).sigmoid() > self.focus_threshold
            )  # [1, D, H, W]
            cavity_wide = (
                cavities_wide.squeeze(0).sigmoid() > self.focus_threshold
            )  # [1, D, H, W]

            # NOTE: Token Selection
            indices = self.model._get_valid_tokens(
                tokens, token_scores, cavity_narrow, cavity_wide
            )  # [Ntoken']
            token_features_list[0] = token_features[indices]  # [Ntoken',]

            # NOTE: Pre-Calculation
            pocket_features, token_features_list = self.model.head.ready_to_calculate(
                multi_scale_features, token_features_list
            )

        out = {
            "pocket_features": pocket_features[0].cpu(),  # [Fh]
            "token_features": token_features_list[0].cpu(),  # [Ntoken', Fh]
        }
        return out
