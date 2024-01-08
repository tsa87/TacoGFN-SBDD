from omegaconf import OmegaConf
import torch

from typing import Union, List, Dict
from torch import Tensor
from torch_geometric.data import Data as PyGData, Batch as PyGBatch

from src.pharmaconet.network import build_model

from .network.head import AffinityHead
from . import pygdata


class AffinityPredictor():
    def __init__(
        self,
        head_path: str,
        device: str = 'cuda',
    ):
        checkpoint = torch.load(head_path, map_location='cpu')
        config = OmegaConf.create(checkpoint['config'])
        model = build_model(config.MODEL.HEAD)
        model.load_state_dict(checkpoint['model'])
        del checkpoint
        model.eval()
        self.model: AffinityHead = model.to(device)
        self.config = config
        self.device = device

    @torch.no_grad()
    def scoring(
        self,
        cache: Dict[str, Tensor],
        smiles: str,
    ) -> Tensor:
        pocket_features: Tensor = cache['pocket_features']
        token_features: Tensor = cache['token_features']
        num_ligands = 1
        ligand_graph = pygdata.smi2graph(smiles).to(self.device)
        pocket_features = pocket_features.repeat(num_ligands, 1)
        return self.model._calculate_affinity_single(pocket_features, token_features, ligand_graph)

    @torch.no_grad()
    def scoring_list(
        self,
        cache: Dict[str, Tensor],
        smiles_list: List[str],
    ) -> Tensor:
        pocket_features: Tensor = cache['pocket_features']
        token_features: Tensor = cache['token_features']
        num_ligands = len(smiles_list)
        ligand_batch = PyGBatch.from_data_list([pygdata.smi2graph(smiles) for smiles in smiles_list]).to(self.device)
        pocket_features = pocket_features.repeat(num_ligands, 1)
        return self.model._calculate_affinity_single(pocket_features, token_features, ligand_batch)

    @torch.no_grad()
    def _scoring(
        self,
        pocket_features: Tensor,
        token_features: Tensor,
        ligand_graph: Union[PyGData, PyGBatch],
    ) -> Tensor:

        if isinstance(ligand_graph, PyGBatch):
            num_ligands = ligand_graph.num_graphs
        else:
            num_ligands = 1
        pocket_features = pocket_features.repeat(num_ligands, 1)
        return self.model._calculate_affinity_single(pocket_features, token_features, ligand_graph)
