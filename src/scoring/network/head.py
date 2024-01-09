import torch
from torch import nn

from typing import Sequence, Tuple, Union
from torch import Tensor
from torch_geometric.data import Data as PyGData, Batch as PyGBatch
from torch_scatter import scatter_sum

from .builder import AFFINITY_HEAD


@AFFINITY_HEAD.register()
class AffinityHead(nn.Module):
    def __init__(
        self,
        ligand_encoder: nn.Module,
        feature_channels: Sequence[int],
        token_feature_dim: int,
        hidden_dim: int,
    ):
        super(AffinityHead, self).__init__()

        self.hidden_dim = hidden_dim
        self.ligand_encoder = ligand_encoder

        # Ready-To-Affinity-Calculation
        self.token_mlp: nn.Module = nn.Sequential(
            nn.SiLU(),
            nn.Linear(token_feature_dim, hidden_dim),
            # nn.LayerNorm(hidden_dim),
        )
        self.pocket_mlp_list: nn.ModuleList = nn.ModuleList([
            nn.Sequential(nn.SiLU(), nn.Conv3d(channels, hidden_dim, 3)) for channels in feature_channels
        ])
        self.pocket_mlp: nn.Module = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim * len(feature_channels), hidden_dim),
        )

        # Concat Layer
        self.concat_layer: nn.Module = nn.Linear(3 * hidden_dim, hidden_dim)
        self.concat_gate: nn.Module = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

        # Affinity Calculation
        ligand_atom_channels: int = ligand_encoder.atom_channels
        ligand_graph_channels: int = ligand_encoder.graph_channels

        # Dimension
        if ligand_atom_channels != hidden_dim:
            self.ligand_layer_atom = nn.Linear(ligand_atom_channels, hidden_dim)
        else:
            self.ligand_layer_atom = nn.Identity()
        if ligand_graph_channels != hidden_dim:
            self.ligand_layer_graph = nn.Linear(ligand_graph_channels, hidden_dim)
        else:
            self.ligand_layer_graph = nn.Identity()

        # Interaction Layer
        self.energy_bias_mlp: nn.Module = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.interaction_mlp: nn.Module = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
        )
        self.pair_energy_layer: nn.Module = nn.Linear(hidden_dim, 1)
        self.pair_energy_gate: nn.Module = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def initialize_weights(self):
        def _init_weight(m):
            if isinstance(m, (nn.Linear, nn.Conv3d)):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(_init_weight)

    def forward(
        self,
        multi_scale_features: Sequence[Tensor],
        token_features_list: Sequence[Tensor],
        ligand_graph_list: Union[Sequence[PyGData], Sequence[PyGBatch]],
    ) -> Tensor:
        """Affinity Prediction

        Args:
            multi_scale_features: Top-Down, List[FloatTensor [N, F_scale, D_scale, H_scale, W_scale]]   - size: N_scale
            token_features_list: List[FloatTensor[Nbox, Ftoken]                                         - size: N
            ligand_graph_list: Union[Sequence[PyGData], Sequence[PyGBatch]]                             - size: N

        Returns:
            affinity: FloatTensor [N, Ngraph]
        """
        assert len(token_features_list) == len(ligand_graph_list)

        pocket_features, token_features_list = \
            self.ready_to_calculate(multi_scale_features, token_features_list)
        num_ligands = ligand_graph_list[0].y.size(0)
        pocket_features = pocket_features.unsqueeze(1).repeat(1, num_ligands, 1)
        return self.calculate_affinity(pocket_features, token_features_list, ligand_graph_list)

    def ready_to_calculate(
        self,
        multi_scale_features: Sequence[Tensor],
        token_features_list: Sequence[Tensor],
    ) -> Tuple[Tensor, Sequence[Tensor]]:
        """Affinity Prediction

        Args:
            multi_scale_features: Top-Down, List[FloatTensor [N, F_scale, D_scale, H_scale, W_scale]]
            token_features_list: List[FloatTensor[Nbox, Ftoken]

        Returns:
            pocket_features: FloatTensor [N, F_hidden]
            token_features_list: List[FloatTensor [Nbox, F_hidden]]
        """
        multi_scale_features = multi_scale_features[::-1]   # Top-Down -> Bottom-Up
        multi_scale_features = [layer(feature) for layer, feature in zip(self.pocket_mlp_list, multi_scale_features)]
        pocket_features: Tensor = self.pocket_mlp(
            torch.cat([feature.mean(dim=(-1, -2, -3)) for feature in multi_scale_features], dim=-1)
        )   # [N, Fh]

        token_features_list = [self.token_mlp(feature) for feature in token_features_list]                                  # List[FloatTensor[Nbox, Fh]]
        out_token_features_list = []
        for feature in token_features_list:
            if feature.size(0) == 0:
                feature = torch.zeros((2 * self.hidden_dim,), dtype=feature.dtype, device=feature.device)
            else:
                feature = torch.cat([feature.sum(0), feature.mean(0)])
            out_token_features_list.append(feature)
        token_features = torch.stack(out_token_features_list)                                           # [N, 2 * Fh]
        pocket_features = torch.cat([pocket_features, token_features], dim=-1)                          # [N, 3 * Fh]
        pocket_features = self.concat_layer(pocket_features) * self.concat_gate(pocket_features)        # [N, Fh]
        return pocket_features, token_features_list

    def calculate_affinity(
        self,
        pocket_features: Tensor,
        token_features_list: Sequence[Tensor],
        ligand_graph_list: Union[Sequence[PyGData], Sequence[PyGBatch]],
    ) -> Tensor:
        """
        pred: [N, Ngraph]    # Ngraph: mini-batch-size, number of ligands per receptor
        """
        num_images = len(token_features_list)
        total_pred_list = [
            self._calculate_affinity_single(
                pocket_features[image_idx],
                token_features_list[image_idx],
                ligand_graph_list[image_idx]
            )
            for image_idx in range(num_images)
        ]
        return torch.stack(total_pred_list)

    def _calculate_affinity_single(
        self,
        pocket_features: Tensor,
        token_features: Tensor,
        ligand_graph: Union[PyGData, PyGBatch]
    ) -> Tensor:
        X, Z = self.ligand_encoder(ligand_graph)
        ligand_atom_features = self.ligand_layer_atom(X)                                   # [Natom, Fh]
        interaction_map = torch.einsum("ik,jk->ijk", ligand_atom_features, token_features)              # [Natom, Nbox, Fh]
        interaction_map = self.interaction_mlp(interaction_map)

        # Element-Wise Calculation
        pair_energy = self.pair_energy_layer(interaction_map) * self.pair_energy_gate(interaction_map)  # [Natom, Nbox, 1]
        if isinstance(ligand_graph, PyGBatch):
            pair_energy = pair_energy.sum((1, 2))                                                       # [Natom,]
            pair_energy = scatter_sum(pair_energy, ligand_graph.batch)                                  # [N,]
        else:
            pair_energy = pair_energy.sum()

        # Graph-Wise Calculation
        ligand_graph_features = self.ligand_layer_graph(Z)                        # [N, Fh]
        bias = self.energy_bias_mlp(torch.cat([pocket_features, ligand_graph_features], dim=-1))        # [N, 1]

        return pair_energy.view(-1) + bias.view(-1)
