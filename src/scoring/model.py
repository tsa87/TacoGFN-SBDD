from collections import OrderedDict
import torch
import torch.nn as nn
from torch import Tensor

from typing import Dict
from numpy.typing import NDArray

from .network.head import AffinityHead

from src.pharmaconet.network.detector import PharmacoFormer
from src.pharmaconet.data import INTERACTION_LIST, constant as C


class AffinityModule(nn.Module):
    def __init__(
        self,
        backbone: PharmacoFormer,
        head: nn.Module,
        focus_threshold: float,
        box_threshold: float,
        absolute_score_threshold: Dict[str, float],
        **kwargs,
    ):
        super(AffinityModule, self).__init__()
        self.backbone: PharmacoFormer = backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        self.head: AffinityHead = head
        self.focus_threshold = focus_threshold
        self.box_threshold = box_threshold
        self.score_threshold: Dict[str, float] = absolute_score_threshold
        self.l2_loss: nn.MSELoss = nn.MSELoss(reduction='mean')
        self.score_distribution: Dict[str, NDArray]

    def initialize_weights(self):
        self.head.initialize_weights()

    def setup_train(self, criterion: nn.Module):
        self.criterion = criterion

    def forward(self, batch, return_loss: bool = True):
        if return_loss:
            return self.forward_train(batch)
        raise NotImplementedError

    def forward_train(self, batch) -> OrderedDict[str, Tensor]:
        pocket_images, tokens_list, ligand_graph_list = batch
        with torch.no_grad():
            # NOTE: Feature Embedding
            multi_scale_features = self.backbone.forward_feature(pocket_images)
            bottom_features = multi_scale_features[-1]
            cavities_narrow, cavities_wide = self.backbone.forward_cavity_extraction(bottom_features)
            token_scores_list, token_features_list = self.backbone.forward_token_prediction(bottom_features, tokens_list)
            cavity_narrow = cavities_narrow.sigmoid() > self.focus_threshold                   # [N, 1, D, H, W]
            cavity_wide = cavities_wide.sigmoid() > self.focus_threshold                       # [N, 1, D, H, W]

            # NOTE: Token Selection
            num_images = len(token_scores_list)
            for idx in range(num_images):
                tokens, token_scores, token_features = tokens_list[idx], token_scores_list[idx], token_features_list[idx]
                indices = self._get_valid_tokens(tokens, token_scores, cavity_narrow[idx], cavity_wide[idx])     # [Ntoken']
                tokens_list[idx] = tokens[indices]                      # [Ntoken',]
                token_features_list[idx] = token_features[indices]      # [Ntoken',]
        pred_affinity = self.head.forward(multi_scale_features, token_features_list, ligand_graph_list)
        true_affinity = torch.stack([graph.y for graph in ligand_graph_list])
        loss = self.l2_loss.forward(pred_affinity, true_affinity)
        losses = OrderedDict()
        losses['loss'] = loss

        return losses

    def _get_valid_tokens(
        self,
        tokens: Tensor,
        token_scores: Tensor,
        cavity_narrow: Tensor,
        cavity_wide: Tensor,
    ) -> Tensor:
        indices = []
        num_tokens = tokens.shape[0]
        for i in range(num_tokens):
            x, y, z, typ = tokens[i].tolist()
            # NOTE: Check the token score
            absolute_score = token_scores[i].item()
            if absolute_score < self.score_threshold[INTERACTION_LIST[int(typ)]]:
                continue
            # NOTE: Check the token exists in cavity
            if typ in C.LONG_INTERACTION:
                if not cavity_wide[0, x, y, z]:
                    continue
            else:
                if not cavity_narrow[0, x, y, z]:
                    continue
            indices.append(i)
        return torch.tensor(indices, device=tokens.device, dtype=torch.long)  # [Ntoken',]
