"""
Model configuration from
https://github.com/recursionpharma/gflownet
"""

from dataclasses import dataclass
from enum import Enum


@dataclass
class GraphTransformerConfig:
    num_heads: int = 2
    ln_type: str = "pre"
    num_mlp_layers: int = 0


@dataclass
class PharmacoConditioningConfig:
    pharmaco_dim: int = 64


@dataclass
class ModelConfig:
    """Generic configuration for models

    Attributes
    ----------
    num_layers : int
        The number of layers in the model
    num_emb : int
        The number of dimensions of the embedding
    """

    num_layers: int = 3
    num_emb: int = 128
    dropout: float = 0
    graph_transformer: GraphTransformerConfig = GraphTransformerConfig()
    pharmaco_cond: PharmacoConditioningConfig = PharmacoConditioningConfig()
