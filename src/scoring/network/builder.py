from torch import nn
from typing import Dict

from src.pharmaconet.network.utils.registry import Registry

AFFINITY_HEAD = Registry('Affinity_head')


def build_model(config: Dict) -> nn.Module:
    registry_key = 'registry'
    module_key = 'name'
    return Registry.build_from_config(config, registry_key, module_key, convert_key_to_lower_case=True, safe_build=True)
