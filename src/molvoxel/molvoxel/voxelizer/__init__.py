from .base import BaseVoxelizer, BaseRandomTransform


def create_random_transform(
    random_translation: float = 0.0,
    random_rotation: bool = False,
    library: str = 'numpy',
    **kwargs,
) -> BaseRandomTransform:
    assert library in ['numba', 'numpy', 'torch']
    if library == 'numba':
        from .numba import RandomTransform
    elif library == 'numpy':
        from .numpy import RandomTransform
    else:
        from .torch import RandomTransform
    return RandomTransform(random_translation, random_rotation, **kwargs)


def create_voxelizer(
    resolution: float = 0.5,
    dimension: int = 64,
    radii_type: str = 'scalar',
    density_type: str = 'gaussian',
    library: str = 'numpy',
    **kwargs,
) -> BaseVoxelizer:
    assert library in ['numba', 'numpy', 'torch']
    if library == 'numba':
        from .numba import Voxelizer
    elif library == 'numpy':
        from .numpy import Voxelizer
    else:
        from .torch import Voxelizer
    return Voxelizer(resolution, dimension, radii_type, density_type, **kwargs)
