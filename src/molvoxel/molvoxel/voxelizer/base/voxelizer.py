import numpy as np

from typing import Tuple, Optional, Union, Type
from numpy.typing import ArrayLike
from abc import ABCMeta, abstractmethod
from .transform import BaseRandomTransform


class BaseVoxelizer(metaclass=ABCMeta):
    LIB = None
    transform_class: Type[BaseRandomTransform] = BaseRandomTransform
    RADII_TYPE_LIST = ['scalar', 'channel-wise', 'atom-wise']
    DENSITY_TYPE_LIST = ['gaussian', 'binary']

    def __init__(
        self,
        resolution: float = 0.5,
        dimension: int = 48,
        radii_type: str = 'scalar',
        density_type: str = 'gaussian',
        **kwargs
    ):
        assert radii_type in self.RADII_TYPE_LIST
        assert density_type in self.DENSITY_TYPE_LIST

        self._resolution: float = resolution
        self._dimension: int = dimension
        self._width: float = resolution * (dimension - 1)

        self._radii_type: str = radii_type
        self._density_type: str = density_type

        self.upper_bound: float = self.width / 2.
        self.lower_bound: float = -1 * self.upper_bound
        self._spatial_dimension: Tuple[int, int, int] = (self._dimension, self._dimension, self._dimension)

        if self._density_type == 'gaussian':
            self._sigma: float = kwargs.get('sigma', 0.5)

    @property
    def radii_type(self) -> str:
        return self._radii_type

    @radii_type.setter
    def radii_type(self, radii_type: str):
        assert radii_type in self.RADII_TYPE_LIST
        self._radii_type = radii_type

    @property
    def is_radii_type_scalar(self):
        return self._radii_type == 'scalar'

    @property
    def is_radii_type_channel_wise(self):
        return self._radii_type == 'channel-wise'

    @property
    def is_radii_type_atom_wise(self):
        return self._radii_type == 'atom-wise'

    @property
    def density_type(self) -> str:
        return self._density_type

    @density_type.setter
    def density_type(self, density_type: str, **kwargs):
        assert density_type in self.DENSITY_TYPE_LIST
        self._density_type = density_type
        if density_type == 'gaussian':
            self._sigma = kwargs.get('sigma', 0.5)

    @property
    def is_density_type_binary(self):
        return self._density_type == 'binary'

    @property
    def is_density_type_gaussian(self):
        return self._density_type == 'gaussian'

    def grid_dimension(self, num_channels: int) -> Tuple[int, int, int, int]:
        return (num_channels, self._dimension, self._dimension, self._dimension)

    @property
    def spatial_dimension(self) -> Tuple[int, int, int]:
        return self._spatial_dimension

    @property
    def resolution(self) -> float:
        return self._resolution

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def width(self) -> float:
        return self._width

    """ Forward """

    def forward(
        self,
        coords: ArrayLike,
        center: Optional[ArrayLike],
        channels: Optional[ArrayLike],
        radii: Union[float, ArrayLike],
        random_translation: float = 0.0,
        random_rotation: bool = False,
        out_grid: Optional[ArrayLike] = None
    ) -> ArrayLike:
        """
        coords: (V, 3)
        center: (3,)
        types: (V, C) or (V,) or None
        radii: scalar or (V, ) of (C, )
        random_translation: float (nonnegative)
        random_rotation: bool

        out_grid: (C,D,H,W)
        """
        if channels is None:
            return self.forward_single(coords, center, radii, random_translation, random_rotation, out_grid)
        elif np.ndim(channels) == 1:
            types = channels
            return self.forward_types(coords, center, types, radii, random_translation, random_rotation, out_grid)
        else:
            features = channels
            return self.forward_features(coords, center, features, radii, random_translation, random_rotation, out_grid)

    __call__ = forward

    @abstractmethod
    def forward_types(
        self,
        coords: ArrayLike,
        center: Optional[ArrayLike],
        types: ArrayLike,
        radii: Union[float, ArrayLike],
        random_translation: float = 0.0,
        random_rotation: bool = False,
        out_grid: Optional[ArrayLike] = None
    ) -> ArrayLike:
        pass

    @abstractmethod
    def forward_features(
        self,
        coords: ArrayLike,
        center: Optional[ArrayLike],
        features: ArrayLike,
        radii: Union[float, ArrayLike],
        random_translation: float = 0.0,
        random_rotation: bool = False,
        out_grid: Optional[ArrayLike] = None
    ) -> ArrayLike:
        pass

    @abstractmethod
    def forward_single(
        self,
        coords: ArrayLike,
        center: Optional[ArrayLike],
        radii: Union[float, ArrayLike],
        random_translation: float = 0.0,
        random_rotation: bool = False,
        out_grid: Optional[ArrayLike] = None
    ) -> ArrayLike:
        pass

    @abstractmethod
    def get_empty_grid(self, num_channels: int, batch_size: Optional[int] = None, init_zero: bool = False) -> ArrayLike:
        pass

    @abstractmethod
    def asarray(self, array: ArrayLike, obj: str) -> ArrayLike:
        pass
