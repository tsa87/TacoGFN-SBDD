import math
import numpy as np
import itertools

from typing import Tuple, Union, Optional, Dict, List
from numpy.typing import NDArray, ArrayLike
from scipy.spatial.distance import cdist

from molvoxel.voxelizer.base import BaseVoxelizer
from .transform import do_random_transform, RandomTransform

NDArrayInt = NDArray[np.int16]
NDArrayFloat = Union[NDArray[np.float32], NDArray[np.float64]]
NDArrayFloat64 = NDArray[np.float64]
NDArrayBool = NDArray[np.bool_]


class Voxelizer(BaseVoxelizer):
    LIB = 'Numpy'
    transform_class = RandomTransform

    def __init__(
        self,
        resolution: float = 0.5,
        dimension: int = 64,
        radii_type: str = 'scalar',
        density_type: str = 'gaussian',
        precision: int = 32,
        blockdim: Optional[int] = None,
        **kwargs
    ):
        super(Voxelizer, self).__init__(resolution, dimension, radii_type, density_type, **kwargs)
        assert precision in [32, 64]
        self.fp = np.float32 if precision == 32 else np.float64
        self._setup_block(blockdim)

    def _setup_block(self, blockdim):
        blockdim = blockdim if blockdim is not None else 8
        self.blockdim = blockdim

        axis = np.arange(self.dimension, dtype=np.float64) * self.resolution - (self.width / 2.)  # cdist only support float64
        self.num_blocks = num_blocks = math.ceil(self.dimension / blockdim)
        if self.num_blocks > 1:
            self.grid = None
            self.grid_block_dict = {}
            for xidx, yidx, zidx in itertools.product(range(self.num_blocks), repeat=3):
                x_axis = axis[xidx * blockdim: (xidx + 1) * blockdim]
                y_axis = axis[yidx * blockdim: (yidx + 1) * blockdim]
                z_axis = axis[zidx * blockdim: (zidx + 1) * blockdim]
                grid_block = np.meshgrid(x_axis, y_axis, z_axis, indexing='ij')
                self.grid_block_dict[(xidx, yidx, zidx)] = np.stack(grid_block, axis=-1)

            self.bounds = [(axis[idx * blockdim] + (self.resolution / 2.))
                           for idx in range(1, num_blocks)]
        else:
            self.grid_block_dict = None
            self.grid = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), axis=-1)

    def get_empty_grid(self, num_channels: int, batch_size: Optional[int] = None, init_zero: bool = False) -> NDArrayFloat:
        if init_zero:
            if batch_size is None:
                return np.zeros(self.grid_dimension(num_channels), dtype=self.fp)
            else:
                return np.zeros((batch_size,) + self.grid_dimension(num_channels), dtype=self.fp)
        else:
            if batch_size is None:
                return np.empty(self.grid_dimension(num_channels), dtype=self.fp)
            else:
                return np.empty((batch_size,) + self.grid_dimension(num_channels), dtype=self.fp)

    """ Forward """

    def forward(
        self,
        coords: NDArrayFloat64,
        center: Optional[NDArrayFloat64],
        channels: Union[NDArrayFloat, NDArrayInt, None],
        radii: Union[float, NDArrayFloat],
        random_translation: float = 0.0,
        random_rotation: bool = False,
        out_grid: Optional[NDArrayFloat] = None
    ) -> NDArrayFloat:
        if channels is None:
            return self.forward_single(coords, center, radii, random_translation, random_rotation, out_grid)
        elif channels.ndim == 1:
            types = channels
            return self.forward_types(coords, center, types, radii, random_translation, random_rotation, out_grid)
        else:
            features = channels
            return self.forward_features(coords, center, features, radii, random_translation, random_rotation, out_grid)
    __call__ = forward

    """ VECTOR """

    def forward_features(
        self,
        coords: NDArrayFloat64,
        center: Optional[NDArrayFloat64],
        features: NDArrayFloat,
        radii: Union[float, NDArrayFloat],
        random_translation: float = 0.0,
        random_rotation: bool = False,
        out_grid: Optional[NDArrayFloat] = None
    ) -> NDArrayFloat:
        """
        coords: (V, 3)
        center: (3,)
        features: (V, C)
        radii: scalar or (V, ) of (C, )
        random_translation: float (nonnegative)
        random_rotation: bool

        out_grid: (C,D,H,W)
        """
        self._check_args_features(coords, features, radii, out_grid)

        # Set Coordinate
        if center is not None:
            coords = coords - center.reshape(1, 3)
        coords = do_random_transform(coords, None, random_translation, random_rotation)

        # DataType
        if coords.dtype != np.float64:     # cdist support only float64
            coords = coords.astype(np.float64)
        if features.dtype != self.fp:
            features = features.astype(self.fp)
        if not np.isscalar(radii):
            radii = radii.astype(self.fp)

        # Set Out
        if out_grid is None:
            C = features.shape[1]
            out_grid = self.get_empty_grid(C)

        # Clipping Overlapped Atoms
        atom_size = radii.max() if self.is_radii_type_channel_wise else radii
        box_overlap = self._get_overlap(coords, atom_size)
        coords, features = coords[box_overlap], features[box_overlap]
        if self.is_radii_type_atom_wise:
            radii = radii[box_overlap]
            atom_size = radii

        # Run
        if self.num_blocks > 1:
            blockdim = self.blockdim
            block_overlap_dict = self._get_overlap_blocks(coords, atom_size)

            for xidx, yidx, zidx in itertools.product(range(self.num_blocks), repeat=3):
                start_x, end_x = xidx * blockdim, (xidx + 1) * blockdim
                start_y, end_y = yidx * blockdim, (yidx + 1) * blockdim
                start_z, end_z = zidx * blockdim, (zidx + 1) * blockdim

                out_grid_block = out_grid[:, start_x:end_x, start_y:end_y, start_z:end_z]

                overlap = block_overlap_dict[(xidx, yidx, zidx)]
                if overlap.shape[0] == 0:
                    out_grid_block.fill(0.)
                    continue

                grid_block = self.grid_block_dict[(xidx, yidx, zidx)]
                coords_block, features_block = coords[overlap], features[overlap]
                radii_block = radii[overlap] if self.is_radii_type_atom_wise else radii
                self._set_grid_features(coords_block, features_block, radii_block, grid_block, out_grid_block)
        else:
            self._set_grid_features(coords, features, radii, self.grid, out_grid)

        return out_grid

    def _check_args_features(self, coords: NDArrayFloat64, features: NDArrayFloat, radii: Union[float, NDArrayFloat],
                             out_grid: Optional[NDArrayFloat] = None):
        V = coords.shape[0]
        C = features.shape[1]
        D = H = W = self.dimension
        assert features.shape[0] == V, f'atom features does not match number of atoms: {features.shape[0]} vs {V}'
        assert features.ndim == 2, f"atom features does not match dimension: {features.shape} vs {(V,'*')}"
        if self.is_radii_type_scalar:
            assert np.isscalar(radii), f'the radii type of voxelizer is `scalar`, radii should be scalar'
        elif self.is_radii_type_channel_wise:
            assert not np.isscalar(radii), f'the radii type of voxelizer is `channel-wise`, radii should be Array[{C},]'
            assert radii.shape == (C,), f'radii does not match dimension (number of channels,): {radii.shape} vs {(C,)}'
        else:
            assert not np.isscalar(radii), f'the radii type of voxelizer is `atom-wise`, radii should be Array[{V},]'
            assert radii.shape == (V,), f'radii does not match dimension (number of atoms,): {radii.shape} vs {(V,)}'
        if out_grid is not None:
            assert out_grid.shape == (C, D, H, W), f'Output grid dimension incorrect: {out_grid.shape} vs {(C,D,H,W)}'

    def _set_grid_features(
        self,
        coords: NDArrayFloat64,
        features: NDArrayFloat,
        radii: Union[float, NDArrayFloat],
        grid: NDArrayFloat,
        out_grid: NDArrayFloat,
    ) -> NDArrayFloat:
        """
        coords: (V, 3)
        features: (V, C)
        radii: scalar or (V, ) or (C, )
        grid: (D, H, W, 3)

        out_grid: (C, D, H, W)
        """
        features = features.T                                               # (V, C) -> (C, V)
        D, H, W, _ = grid.shape
        grid = grid.reshape(-1, 3)                                          # (DHW, 3)
        if self.is_radii_type_channel_wise:
            if out_grid.data.contiguous:
                _out_grid = out_grid.reshape(-1, D * H * W)
                for type_idx in range(features.shape[0]):
                    typ = features[type_idx]                                # (V,)
                    res = self._calc_grid(coords, radii[type_idx], grid)    # (V, DHW)
                    np.matmul(typ, res, out=_out_grid[type_idx])                 # (V,) @ (V, DHW) -> (DHW) = (D, H, W)
            else:
                for type_idx in range(features.shape[0]):
                    typ = features[type_idx]                                # (V,)
                    res = self._calc_grid(coords, radii[type_idx], grid)    # (V, DHW)
                    out_grid[type_idx] = np.matmul(typ, res).reshape(D, H, W)    # (V,) @ (V, DHW) -> (DHW) = (D, H, W)
        else:
            if out_grid.data.contiguous:
                res = self._calc_grid(coords, radii, grid)                  # (V, DHW)
                np.matmul(features, res, out=out_grid.reshape(-1, D * H * W))        # (C, V) @ (V, DHW) -> (C, DHW) = (C, D, H, W)
            else:
                res = self._calc_grid(coords, radii, grid)                  # (V, DHW)
                out_grid[:] = np.matmul(features, res).reshape(-1, D, H, W)      # (C, V) @ (V, DHW) -> (C, DHW) = (C, D, H, W)
        return out_grid

    """ INDEX """

    def forward_types(
        self,
        coords: NDArrayFloat64,
        center: Optional[NDArrayFloat64],
        types: NDArrayInt,
        radii: Union[float, NDArrayFloat],
        random_translation: float = 0.0,
        random_rotation: bool = False,
        out_grid: Optional[NDArrayFloat] = None
    ) -> NDArrayFloat:
        """
        coords: (V, 3)
        center: (3,)
        types: (V,)
        radii: scalar or (V, ) of (C, )
        random_translation: float (nonnegative)
        random_rotation: bool

        out_grid: (C,D,H,W)
        """
        self._check_args_types(coords, types, radii, out_grid)

        # Set Coordinate
        if center is not None:
            coords = coords - center.reshape(1, 3)
        coords = self.do_random_transform(coords, None, random_translation, random_rotation)

        # DataType
        coords = self._dtypechange(coords, np.float64)
        types = self._dtypechange(types, np.int16)
        if not np.isscalar(radii):
            radii = self._dtypechange(radii, self.fp)

        # Set Out
        if out_grid is None:
            if self.is_radii_type_channel_wise:
                C = radii.shape[0]
            else:
                C = np.max(types) + 1
            out_grid = self.get_empty_grid(C, init_zero=True)
        else:
            out_grid.fill(0.)

        # Clipping Overlapped Atoms
        if self.is_radii_type_channel_wise:
            radii = radii[types]           # (C, ) -> (V, )
        atom_size = radii
        box_overlap = self._get_overlap(coords, atom_size)
        coords, types = coords[box_overlap], types[box_overlap]
        radii = radii[box_overlap] if not np.isscalar(radii) else radii
        atom_size = radii

        # Run
        if self.num_blocks > 1:
            blockdim = self.blockdim
            block_overlap_dict = self._get_overlap_blocks(coords, atom_size)

            for xidx, yidx, zidx in itertools.product(range(self.num_blocks), repeat=3):
                start_x, end_x = xidx * blockdim, (xidx + 1) * blockdim
                start_y, end_y = yidx * blockdim, (yidx + 1) * blockdim
                start_z, end_z = zidx * blockdim, (zidx + 1) * blockdim

                out_grid_block = out_grid[:, start_x:end_x, start_y:end_y, start_z:end_z]

                overlap = block_overlap_dict[(xidx, yidx, zidx)]
                if overlap.shape[0] == 0:
                    continue

                grid_block = self.grid_block_dict[(xidx, yidx, zidx)]
                coords_block, types_block = coords[overlap], types[overlap]
                radii_block = radii[overlap] if not np.isscalar(radii) else radii
                self._set_grid_types(coords_block, types_block, radii_block, grid_block, out_grid_block)
        else:
            self._set_grid_types(coords, types, radii, self.grid, out_grid)

        return out_grid

    def _check_args_types(self, coords: NDArrayFloat64, types: NDArrayInt, radii: Union[float, NDArrayFloat],
                          out_grid: Optional[NDArrayFloat] = None):
        V = coords.shape[0]
        C = np.max(types) + 1
        D = H = W = self.dimension
        assert types.shape == (V,), f"types does not match dimension: {types.shape} vs {(V,)}"
        if self.is_radii_type_scalar:
            assert np.isscalar(radii), f'the radii type of voxelizer is `scalar`, radii should be scalar'
        elif self.is_radii_type_channel_wise:
            assert not np.isscalar(radii), f'the radii type of voxelizer is `channel-wise`, radii should be Array[{C},]'
            assert radii.shape == (C,), f'radii does not match dimension (number of channels,): {radii.shape} vs {(C,)}'
        else:
            assert not np.isscalar(radii), f'the radii type of voxelizer is `atom-wise`, radii should be Array[{V},]'
            assert radii.shape == (V,), f'radii does not match dimension (number of atoms,): {radii.shape} vs {(V,)}'
        if out_grid is not None:
            assert out_grid.shape[0] >= C, f'Output channel is less than number of types: {out_grid.shape[0]} < {C}'
            assert out_grid.shape[1:] == (D, H, W), f'Output grid dimension incorrect: {out_grid.shape} vs {("*",D,H,W)}'

    def _set_grid_types(
        self,
        coords: NDArrayFloat64,
        types: NDArrayInt,
        radii: Union[float, NDArrayFloat],
        grid: NDArrayFloat,
        out_grid: NDArrayFloat,
    ) -> NDArrayFloat:
        """
        coords: (V, 3)
        types: (V,)
        radii: scalar or (V, )
        grid: (D, H, W, 3)

        out_grid: (C, D, H, W)
        """
        D, H, W, _ = grid.shape
        grid = grid.reshape(-1, 3)
        res = self._calc_grid(coords, radii, grid)          # (V, D*H*W)
        res = res.reshape(-1, D, H, W)                      # (V, D, H, W)
        for vidx, typ in enumerate(types):
            out_grid[typ] += res[vidx]
        return out_grid

    """ SINGLE """

    def forward_single(
        self,
        coords: NDArrayFloat64,
        center: Optional[NDArrayFloat64],
        radii: Union[float, NDArrayFloat],
        random_translation: float = 0.0,
        random_rotation: bool = False,
        out_grid: Optional[NDArrayFloat] = None
    ) -> NDArrayFloat:
        """
        coords: (V, 3)
        center: (3,)
        radii: scalar or (V, )
        random_translation: float (nonnegative)
        random_rotation: bool

        out_grid: (1,D,H,W)
        """
        self._check_args_single(coords, radii, out_grid)

        # Set Coordinate
        if center is not None:
            coords = coords - center.reshape(1, 3)
        coords = self.do_random_transform(coords, None, random_translation, random_rotation)

        # DataType
        coords = self._dtypechange(coords, np.float64)
        if not np.isscalar(radii):
            radii = self._dtypechange(radii, self.fp)

        # Set Out
        if out_grid is None:
            out_grid = self.get_empty_grid(1, init_zero=True)
        else:
            out_grid.fill(0.)

        # Clipping Overlapped Atoms
        atom_size = radii
        box_overlap = self._get_overlap(coords, atom_size)
        coords = coords[box_overlap]
        radii = radii[box_overlap] if not np.isscalar(radii) else radii
        atom_size = radii

        # Run
        if self.num_blocks > 1:
            blockdim = self.blockdim
            block_overlap_dict = self._get_overlap_blocks(coords, atom_size)

            for xidx, yidx, zidx in itertools.product(range(self.num_blocks), repeat=3):
                start_x, end_x = xidx * blockdim, (xidx + 1) * blockdim
                start_y, end_y = yidx * blockdim, (yidx + 1) * blockdim
                start_z, end_z = zidx * blockdim, (zidx + 1) * blockdim

                out_grid_block = out_grid[:, start_x:end_x, start_y:end_y, start_z:end_z]

                overlap = block_overlap_dict[(xidx, yidx, zidx)]
                if overlap.shape[0] == 0:
                    continue

                grid_block = self.grid_block_dict[(xidx, yidx, zidx)]
                coords_block = coords[overlap]
                radii_block = radii[overlap] if not np.isscalar(radii) else radii
                self._set_grid_single(coords_block, radii_block, grid_block, out_grid_block)
        else:
            self._set_grid_single(coords, radii, self.grid, out_grid)

        return out_grid

    def _check_args_single(self, coords: NDArrayFloat64, radii: Union[float, NDArrayFloat],
                           out_grid: Optional[NDArrayFloat] = None):
        V = coords.shape[0]
        D = H = W = self.dimension
        assert not self.is_radii_type_channel_wise, 'Channel-Wise Radii Type is not supported'
        if self.is_radii_type_scalar:
            assert np.isscalar(radii), 'the radii type of voxelizer is `scalar`, radii should be scalar'
        else:
            assert not np.isscalar(radii), f'the radii type of voxelizer is `atom-wise`, radii should be Array[{V},]'
            assert radii.shape == (V,), f'radii does not match dimension (number of atoms,): {radii.shape} vs {(V,)}'
        if out_grid is not None:
            assert out_grid.shape[0] == 1, 'Output channel should be 1'
            assert out_grid.shape[1:] == (D, H, W), f'Output grid dimension incorrect: {out_grid.shape} vs {("*",D,H,W)}'

    def _set_grid_single(
        self,
        coords: NDArrayFloat64,
        radii: Union[float, NDArrayFloat],
        grid: NDArrayFloat,
        out_grid: NDArrayFloat,
    ) -> NDArrayFloat:
        """
        coords: (V, 3)
        types: (V,)
        radii: scalar or (V, )
        grid: (D, H, W, 3)

        out_grid: (1, D, H, W)
        """
        D, H, W, _ = grid.shape
        grid = grid.reshape(-1, 3)
        res = self._calc_grid(coords, radii, grid)          # (V, D*H*W)
        res = res.reshape(-1, D, H, W)                      # (V, D, H, W)
        np.sum(res, axis=0, keepdims=True, out=out_grid)
        return out_grid

    """ COMMON BLOCK DIVISION """

    def _get_overlap(
        self,
        coords: NDArrayFloat64,
        atom_size: Union[NDArrayFloat, float],
    ) -> NDArrayInt:
        if np.isscalar(atom_size):
            lb_overlap = np.greater(coords, self.lower_bound - atom_size).all(axis=-1)  # (V,)
            ub_overlap = np.less(coords, self.upper_bound + atom_size).all(axis=-1)     # (V,)
        else:
            atom_size = np.expand_dims(atom_size, 1)
            lb_overlap = np.greater(coords + atom_size, self.lower_bound).all(axis=-1)  # (V,)
            ub_overlap = np.less(coords - atom_size, self.upper_bound).all(axis=-1)     # (V,)
        overlap = np.logical_and(lb_overlap, ub_overlap)                                # (V,)
        return np.where(overlap)

    def _get_overlap_blocks(
        self,
        coords: NDArrayFloat64,
        atom_size: Union[NDArray, float]
    ) -> Dict[Tuple[int, int, int], NDArrayInt]:

        def get_axis_overlap_list(coord_1d, atom_size) -> List[NDArrayBool]:
            overlaps = [None] * self.num_blocks
            for i in range(self.num_blocks):
                if i == 0:
                    upper = np.less(coord_1d, self.bounds[i] + atom_size)        # (V,)
                    overlaps[i] = upper
                elif i == self.num_blocks - 1:
                    lower = np.greater(coord_1d, self.bounds[i - 1] - atom_size)   # (V,)
                    overlaps[i] = lower
                else:
                    lower = np.greater(coord_1d, self.bounds[i - 1] - atom_size)   # (V,)
                    upper = np.less(coord_1d, self.bounds[i] + atom_size)        # (V,)
                    overlaps[i] = np.logical_and(lower, upper)
            return overlaps

        overlap_dict = {key: None for key in self.grid_block_dict.keys()}
        x, y, z = np.split(coords, 3, axis=1)
        if not np.isscalar(atom_size):
            atom_size = np.expand_dims(atom_size, 1)
        x_overlap_list = get_axis_overlap_list(x, atom_size)
        y_overlap_list = get_axis_overlap_list(y, atom_size)
        z_overlap_list = get_axis_overlap_list(z, atom_size)
        for xidx, yidx, zidx in itertools.product(range(self.num_blocks), repeat=3):
            x_overlap = x_overlap_list[xidx]
            y_overlap = y_overlap_list[yidx]
            z_overlap = z_overlap_list[zidx]
            overlap_dict[(xidx, yidx, zidx)] = np.where(x_overlap & y_overlap & z_overlap)[0]
        return overlap_dict

    """ COMMON - GRID CALCULATION """

    def _calc_grid(
        self,
        coords: NDArrayFloat64,
        radii: Union[float, NDArrayFloat],
        grid: NDArrayFloat,
    ) -> NDArrayFloat:
        """
        coords: (V, 3)
        radii: scalar or (V, )
        grid: (D*H*W, 3)

        out_grid: (V, D*H*W)
        """
        dist = cdist(coords, grid)              # (V, DHW)
        dist = dist.astype(self.fp)          # np.float64 -> self.fp
        if not np.isscalar(radii):
            radii = np.expand_dims(radii, -1)
        dr = np.divide(dist, radii)
        if self.is_density_type_gaussian:
            return self.__calc_grid_density_gaussian(dr)
        else:
            return self.__calc_grid_density_binary(dr)

    def __calc_grid_density_binary(self, dr: NDArrayFloat) -> NDArrayFloat:
        return np.less_equal(dr, 1., dr)

    def __calc_grid_density_gaussian(self, dr: NDArrayFloat) -> NDArrayFloat:
        out_grid = np.exp(-0.5 * ((dr / self._sigma) ** 2))
        out_grid[dr > 1.] = 0
        return out_grid

    @staticmethod
    def _dtypechange(array: NDArray, dtype) -> NDArray:
        if array.dtype != dtype:
            return array.astype(dtype)
        else:
            return array

    @classmethod
    def _asarray(cls, array: ArrayLike, dtype) -> NDArray:
        if isinstance(array, np.ndarray):
            return cls._dtypechange(array, dtype)
        else:
            return np.array(array, dtype=dtype)

    def asarray(self, array: ArrayLike, obj) -> NDArray:
        if obj in ['coords', 'center']:
            return self._asarray(array, np.float64)
        elif obj in ['features', 'radii']:
            return self._asarray(array, self.fp)
        elif obj == 'types':
            return self._asarray(array, np.int16)
        raise ValueError("obj should be ['coords', 'center', 'radii', types', 'features']")

    @staticmethod
    def do_random_transform(coords, center, random_translation, random_rotation):
        return do_random_transform(coords, center, random_translation, random_rotation)
