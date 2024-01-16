import math
import torch
import numpy as np
import itertools

from torch import Tensor, FloatTensor, LongTensor, BoolTensor
from typing import Tuple, Union, Optional, Dict, List

from molvoxel.voxelizer.base import BaseVoxelizer
from .transform import do_random_transform, RandomTransform


class Voxelizer(BaseVoxelizer):
    LIB = 'PyTorch'
    transform_class = RandomTransform

    def __init__(
        self,
        resolution: float = 0.5,
        dimension: int = 64,
        radii_type: str = 'scalar',
        density_type: str = 'gaussian',
        device: str = 'cpu',
        blockdim: Optional[int] = None,
        **kwargs,
    ):
        super(Voxelizer, self).__init__(resolution, dimension, radii_type, density_type, **kwargs)
        self.device = torch.device(device)
        self.gpu = (self.device != torch.device('cpu'))
        self._setup_block(blockdim)

    def _setup_block(self, blockdim):
        if blockdim is None:
            if self.gpu:
                blockdim = math.ceil(self.dimension / 2)
            else:
                blockdim = 8
        self.blockdim = blockdim

        axis = torch.arange(self.dimension, dtype=torch.float, device=self.device) * self.resolution - (self.width / 2.)
        self.num_blocks = num_blocks = math.ceil(self.dimension / blockdim)
        if self.num_blocks > 1:
            self.grid = None
            self.grid_block_dict: Dict[Tuple[int, int, int], FloatTensor] = {}
            for xidx, yidx, zidx in itertools.product(range(self.num_blocks), repeat=3):
                x_axis = axis[xidx * blockdim: (xidx + 1) * blockdim]
                y_axis = axis[yidx * blockdim: (yidx + 1) * blockdim]
                z_axis = axis[zidx * blockdim: (zidx + 1) * blockdim]
                grid_block = torch.stack(torch.meshgrid([x_axis, y_axis, z_axis], indexing='ij'), dim=-1)
                self.grid_block_dict[(xidx, yidx, zidx)] = grid_block

            self.bounds = [(axis[idx * blockdim].item() + (self.resolution / 2.))
                           for idx in range(1, num_blocks)]
        else:
            self.grid_block_dict = None
            self.grid = torch.stack(torch.meshgrid([axis, axis, axis], indexing='ij'), dim=-1)

    def get_empty_grid(self, num_channels: int, batch_size: Optional[int] = None, init_zero: bool = False) -> FloatTensor:
        if init_zero:
            if batch_size is None:
                return torch.zeros(self.grid_dimension(num_channels), device=self.device)
            else:
                return torch.zeros((batch_size,) + self.grid_dimension(num_channels), device=self.device)
        else:
            if batch_size is None:
                return torch.empty(self.grid_dimension(num_channels), device=self.device)
            else:
                return torch.empty((batch_size,) + self.grid_dimension(num_channels), device=self.device)

    """ DEVICE """

    def to(self, device, update_blockdim: bool = True, blockdim: Optional[int] = None):
        device = torch.device(device)
        if device == self.device:
            return
        self.device = device
        self.gpu = (device != torch.device('cpu'))

        if update_blockdim:
            self._setup_block(blockdim)
        return self

    def cuda(self, update_blockdim: bool = True, blockdim: Optional[int] = None):
        return self.to('cuda', update_blockdim, blockdim)

    def cpu(self, update_blockdim: bool = True, blockdim: Optional[int] = None):
        return self.to('cpu', update_blockdim, blockdim)

    """ Forward """

    def forward(
        self,
        coords: FloatTensor,
        center: Optional[FloatTensor],
        channels: Union[FloatTensor, LongTensor, None],
        radii: Union[float, FloatTensor],
        random_translation: float = 0.0,
        random_rotation: bool = False,
        out_grid: Optional[FloatTensor] = None
    ) -> FloatTensor:
        if channels is None:
            return self.forward_single(coords, center, radii, random_translation, random_rotation, out_grid)
        elif channels.dim() == 1:
            types = channels
            return self.forward_types(coords, center, types, radii, random_translation, random_rotation, out_grid)
        else:
            features = channels
            return self.forward_features(coords, center, features, radii, random_translation, random_rotation, out_grid)

    __call__ = forward

    """ VECTOR """
    @torch.no_grad()
    def forward_features(
        self,
        coords: FloatTensor,
        center: Optional[FloatTensor],
        features: FloatTensor,
        radii: Union[float, FloatTensor],
        random_translation: float = 0.0,
        random_rotation: bool = False,
        out_grid: Optional[FloatTensor] = None
    ) -> FloatTensor:
        """unsqueeze
        coords: (V, 3)
        center: (3,)
        features: (V, C)
        radii: scalar or (V, ) of (C, )
        random_translation: float (nonnegative)
        random_rotation: bool

        out_grid: (C,D,H,W)
        """
        coords = self.asarray(coords, 'coords')
        center = self.asarray(center, 'center')
        features = self.asarray(features, 'features')
        if not isinstance(radii, float):
            radii = self.asarray(radii, 'radii')
        self._check_args_features(coords, features, radii, out_grid)

        # Set Coordinate
        if center is not None:
            coords = coords - center.view(1, 3)
        coords = do_random_transform(coords, None, random_translation, random_rotation)

        # Set Out
        if out_grid is None:
            C = features.size(1)
            out_grid = self.get_empty_grid(C)

        # Clipping Overlapped Atoms

        atom_size = radii.max().item() if self.is_radii_type_channel_wise else radii
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
                if overlap.size(0) == 0:
                    out_grid_block.fill_(0.)
                    continue

                grid_block = self.grid_block_dict[(xidx, yidx, zidx)]
                coords_block, features_block = coords[overlap], features[overlap]
                radii_block = radii[overlap] if self.is_radii_type_atom_wise else radii
                self._set_grid_features(coords_block, features_block, radii_block, grid_block, out_grid_block)
        else:
            self._set_grid_features(coords, features, radii, self.grid, out_grid)

        return out_grid

    def _check_args_features(self, coords: FloatTensor, features: FloatTensor, radii: Union[float, FloatTensor],
                             out_grid: Optional[FloatTensor] = None):
        V = coords.shape[0]
        C = features.shape[1]
        D = H = W = self.dimension
        assert features.shape[0] == V, f'atom features does not match number of atoms: {features.shape[0]} vs {V}'
        assert features.ndim == 2, f"atom features does not match dimension: {features.shape} vs {(V,'*')}"
        if self.is_radii_type_scalar:
            assert np.isscalar(radii), f'the radii type of voxelizer is `scalar`, radii should be scalar'
        elif self.is_radii_type_channel_wise:
            assert not np.isscalar(radii), f'the radii type of voxelizer is `channel-wise`, radii should be Tensor[{C},]'
            assert radii.shape == (C,), f'radii does not match dimension (number of channels,): {radii.shape} vs {(C,)}'
        else:
            assert not np.isscalar(radii), f'the radii type of voxelizer is `atom-wise`, radii should be Tensor[{V},]'
            assert radii.shape == (V,), f'radii does not match dimension (number of atoms,): {radii.shape} vs {(V,)}'
        if out_grid is not None:
            assert out_grid.shape == (C, D, H, W), f'Output grid dimension incorrect: {out_grid.shape} vs {(C,D,H,W)}'

    def _set_grid_features(
        self,
        coords: FloatTensor,
        features: FloatTensor,
        radii: Union[float, FloatTensor],
        grid: FloatTensor,
        out_grid: FloatTensor,
    ) -> FloatTensor:
        """
        coords: (V, 3)
        features: (V, C)
        radii: scalar or (V, ) or (C, )
        grid: (D, H, W, 3)

        out_grid: (C, D, H, W)
        """
        features = features.T                                               # (V, C) -> (C, V)
        D, H, W, _ = grid.size()
        grid = grid.view(-1, 3)                                             # (DHW, 3)
        if self.is_radii_type_channel_wise:
            if out_grid.is_contiguous():
                _out_grid = out_grid.view(-1, D * H * W)
                for type_idx in range(features.size(0)):
                    typ = features[type_idx]                                # (V,)
                    res = self._calc_grid(coords, radii[type_idx], grid)    # (V, DHW)
                    torch.matmul(typ, res, out=_out_grid[type_idx])         # (V,) @ (V, DHW) -> (DHW) = (D, H, W)
            else:
                for type_idx in range(features.size(0)):
                    typ = features[type_idx]                                # (V,)
                    res = self._calc_grid(coords, radii[type_idx], grid)    # (V, DHW)
                    out_grid[type_idx] = torch.matmul(typ, res).view(D, H, W)    # (D, H, W)
        else:
            if out_grid.is_contiguous():
                res = self._calc_grid(coords, radii, grid)                  # (V, DHW)
                torch.mm(features, res, out=out_grid.view(-1, D * H * W))       # (V,C) @ (V, DHW) -> (C, DHW)
            else:
                res = self._calc_grid(coords, radii, grid)                  # (V, DHW)
                out_grid[:] = torch.mm(features, res).view(-1, D, H, W)     # (C, D, H, W)
        return out_grid

    """ INDEX """
    @torch.no_grad()
    def forward_types(
        self,
        coords: FloatTensor,
        center: Optional[FloatTensor],
        types: FloatTensor,
        radii: Union[float, FloatTensor],
        random_translation: float = 0.0,
        random_rotation: bool = False,
        out_grid: Optional[FloatTensor] = None
    ) -> FloatTensor:
        """
        coords: (V, 3)
        center: (3,)
        types: (V,)
        radii: scalar or (V, ) of (C, )
        random_translation: float (nonnegative)
        random_rotation: bool

        out_grid: (C,D,H,W)
        """
        coords = self.asarray(coords, 'coords')
        center = self.asarray(center, 'center')
        types = self.asarray(types, 'types')
        if not isinstance(radii, float):
            radii = self.asarray(radii, 'radii')
        self._check_args_types(coords, types, radii, out_grid)

        # Set Coordinate
        if center is not None:
            coords = coords - center.view(1, 3)
        coords = do_random_transform(coords, None, random_translation, random_rotation)

        # Set Out
        if out_grid is None:
            if self.is_radii_type_channel_wise:
                C = radii.size(0)
            else:
                C = torch.max(types).item() + 1
            out_grid = self.get_empty_grid(C, init_zero=True)
        else:
            out_grid.fill_(0.)

        # Clipping Overlapped Atoms
        if self.is_radii_type_channel_wise:
            radii = radii[types]           # (C, ) -> (V, )
        atom_size = radii
        box_overlap = self._get_overlap(coords, atom_size)
        coords, types = coords[box_overlap], types[box_overlap]
        radii = radii[box_overlap] if isinstance(radii, Tensor) else radii
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
                if overlap.size(0) == 0:
                    continue

                grid_block = self.grid_block_dict[(xidx, yidx, zidx)]
                coords_block, types_block = coords[overlap], types[overlap]
                radii_block = radii[overlap] if isinstance(radii, Tensor) else radii
                self._set_grid_types(coords_block, types_block, radii_block, grid_block, out_grid_block)
        else:
            self._set_grid_types(coords, types, radii, self.grid, out_grid)

        return out_grid

    def _check_args_types(self, coords: FloatTensor, types: FloatTensor, radii: Union[float, FloatTensor],
                          out_grid: Optional[FloatTensor] = None):
        V = coords.size(0)
        C = torch.max(types).item() + 1
        D = H = W = self.dimension
        assert types.shape == (V,), f"types does not match dimension: {types.shape} vs {(V,)}"
        if self.is_radii_type_scalar:
            assert np.isscalar(radii), f'the radii type of voxelizer is `scalar`, radii should be scalar'
        elif self.is_radii_type_channel_wise:
            assert not np.isscalar(radii), f'the radii type of voxelizer is `channel-wise`, radii should be Tensor[{C},]'
            assert radii.shape == (C,), f'radii does not match dimension (number of channels,): {radii.shape} vs {(C,)}'
        else:
            assert not np.isscalar(radii), f'the radii type of voxelizer is `atom-wise`, radii should be Tensor[{V},]'
            assert radii.shape == (V,), f'radii does not match dimension (number of atoms,): {radii.shape} vs {(V,)}'
        if out_grid is not None:
            assert out_grid.shape[0] >= C, f'Output channel is less than number of types: {out_grid.shape[0]} < {C}'
            assert out_grid.shape[1:] == (D, H, W), f'Output grid dimension incorrect: {out_grid.shape} vs {("*",D,H,W)}'

    def _set_grid_types(
        self,
        coords: FloatTensor,
        types: FloatTensor,
        radii: Union[float, FloatTensor],
        grid: FloatTensor,
        out_grid: FloatTensor,
    ) -> FloatTensor:
        """
        coords: (V, 3)
        types: (V,)
        radii: scalar or (V, )
        grid: (D, H, W, 3)

        out_grid: (C, D, H, W)
        """
        D, H, W, _ = grid.size()
        grid = grid.view(-1, 3)
        res = self._calc_grid(coords, radii, grid)          # (V, D*H*W)
        res = res.view(-1, D, H, W)                         # (V, D, H, W)
        types = types.view(-1, 1, 1, 1).expand(res.size())  # (V, D, H, W)
        return out_grid.scatter_add_(0, types, res)

    """ SINGLE """
    @torch.no_grad()
    def forward_single(
        self,
        coords: FloatTensor,
        center: Optional[FloatTensor],
        radii: Union[float, FloatTensor],
        random_translation: float = 0.0,
        random_rotation: bool = False,
        out_grid: Optional[FloatTensor] = None
    ) -> FloatTensor:
        """
        coords: (V, 3)
        center: (3,)
        radii: scalar or (V, )
        random_translation: float (nonnegative)
        random_rotation: bool

        out_grid: (C,D,H,W)
        """
        coords = self.asarray(coords, 'coords')
        center = self.asarray(center, 'center')
        if not isinstance(radii, float):
            radii = self.asarray(radii, 'radii')
        self._check_args_single(coords, radii, out_grid)

        # Set Coordinate
        if center is not None:
            coords = coords - center.view(1, 3)
        coords = do_random_transform(coords, None, random_translation, random_rotation)

        # Set Out
        if out_grid is None:
            out_grid = self.get_empty_grid(1, init_zero=True)
        else:
            out_grid.fill_(0.)

        # Clipping Overlapped Atoms
        atom_size = radii
        box_overlap = self._get_overlap(coords, atom_size)
        coords = coords[box_overlap]
        radii = radii[box_overlap] if isinstance(radii, Tensor) else radii
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
                if overlap.size(0) == 0:
                    continue

                grid_block = self.grid_block_dict[(xidx, yidx, zidx)]
                coords_block = coords[overlap]
                radii_block = radii[overlap] if isinstance(radii, Tensor) else radii
                self._set_grid_single(coords_block, radii_block, grid_block, out_grid_block)
        else:
            self._set_grid_single(coords, radii, self.grid, out_grid)

        return out_grid

    def _check_args_single(self, coords: FloatTensor, radii: Union[float, FloatTensor],
                           out_grid: Optional[FloatTensor] = None):
        V = coords.size(0)
        D = H = W = self.dimension
        assert not self.is_radii_type_channel_wise, 'Channel-Wise Radii Type is not supported'
        if self.is_radii_type_scalar:
            assert np.isscalar(radii), f'the radii type of voxelizer is `scalar`, radii should be scalar'
        else:
            assert not np.isscalar(radii), f'the radii type of voxelizer is `atom-wise`, radii should be Tensor[{V},]'
            assert radii.shape == (V,), f'radii does not match dimension (number of atoms,): {radii.shape} vs {(V,)}'
        if out_grid is not None:
            assert out_grid.shape[0] == 1, 'Output channel should be 1'
            assert out_grid.shape[1:] == (D, H, W), f'Output grid dimension incorrect: {out_grid.shape} vs {("*",D,H,W)}'

    def _set_grid_single(
        self,
        coords: FloatTensor,
        radii: Union[float, FloatTensor],
        grid: FloatTensor,
        out_grid: FloatTensor,
    ) -> FloatTensor:
        """
        coords: (V, 3)
        radii: scalar or (V, )
        grid: (D, H, W, 3)

        out_grid: (1, D, H, W)
        """
        D, H, W, _ = grid.size()
        grid = grid.view(-1, 3)
        res = self._calc_grid(coords, radii, grid)          # (V, D*H*W)
        res = res.view(-1, D, H, W)                         # (V, D, H, W)
        torch.sum(res, dim=0, keepdim=True, out=out_grid)
        return out_grid

    """ COMMON BLOCK DIVISION """

    def _get_overlap(
        self,
        coords: FloatTensor,
        atom_size: Union[FloatTensor, float],
    ) -> LongTensor:
        if isinstance(atom_size, Tensor) and atom_size.dim() == 1:
            atom_size = atom_size.unsqueeze(1)
            lb_overlap = torch.greater(coords + atom_size, self.lower_bound).all(dim=-1)    # (V,)
            ub_overlap = torch.less(coords - atom_size, self.upper_bound).all(dim=-1)       # (V,)
        else:
            lb_overlap = torch.greater(coords, self.lower_bound - atom_size).all(dim=-1)    # (V,)
            ub_overlap = torch.less(coords, self.upper_bound + atom_size).all(dim=-1)       # (V,)
        overlap = lb_overlap.logical_and_(ub_overlap)                                       # (V,)
        return torch.where(overlap)

    def _get_overlap_blocks(
        self,
        coords: FloatTensor,
        atom_size: Union[FloatTensor, float]
    ) -> Dict[Tuple[int, int, int], LongTensor]:

        def get_axis_overlap_list(coord_1d, atom_size) -> List[BoolTensor]:
            overlaps = [None] * self.num_blocks
            for i in range(self.num_blocks):
                if i == 0:
                    upper = torch.less(coord_1d, self.bounds[i] + atom_size)        # (V,)
                    overlaps[i] = upper
                elif i == self.num_blocks - 1:
                    lower = torch.greater(coord_1d, self.bounds[i - 1] - atom_size)   # (V,)
                    overlaps[i] = lower
                else:
                    lower = torch.greater(coord_1d, self.bounds[i - 1] - atom_size)   # (V,)
                    upper = torch.less(coord_1d, self.bounds[i] + atom_size)        # (V,)
                    overlaps[i] = lower.logical_and_(upper)
            return overlaps

        overlap_dict = {key: None for key in self.grid_block_dict.keys()}
        x, y, z = torch.unbind(coords, dim=-1)

        x_overlap_list = get_axis_overlap_list(x, atom_size)
        y_overlap_list = get_axis_overlap_list(y, atom_size)
        z_overlap_list = get_axis_overlap_list(z, atom_size)
        for xidx, yidx, zidx in itertools.product(range(self.num_blocks), repeat=3):
            x_overlap = x_overlap_list[xidx]
            y_overlap = y_overlap_list[yidx]
            z_overlap = z_overlap_list[zidx]
            overlap_dict[(xidx, yidx, zidx)] = torch.where(x_overlap & y_overlap & z_overlap)[0]
        return overlap_dict

    """ COMMON - GRID CALCULATION """

    def _calc_grid(
        self,
        coords: FloatTensor,
        radii: Union[float, FloatTensor],
        grid: FloatTensor,
    ) -> FloatTensor:
        """
        coords: (V, 3)
        radii: scalar or (V, )
        grid: (D*H*W, 3)

        out_grid: (V, D*H*W)
        """
        dist = torch.cdist(coords, grid)                    # (V, D, H, W)
        if isinstance(radii, Tensor):
            radii = radii.unsqueeze(-1)
        dr = dist.div_(radii)                               # (V, D, H, W)
        if self.is_density_type_gaussian:
            return self.__calc_grid_density_gaussian(dr)
        else:
            return self.__calc_grid_density_binary(dr)

    def __calc_grid_density_binary(self, dr: FloatTensor) -> FloatTensor:
        return dr.le_(1.0)

    def __calc_grid_density_gaussian(self, dr: FloatTensor) -> FloatTensor:
        out_grid = dr.div_(self._sigma).pow_(2).mul_(-0.5).exp_()
        out_grid.masked_fill_(dr > 1.0, 0)
        return out_grid

    def asarray(self, array, obj):
        if isinstance(array, np.ndarray):
            array = torch.from_numpy(array)
        if isinstance(array, torch.Tensor):
            if obj in ['coords', 'center', 'features', 'radii']:
                return array.to(device=self.device, dtype=torch.float)
            elif obj == 'types':
                return array.to(device=self.device, dtype=torch.long)
        else:
            if obj in ['coords', 'center', 'features', 'radii']:
                return torch.tensor(array, dtype=torch.float, device=self.device)
            elif obj == 'types':
                return torch.tensor(array, dtype=torch.long, device=self.device)
        raise ValueError("obj should be ['coords', center', 'types', 'features', 'radii']")

    @staticmethod
    def do_random_transform(coords, center, random_translation, random_rotation):
        return do_random_transform(coords, center, random_translation, random_rotation)
