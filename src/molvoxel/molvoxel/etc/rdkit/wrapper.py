import rdkit
from rdkit import Chem
import numpy as np

from rdkit.Chem import Mol
from typing import Optional, Union, Dict, List, Any
from numpy.typing import ArrayLike

from molvoxel.voxelizer import BaseVoxelizer
from .pointcloud import MolPointCloudMaker, MolSystemPointCloudMaker, ComplexPointCloudMaker

class MolWrapper() :
    def __init__(self, pointcloudmaker: MolPointCloudMaker, voxelizer: BaseVoxelizer, visualizer: Optional[Any] = None) :
        self.maker = pointcloudmaker
        self.voxelizer = voxelizer
        self.visualizer = visualizer
        self.num_channels = self.maker.num_channels
        self.channel_type = self.maker.channel_type
        self.grid_dimension = self.voxelizer.grid_dimension(self.num_channels)
        self.resolution = self.voxelizer.resolution

    def run(
        self,
        rdmol: Mol,
        center: Optional[ArrayLike] = None,
        radii: Union[float, ArrayLike] = 1.0,
        random_translation: float = 0.0,
        random_rotation: bool = False,
        out_grid: Optional[ArrayLike] = None,
        **kwargs
    ) -> ArrayLike:
        maker, voxelizer = self.maker, self.voxelizer
        coords, channels = maker.run(rdmol, **kwargs)

        if out_grid is not None :
            assert np.shape(out_grid) == self.grid_dimension
        
        coords = voxelizer.asarray(coords, 'coords')
        center = voxelizer.asarray(center, 'center') if center is not None else center
        channels = voxelizer.asarray(channels, self.channel_type)
        radii = radii if np.isscalar(radii) else voxelizer.asarray(radii, 'radii')

        return voxelizer.forward(coords, center, channels, radii, random_translation, random_rotation, out_grid=out_grid)

    def get_coords(self, rdmol) :
        coords = self.maker.get_coords(rdmol)
        return self.voxelizer.asarray(coords, 'coords')

    def get_channels(self, rdmol) :
        channels = self.maker.get_channels(rdmol)
        return self.voxelizer.asarray(channels, self.channel_type)

    def split_channel(self, image: ArrayLike) -> Dict[str, ArrayLike]:
        return self.maker.split_channel(image)

    def get_empty_grid(self, batch_size: Optional[int] = None, init_zero: bool = False) -> ArrayLike :
        return self.voxelizer.get_empty_grid(self.num_channels, batch_size, init_zero)
    
    def visualize(self, pse_path: str, rdmol: Mol, image: ArrayLike, center: Optional[ArrayLike], new_coords: Optional[ArrayLike] = None) :
        assert self.visualizer is not None
        channel_dict = self.split_channel(image)
        if center is None :
            center = self.voxelizer.asarray([0,0,0], 'center')
        self.visualizer.visualize_mol(pse_path, rdmol, channel_dict, center, self.resolution, new_coords)

class MolSystemWrapper(MolWrapper) :
    def __init__(self, pointcloudmaker: MolSystemPointCloudMaker, voxelizer: BaseVoxelizer, \
            name_list: Optional[List[str]] = None, visualizer: Optional[Any] = None) :
        self.maker = pointcloudmaker
        self.voxelizer = voxelizer
        self.visualizer = visualizer
        self.num_channels = self.maker.num_channels
        self.channel_type = self.maker.channel_type
        self.grid_dimension = self.voxelizer.grid_dimension(self.num_channels)
        self.resolution = self.voxelizer.resolution

        self.name_list = name_list

    def run(
        self,
        rdmol_list: List[Mol],
        center: Optional[ArrayLike] = None,
        radii: Union[float, List[float], List[ArrayLike]] = 1.0,
        random_translation: float = 0.0,
        random_rotation: bool = False,
        out_grid: Optional[ArrayLike] = None,
        **kwargs,
    ) -> ArrayLike:
        maker, voxelizer = self.maker, self.voxelizer
        coords, channels = maker.run(rdmol_list, **kwargs)

        if out_grid is not None :
            assert np.shape(out_grid) == self.grid_dimension
        
        if voxelizer.is_radii_type_scalar :
            pass
        elif voxelizer.is_radii_type_atom_wise :
            if isinstance(radii, list) :
                assert len(radii) == len(rdmol_list)
                radii_list = [[r] * rdmol.GetNumAtoms() for r, rdmol in zip(radii, rdmol_list)]
                radii = np.concatenate(radii_list, dtype=np.float32)
        else :
            if isinstance(radii, list) :
                radii = np.concatenate(radii, dtype=np.float32)

        coords = voxelizer.asarray(coords, 'coords')
        center = voxelizer.asarray(center, 'center') if center is not None else center
        channels = voxelizer.asarray(channels, maker.channel_type)
        radii = radii if np.isscalar(radii) else voxelizer.asarray(radii, 'radii')
        return voxelizer.forward(coords, center, channels, radii, random_translation, random_rotation, out_grid=out_grid)

    def get_coords(self, rdmol_list) :
        coords = self.maker.get_coords(rdmol_list)
        return self.voxelizer.asarray(coords, 'coords')

    def get_channels(self, rdmol_list) :
        channels= self.maker.get_channels(rdmol_list)
        return self.voxelizer.asarray(channels, self.channel_type)

    def split_channel(self, image: ArrayLike) -> List[Dict[str, ArrayLike]]:
        return self.maker.split_channel(image)

    def visualize(
        self,
        pse_path: str,
        rdmol_list: List[Mol],
        image: ArrayLike,
        center: Optional[ArrayLike],
        new_coords: Optional[ArrayLike] = None,
    ) :
        assert self.visualizer is not None
        assert self.name_list is not None, 'name_list should be set'
        channel_dict_list = self.split_channel(image)
        if center is None :
            center = self.voxelizer.asarray([0,0,0], 'center')
        if new_coords is not None :
            new_coords_list = []
            offset = 0
            for rdmol in rdmol_list :
                num_atoms = rdmol.GetNumAtoms()
                new_coords_list.append(new_coords[offset : offset + num_atoms])
                offset += num_atoms
        else :
            new_coords_list = None
        self.visualizer.visualize_system(pse_path, rdmol_list, self.name_list, channel_dict_list, center, self.resolution, new_coords_list)

class ComplexWrapper(MolSystemWrapper) :
    def __init__(self, pointcloudmaker: ComplexPointCloudMaker, voxelizer: BaseVoxelizer, visualizer: Optional[Any] = None) :
        super(ComplexWrapper, self).__init__(pointcloudmaker, voxelizer, ['Ligand', 'Protein'], visualizer)

    def run(
        self,
        ligand_rdmol: Mol,
        protein_rdmol: Mol,
        center: Optional[ArrayLike] = None,
        radii: Union[float, List[float], List[ArrayLike]] = 1.0,
        random_translation: float = 0.0,
        random_rotation: bool = False,
        out_grid: Optional[ArrayLike] = None,
        **kwargs,
    ) -> ArrayLike:
        return super().run([ligand_rdmol, protein_rdmol], center, radii, random_translation, random_rotation, out_grid, **kwargs)

    def get_coords(self, ligand_rdmol, protein_rdmol) :
        return super().get_coords([ligand_rdmol, protein_rdmol])

    def get_channels(self, ligand_rdmol, protein_rdmol) :
        return super().get_channels([ligand_rdmol, protein_rdmol])

    def visualize(
        self,
        pse_path: str,
        ligand_rdmol: Mol,
        protein_rdmol: Mol,
        image: ArrayLike,
        center: Optional[ArrayLike],
        new_coords: Optional[ArrayLike] = None,
    ) :
        assert self.visualizer is not None
        ligand_channel_dict, protein_channel_dict = self.split_channel(image)
        if center is None :
            center = self.voxelizer.asarray([0,0,0], 'center')
        if new_coords is not None :
            num_ligand_atoms = ligand_rdmol.GetNumAtoms()
            ligand_new_coords = new_coords[:num_ligand_atoms]
            protein_new_coords = new_coords[num_ligand_atoms:]
        else :
            ligand_new_coords = None
            protein_new_coords = None
        self.visualizer.visualize_complex(pse_path, ligand_rdmol, protein_rdmol, ligand_channel_dict, protein_channel_dict,
                          center, self.resolution, ligand_new_coords, protein_new_coords)
