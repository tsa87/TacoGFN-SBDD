from rdkit import Chem
import numpy as np

from rdkit.Chem import Atom, Bond, Mol
from typing import Any, Dict, Tuple, List
from numpy.typing import ArrayLike, NDArray

from .getter import *

NDArrayInt = NDArray[np.int16]
NDArrayFloat32 = NDArray[np.float32]
NDArrayFloat64 = NDArray[np.float64]

class PointCloudMaker() :
    def __init__(self, channels: List[str]) :
        self.channels = channels
        self.num_channels = len(channels)

    def split_channel(self, image: ArrayLike) -> Dict[str, ArrayLike] :
        assert np.shape(image)[0] == self.num_channels
        return {channel_name: channel for channel_name, channel in zip(self.channels, image)}

    def run(self, *args, **kwargs) :
        raise NotImplemented

    def __call__(self, *args, **kwargs) :
        return self.run(*args, **kwargs)

""" MOLECULE"""
class MolPointCloudMaker(PointCloudMaker) :
    def __init__(self,
            atom_getter: AtomChannelGetter,
            bond_getter: Optional[BondChannelGetter] = None,
            channel_type: str = 'features',
        ) :
        assert channel_type in ['features', 'types'], f"channel_type(input: {channel_type}) should be 'features' or 'types'"
        self.channel_type = channel_type
        self.use_features = use_features = (channel_type == 'features')
        if use_features :
            assert atom_getter.CHANNEL_TYPE in ['TYPE', 'FEATURE']
            if bond_getter is not None :
                assert bond_getter.CHANNEL_TYPE in ['TYPE', 'FEATURE']
        else :
            assert atom_getter.CHANNEL_TYPE == 'TYPE'
            if bond_getter is not None :
                assert bond_getter.CHANNEL_TYPE == 'TYPE'

        self.atom_getter = atom_getter
        self.bond_getter = bond_getter
        if bond_getter is not None :
            self.use_bond = True
            self.num_atom_channels = atom_getter.num_channels
            self.num_bond_channels = bond_getter.num_channels
            channels = atom_getter.channels + bond_getter.channels
        else :
            self.use_bond = False
            self.num_atom_channels = atom_getter.num_channels
            channels = atom_getter.channels
        super(MolPointCloudMaker, self).__init__(channels)

        if use_features :
            self.setup_features()
        else :
            self.setup_types()

    def run(self, rdmol: Mol, **kwargs) -> Tuple[NDArrayFloat64, NDArray] :
        coords = kwargs.get('kwargs', self.get_coords(rdmol))
        channels = kwargs.get('channels', self.get_channels(rdmol, **kwargs))
        return coords, channels

    def __call__(self, *args, **kwargs) :
        return self.run(*args, **kwargs)

    def get_coords(self, rdmol: Mol) -> NDArrayFloat32:
        conf = rdmol.GetConformer()
        atom_coords = conf.GetPositions()
        if self.use_bond :
            beginatoms = [bond.GetBeginAtomIdx() for bond in rdmol.GetBonds()]
            endatoms = [bond.GetEndAtomIdx() for bond in rdmol.GetBonds()]
            bond_coords = (atom_coords[beginatoms] + atom_coords[endatoms]) / 2
            coords = np.concatenate([atom_coords, bond_coords], axis=0)
        else :
            coords = atom_coords
        return coords

    def get_channels(self, rdmol: Mol, out: Optional[NDArray] = None, **kwargs) -> NDArray:
        if self.use_features :
            return self.get_features(rdmol, out, **kwargs)
        else :
            return self.get_types(rdmol, out, **kwargs)

    """ FEATURES """
    def setup_features(self) :
        self.atom_st = 0
        self.atom_end = self.atom_st + self.num_atom_channels
        if self.use_bond :
            self.bond_st = self.atom_end
            self.bond_end = self.bond_st + self.num_bond_channels

    def get_features(self, rdmol: Mol, out: Optional[NDArrayFloat32] = None, **kwargs) -> NDArrayFloat32:
        if out is None :
            if self.use_bond :
                size = (rdmol.GetNumAtoms() + rdmol.GetNumBonds(), self.num_channels)
            else :
                size = (rdmol.GetNumAtoms(), self.num_channels)
            out = np.zeros(size, dtype=np.float32)
        else :
            out.fill(0)
        return self._get_features(rdmol, out, **kwargs)

    def _get_features(self, rdmol: Mol, out: NDArrayFloat32, **kwargs) -> NDArrayFloat32:
        if self.use_bond :
            num_atoms = rdmol.GetNumAtoms()
            num_bonds = rdmol.GetNumBonds()
            atom_st, atom_end = self.atom_st, self.atom_end
            bond_st, bond_end = self.bond_st, self.bond_end
            atom_features = [self.__get_atom_feature(atom, **kwargs) for atom in rdmol.GetAtoms()]
            bond_features = [self.__get_bond_feature(bond, **kwargs) for bond in rdmol.GetBonds()]
            if num_atoms > 0 :
                out[:num_atoms, atom_st:atom_end] = atom_features
            if num_bonds > 0 :
                out[num_atoms:, bond_st:bond_end] = bond_features
        else :
            num_atoms = rdmol.GetNumAtoms()
            atom_st, atom_end = self.atom_st, self.atom_end
            atom_features = [self.__get_atom_feature(atom, **kwargs) for atom in rdmol.GetAtoms()]
            if num_atoms > 0 :
                out[:, atom_st:atom_end] = atom_features
        return out

    def __get_atom_feature(self, atom: Atom, **kwargs) -> ArrayLike:
        return self.atom_getter.get_feature(atom, **kwargs)
    def __get_bond_feature(self, bond: Bond, **kwargs) -> ArrayLike:
        assert self.bond_getter is not None
        return self.bond_getter.get_feature(bond, **kwargs)

    """ TYPES """
    def setup_types(self) :
        self.atom_start_index = 0
        if self.use_bond :
            self.bond_start_index = self.atom_start_index + self.num_atom_channels

    def get_types(self, rdmol: Mol, out: Optional[NDArrayInt] = None, **kwargs) -> NDArrayInt:
        assert self.use_features is False
        if out is None :
            if self.use_bond :
                size = (rdmol.GetNumAtoms() + rdmol.GetNumBonds(),)
            else :
                size = (rdmol.GetNumAtoms(),)
            out = np.empty(size, dtype=np.int16)
        else :
            out.fill(0)
        return self._get_types(rdmol, out, **kwargs)

    def _get_types(self, rdmol: Mol, out: NDArrayInt, **kwargs) -> NDArrayInt :
        if self.use_bond :
            num_atoms = rdmol.GetNumAtoms()
            num_bonds = rdmol.GetNumBonds()
            atom_types = [self.__get_atom_type(atom, **kwargs) for atom in rdmol.GetAtoms()]
            bond_types = [self.__get_bond_type(bond, **kwargs) for bond in rdmol.GetBonds()]
            if num_atoms > 0 :
                out[:num_atoms] = atom_types
            if num_bonds > 0 :
                out[num_atoms:] = bond_types
        else :
            out[:] = [self.__get_atom_type(atom, **kwargs) for atom in rdmol.GetAtoms()]
        return out
    def __get_atom_type(self, atom: Atom, **kwargs) -> int:
        return self.atom_getter.get_type(atom, **kwargs) + self.atom_start_index
    def __get_bond_type(self, bond: Bond, **kwargs) -> int:
        assert self.bond_getter is not None
        return self.bond_getter.get_type(bond, **kwargs) + self.bond_start_index

class _MolElementPointCloudMaker(MolPointCloudMaker) :
    def __init__(self, atom_getter, bond_getter, channel_type, start_index) :
        self.start_index = start_index
        super(_MolElementPointCloudMaker, self).__init__(atom_getter, bond_getter, channel_type)

    """ FEATURES """
    def setup_features(self) :
        self.atom_st = self.start_index
        self.atom_end = self.atom_st + self.num_atom_channels
        if self.use_bond :
            self.bond_st = self.atom_end
            self.bond_end = self.bond_st + self.num_bond_channels

    """ TYPES """
    def setup_types(self) :
        self.atom_start_index = self.start_index
        if self.use_bond :
            self.bond_start_index = self.atom_start_index + self.num_atom_channels

class MolSystemPointCloudMaker(PointCloudMaker) :
    def __init__(self, *args, channel_type: str = 'features') :
        ## type: self, *args: Tuple[AtomChannelGetter, Optional[BondChannelGetter]], mode: str
        assert channel_type in ['features', 'types'], f"channel_type(input: {channel_type}) should be 'features' or 'types'"
        self.channel_type = channel_type
        self.use_features = (channel_type == 'features')

        self.maker_list = []
        channel_offset = 0
        channels = []
        for arg in args :
            if isinstance(arg, MolPointCloudMaker) :
                maker = _MolElementPointCloudMaker(arg.atom_getter, arg.bond_getter, channel_type, channel_offset)
            else :
                atom_getter, bond_getter = arg
                maker = _MolElementPointCloudMaker(atom_getter, bond_getter, channel_type, channel_offset)
            self.maker_list.append(maker)
            channel_offset += maker.num_channels
            channels += maker.channels
        super(MolSystemPointCloudMaker, self).__init__(channels)
    
    def run(self, rdmol_list: List[Mol], **kwargs) -> Tuple[NDArrayFloat64, NDArray]:
        coords = kwargs.get('kwargs', self.get_coords(rdmol_list))
        channels = kwargs.get('channels', self.get_channels(rdmol_list, **kwargs))
        return coords, channels

    def get_coords(self, rdmol_list: Mol) -> NDArrayFloat64:
        coords_list: List[NDArrayFloat64] = []
        for rdmol, maker in zip(rdmol_list, self.maker_list) :
            conf = rdmol.GetConformer()
            atom_coords = conf.GetPositions()
            coords_list.append(atom_coords)
            if maker.use_bond :
                beginatoms = [bond.GetBeginAtomIdx() for bond in rdmol.GetBonds()]
                endatoms = [bond.GetEndAtomIdx() for bond in rdmol.GetBonds()]
                bond_coords = (atom_coords[beginatoms] + atom_coords[endatoms]) / 2
                coords_list.append(bond_coords)
        coords = np.concatenate(coords_list, axis=0)
        return coords

    def get_channels(self, rdmol_list: List[Mol], out: Optional[NDArray] = None, **kwargs) -> NDArray:
        if self.use_features :
            return self.get_features(rdmol_list, out, **kwargs)
        else :
            return self.get_types(rdmol_list, out, **kwargs)

    def split_channel(self, image) -> List[Dict[str, ArrayLike]]:
        channel_offset = 0
        grid_dist_list = []
        for maker in self.maker_list :
            mol_image = image[channel_offset: channel_offset + maker.num_channels]
            grid_dist_list.append(maker.split_channel(mol_image))
            channel_offset += maker.num_channels
        return grid_dist_list

    def get_features(self, rdmol_list: List[Mol], out: Optional[NDArrayFloat32], **kwargs) -> NDArrayFloat32:
        if out is None :
            num_objects = 0
            for rdmol, maker in zip(rdmol_list, self.maker_list) :
                if maker.use_bond :
                    num_objects += rdmol.GetNumAtoms() + rdmol.GetNumBonds()
                else :
                    num_objects += rdmol.GetNumAtoms()
            size = (num_objects, self.num_channels)
            out = np.zeros(size, dtype=np.float32)
        else :
            out.fill(0)
        return self._get_features(rdmol_list, out, **kwargs)

    def _get_features(self, rdmol_list: List[Mol], out: NDArrayFloat32, **kwargs) -> NDArrayFloat32 :
        object_offset = 0
        for rdmol, maker in zip(rdmol_list, self.maker_list) :
            if maker.use_bond :
                num_objects = rdmol.GetNumAtoms() + rdmol.GetNumBonds()
            else :
                num_objects = rdmol.GetNumAtoms()
            maker._get_features(rdmol, out[object_offset:object_offset+num_objects], **kwargs) 
            object_offset += num_objects
        return out

    def get_types(self, rdmol_list: List[Mol], out: Optional[NDArrayInt], **kwargs) -> NDArrayInt:
        assert self.use_features is False
        if out is None :
            num_objects = 0
            for rdmol, maker in zip(rdmol_list, self.maker_list) :
                if maker.use_bond :
                    num_objects+= rdmol.GetNumAtoms() + rdmol.GetNumBonds()
                else :
                    num_objects += rdmol.GetNumAtoms()
            size = (num_objects,)
            out = np.empty(size, dtype=np.int16)
        return self._get_types(rdmol_list, out, **kwargs)
    
    def _get_types(self, rdmol_list: List[Mol], out: NDArrayInt, **kwargs) -> NDArrayInt :
        object_offset = 0
        for rdmol, maker in zip(rdmol_list, self.maker_list) :
            if maker.use_bond :
                num_objects = rdmol.GetNumAtoms() + rdmol.GetNumBonds()
            else :
                num_objects = rdmol.GetNumAtoms()
            maker._get_types(rdmol, out[object_offset:object_offset+num_objects], **kwargs) 
            object_offset += num_objects
        return out

class ComplexPointCloudMaker(MolSystemPointCloudMaker) :
    def __init__(
            self,
            ligand_atom_getter: AtomChannelGetter,
            ligand_bond_getter: Optional[BondChannelGetter],
            protein_atom_getter: AtomChannelGetter,
            protein_bond_getter: Optional[BondChannelGetter],
            channel_type: str = 'features'
    ) :
        super(ComplexPointCloudMaker, self).__init__(
                (ligand_atom_getter, ligand_bond_getter),
                (protein_atom_getter, protein_bond_getter),
                channel_type
        )
