import os
import sys
from rdkit import Chem

from molvoxel.voxelizer.numpy import Voxelizer
from molvoxel.etc.rdkit.wrapper import MolWrapper, MolSystemWrapper, ComplexWrapper
from molvoxel.etc.rdkit.pointcloud import MolPointCloudMaker, MolSystemPointCloudMaker
from molvoxel.etc.rdkit.getter import AtomTypeGetter, BondTypeGetter

save_dir = 'result_rdkit'
if '-y' in sys.argv :
    pymol = True
    from molvoxel.etc.pymol import Visualizer
    visualizer = Visualizer()
    os.system(f'mkdir -p {save_dir}')
else :
    pymol = False
    visualizer = None

voxelizer = Voxelizer(dimension = 32)

""" LOAD DATA """
ligand_path = './10gs/10gs_ligand.sdf'
protein_path = './10gs/10gs_protein_nowater.pdb'

ligand_rdmol = Chem.SDMolSupplier(ligand_path)[0]
protein_rdmol = Chem.MolFromPDBFile(protein_path)

ligand_center = ligand_rdmol.GetConformer().GetPositions().mean(axis=0)

""" SINGLE MOL TEST """
rdmol = ligand_rdmol
center = ligand_center
atom_getter = AtomTypeGetter(['C', 'N', 'O', 'S'])
bond_getter = BondTypeGetter.default()

pointcloudmaker = MolPointCloudMaker(atom_getter, None, channel_type='types')
wrapper = MolWrapper(pointcloudmaker, voxelizer, visualizer)
image = wrapper.run(rdmol, center, radii=1.5)
if pymol :
    wrapper.visualize(f'{save_dir}/types.pse', ligand_rdmol, image, center)

pointcloudmaker = MolPointCloudMaker(atom_getter, bond_getter, channel_type='types')
wrapper = MolWrapper(pointcloudmaker, voxelizer, visualizer)
image = wrapper.run(rdmol, center, radii=1.5)
if pymol :
    wrapper.visualize(f'{save_dir}/bond_types.pse', ligand_rdmol, image, center)

pointcloudmaker = MolPointCloudMaker(atom_getter, None, channel_type='features')
wrapper = MolWrapper(pointcloudmaker, voxelizer, visualizer)
image = wrapper.run(rdmol, center, radii=1.5)
if pymol :
    wrapper.visualize(f'{save_dir}/features.pse', ligand_rdmol, image, center)

pointcloudmaker = MolPointCloudMaker(atom_getter, bond_getter, channel_type='features')
wrapper = MolWrapper(pointcloudmaker, voxelizer, visualizer)
image = wrapper.run(rdmol, center, radii=1.5)
if pymol :
    wrapper.visualize(f'{save_dir}/bond_features.pse', ligand_rdmol, image, center)

unknown_atom_getter = AtomTypeGetter(['C', 'N', 'O'], unknown=True)
pointcloudmaker = MolPointCloudMaker(unknown_atom_getter, bond_getter, channel_type='types')
wrapper = MolWrapper(pointcloudmaker, voxelizer, visualizer)
image = wrapper.run(rdmol, center, radii=1.5)
if pymol :
    wrapper.visualize(f'{save_dir}/unknownS.pse', ligand_rdmol, image, center)

""" SYSTEM TEST """
rdmol_list = [ligand_rdmol, protein_rdmol]
center = ligand_center
atom_getter = AtomTypeGetter(['C', 'N', 'O', 'S'])
bond_getter = BondTypeGetter.default()

pointcloudmaker = MolSystemPointCloudMaker([atom_getter, None], [atom_getter, bond_getter], channel_type='types')
wrapper = MolSystemWrapper(pointcloudmaker, voxelizer, ['Ligand', 'Protein'], visualizer)
image = wrapper.run(rdmol_list, center, radii=1.5)
if pymol :
    wrapper.visualize(f'{save_dir}/system.pse', [ligand_rdmol, protein_rdmol], image, center)

""" COMPLEX TEST """
pointcloudmaker = MolSystemPointCloudMaker([atom_getter, bond_getter], [atom_getter, None], channel_type='features')
wrapper = ComplexWrapper(pointcloudmaker, voxelizer, visualizer)
image = wrapper.run(ligand_rdmol, protein_rdmol, center, radii=1.5)
if pymol :
    wrapper.visualize(f'{save_dir}/complex.pse', ligand_rdmol, protein_rdmol, image, center)

