import sys
import os
import numpy as np
from rdkit import Chem
import torch

from utils import apply_coord

def main(Voxelizer, RandomTransform, pymol, device) :
    if pymol :
        from molvoxel.etc.pymol import Visualizer

    """ SET FUNCTION """
    def test(ligand_rdmol, protein_rdmol, atom_radii, save_dir) :
        if pymol :
            os.system(f'mkdir -p {save_dir}')
            visualizer = Visualizer()
        else :
            visualizer = None

        ligand_coords = torch.from_numpy(ligand_rdmol.GetConformer().GetPositions()).float()
        protein_coords = torch.from_numpy(protein_rdmol.GetConformer().GetPositions()).float()
        ligand_center = ligand_coords.mean(axis=0)
        center = ligand_center

        ligand_atom_radii = atom_radii[:ligand_rdmol.GetNumAtoms()]
        protein_atom_radii = atom_radii[ligand_rdmol.GetNumAtoms():]

        voxelizer = Voxelizer(device = device) #resolution=0.5, dimension=64, atom_scale=1.5, radii_type='scalar', density='gaussian'
        voxelizer_small = Voxelizer(0.5, 16, blockdim = 16, device=device)
        voxelizer_hr = Voxelizer(0.4, 64, device=device)

        transform = RandomTransform(random_translation=0.5, random_rotation=True)

        ligand_grid = voxelizer.get_empty_grid(1)
        protein_grid = voxelizer.get_empty_grid(1)

        print('Test 1: Radii Type: Scalar (Default), Density: Gaussian (Default)')
        test_name = 'ref'
        ligand_image = voxelizer.forward_single(ligand_coords, center, radii=1.0, out_grid=ligand_grid)
        protein_image = voxelizer.forward_single(protein_coords, center, radii=1.0, out_grid=protein_grid)
        assert ligand_image is ligand_grid, 'INPLACE FAILE'
        assert protein_image is protein_grid, 'INPLACE FAILE'
        if pymol :
            assert visualizer is not None
            visualizer.visualize_complex(f'{save_dir}/{test_name}.pse', ligand_rdmol, protein_rdmol, {'Atom': ligand_image.squeeze(0)}, {'Atom': protein_image.squeeze(0)}, center, resolution=voxelizer.resolution)

        print('Test 2: Small (One Block)')
        test_name = 'small'
        ligand_image = voxelizer_small.forward_single(ligand_coords, center, radii=1.0)
        protein_image = voxelizer_small.forward_single(protein_coords, center, radii=1.0)
        if pymol :
            assert visualizer is not None
            visualizer.visualize_complex(f'{save_dir}/{test_name}.pse', ligand_rdmol, protein_rdmol, {'Atom': ligand_image.squeeze(0)}, {'Atom': protein_image.squeeze(0)}, center, resolution=voxelizer.resolution)

        print('Test 3: High Resolution')
        test_name = 'hr'
        ligand_image = voxelizer_hr.forward_single(ligand_coords, center, radii=1.0, out_grid=ligand_grid)
        protein_image = voxelizer_hr.forward_single(protein_coords, center, radii=1.0, out_grid=protein_grid)
        assert ligand_image is ligand_grid, 'INPLACE FAILE'
        assert protein_image is protein_grid, 'INPLACE FAILE'
        if pymol :
            assert visualizer is not None
            visualizer.visualize_complex(f'{save_dir}/{test_name}.pse', ligand_rdmol, protein_rdmol, {'Atom': ligand_image.squeeze(0)}, {'Atom': protein_image.squeeze(0)}, center, resolution=voxelizer.resolution)

        print('Test 4: Radii Type: Atom-Wise')
        test_name = 'atom-wise'
        voxelizer.radii_type = 'atom-wise'
        ligand_image = voxelizer.forward_single(ligand_coords, center, radii=ligand_atom_radii, out_grid=ligand_grid)
        protein_image = voxelizer.forward_single(protein_coords, center, radii=protein_atom_radii, out_grid=protein_grid)
        assert ligand_image is ligand_grid, 'INPLACE FAILE'
        assert protein_image is protein_grid, 'INPLACE FAILE'
        if pymol :
            assert visualizer is not None
            visualizer.visualize_complex(f'{save_dir}/{test_name}.pse', ligand_rdmol, protein_rdmol, {'Atom': ligand_image.squeeze(0)}, {'Atom': protein_image.squeeze(0)}, center, resolution=voxelizer.resolution)

        print('Test 5: Density: Binary')
        test_name = 'binary'
        voxelizer.density = 'binary'
        voxelizer.radii_type = 'scalar'
        ligand_image = voxelizer.forward_single(ligand_coords, center, radii=1.0, out_grid=ligand_grid)
        protein_image = voxelizer.forward_single(protein_coords, center, radii=1.0, out_grid=protein_grid)
        assert ligand_image is ligand_grid, 'INPLACE FAILE'
        assert protein_image is protein_grid, 'INPLACE FAILE'
        if pymol :
            assert visualizer is not None
            visualizer.visualize_complex(f'{save_dir}/{test_name}.pse', ligand_rdmol, protein_rdmol, {'Atom': ligand_image.squeeze(0)}, {'Atom': protein_image.squeeze(0)}, center, resolution=voxelizer.resolution)

        print('Test 6: Random Transform')
        test_name = 'transform'
        voxelizer.density = 'gaussian'
        T = transform.get_transform()
        print(center)
        new_ligand_coords = T(ligand_coords, center)
        print(center)
        new_protein_coords = T(protein_coords, center)
        print(center)
        ligand_rdmol, protein_rdmol = apply_coord(ligand_rdmol, new_ligand_coords), apply_coord(protein_rdmol, new_protein_coords)

        ligand_image = voxelizer.forward_single(new_ligand_coords, center, radii=1.0, out_grid=ligand_grid)
        protein_image = voxelizer.forward_single(new_protein_coords, center, radii=1.0, out_grid=protein_grid)
        assert ligand_image is ligand_grid, 'INPLACE FAILE'
        assert protein_image is protein_grid, 'INPLACE FAILE'
        if pymol :
            assert visualizer is not None
            visualizer.visualize_complex(f'{save_dir}/{test_name}.pse', ligand_rdmol, protein_rdmol, {'Atom': ligand_image.squeeze(0)}, {'Atom': protein_image.squeeze(0)}, center, resolution=voxelizer.resolution)

    """ LOAD DATA """
    ligand_path = './10gs/10gs_ligand.sdf'
    protein_path = './10gs/10gs_protein_nowater.pdb'

    ligand_rdmol = Chem.SDMolSupplier(ligand_path)[0]
    protein_rdmol = Chem.MolFromPDBFile(protein_path)

    """ TEST """
    num_atoms = ligand_rdmol.GetNumAtoms() + protein_rdmol.GetNumAtoms()
    atom_radii = np.ones((num_atoms,))
    atom_radii[:ligand_rdmol.GetNumAtoms()] = 2.0

    save_dir = 'result_single'
    test(ligand_rdmol, protein_rdmol, atom_radii, save_dir)

if __name__ == '__main__' :
    if '-y' in sys.argv :
        pymol = True
    else :
        pymol = False

    if '-g' in sys.argv :
        device = 'cuda'
    else :
        device = 'cpu'

    from molvoxel.voxelizer.torch import Voxelizer, RandomTransform
    main(Voxelizer, RandomTransform, pymol, device)
