import numpy as np
from rdkit import Chem
from molvoxel.etc.rdkit.wrapper import ComplexWrapper
from molvoxel.etc.rdkit.pointcloud import MolSystemPointCloudMaker
from molvoxel.etc.rdkit.getter import AtomTypeGetter, BondTypeGetter, AtomFeatureGetter
import time


def run_test(voxelizer, grid, coords, center, channels, radii, random_translation=0.5, random_rotation=True):
    batch_size = grid.shape[0]
    for i in range(batch_size):
        voxelizer.forward(coords, center, channels, radii, random_translation, random_rotation, out_grid=grid[i])
    return grid


def main(voxelizer):
    batch_size = 16
    num_iteration = 25
    num_trial = 5

    """ LOAD DATA """
    ligand_path = './10gs/10gs_ligand.sdf'
    protein_path = './10gs/10gs_protein_nowater.pdb'

    ligand_rdmol = Chem.SDMolSupplier(ligand_path)[0]
    protein_rdmol = Chem.MolFromPDBFile(protein_path)

    atom_getter = AtomTypeGetter(['C', 'N', 'O', 'S'])
    bond_getter = BondTypeGetter.default()
    pointcloudmaker_types = MolSystemPointCloudMaker([atom_getter, bond_getter], [atom_getter, bond_getter], channel_type='types')
    pointcloudmaker_features = MolSystemPointCloudMaker([atom_getter, bond_getter], [atom_getter, bond_getter], channel_type='features')
    wrapper_types = ComplexWrapper(pointcloudmaker_types, voxelizer)
    wrapper_features = ComplexWrapper(pointcloudmaker_features, voxelizer)

    ligand_coords = ligand_rdmol.GetConformer().GetPositions()
    ligand_center = ligand_coords.mean(0)
    center = ligand_center

    coords = wrapper_types.get_coords(ligand_rdmol, protein_rdmol)
    types = wrapper_types.get_channels(ligand_rdmol, protein_rdmol)
    features = wrapper_features.get_channels(ligand_rdmol, protein_rdmol)
    grid = wrapper_types.get_empty_grid(batch_size)
    center = voxelizer.asarray(center, 'center')

    single_grid = voxelizer.get_empty_grid(1, batch_size)

    """ SANITY CHECK """
    print('Sanity Check')
    for _ in range(2):
        single_out = run_test(voxelizer, single_grid, coords, center, None, 1.0, random_translation=0.0, random_rotation=False).tolist()
        type_out = run_test(voxelizer, grid, coords, center, types, 1.0, random_translation=0.0, random_rotation=False).tolist()
        feature_out = run_test(voxelizer, grid, coords, center, features, 1.0, random_translation=0.0, random_rotation=False).tolist()
        type_out = np.array(type_out)
        feature_out = np.array(feature_out)
        assert np.less(np.abs(type_out - type_out[0]), 1e-5).all(), 'REPRODUCTION FAIL'
        assert np.less(np.abs(feature_out - feature_out[0]), 1e-5).all(), 'REPRODUCTION FAIL'
        assert np.less(np.abs(type_out[0] - feature_out[0]), 1e-5).all(), 'REPRODUCTION FAIL'
    print('PASS\n')

    """ ATOM TYPE """
    print('Test Atom SINGLE')
    st_tot = time.time()
    for i in range(num_trial):
        print(f'trial {i}')
        st = time.time()
        for _ in range(num_iteration):
            single_grid = run_test(voxelizer, single_grid, coords, center, None, 1.0)
        end = time.time()
        print(f'total time {(end-st)}')
        print(f'time per run {(end-st) / batch_size / num_iteration}')
        print()
    end_tot = time.time()
    print(f'times per run {(end_tot-st_tot) / batch_size / num_iteration / num_trial}\n')

    """ ATOM TYPE """
    print('Test Atom Type')
    st_tot = time.time()
    for i in range(num_trial):
        print(f'trial {i}')
        st = time.time()
        for _ in range(num_iteration):
            grid = run_test(voxelizer, grid, coords, center, types, 1.0)
        end = time.time()
        print(f'total time {(end-st)}')
        print(f'time per run {(end-st) / batch_size / num_iteration}')
        print()
    end_tot = time.time()
    print(f'times per run {(end_tot-st_tot) / batch_size / num_iteration / num_trial}\n')

    """ ATOM FEATURE """
    print('Test Atom Feature')
    st_tot = time.time()
    for i in range(num_trial):
        print(f'trial {i}')
        st = time.time()
        for _ in range(num_iteration):
            grid = run_test(voxelizer, grid, coords, center, features, 1.0)
        end = time.time()
        print(f'total time {(end-st)}')
        print(f'time per run {(end-st) / batch_size / num_iteration}')
        print()
    end_tot = time.time()
    print(f'times per run {(end_tot-st_tot) / batch_size / num_iteration / num_trial}')


if __name__ == '__main__':
    from molvoxel.voxelizer.numpy import Voxelizer
    resolution = 0.5
    dimension = 48
    voxelizer = Voxelizer(resolution, dimension)
    main(voxelizer)
