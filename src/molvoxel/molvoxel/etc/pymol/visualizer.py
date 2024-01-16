import os
from rdkit import Chem
import tempfile
from pathlib import Path

from rdkit.Chem import Mol
from numpy.typing import ArrayLike
from typing import Dict, Optional, List

import pymol
from pymol import cmd

from .dx import write_grid_to_dx_file
from .atom import ATOMSYMBOL

PROTEIN = 'Protein'
CARTOON = 'Cartoon'
LIGAND = 'Ligand'
MOLECULE = 'Molecule'

_ligand_grid_color_dict = ATOMSYMBOL.copy()
_protein_grid_color_dict = ATOMSYMBOL.copy()
_protein_grid_color_dict['C'] = 'aqua'

_molecule_grid_color_dict = _ligand_grid_color_dict


class Visualizer():
    def visualize_mol(
        self,
        pse_path: str,
        rdmol: Mol,
        grid_dict: Dict[str, ArrayLike],
        center: ArrayLike,
        resolution: float,
        new_coords: Optional[ArrayLike] = None,
        grid_color_dict: Optional[Dict[str, str]] = None,
    ):
        if new_coords is not None:
            rdmol = self._apply_coords(rdmol, new_coords)
        if grid_color_dict is None:
            grid_color_dict = _molecule_grid_color_dict
        self._launch_pymol()

        temp_dir = tempfile.TemporaryDirectory()
        temp_dirpath = Path(temp_dir.name)
        temp_mol_path = str(temp_dirpath / f'{MOLECULE}.sdf')
        temp_grid_path = str(temp_dirpath / 'grid.dx')

        self._save_rdmol(rdmol, temp_mol_path)
        cmd.load(temp_mol_path)
        cmd.color('green', MOLECULE)

        dx_dict = []
        for key, grid in grid_dict.items():
            write_grid_to_dx_file(temp_grid_path, grid, center, resolution)
            cmd.load(temp_grid_path)
            dx = key
            cmd.set_name('grid', dx)
            color = grid_color_dict.get(key, None)
            if color is not None:
                cmd.color(color, dx)
            dx_dict.append(dx)
        cmd.group('Voxel', ' '.join(dx_dict))

        temp_dir.cleanup()

        cmd.enable('all')

        cmd.hide('everything', 'all')
        cmd.show('sticks', MOLECULE)
        cmd.show('everything', 'Voxel')
        cmd.util.cnc('all')

        cmd.bg_color('black')
        cmd.set('dot_width', 2.5)

        cmd.save(pse_path)

    def visualize_complex(
        self,
        pse_path: str,
        ligand_rdmol: Mol,
        protein_rdmol: Mol,
        ligand_grid_dict: Dict[str, ArrayLike],
        protein_grid_dict: Dict[str, ArrayLike],
        center: ArrayLike,
        resolution: str,
        ligand_new_coords: Optional[ArrayLike] = None,
        protein_new_coords: Optional[ArrayLike] = None,
        ligand_grid_color_dict: Optional[Dict[str, str]] = None,
        protein_grid_color_dict: Optional[Dict[str, str]] = None,
    ):
        if ligand_grid_color_dict is None:
            ligand_grid_color_dict = _ligand_grid_color_dict
        if protein_grid_color_dict is None:
            protein_grid_color_dict = _protein_grid_color_dict

        if ligand_new_coords is not None:
            ligand_rdmol = self._apply_coord(ligand_rdmol, ligand_new_coords)
        if protein_new_coords is not None:
            protein_rdmol = self._apply_coord(protein_rdmol, protein_new_coords)
        self._launch_pymol()
        cmd.set_color('aqua', '[0, 150, 255]')

        temp_dir = tempfile.TemporaryDirectory()
        temp_dirpath = Path(temp_dir.name)
        temp_ligand_path = str(temp_dirpath / f'{LIGAND}.sdf')
        temp_protein_path = str(temp_dirpath / f'{PROTEIN}.pdb')
        temp_grid_path = str(temp_dirpath / 'grid.dx')

        self._save_rdmol(ligand_rdmol, temp_ligand_path)
        cmd.load(temp_ligand_path)
        cmd.color('green', LIGAND)

        self._save_rdmol(protein_rdmol, temp_protein_path)
        cmd.load(temp_protein_path)
        cmd.copy(CARTOON, PROTEIN)
        cmd.color('aqua', PROTEIN)
        cmd.color('cyan', CARTOON)

        cmd.group('Molecule', f'{LIGAND} {PROTEIN} {CARTOON}')

        ligand_dx_dict = []
        for key, grid in ligand_grid_dict.items():
            write_grid_to_dx_file(temp_grid_path, grid, center, resolution)
            cmd.load(temp_grid_path)
            dx = 'Ligand_' + key
            cmd.set_name('grid', dx)
            color = ligand_grid_color_dict.get(key, None)
            if color is not None:
                cmd.color(color, dx)
            ligand_dx_dict.append(dx)
        cmd.group('LigandVoxel', ' '.join(ligand_dx_dict))

        protein_dx_dict = []
        for key, grid in protein_grid_dict.items():
            write_grid_to_dx_file(temp_grid_path, grid, center, resolution)
            cmd.load(temp_grid_path)
            dx = 'Protein_' + key
            cmd.set_name('grid', dx)
            color = protein_grid_color_dict.get(key, None)
            if color is not None:
                cmd.color(color, dx)
            protein_dx_dict.append(dx)
        cmd.group('ProteinVoxel', ' '.join(protein_dx_dict))
        cmd.group('Voxel', 'LigandVoxel ProteinVoxel')

        temp_dir.cleanup()

        cmd.enable('all')

        cmd.hide('everything', 'all')
        cmd.show('sticks', LIGAND)
        cmd.show('sticks', PROTEIN)
        cmd.show('cartoon', CARTOON)
        cmd.show('everything', 'Voxel')
        cmd.util.cnc('all')

        cmd.disable(CARTOON)
        cmd.bg_color('black')
        cmd.set('dot_width', 2.5)

        cmd.save(pse_path)

    def visualize_system(
        self,
        pse_path: str,
        rdmol_list: List[Mol],
        name_list: List[str],
        grid_dict_list: List[Dict[str, ArrayLike]],
        center: ArrayLike,
        resolution: str,
        new_coords_list: Optional[List[ArrayLike]]
    ):
        if new_coords_list is not None:
            for i in range(len(rdmol_list)):
                rdmol_list[i] = self._apply_coord(rdmol_list[i], new_coords_list[i])

        self._launch_pymol()
        temp_dir = tempfile.TemporaryDirectory()
        temp_dirpath = Path(temp_dir.name)
        temp_grid_path = str(temp_dirpath / 'grid.dx')

        for rdmol, grid_dict, name in zip(rdmol_list, grid_dict_list, name_list):
            temp_mol_path = str(temp_dirpath / f'{name}.sdf')

            self._save_rdmol(rdmol, temp_mol_path)
            cmd.load(temp_mol_path)
            cmd.color('green', name)

            cmd.group('Molecule', f'{name}')

            dx_dict = []
            for key, grid in grid_dict.items():
                write_grid_to_dx_file(temp_grid_path, grid, center, resolution)
                cmd.load(temp_grid_path)
                dx = f'{name}_' + key
                cmd.set_name('grid', dx)
                if key in molecule_grid_color_dict:
                    cmd.color(molecule_grid_color_dict[key], dx)
                dx_dict.append(dx)
            cmd.group(f'{name}Voxel', ' '.join(dx_dict))
            cmd.group(f'Voxel', f'{name}Voxel')

        temp_dir.cleanup()

        cmd.enable('all')
        cmd.hide('everything', 'all')
        cmd.show('sticks', 'Molecule')
        cmd.show('everything', 'Voxel')
        cmd.util.cnc('all')

        cmd.bg_color('black')
        cmd.set('dot_width', 2.5)
        cmd.save(pse_path)

    @staticmethod
    def _launch_pymol():
        pymol.pymol_argv = ['pymol', '-pcq']
        pymol.finish_launching(args=['pymol', '-pcq', '-K'])
        cmd.reinitialize()
        cmd.feedback('disable', 'all', 'everything')

    @staticmethod
    def _apply_coords(rdmol: Mol, coords: ArrayLike) -> Mol:
        rdmol = Chem.Mol(rdmol)
        conf = rdmol.GetConformer()
        for i in range(rdmol.GetNumAtoms()):
            conf.SetAtomPosition(i, coords[i].tolist())
        return rdmol

    @staticmethod
    def _save_rdmol(rdmol, save_path, coords=None):
        rdmol = Chem.Mol(rdmol)
        if coords is not None:
            conf = rdmol.GetConformer()
            for i in range(rdmol.GetNumAtoms()):
                conf.SetAtomPosition(i, coords[i].tolist())

        ext = os.path.splitext(save_path)[-1]
        assert ext in ['.pdb', '.sdf']
        if ext == '.pdb':
            w = Chem.PDBWriter(save_path)
        else:
            w = Chem.SDWriter(save_path)
        w.write(rdmol)
        w.close()
