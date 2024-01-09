from openbabel import pybel

from openbabel.pybel import Molecule
from openbabel.pybel import ob

import torch
from torch_geometric.data import Data as PyGData


ATOM_DICT = {
    6: 0,
    7: 1,
    8: 2,
    9: 3,
    15: 4,
    16: 5,
    17: 6,
    35: 7,
    53: 8,
    -1: 9,  # UNKNOWN
}
NUM_ATOM_TYPES = 10

BOND_DICT = {
    1: 0,
    2: 1,
    3: 2,
    1.5: 3,     # AROMATIC
    -1: 4,      # UNKNOWN
}
NUM_BOND_TYPES = 5


def file2datapoint(path: str, filetype: str = 'pdb') -> PyGData:
    pbmol: Molecule = next(pybel.readfile(filetype, path))
    graph: PyGData = mol2graph(pbmol)
    y = float(pbmol.data['REMARK'].split()[2])
    graph['y'] = y
    return graph


def file2graph(path: str, filetype: str = 'pdb') -> PyGData:
    pbmol: Molecule = next(pybel.readfile(filetype, path))
    graph: PyGData = mol2graph(pbmol)
    return graph


def smi2graph(smiles: str) -> PyGData:
    return mol2graph(pybel.readstring('smi', smiles))


def mol2graph(pbmol: pybel.Molecule) -> PyGData:
    obmol: ob.OBMol = pbmol.OBMol
    atom_features = []
    pos = []
    for pbatom in pbmol.atoms:
        atom_features.append(ATOM_DICT.get(pbatom.atomicnum, 9))
        pos.append(pbatom.coords)

    edge_index = []
    edge_type = []
    for obbond in ob.OBMolBondIter(obmol):
        obbond: ob.OBBond
        edge_index.append((obbond.GetBeginAtomIdx() - 1, obbond.GetEndAtomIdx() - 1))
        if obbond.IsAromatic():
            edge_type.append(3)
        else:
            edge_type.append(BOND_DICT.get(obbond.GetBondOrder(), 4))

    return PyGData(
        x=torch.LongTensor(atom_features),
        edge_index=torch.LongTensor(edge_index).T,
        edge_attr=torch.LongTensor(edge_type),
    )
