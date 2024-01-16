from rdkit import Chem
from rdkit.Chem import Mol
from numpy.typing import ArrayLike

def apply_coord(rdmol: Mol, coords: ArrayLike) -> Mol :
    rdmol = Chem.Mol(rdmol)
    conf = rdmol.GetConformer()
    for i in range(rdmol.GetNumAtoms()) :
        conf.SetAtomPosition(i, coords[i].tolist())
    return rdmol

