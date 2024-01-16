from rdkit import Chem
from rdkit.Chem import BondType

from typing import Sequence, Optional
from rdkit.Chem import Atom, Bond

from .base import ChannelGetter, TypeGetter, FeatureGetter

""" ATOM """
AtomChannelGetter = ChannelGetter
class AtomFeatureGetter(FeatureGetter) :
    pass
class AtomTypeGetter(TypeGetter) :
    def __init__(self, symbols: Sequence[str], symbol_names: Optional[Sequence[str]] = None, unknown: bool = False) :
        if symbol_names is None :
            symbol_names = symbols
        super(AtomTypeGetter, self).__init__(symbols, symbol_names, unknown)

    def get_type(self, atom: Atom, **kwargs) -> int :
        return super().get_type(atom.GetSymbol(), **kwargs)

""" BOND """
BondChannelGetter = ChannelGetter
class BondFeatureGetter(FeatureGetter) :
    pass
class BondTypeGetter(TypeGetter) :
    def __init__(self, bondtypes: Sequence[BondType], bondtype_names: Optional[Sequence[str]] = None, unknown: bool = False) :
        if bondtype_names is None :
            bondtype_names = [str(bt) for bt in bondtypes]
        super(BondTypeGetter, self).__init__(bondtypes, bondtype_names, unknown)

    def get_type(self, bond: Bond, **kwargs) -> int :
        return super().get_type(bond.GetBondType(), **kwargs)

    @classmethod
    def default(cls) :
        bondtypes = [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC]
        names = ['SingleBond', 'DoubleBond', 'TripleBond', 'AromaticBond']
        return cls(bondtypes, names)

