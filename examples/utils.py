from rdkit import Chem

from src.tacogfn.data import pharmacophore
from src.tacogfn.data.pharmacophore import PharmacoDB
from src.tacogfn.data.pocket import PocketDB
from src.tacogfn.envs import frag_mol_env
from src.tacogfn.tasks.seh_frag import SOME_MOLS


def get_example_pharmacophore_datalist():
    db = PharmacoDB("../misc/pharmacophores_db.lmdb")
    ids = db.all_id[:10]
    keys = db.get_idxs_from_keys(ids)
    data_list = db.get_datalist_from_idxs(keys)
    return data_list


def get_example_pockets():
    db = PocketDB("../misc/pocket_db.lmdb")
    ids = db.all_id[:10]
    keys = db.get_idxs_from_keys(ids)
    data_list = db.get_from_idxs(keys)
    return data_list


def get_example_molecule_datalist(ctx):
    mols = [Chem.MolFromSmiles(s) for s in SOME_MOLS]
    graphs = [ctx.mol_to_graph(mols[i]) for i in range(len(mols))]
    molecule_data_list = [ctx.graph_to_Data(graphs[i]) for i in range(len(graphs))]
    return molecule_data_list
