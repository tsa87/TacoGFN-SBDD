from rdkit import Chem

from src.tacogfn.data import pharmacophore
from src.tacogfn.data.pharmacophore import PharmacoDB
from src.tacogfn.envs import frag_mol_env
from src.tacogfn.tasks.seh_frag import SOME_MOLS


def get_example_pharmacophore_datalist():
    db = PharmacoDB("../misc/pharmacophores_db.lmdb")

    ids = [
        "1a0q",
        "1a0t",
        "1a1b",
        "1a1c",
        "1a1e",
        "1a2c",
        "1a3e",
        "1a4g",
        "1a4h",
        "1a4k",
    ]

    pharmacophores = [db.get_pharmacophore(id) for id in ids]

    data_list = pharmacophore.PharmacophoreGraphDataset(pharmacophores)

    return data_list


def get_example_molecule_datalist(ctx):
    mols = [Chem.MolFromSmiles(s) for s in SOME_MOLS]
    graphs = [ctx.mol_to_graph(mols[i]) for i in range(len(mols))]
    molecule_data_list = [ctx.graph_to_Data(graphs[i]) for i in range(len(graphs))]
    return molecule_data_list
