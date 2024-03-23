import argparse
from argparse import ArgumentParser
import os
import tempfile

from MolKit import Read
from AutoDockTools.MoleculePreparation import AD4ReceptorPreparation


def convert_receptor_pdbqt(receptor_filename, outputfilename, hetatom='off'):
    import sys
    fd, tmp_file = tempfile.mkstemp(suffix=os.path.basename(receptor_filename))
    if hetatom == 'off':
        os.system(f'grep -v HETATM {receptor_filename} > {tmp_file}')
    else:
        os.system(f'grep -v HOH {receptor_filename} > {tmp_file}')

    receptor_filename = tmp_file

    repairs = 'hydrogens'
    charges_to_add = 'gasteiger'
    preserve_charge_types = None
    cleanup = "nphs_lps_deleteAltB"
    mode = 'automatic'
    delete_single_nonstd_residues = None
    dictionary = None
    unique_atom_names = False

    mols = Read(receptor_filename)
    mol = mols[0]
    if unique_atom_names:  # added to simplify setting up covalent dockings 8/2014
        for at in mol.allAtoms:
            if mol.allAtoms.get(at.name) > 1:
                at.name = at.name + str(at._uniqIndex + 1)
    preserved = {}
    has_autodock_element = False
    if charges_to_add and preserve_charge_types:
        if hasattr(mol, 'allAtoms') and not hasattr(mol.allAtoms[0], 'autodock_element'):
            file_name, file_ext = os.path.splitext(receptor_filename)
            if file_ext == '.pdbqt':
                has_autodock_element = True
        if preserve_charge_types and not has_autodock_element:
            print('prepare_receptor4: input format does not have autodock_element SO unable to preserve charges on ' + preserve_charge_types)
            print('exiting...')
            sys.exit(1)
        preserved_types = preserve_charge_types.split(',')
        for t in preserved_types:
            if not len(t):
                continue
            ats = mol.allAtoms.get(lambda x: x.autodock_element == t)
            for a in ats:
                if a.chargeSet is not None:
                    preserved[a] = [a.chargeSet, a.charge]

    if len(mols) > 1:
        ctr = 1
        for m in mols[1:]:
            ctr += 1
            if len(m.allAtoms) > len(mol.allAtoms):
                mol = m
    mol.buildBondsByDistance()
    alt_loc_ats = mol.allAtoms.get(lambda x: "@" in x.name)
    len_alt_loc_ats = len(alt_loc_ats)
    if len_alt_loc_ats:
        print("WARNING!", mol.name, "has", len_alt_loc_ats, ' alternate location atoms!\nUse prepare_pdb_split_alt_confs.py to create pdb files containing a single conformation.\n')

    RPO = AD4ReceptorPreparation(mol, mode, repairs, charges_to_add,
                                 cleanup, outputfilename=outputfilename,
                                 preserved=preserved,
                                 delete_single_nonstd_residues=delete_single_nonstd_residues,
                                 dict=dictionary)

    if charges_to_add:
        for atom, chargeList in list(preserved.items()):
            atom._charges[chargeList[0]] = chargeList[1]
            atom.chargeSet = chargeList[0]

    os.close(fd)


def get_parser():
    parser = ArgumentParser(
        prog='PDBQT-OpenBabel',
        description="Python Embedding for Openbabel PDBQT Conversion (Seonghwan Seo)",
        formatter_class=argparse.MetavarTypeHelpFormatter
    )
    input_group = parser.add_argument_group('Input')
    input_group.add_argument('input', type=str, help='input file name')

    out_group = parser.add_argument_group('Output')
    out_group.add_argument('-o', '--out', type=str, help='output file name', required=True)
    out_group.add_argument('--hetatom', type=str, default='off', choices=('on', 'off'), help='remain hetatoms in protein (default off)')
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    convert_receptor_pdbqt(args.input, args.out, args.hetatom)
