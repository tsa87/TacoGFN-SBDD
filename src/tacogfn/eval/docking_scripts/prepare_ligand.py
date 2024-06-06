import argparse
from argparse import ArgumentParser
import os
import tempfile
import sys

from AutoDockTools.MoleculePreparation import AD4LigandPreparation
from MolKit import Read


def convert_ligand_pdbqt(ligand_filename, outputfilename, addH):
    extension = os.path.splitext(ligand_filename)[-1][1:]
    fd = None
    if addH == 'on':
        fd, tmp_file = tempfile.mkstemp(suffix='.mol2')
        os.system(f'obabel {ligand_filename} -omol2 -O {tmp_file} -p 7.4')
        ligand_filename = tmp_file
    elif extension not in ('mol2', 'pdb'):
        fd, tmp_file = tempfile.mkstemp(suffix='.mol2')
        os.system(f'obabel {ligand_filename} -omol -O {tmp_file}')
        ligand_filename = tmp_file

    # initialize required parameters
    repairs = ""
    charges_to_add = 'gasteiger'
    preserve_charge_types = ''
    cleanup = "nphs_lps"
    allowed_bonds = "backbone"
    root = 'auto'
    check_for_fragments = False
    bonds_to_inactivate = ""
    inactivate_all_torsions = False
    attach_nonbonded_fragments = False
    attach_singletons = False
    mode = 'automatic'
    dict = None

    mols = Read(ligand_filename)
    mol = mols[0]
    if len(mols) > 1:
        ctr = 1
        for m in mols[1:]:
            ctr += 1
            if len(m.allAtoms) > len(mol.allAtoms):
                print("mol set to ", ctr, "th molecule with", len(mol.allAtoms), "atoms")
    coord_dict = {}
    for a in mol.allAtoms:
        coord_dict[a] = a.coords

    mol.buildBondsByDistance()
    if charges_to_add is not None:
        preserved = {}
        preserved_types = preserve_charge_types.split(',')
        for t in preserved_types:
            if not len(t):
                continue
            ats = mol.allAtoms.get(lambda x: x.autodock_element == t)
            for a in ats:
                if a.chargeSet is not None:
                    preserved[a] = [a.chargeSet, a.charge]

    LPO = AD4LigandPreparation(mol, mode, repairs, charges_to_add,
                               cleanup, allowed_bonds, root,
                               outputfilename=outputfilename,
                               dict=dict, check_for_fragments=check_for_fragments,
                               bonds_to_inactivate=bonds_to_inactivate,
                               inactivate_all_torsions=inactivate_all_torsions,
                               attach_nonbonded_fragments=attach_nonbonded_fragments,
                               attach_singletons=attach_singletons)
    # do something about atoms with too many bonds (?)
    # FIX THIS: could be peptide ligand (???)
    #          ??use isPeptide to decide chargeSet??
    if charges_to_add is not None:
        # restore any previous charges
        for atom, chargeList in preserved.items():
            atom._charges[chargeList[0]] = chargeList[1]
            atom.chargeSet = chargeList[0]
    bad_list = []
    for a in mol.allAtoms:
        if a in coord_dict and a.coords != coord_dict[a]:
            bad_list.append(a)
    if len(bad_list):
        print(len(bad_list), ' atom coordinates changed!')
        for a in bad_list:
            print(a.name, ":", coord_dict[a], ' -> ', a.coords)
    if mol.returnCode != 0:
        sys.stderr.write(mol.returnMsg + "\n")
    if fd is not None:
        os.close(fd)
    return mol.returnCode


def get_parser():
    parser = ArgumentParser(
        prog='PDBQT-OpenBabel',
        description="Python Embedding for Openbabel PDBQT Conversion",
        formatter_class=argparse.MetavarTypeHelpFormatter
    )
    input_group = parser.add_argument_group('Input')
    input_group.add_argument('input', type=str, help='input file name')

    out_group = parser.add_argument_group('Output')
    out_group.add_argument('-o', '--out', type=str, help='output file name', required=True)

    misc_group = parser.add_argument_group('Misc (optional)')
    misc_group.add_argument('--addH', type=str, default='on', choices=('on', 'off'), help='automatically add hydrogens in ligands (on by default)')
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    code = convert_ligand_pdbqt(args.input, args.out, args.addH)
    sys.exit(code)
