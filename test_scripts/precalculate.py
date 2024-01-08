import argparse
import openbabel
import torch

import sys
sys.path.append('.')
sys.path.append('..')
from src.scoring import PrecalculationModule


MODEL_PATH = './model_weights/model.tar'
HEAD_PATH = './model_weights/base_head.pth'


class Precalculation_ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__('pharmacophore modeling script')
        self.formatter_class = argparse.ArgumentDefaultsHelpFormatter

        # model config
        cfg_args = self.add_argument_group('config')
        cfg_args.add_argument('--pharmaconet_path', type=str, help='path of pharmaconet (.tar)', default=MODEL_PATH)
        cfg_args.add_argument('--head_path', type=str, help='path of head (.tar)', default=HEAD_PATH)

        # system config
        sys_args = self.add_argument_group('system')
        sys_args.add_argument('-r', '--receptor', type=str, help='path of receptor pdb file (.pdb)', required=True)
        sys_args.add_argument('--autobox_ligand', type=str, help='path of ligand to define the center of box (.sdf, .pdb, .mol2)')
        sys_args.add_argument('--center', nargs='+', type=float, help='coordinate of the center')
        sys_args.add_argument('-c', '--cache_path', type=str, help='path to save cache file (.tar)', required=True)

        # system config
        env_args = self.add_argument_group('environment')
        env_args.add_argument('--cuda', action='store_true', help='use gpu acceleration with CUDA')


def main(args):
    if args.autobox_ligand is not None:
        print(f'Using center of {args.autobox_ligand} as center of box')
    else:
        assert args.center is not None, \
            'No Center!. Enter the input `--autobox_ligand <LIGAND_PATH>` or `--center x y z`'
        assert len(args.center) == 3, \
            'Wrong Center!. The arguments for center coordinates should be 3. (ex. --center 1.00 2.00 -1.50)'
        print(f'Using center {tuple(args.center)}')
    predictor = PrecalculationModule(args.pharmaconet_path, args.head_path, 'cuda' if args.cuda else 'cpu')

    if args.autobox_ligand is not None:
        cache = predictor.run(args.receptor, ref_ligand_path=args.autobox_ligand)
    else:
        cache = predictor.run(args.receptor, center=tuple(args.center))
    torch.save(cache, args.cache_path)


if __name__ == '__main__':
    parser = Precalculation_ArgParser()
    args = parser.parse_args()
    main(args)
