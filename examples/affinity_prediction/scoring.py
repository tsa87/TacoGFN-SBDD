import argparse
from openbabel import pybel
import torch
import sys

sys.path.append('.')
sys.path.append('..')
from src.scoring.scoring_module import AffinityPredictor


HEAD_PATH = "./model_weights/20240117_500.pth"


class Scoring_ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__('pharmacophore modeling script')
        self.formatter_class = argparse.ArgumentDefaultsHelpFormatter

        # system config
        sys_args = self.add_argument_group('system')
        sys_args.add_argument('-c', '--cache_path', type=str, help='path to receptor cache file (.tar)', required=True)
        sys_args.add_argument('-l', '--ligand', type=str, help='ligand smiles', required=True)

        # system config
        env_args = self.add_argument_group('environment')
        env_args.add_argument('--cuda', action='store_true', help='use gpu acceleration with CUDA')


def main(args):
    # NOTE: SET MODEL
    device = 'cuda' if args.cuda else 'cpu'
    predictor = AffinityPredictor(HEAD_PATH, device)
    cache = torch.load(args.cache_path, map_location=device)
    score = float(predictor.scoring(cache, args.ligand))
    print(score)
    return score


if __name__ == '__main__':
    parser = Scoring_ArgParser()
    args = parser.parse_args()
    main(args)
