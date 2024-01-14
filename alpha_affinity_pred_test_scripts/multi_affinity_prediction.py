import pathlib
from openbabel import pybel
import torch
import time
import sys

sys.path.append('.')
sys.path.append('..')
from src.scoring.scoring_module import AffinityPredictor


HEAD_PATH = "./model_weights/base_head.pth"


if __name__ == '__main__':
    # NOTE: SET MODEL
    device = 'cpu'
    predictor = AffinityPredictor(HEAD_PATH, device)

    # NOTE: LOAD RECEPTOR CACHE
    cache = torch.load('./1a0g_A_rec.tar', map_location=device)

    # NOTE: Setup SMILES List
    smiles_list = ['c1ccccc1', 'C1CCCCC1', 'CCCCCC']

    # NOTE: Single Graph Calculation
    st = time.time()
    for smiles in smiles_list:
        score = predictor.scoring(cache, smiles)
    end = time.time()
    print((end - st) / len(smiles_list))

    # NOTE: Batch Calculation
    st = time.time()
    score_list = predictor.scoring_list(cache, smiles_list)
    end = time.time()
    print((end - st) / len(smiles_list))
