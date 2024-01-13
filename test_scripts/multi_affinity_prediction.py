import pathlib
from openbabel import pybel
import torch
import time
import sys

sys.path.append('.')
sys.path.append('..')
from src.scoring.scoring_module import AffinityPredictor


HEAD_PATH = "./model_weights/base_100_per_pocket.pth"


if __name__ == '__main__':
    # NOTE: SET MODEL
    device = 'cpu'
    predictor = AffinityPredictor(HEAD_PATH, device)

    # NOTE: LOAD RECEPTOR CACHE
    cache = torch.load('./test_scripts/1a0g_A_rec.tar', map_location=device)

    # NOTE: Setup SMILES List
    root_path = pathlib.Path('/home/shwan/DATA/random_docking/result/pdb/1a0g_A_rec/')
    files = [path for path in root_path.iterdir()][:100]

    smiles_list = []
    affinity_list = []
    for file in files:
        pbmol = next(pybel.readfile('pdb', str(file)))
        aff = pbmol.data['REMARK'].split()[2]
        smiles_list.append(pbmol.write('smi').split()[0])
        affinity_list.append(float(aff))
    # smiles_list = ['c1ccccc1', 'C1CCCCC1', 'CCCCCC']

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

    # NOTE: Single Graph Calculation
    for smiles, affinity in zip(smiles_list, affinity_list):
        score = float(predictor.scoring(cache, smiles))
        print(f'{smiles}, {score: .1f}, {affinity: .1f}')
