"""
Misc helper from 
https://github.com/recursionpharma/gflownet
"""

import logging
import sys

import torch
from rdkit import Chem
from rdkit.Chem import AllChem


def create_logger(
    name="logger", loglevel=logging.INFO, logfile=None, streamHandle=True
):
    logger = logging.getLogger(name)
    logger.setLevel(loglevel)
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - {} - %(message)s".format(name),
        datefmt="%d/%m/%Y %H:%M:%S",
    )

    handlers = []
    if logfile is not None:
        handlers.append(logging.FileHandler(logfile, mode="a"))
    if streamHandle:
        handlers.append(logging.StreamHandler(stream=sys.stdout))

    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def get_reference_fps():
    crossdock = torch.load(
        "dataset/archived/crossdock_docking_scores/all_crossdock_100k.pt"
    )
    crossdock_smiles = list(set([v[1] for v in crossdock]))
    crossdock_mols = [Chem.MolFromSmiles(s) for s in crossdock_smiles]
    crossdock_fps = [Chem.RDKFingerprint(m) for m in crossdock_mols]
    return crossdock_fps
