import ast
import cProfile
import hashlib
import logging
import os
import pstats
import sys
import time
from functools import wraps

import numpy as np
import torch
from rdkit.Chem import Descriptors

from src.tacogfn.utils import sascore


class GeneratedStore(object):
    def __init__(self, cfg, pdb_id):
        self.cfg = cfg
        self.pdb_id = pdb_id
        self.generated_molecules = []
        self.docking_scores = []
        self.save_interval = 60  # Save every 60 seconds
        self.last_save_time = time.time()
        self.save_count = 0

    def push(self, mols, scores):
        self.generated_molecules.extend(mols)
        self.docking_scores.extend([s.item() for s in scores])

        current_time = time.time()
        if current_time - self.last_save_time >= self.save_interval:
            self.save()
            self.last_save_time = current_time

    def save(self):
        self.save_count += 1
        save_file = os.path.join(
            self.cfg.log_dir, f"generated_molecules_{self.save_count}.pt"
        )
        torch.save(
            {
                "generated_molecules": self.generated_molecules,
                "docking_scores": self.docking_scores,
            },
            save_file,
        )
        if self.save_count > 1:
            os.remove(
                os.path.join(
                    self.cfg.log_dir, f"generated_molecules_{self.save_count-1}.pt"
                )
            )

    def get_top_avg(self, top=100):
        sort_idxs = np.argsort(self.docking_scores)

        top_mols = [self.generated_molecules[i] for i in sort_idxs[:top]]

        top_qed = np.array([Descriptors.qed(mol) for mol in top_mols])
        top_sa = np.array([(10 - sascore.calculateScore(mol)) / 9 for mol in top_mols])
        top_ds = np.array(self.docking_scores)[sort_idxs[:top]]

        success_rate = (top_qed > 0.25) & (top_sa > 0.59) & (top_ds < -8.18)
        return {
            f"top_{top}_qed": np.mean(top_qed),
            f"top_{top}_sa": np.mean(top_sa),
            f"top_{top}": np.mean(top_ds),
            f"top_{top}_success_rate": np.mean(success_rate),
        }


def time_profile(
    output_file=None, sort_by="cumulative", lines_to_print=None, strip_dirs=False
):
    """A time profiler decorator.
    Inspired by and modified the profile decorator of Giampaolo Rodola:
    http://code.activestate.com/recipes/577817-profile-decorator/
    Args:
        output_file: str or None. Default is None
            Path of the output file. If only name of the file is given, it's
            saved in the current directory.
            If it's None, the name of the decorated function is used.
        sort_by: str or SortKey enum or tuple/list of str/SortKey enum
            Sorting criteria for the Stats object.
            For a list of valid string and SortKey refer to:
            https://docs.python.org/3/library/profile.html#pstats.Stats.sort_stats
        lines_to_print: int or None
            Number of lines to print. Default (None) is for all the lines.
            This is useful in reducing the size of the printout, especially
            that sorting by 'cumulative', the time consuming operations
            are printed toward the top of the file.
        strip_dirs: bool
            Whether to remove the leading path info from file names.
            This is also useful in reducing the size of the printout
    Returns:
        Profile of the decorated function
    """

    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            _output_file = output_file or func.__name__ + ".prof"
            pr = cProfile.Profile()
            pr.enable()
            retval = func(*args, **kwargs)
            pr.disable()
            pr.dump_stats(_output_file)

            with open(_output_file, "w") as f:
                ps = pstats.Stats(pr, stream=f)
                if strip_dirs:
                    ps.strip_dirs()
                if isinstance(sort_by, (tuple, list)):
                    ps.sort_stats(*sort_by)
                else:
                    ps.sort_stats(sort_by)
                ps.print_stats(lines_to_print)
            return retval

        return wrapper

    return inner
