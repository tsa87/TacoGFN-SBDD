import os
import torch
import time
import numpy as np


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
        save_file = os.path.join(self.cfg.log_dir, f"generated_molecules_{self.save_count}.pt")
        torch.save(
            {
                "generated_molecules": self.generated_molecules,
                "docking_scores": self.docking_scores,
            },
            save_file,
        )
        if self.save_count > 1:
            os.remove(os.path.join(self.cfg.log_dir, f"generated_molecules_{self.save_count-1}.pt"))
            
    def get_top_avg(self, top=100):
        return np.mean(sorted(self.docking_scores)[:top])
        