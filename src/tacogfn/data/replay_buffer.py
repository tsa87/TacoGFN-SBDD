"""
Replay buffer from:
https://github.com/recursionpharma/gflownet
"""

import heapq
from typing import List

import numpy as np
import torch

from src.tacogfn.config import Config

class ReplayBuffer(object):
    def __init__(self, cfg: Config, rng: np.random.Generator = None):
        self.capacity = cfg.replay.capacity
        self.warmup = cfg.replay.warmup
        assert (
            self.warmup <= self.capacity
        ), "ReplayBuffer warmup must be smaller than capacity"

        self.buffer: List[tuple] = []
        self.position = 0
        self.rng = rng

    def push(self, *args):
        if len(self.buffer) == 0:
            self._input_size = len(args)
        else:
            assert self._input_size == len(
                args
            ), "ReplayBuffer input size must be constant"
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = args
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        idxs = self.rng.choice(len(self.buffer), batch_size)
        out = list(zip(*[self.buffer[idx] for idx in idxs]))
        for i in range(len(out)):
            # stack if all elements are numpy arrays or torch tensors
            # (this is much more efficient to send arrays through multiprocessing queues)
            if all([isinstance(x, np.ndarray) for x in out[i]]):
                out[i] = np.stack(out[i], axis=0)
            elif all([isinstance(x, torch.Tensor) for x in out[i]]):
                out[i] = torch.stack(out[i], dim=0)
        return tuple(out)

    def __len__(self):
        return len(self.buffer)



class RewardPrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, cfg: Config, rng: np.random.Generator = None):
        super().__init__(cfg, rng)
        
    def push(self, *args):
        score = args[2].item()
        
        if len(self.buffer) == 0:
            self._input_size = len(args)
        else:
            assert self._input_size == len(
                args
            ), "ReplayBuffer input size must be constant"

        if len(self.buffer) < self.capacity:
            # Add the new element if there is still space.
            heapq.heappush(self.buffer, (score, args))
        else:
            # If the buffer is full, push the new element and pop the smallest if the new is better.
            if score > self.buffer[0][0]:  # Check if the new score is better than the smallest score in the heap.
                heapq.heapreplace(self.buffer, (score, args))
                
    def sample(self, batch_size):
        idxs = self.rng.choice(len(self.buffer), batch_size)
        out = list(zip(*[self.buffer[idx][1] for idx in idxs]))
        for i in range(len(out)):
            # stack if all elements are numpy arrays or torch tensors
            # (this is much more efficient to send arrays through multiprocessing queues)
            if all([isinstance(x, np.ndarray) for x in out[i]]):
                out[i] = np.stack(out[i], axis=0)
            elif all([isinstance(x, torch.Tensor) for x in out[i]]):
                out[i] = torch.stack(out[i], dim=0)
        return tuple(out)
