"""Restore policy RNG streams after isolated evaluation."""
from contextlib import contextmanager
import random
import numpy as np
import torch

@contextmanager
def evaluation_rng(seed):
    states = random.getstate(), np.random.get_state(), torch.get_rng_state()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        random.setstate(states[0])
        np.random.set_state(states[1])
        torch.set_rng_state(states[2])

