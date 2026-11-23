import torch
import numpy as np
import random

def seed_all(seed_value):
    pass
    ## YOUR CODE HERE
    # -- start --
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)  # CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)  # every GPU, not only the current one
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed_value)  # Apple GPU (Metal), MacBook
    try:
        import torch_xla.core.xla_model as xla_model  # TPU, optional package
    except ImportError:
        xla_model = None
    if xla_model is not None:
        xla_model.set_rng_state(seed_value)
    torch.backends.cudnn.deterministic = True
    # -- end --