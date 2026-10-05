import random
import numpy as np
import torch


def set_seed(seed: int = 912):
    #ref: https://docs.pytorch.org/docs/2.14/notes/randomness.html
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True) # disable if false
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)