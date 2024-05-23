import random
import numpy as np
import torch


def seed(seed_num, hparams):
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.backends.cudnn.benchmark = False
    if hparams.use_deterministic_torch_algorithms:
        torch.use_deterministic_algorithms(True)

