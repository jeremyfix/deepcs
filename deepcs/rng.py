# coding: utf-8

# Standard imports
import random
import os

# External imports
import torch
import numpy as np


def seed_torch(seed=42):
    """
    Function allowing reproducible experiments by deterministically
    setting the random seeds
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
