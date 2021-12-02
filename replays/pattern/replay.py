# Copyright (c) 2021, Xu Chen, FUNLab, Xiamen University
# All rights reserved.

"""Replays for emulator"""

import os
import random
import numpy as np
from glob import glob
from torch.utils.data import Dataset

from common import npz_load

class UniformReplay(Dataset):
    """Uniform sampling replay."""

    def __init__(self, fpaths):
        
        # extract fpaths
        GU_fpath, ABS_fpath, CGU_fpath = fpaths

        self.P_GUs  = np.expand_dims(npz_load(GU_fpath), axis=1)
        self.P_ABSs = np.expand_dims(npz_load(ABS_fpath), axis=1)
        self.P_CGUs = np.expand_dims(npz_load(CGU_fpath), axis=1)
        assert len(self.P_GUs) == len(self.P_ABSs)
        assert len(self.P_ABSs) == len(self.P_CGUs)

    def __getitem__(self, index):
        return (
            self.P_GUs[index],
            self.P_ABSs[index],
            self.P_CGUs[index]
        )

    def __len__(self):
        return len(self.P_GUs)