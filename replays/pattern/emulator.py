# Copyright (c) 2021, Xu Chen, FUNLab, Xiamen University
# All rights reserved.

"""Replays for emulator"""

import os
import random
import numpy as np
from common import npz_load

class UniformReplay(object):
    """Uniform sampling replay."""

    def __init__(self, K, max_size):

        self.K = K
        self.max_size = max_size
        self.size = 0
        self.ptr = 0

        shape = (max_size, 1, K, K)
        self.data = {
            "P_GUs" : np.zeros(shape, dtype=np.float32),
            "P_ABSs": np.zeros(shape, dtype=np.float32),
            "P_CGUs": np.zeros(shape, dtype=np.float32)
        }

    def add(self, data):
        """
        Add a single transition of data.
        """
        P_GU, P_ABS, _, P_CGU = data

        ptr = self.ptr
        K = self.K
        self.data["P_GUs"][ptr] = P_GU.reshape(1, K, K)
        self.data["P_ABSs"][ptr] = P_ABS.reshape(1, K, K)
        self.data["P_CGUs"][ptr] = P_CGU.reshape(1, K, K)

        self.ptr = (ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def shuffle(self):
        """
        Shuffle data.
        """
        perm = np.arange(self.size)
        np.random.shuffle(perm)

        self.data["P_GUs"] = self.data["P_GUs"][perm]
        self.data["P_ABSs"] = self.data["P_ABSs"][perm]
        self.data["P_CGUs"] = self.data["P_CGUs"][perm]

        print("shuffled...")

    def sample(self, batch_size):
        """
        Sample `batch_size` samples.
        """
        idc = np.random.randint(0, self.size, size=batch_size)

        sample = {}
        for k, v in self.data.items():
            sample[k] = v[idc]

        return sample

    def data_loader(self, batch_size):
        """
        Load all data in iterator style.
        """
        cur = 0
        while True:
            if cur >= self.size - 1:
                break
            else:
                end = min(cur + batch_size, self.size)
                idc = np.arange(cur, end)
                cur = end
                sample = {}
                for k, v in self.data.items():
                    sample[k] = v[idc]
                yield sample

    def paste(self, data):
        """
        Paste a serie of data to replay.
        """
        P_GUs, P_ABSs, P_CGUs = data
        size = len(P_GUs)
        ptr = self.ptr

        self.data["P_GUs"][ptr: ptr+size] = P_GUs
        self.data["P_ABSs"][ptr: ptr+size] = P_ABSs
        self.data["P_CGUs"][ptr: ptr+size] = P_CGUs

        self.ptr = (ptr + size) % self.max_size
        self.size = min(self.size + size, self.max_size)

    def is_full(self):
        return self.size == self.max_size



