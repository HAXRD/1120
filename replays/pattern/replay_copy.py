# Copyright (c) 2021, Xu Chen, FUNLab, Xiamen University
# All rights reserved.

"""Replays for emulator"""

import os
import random
import numpy as np
from glob import glob

from common import npz_load, npz_save

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

        idx = self.ptr
        K = self.K
        self.data["P_GUs"][idx] = P_GU.reshape(1, K, K)
        self.data["P_ABSs"][idx] = P_ABS.reshape(1, K, K)
        self.data["P_CGUs"][idx] = P_CGU.reshape(1, K, K)

        self.ptr = (idx + 1) % self.max_size
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

    def save_to_npz(self, file_size_limit, fdir):

        def _save(dir, prefix, pname, idx, arr):
            fpath = os.path.join(dir, f"{prefix}_{pname}_{idx}.npz")
            npz_save(arr, fpath)
            print(f"[train | save val replays] saved to '{fpath}' (size: ({len(arr)})).")

        K = self.K
        cur_size = self.size
        i = 0
        idx = 0
        while cur_size > 0:
            size = min(cur_size, file_size_limit)
            P_GUs  = np.zeros((size, K, K), dtype=np.float32)
            P_ABSs = np.zeros((size, K, K), dtype=np.float32)
            P_CGUs = np.zeros((size, K, K), dtype=np.float32)

            P_GUs  = self.data["P_GUs"][i: i+size].squeeze(1)
            P_ABSs = self.data["P_ABSs"][i: i+size].squeeze(1)
            P_CGUs = self.data["P_CGUs"][i: i+size].squeeze(1)

            _save(fdir, "val", "GUs", idx, P_GUs)
            _save(fdir, "val", "ABSs", idx, P_ABSs)
            _save(fdir, "val", "CGUs", idx, P_CGUs)
            i += size
            idx += 1
            cur_size -= size

    def load_from_npz(self, fdir):

        P_GUs_fnames, P_ABSs_fnames, P_CGUs_fnames = [
            sorted(glob(os.path.join(fdir, f"val_{pname}_*.npz")))
            for pname in ["GUs", "ABSs", "CGUs"]
        ]

        for _GU_fname, _ABS_fname, _CGU_fname in zip(P_GUs_fnames, P_ABSs_fnames, P_CGUs_fnames):
            P_GUs, P_ABSs, P_CGUs = [npz_load(fpath) for fpath in [_GU_fname, _ABS_fname, _CGU_fname]]

            P_GUs = np.expand_dims(P_GUs, axis=1)
            P_ABSs = np.expand_dims(P_ABSs, axis=1)
            P_CGUs = np.expand_dims(P_CGUs, axis=1)

            data = P_GUs, P_ABSs, P_CGUs
            self.paste(data)

        print(f"[train | emulator val replay] preloaded from dir '{fdir}'.")

class PrioritizedReplay(object):
    """
    Prioritized experience replay for emulator, for which the priority
    is based on the absolute element-wise difference between 'P_CGU'
    and 'P_rec_CGU'.

    Revised code from reference: https://nn.labml.ai/rl/dqn/replay_buffer.html
    """
    def __init__(self, K, max_size, alpha):

        self.K = K
        self.max_size = max_size
        self.alpha = alpha
        self.size = 0
        self.ptr = 0

        self.priority_sum = [0 for _ in range(2 * max_size)]
        self.priority_min = [float('inf') for _ in range(2 * max_size)]

        self.max_priority = 1.

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

        idx = self.ptr
        K = self.K
        self.data["P_GUs"][idx] = P_GU.reshape(1, K, K)
        self.data["P_ABSs"][idx] = P_ABS.reshape(1, K, K)
        self.data["P_CGUs"][idx] = P_CGU.reshape(1, K, K)

        self.ptr = (idx + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        priority_alpha = self.max_priority ** self.alpha

        self._set_priority_min(idx, priority_alpha)
        self._set_priority_sum(idx, priority_alpha)

    def _set_priority_min(self, idx, priority_alpha):

        idx += self.max_size
        self.priority_min[idx] = priority_alpha

        while idx >= 2:
            idx //= 2
            self.priority_min[idx] = min(self.priority_min[2 * idx], self.priority_min[2 * idx + 1])

    def _set_priority_sum(self, idx, priority):

        idx += self.max_size
        self.priority_sum[idx] = priority

        while idx >= 2:
            idx //= 2
            self.priority_sum[idx] = self.priority_sum[2 * idx] + self.priority_sum[2 * idx + 1]

    def _sum(self):
        return self.priority_sum[1]

    def _min(self):
        return self.priority_min[1]

    def find_prefix_sum_idx(self, prefix_sum):

        idx = 1
        while idx < self.max_size:
            if self.priority_sum[idx * 2] > prefix_sum:
                idx = 2 * idx
            else:
                prefix_sum -= self.priority_sum[idx * 2]
                idx = 2 * idx + 1

        return idx - self.max_size

    def sample(self, batch_size, beta):

        sample = {
            "weights": np.zeros(batch_size, dtype=np.float32),
            "indices": np.zeros(batch_size, dtype=np.int32)
        }

        for i in range(batch_size):
            p = random.random() * self._sum()
            idx = self.find_prefix_sum_idx(p)
            sample["indices"][i] = idx

        prob_min = self._min() / self._sum()

        max_weight = (prob_min * self.size) ** (-beta)

        for i in range(batch_size):
            idx = sample["indices"][i]

            prob = self.priority_sum[idx + self.max_size] / self._sum()

            weight = (prob * self.size) ** (-beta)
            sample["weights"][i] = weight / max_weight

        for k, v in self.data.items():
            sample[k] = v[sample["indices"]]

        return sample

    def update_priorities(self, indices, priorities):

        for idx, priority in zip (indices, priorities):
            self.max_priority = max(self.max_priority, priority)
            priority_alpha = priority ** self.alpha

            self._set_priority_min(idx, priority_alpha)
            self._set_priority_sum(idx, priority_alpha)

    def is_full(self):
        return self.max_size == self.size

    def save_to_npz(self, file_size_limit, fdir):

        def _save(dir, prefix, pname, idx, arr):
            fpath = os.path.join(dir, f"{prefix}_{pname}_{idx}.npz")
            npz_save(arr, fpath)
            print(f"[train | save replays] saved to '{fpath}' (size: ({len(arr)})).")

        K = self.K
        cur_size = self.size
        i = 0
        idx = 0
        while cur_size > 0:
            size = min(cur_size, file_size_limit)
            P_GUs  = np.zeros((size, K, K), dtype=np.float32)
            P_ABSs = np.zeros((size, K, K), dtype=np.float32)
            P_CGUs = np.zeros((size, K, K), dtype=np.float32)

            P_GUs  = self.data["P_GUs"][i: i+size].squeeze(1)
            P_ABSs = self.data["P_ABSs"][i: i+size].squeeze(1)
            P_CGUs = self.data["P_CGUs"][i: i+size].squeeze(1)

            _save(fdir, "train", "GUs", idx, P_GUs)
            _save(fdir, "train", "ABSs", idx, P_ABSs)
            _save(fdir, "train", "CGUs", idx, P_CGUs)
            i += size
            idx += 1
            cur_size -= size

    def load_from_npz(self, fdir):

        P_GUs_fnames, P_ABSs_fnames, P_CGUs_fnames = [
            sorted(glob(os.path.join(fdir, f"train_{pname}_*.npz")))
            for pname in ["GUs", "ABSs", "CGUs"]
        ]

        for _GU_fname, _ABS_fname, _CGU_fname in zip(P_GUs_fnames, P_ABSs_fnames, P_CGUs_fnames):
            P_GUs, P_ABSs, P_CGUs = [npz_load(fpath) for fpath in [_GU_fname, _ABS_fname, _CGU_fname]]

            P_GUs = np.expand_dims(P_GUs, axis=1)
            P_ABSs = np.expand_dims(P_ABSs, axis=1)
            P_CGUs = np.expand_dims(P_CGUs, axis=1)

            for _P_GU, _P_ABS, _P_CGU in zip(P_GUs, P_ABSs, P_CGUs):

                data = _P_GU, _P_ABS, None, _P_CGU
                self.add(data)

        print(f"[train | emulator val replay] preloaded from dir '{fdir}'.")