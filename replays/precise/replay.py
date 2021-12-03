# Copyright (c) 2021, Xu Chen, FUNLab, Xiamen University
# All rights reserved.

"""Replays for DQN"""

import os
import random
import numpy as np
from collections import namedtuple

Transition = namedtuple(
    "Transition", ("state", "action", "mask", "reward", "next_state", "next_state_action_filter")
)
class UniformReplay(object):
    """Uniform sampling replay."""

    def __init__(self, max_size):

        self.max_size = max_size
        self.ptr = 0

        self.data = []

    def add(self, *args):
        """
        Add a single trainsition data.
        """

        if len(self.data) < self.max_size:
            self.data.append(None)
        self.data[self.ptr] = Transition(*args)
        self.ptr = (self.ptr + 1) % self.max_size

    def sample(self, batch_size):
        """
        Sample `batch_size` samples.
        """
        return random.sample(self.data, batch_size)

    def __len__(self):
        return len(self.data)
