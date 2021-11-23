# Copyright (c) 2021, Xu Chen, FUNLab, Xiamen University
# All rights reserved.

class BaseScenario(object):
    """Defines scenario upon which the world is built."""
    # create elements of the world
    def make_world(self):
        raise NotImplementedError()
    # create initial conditions of the world
    def reset_world(self):
        raise NotImplementedError()
