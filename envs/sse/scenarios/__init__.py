# Copyright (c) 2021, Xu Chen, FUNLab, Xiamen University
# All rights reserved.

import os
import imp

def load(name):
    pname = os.path.join(os.path.dirname(__file__), name)
    return imp.load_source('', pname)
