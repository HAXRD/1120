# Copyright (c) 2021, Xu Chen, FUNLab, Xiamen University
# All rights reserved.
# NOTE: Only precise information are considered in this piece of code.

import argparse
import math

import numpy as np
from numba import jit

DIRECTIONs_2D = [
    np.array([0, 0], dtype=np.float32),
    np.array([0, 1], dtype=np.float32),
    np.array([1, 0], dtype=np.float32),
    np.array([0, -1], dtype=np.float32),
    np.array([-1, 0], dtype=np.float32)
]

DIRECTIONs_3D = [
    np.array([0, 0, 0], dtype=np.float32),
    np.array([0, 1, 0], dtype=np.float32),
    np.array([1, 0, 0], dtype=np.float32),
    np.array([0, -1, 0], dtype=np.float32),
    np.array([-1, 0, 0], dtype=np.float32),
    np.array([0, 0, 1], dtype=np.float32),
    np.array([0, 0, -1], dtype=np.float32)
]

@jit(nopython=True)
def compute_2D_distance(pos1, pos2):
    assert pos1.shape == (2,)
    assert pos2.shape == (2,)
    return np.sqrt(np.sum((pos1 - pos2)**2))

def compute_R_2D_NLoS(PL, h_ABS, h_GU, f_c):
    """
    According to 3GPP empirical formula:
        PL_NLoS = -17.5 + (46 - 7 * log_10(h_UAV)) * log_10(d_3D) + 20 * log_10(40 * pi * f_c / 3)
    """
    comp1 = -17.5
    mult2 = 46 - 7 * math.log10(h_ABS)
    mult3 = 20
    comp3 = math.log10(40 * math.pi * f_c / 3)

    radius = ((10 ** ((PL - comp1 - mult3 * comp3) / mult2)) ** 2 - (h_ABS - h_GU) ** 2) ** 0.5
    return radius

def compute_R_2D_LoS(PL, h_ABS, h_GU, f_c):
    """
    According to 3GPP empirical formula:
        PL_LoS = 28. + 22 * log_10(d_3D) + 20 * log_10(f_c)
    """

    comp1 = 28.
    mult2 = 22
    mult3 = 20
    comp3 = math.log10(f_c)

    radius = ((10 ** ((PL - comp1 - mult3 * comp3) / mult2)) ** 2 - (h_ABS - h_GU) ** 2) ** 0.5
    return radius

if __name__ == "__main__":
    # parsing args
    parser = argparse.ArgumentParser()
    parser.add_argument("--PL", type=float, default=85)
    parser.add_argument("--h_ABS", type=float, default=90)
    parser.add_argument("--h_GU", type=float, default=1)
    parser.add_argument("--f_c", type=float, default=2)
    args = parser.parse_args()

    PL = args.PL
    h_ABS = args.h_ABS
    h_GU = args.h_GU
    f_c = args.f_c

    R_2D_NLoS = compute_R_2D_NLoS(PL, h_ABS, h_GU, f_c)
    R_2D_LoS = compute_R_2D_LoS(PL, h_ABS, h_GU, f_c)

    print(f"R_2D_NLoS: {R_2D_NLoS}, R_2D_LoS: {R_2D_LoS}")
