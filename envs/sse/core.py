# Copyright (c) 2021, Xu Chen, FUNLab, Xiamen University
# All rights reserved.
# NOTE: Only precise information are considered in this piece of code.

import os
import scipy.io as sio
import numpy as np
import random
from gym.spaces import Discrete
from collections import namedtuple
from sklearn.cluster import KMeans

from envs.sse.common import compute_2D_distance, compute_R_2D_LoS, compute_R_2D_NLoS, DIRECTIONs_3D

COLORs = {
    'grey'  : np.array([0.5, 0.5, 0.5]), # 'off' GU
    'red'   : np.array([0.9, 0.0, 0.0]), # not covered GU
    'green' : np.array([0.0, 0.9, 0.0]), # covered GU
    'blue'  : np.array([0.0, 0.0, 0.9]), # ABS
    'orange': np.array([1.0, 0.5, 0.0])  # BM color
}

class Entity(object):
    """Basic class for all entities in World class"""
    def __init__(self,
                 id, name,
                 pos,
                 size=8.):

        # basic properties
        self.id    = id     # unique identifier
        self.name  = name   # f'{subclass_type}-{id}'
        self.pos   = pos    # shape==(3, ), 3D position
        assert pos.shape == (3, )
        # visualization properties
        self.color = None
        self.size  = size

class BM(Entity):
    """Build mesh class"""
    def __init__(self,
                 id, name,
                 pos,
                 size):
        super(BM, self).__init__(id, name, pos, size)
        self.color = COLORs['orange']


# namedtuple for storing GU covered by which ABS(s) info
CoverTuple = namedtuple('CoverTuple', ('id', 'distance_2D'))
class GU(Entity):
    """Ground user class"""
    def __init__(self,
                 id, name,
                 pos):
        super(GU, self).__init__(id, name, pos)
        self.color = COLORs['red']

        self.ON = True
        self.action = np.zeros((3,), dtype=np.float32)
        self.covered_by = []    # (list[CoverTuple]) a list of tuples that include which ABSs are covering this GU along with the 2D distance between them

    def sort_covered_by(self):
        """Sort the `covered_by` list by 2D distance."""
        self.covered_by = sorted(self.covered_by, key=lambda x: x.distance_2D)

class ABS(Entity):
    """Aerial base station / Unmanned aerial vehicles"""
    def __init__(self,
                 id, name,
                 pos):
        super(ABS, self).__init__(id, name, pos)
        self.color = COLORs['blue']

        self.action = np.zeros((3, ), dtype=np.float32)

        self.recomputed = True
        self.R_2D_NLoS = 0.
        self.R_2D_LoS = 0.



class World(object):
    """
    Site-Specific World object contains
    - building meshes (BMs)
    - ground users (GUs)
    - aerial base stations / unmanned aerial vehicles (ABSs/UAVs)

    NOTE: Only the BMs' info is initialized during instantiation.
    """

    def __init__(self, path_to_load_BMs, seed=0):

        #### seed ####
        self.np_rng = np.random.RandomState(0)
        self.seed = seed if seed is not None else 0
        self.np_rng.seed(self.seed)

        #### entities ####
        # load terrain
        world_len, mesh_len, N, grids = self._load_BMs(path_to_load_BMs)
        self.world_len = world_len
        self.mesh_len  = mesh_len           # side len of each BM
        self.n_BM = N                       # num of BMs
        self.M = int(world_len // mesh_len) # num of maximum meshes along the world side
        self.grids = grids                  # shape==(M, M), the value of each elem is the correponding height

        self.dim_pos = 3
        self.dim_color = 3
        self.episode_duration = 0.
        self.step_duration = 0.
        # TODO: 3GPP equations
        self.f_c = 0.           # carrier frequency
        self.p_t = 0.           # maximum transmit power (in log)
        self.p_r = 0.           # minimum receive power (in log)

        # BMs
        self.BMs = self._process_BMs()
        assert len(self.BMs) == self.n_BM

        # GUs
        self.GUs = []
        self.n_GU = 0
        self.h_GU = 0.      # default height
        self.v_GU = 0.
        self.random_on_off = False
        self.p_on = 1.

        # Coverage rates (CR)
        self.CR_new = 0.
        self.CR_old = 0.

        # ABSs
        self.ABSs = []
        self.n_ABS = 0
        self.h_ABS = 0.     # default height
        self.v_ABS = 0.

        # 'pattern'-style only params
        self.granularity = None
        self.K = None

        # 'precise'-style only params
        self.action_space = Discrete(len(DIRECTIONs_3D))


    ############### public methods ###############
    def gen_1_2D_position(self, AVOID_COLLISION=True):
        """Generate 1 pair of 2D position."""
        world_len = self.world_len
        mesh_len = self.mesh_len
        M = self.M
        while True:
            x, y = world_len * self.np_rng.rand(2)
            if AVOID_COLLISION:
                x_idx, y_idx = np.clip(np.array([x, y]) // mesh_len, 0, M - 1).astype(np.int32)
                if self.grids[x_idx, y_idx] == 0.:
                    return np.array([x, y], dtype=np.float32)
            else:
                return np.array([x, y], dtype=np.float32)

    def walk(self):
        """All GUs do random walk."""
        v_GU = self.v_GU
        episode_duration = self.episode_duration
        world_len = self.world_len
        mesh_len = self.mesh_len
        M = self.M

        for _gu in self.GUs:
            assert isinstance(_gu, GU)
            next_pos = np.clip(_gu.pos + _gu.action * v_GU * episode_duration, 0, world_len)
            next_x_idx, next_y_idx = np.clip((next_pos[:2] // mesh_len).astype(np.int32), 0, M - 1)
            if self.grids[next_x_idx, next_y_idx] == 0.:
                _gu.pos = next_pos

            # randomly turn on/off
            if self.random_on_off:
                if random.random() < self.p_on:
                    _gu.ON = True
                else:
                    _gu.ON = False

        # update covered state
        self.update()

    def dispatch(self):
        """All ABSs perform their actions."""
        v_ABS = self.v_ABS
        step_duration = self.step_duration
        world_len = self.world_len
        mesh_len = self.mesh_len
        M = self.M

        for _abs in self.ABSs:
            assert isinstance(_abs, ABS)
            next_pos = np.clip(_abs.pos + _abs.action * v_ABS * step_duration, 0, world_len)
            next_x_idx, next_y_idx = np.clip((next_pos[:2] // mesh_len).astype(np.int32), 0, M - 1)
            if self.grids[next_x_idx, next_y_idx] <= next_pos[-1]:
                _abs.pos = next_pos

        # update covered state
        self.update()

    def update(self):
        """Update the GUs' covered state."""
        # reset all ABSs not recomputed radias
        for _abs in self.ABSs:
            assert isinstance(_abs, ABS)
            _abs.recomputed = False

        #### update all GUs' covered_by lists ####
        for _gu in self.GUs:
            assert isinstance(_gu, GU)
            # empty out covered_by list
            _gu.covered_by = []
            # recompute covered by list
            for _abs in self.ABSs:
                assert isinstance(_abs, ABS)
                distance_2D = compute_2D_distance(_gu.pos[:2], _abs.pos[:2])
                # compute LoS & NLoS 2D radius for only once
                if _abs.recomputed == False:
                    R_2D_LoS, R_2D_NLoS = self._compute_Rs(_abs, _gu)
                    _abs.R_2D_NLoS = R_2D_NLoS
                    _abs.R_2D_LoS  = R_2D_LoS
                    _abs.recomputed = True

                # the GU is covered by an ABs only under 2 conditions:
                # 1. within the NLoS range
                # 2. within the ring between NLoS and LoS range and is LoS
                if distance_2D <= R_2D_NLoS or \
                  (distance_2D >  R_2D_NLoS and distance_2D <= R_2D_LoS and self._judge_is_LoS(_abs, _gu)):
                  # update `covered_by` property
                  _gu.covered_by.append(CoverTuple(id=_abs.id, distance_2D=distance_2D))
            # change covered GUs' color
            if not _gu.ON: # disabled
                _gu.color = COLORs['grey']
            elif len(_gu.covered_by) == 0:
                _gu.color = COLORs['red']
            else:
                _gu.color = COLORs['green']

        # print(f"[env | update] updated GUs' `covered_by` list.")
        
        #### update CRs ####
        self.CR_old = self.CR_new
        self.CR_new = self.n_covered_ON_GU / self.n_ON_GU

    def compute_KMEANS_centers(self, gus, seed=0):
        """compute KMEANS centers w/ given GUs."""
        locs_2D_gus = np.stack([gu.pos[:2] for gu in gus if gu.ON], axis=0)
        assert locs_2D_gus.shape == (self.n_ON_GU, 2)
        kmeans = KMeans(n_clusters=self.n_ABS, random_state=seed).fit(locs_2D_gus)
        centers = kmeans.cluster_centers_.astype(np.float32)
        assert centers.shape == (self.n_ABS, 2)
        return centers

    ############### private methods ###############
    def _load_BMs(self, path_to_load_BMs):
        """Load BMs's info from mat file."""
        if not os.path.exists(path_to_load_BMs) or not os.path.isfile(path_to_load_BMs):
            raise FileNotFoundError()
        else:
            mat = sio.loadmat(path_to_load_BMs)
            return (
                mat["world_len"].item(),
                mat["mesh_len"].item(),
                mat["N"].item(),
                mat["grids"]
            )

    def _process_BMs(self):
        """Process mat file into a list of BM objects."""
        BMs = []
        k = 0
        M = self.M
        for i in range(M):
            for j in range(M):
                h = self.grids[i, j]
                if h > 0.:
                    pos = np.zeros((3,), dtype=np.float32)
                    pos[:2] = self.mesh_len * (np.array([i, j], dtype=np.float32) + 0.5)
                    pos[-1] = h
                    BMs.append(BM(k, f"BM-{k}", pos, self.mesh_len))
        return BMs

    def _judge_is_LoS(self, abs, gu):
        """Judge if a given pair of ABS and GU is LoS."""
        assert isinstance(abs, ABS)
        assert isinstance(gu, GU)
        assert abs.pos[-1] > gu.pos[-1]

        mesh_len = self.mesh_len
        M = self.M

        dx, dy = (gu.pos - abs.pos)[:2]

        # calculate `x` value of the intersection between
        # abs-gu line and the given BM edge line (by
        # providing its `y` value)
        def _cal_x(y):
            return abs.pos[0] + (y - abs.pos[1]) * dx / dy
        def _cal_y(x):
            return abs.pos[1] + (x - abs.pos[0]) * dy / dx

        abs_x_i, abs_y_i = np.clip(abs.pos[:2] // mesh_len, 0, M - 1).astype(np.int32)
        gu_x_i, gu_y_i = np.clip(gu.pos[:2] // mesh_len, 0, M - 1).astype(np.int32)

        for _idx_i in range(min(abs_x_i, gu_x_i), max(abs_x_i, gu_x_i) + 1):
            for _idx_j in range(min(abs_y_i, gu_y_i), max(abs_y_i, gu_y_i) + 1):
                # taken by a BM
                if self.grids[_idx_i, _idx_j] > 0.:
                    w_x, e_x = _idx_i * mesh_len, (_idx_i+1) * mesh_len
                    s_y, n_y = _idx_j * mesh_len, (_idx_j+1) * mesh_len

                    # compute 2D projection intersection point as `point`
                    point = None
                    # different conditions for intersections (take `ABS` as origin)
                    # axises
                    if dx == 0 and dy > 0:      # GU: y-axis +
                        point = np.array([abs.pos[0], n_y])
                    elif dx == 0 and dy < 0:    # GU: y-axis -
                        point = np.array([abs.pos[0], s_y])
                    elif dx > 0 and dy == 0:    # GU: x-axis +
                        point = np.array([e_x, abs.pos[1]])
                    elif dx < 0 and dy == 0:    # GU: x-axis -
                        point = np.array([w_x, abs.pos[1]])
                    elif dx > 0 and dy > 0:     # GU: 1st quad
                        point_1 = np.array([_cal_x(n_y), n_y])
                        point_2 = np.array([e_x, _cal_y(e_x)])
                        if point_1[0] >= w_x and point_1[0] <= e_x:
                            point = point_1
                        elif point_2[1] >= s_y and point_2[1] <= n_y:
                            point = point_2
                        else:
                            pass # LoS
                    elif dx < 0 and dy > 0:     # GU: 2nd quad
                        point_1 = np.array([_cal_x(n_y), n_y])
                        point_4 = np.array([w_x, _cal_y(w_x)])
                        if point_1[0] >= w_x and point_1[0] <= e_x:
                            point = point_1
                        elif point_4[1] >= s_y and point_4[1] <= n_y:
                            point = point_4
                        else:
                            pass # LoS
                    elif dx < 0 and dy < 0:     # GU: 3rd quad
                        point_3 = np.array([_cal_x(s_y), s_y])
                        point_4 = np.array([w_x, _cal_y(w_x)])
                        if point_3[0] >= w_x and point_3[0] <= e_x:
                            point = point_3
                        elif point_4[1] >= s_y and point_4[1] <= n_y:
                            point = point_4
                        else:
                            pass # LoS
                    elif dx > 0 and dy < 0:     # GU: 4th quad
                        point_2 = np.array([e_x, _cal_y(e_x)])
                        point_3 = np.array([_cal_x(s_y), s_y])
                        if point_2[1] >= s_y and point_2[1] <= n_y:
                            point = point_2
                        elif point_3[0] >= w_x and point_3[0] <= e_x:
                            point = point_3
                        else:
                            pass # LoS
                    # check vertical intersection
                    if point is not None:
                        d = compute_2D_distance(abs.pos[:2], point)
                        D = compute_2D_distance(abs.pos[:2], gu.pos[:2])
                        dH = abs.pos[-1] - gu.pos[-1]
                        dh = dH - dH*d/D + gu.pos[-1]
                        if dh < self.grids[_idx_i, _idx_j]: # 3D wise intersection
                            return False # NLoS
        return True # LoS

    def _compute_Rs(self, abs, gu):
        """
        Compute LoS & NLoS 2D radias.
        """
        assert isinstance(abs, ABS)
        assert isinstance(gu, GU)

        path_loss = self.p_t - self.p_r
        f_c = self.f_c
        h_ABS = abs.pos[-1]
        h_GU = gu.pos[-1]

        R_2D_NLoS = compute_R_2D_NLoS(path_loss, h_ABS, h_GU, f_c)
        R_2D_LoS = compute_R_2D_LoS(path_loss, h_ABS, h_GU, f_c)

        # print(f"[env | LoS & NLoS] LoS (2D): {R_2D_LoS}, NLoS (2D): {R_2D_NLoS}")
        return (
            R_2D_LoS, R_2D_NLoS
        )

    @property
    def entities(self):
        return self.BMs + self.GUs + self.ABSs

    @property
    def n_ON_GU(self):
        n = 0
        for _gu in self.GUs:
            if _gu.ON:
                n += 1
        return n

    @property
    def n_covered_ON_GU(self):
        n = 0
        for _gu in self.GUs:
            if _gu.ON and len(_gu.covered_by) > 0:
                n += 1
        return n
