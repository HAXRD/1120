# Copyright (c) 2021, Xu Chen, FUNLab, Xiamen University
# All rights reserved.

import numpy as np
import random
from envs.sse.core import World, GU, ABS, BM
from envs.sse.scenario import BaseScenario

class Scenario(BaseScenario):
    """
    Realization of 'pattern' style site-specific environment.
    """
    def make_world(self, args, is_base=False, seed=0):
        """
        Create a world object with given specs.
        :param args   : (namespace) specs of the world;
        :param is_base: (bool) if have any BMs;
        """
        #### world ####
        path_to_load_BMs = args.base_BMs_fname if is_base else args.BMs_fname
        world = World(path_to_load_BMs, seed)

        ## set world properties ##
        world.episode_duration = args.episode_duration
        world.step_duration = args.step_duration

        # 3GPP empirical formula
        world.f_c = args.f_c
        world.p_t = args.p_t
        world.p_r = args.p_r

        # GUs
        world.n_GU = args.n_GU
        world.h_GU = args.h_GU
        world.v_GU = args.v_GU
        world.GUs = [GU(id=i, name=f"GU-{i}", pos=np.zeros((3,), dtype=np.float32)) for i in range(world.n_GU)]
        world.random_on_off = args.random_on_off
        world.p_on = args.p_on

        # ABSs
        world.n_ABS = args.n_ABS
        world.h_ABS = args.h_ABS
        world.v_ABS = args.v_ABS
        world.ABSs = [ABS(id=i, name=f"ABS-{i}", pos=np.zeros((3,), dtype=np.float32)) for i in range(world.n_ABS)]

        # 'pattern' style only params
        world.granularity = args.granularity
        world.K = args.K

        ## reset world ##
        self.reset_world(world)
        world.update()  # update world's entities coverage status

        return world

    def reset_world(self, world):
        """
        Set random initial positions of GUs & ABSs in world.
        """
        assert isinstance(world, World)
        for _abs in world.ABSs:
            _abs.pos[:2] = world.gen_1_2D_position(AVOID_COLLISION=False)
            _abs.pos[-1] = world.h_ABS
        for _gu in world.GUs:
            _gu.pos[:2] = world.gen_1_2D_position(AVOID_COLLISION=True)
            _gu.pos[-1] = world.h_GU

            if world.random_on_off:
                if random.random() < world.p_on:
                    _gu.ON = True
                else:
                    _gu.ON = False
        # print(f'[env | init] initialization done.')

        #### reset CRs to 0.s ####
        world.CR_old = 0.
        world.CR_new = 0.


    ############## Getters ##############
    def get_P_GU(self, world, NORMALIZED=False):
        """Get pattern-style GU info, aka. P_GU."""
        assert isinstance(world, World)
        granularity = world.granularity
        K = world.K

        P_GU = np.zeros((K, K), dtype=np.float32)
        for _gu in world.GUs:
            assert isinstance(_gu, GU)
            if _gu.ON:
                x_idx, y_idx = np.clip(_gu.pos[:2] // granularity, 0, K - 1).astype(np.int32)
                P_GU[x_idx, y_idx] += 1.

        P_GU /= world.n_ON_GU if NORMALIZED else 1.
        return P_GU

    def get_P_ABS(self, world):
        """Get pattern-style ABS info, aka. P_ABS."""
        assert isinstance(world, World)
        granularity = world.granularity
        K = world.K

        P_ABS = np.zeros((K, K), dtype=np.float32)
        for _abs in world.ABSs:
            x_idx, y_idx = np.clip(_abs.pos[:2] // granularity, 0, K - 1).astype(np.int32)
            P_ABS[x_idx, y_idx] += 1.

        return P_ABS

    def get_P_CGU(self, world, NORMALIZED=False):
        """Get pattern-style CGU info, aka. P_CGU."""
        assert isinstance(world, World)
        granularity = world.granularity
        K = world.K

        P_CGU = np.zeros((K, K), dtype=np.float32)
        for _gu in world.GUs:
            assert isinstance(_gu, GU)
            if _gu.ON and len(_gu.covered_by) > 0:
                x_idx, y_idx = np.clip(_gu.pos[:2] // granularity, 0, K - 1).astype(np.int32)
                P_CGU[x_idx, y_idx] += 1.

        P_CGU /= world.n_ON_GU if NORMALIZED else 1.
        return P_CGU

    def find_KMEANS_P_ABS(self, world, seed=0):
        """Get pattern-style KMEANS ABS placement."""
        assert isinstance(world, World)
        granularity = world.granularity
        K = world.K

        P_kmeans_centers = np.zeros((K, K), dtype=np.float32)
        kmeans_centers = world.compute_KMEANS_centers(world.GUs, seed)

        for loc in kmeans_centers:
            i, j = np.clip(loc // granularity, 0, K - 1).astype(np.int32)
            P_kmeans_centers[i, j] += 1.

        return P_kmeans_centers

    ############## Getters with augmentation ##############
    def get_P_GU_with_augmentation(self, world, abs_id, NORMALIZED=False):
        """Get pattern-style GU info with selected abs_id"""
        return self.get_P_GU(world, NORMALIZED)

    def get_P_ABS_with_augmentation(self, world, abs_id):
        """Get pattern-style ABS info with selected abs_id"""
        assert isinstance(world, World)
        granularity = world.granularity
        K = world.K

        P_ABS = np.zeros((K, K), dtype=np.float32)
        for _abs in world.ABSs:
            assert isinstance(_abs, ABS)
            if _abs.id == abs_id:
                x_idx, y_idx = np.clip(_abs.pos[:2] // granularity, 0, K - 1).astype(np.int32)
                P_ABS[x_idx, y_idx] += 1.
                break

        return P_ABS

    def get_P_CGU_with_augmentation(self, world, abs_id, NORMALIZED=False):
        """Get pattern-style CGU info with selected abs_id"""
        assert isinstance(world, World)
        granularity = world.granularity
        K = world.K

        P_CGU = np.zeros((K, K), dtype=np.float32)
        for _gu in world.GUs:
            assert isinstance(_gu, GU)
            if _gu.ON and abs_id in [item.id for item in _gu.covered_by]:
                x_idx, y_idx = np.clip(_gu.pos[:2] // granularity, 0, K - 1).astype(np.int32)
                P_CGU[x_idx, y_idx] += 1.

        P_CGU /= world.n_ON_GU if NORMALIZED else 1.
        return P_CGU

    ############## Setters ##############