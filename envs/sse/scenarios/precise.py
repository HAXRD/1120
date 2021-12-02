# Copyright (c) 2021, Xu Chen, FUNLab, Xiamen University
# All rights reserved.

import math
import random
import numpy as np
from envs.sse.core import World, GU, ABS, BM
from envs.sse.scenario import BaseScenario

def get_onehot(n_classes,):
    """
    Get onehot encoding, e.g.,
        n = 3, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    :param n: # of classes
    :return encodings: shape=(n, n)
    """
    targets = np.arange(n_classes).reshape(-1)
    encodings = np.eye(n_classes)[targets]

    return encodings

class Scenario(BaseScenario):
    """
    Realization of 'precise' style site-specific environment.

    NOTE: the positional values are truncated into those
    without decimal part.

    Used for Deep Q-learning.
        state : (g, ID_ABS), g is all ABSs' location, ID_ABS is the
        one-hot encoding of a specific ABS
        action: the ABSs' actions, shape=(1,),
        reward: according to overall CR status
                1,      if CR_new > CR_old
                -0.1,   if CR_new = CR_old
                -1,     if CR_new < CR_old
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
        world.PL = args.PL

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

        #### reset GUs & ABSs locations randomly ####
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
    def get_states(self, world, POS_DIM=2):
        """
        Get the list of states for each agent (ABS).
        :return states: (list), the element of it has the shape of (3*n_ABS,)
        """
        assert isinstance(world, World)

        states = []
        id_encodings = get_onehot(world.n_ABS)

        locations = []
        for _abs in world.ABSs:
            assert isinstance(_abs, ABS)
            locations.append(_abs.pos[:POS_DIM])
        locations = np.array(locations).reshape(-1)
        assert locations.shape == (2 * world.n_ABS, )
        for _encoding in id_encodings:
            state = np.concatenate([locations, _encoding], axis=0).astype(np.int32) # remove decimal part
            state = state.astype(np.float32) # convert to float
            assert state.shape == (2 * world.n_ABS + world.n_ABS, )

            states.append(state)

        return states

    def get_rewards(self, world):
        """
        Get the list of rewards for each agent (ABS).

        NOTE: based on total # of covered GUs

        :return rewards: (list), the element of each is a float.
        """
        assert isinstance(world, World)

        reward = world.n_covered_ON_GU / world.n_ABS
        rewards = [reward for _ in range(world.n_ABS)]

        return rewards

    def get_costs(self, world):
        """
        Get the list of costs for each agent (ABS).

        NOTE: based on the # of GUs covered by the current agent

        :return costs: (list), each element is a float.
        """
        assert isinstance(world, World)

        costs = [0 for _ in range(world.n_ABS)]
        for _gu in world.GUs:
            assert isinstance(_gu, GU)
            if _gu.ON:
                if len(_gu.covered_by) > 0:
                    for _item in _gu.covered_by:
                        costs[_item.id] += 1.

        return costs

    ############## Utils ##############
    def sample_actions(self, world):
        """
        Sample a list of actions for ABSs (for epsilon greedy strategy).

        :return actions: (list), each element is an integer.
        """
        assert isinstance(world, World)

        actions = [world.action_space.sample() for _ in range(world.n_ABS)]
        return actions

