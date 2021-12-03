# Copyright (c) 2021, Xu Chen, FUNLab, Xiamen University
# All rights reserved.

import math
import random
import numpy as np
from envs.sse.core import World, GU, ABS, BM
from envs.sse.scenario import BaseScenario
from envs.sse.common import DIRECTIONs_3D

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

    def reset_world(self, world, ABS_KMEANS=True, seed=0):
        """
        Set random initial positions of GUs & ABSs in world.
        """
        assert isinstance(world, World)

        #### reset GUs & ABSs locations randomly ####
        for _gu in world.GUs:
            _gu.pos[:2] = world.gen_1_2D_position(AVOID_COLLISION=True)
            _gu.pos[-1] = world.h_GU

            if world.random_on_off:
                if random.random() < world.p_on:
                    _gu.ON = True
                else:
                    _gu.ON = False
        if ABS_KMEANS:
            kmeans_centers = world.compute_KMEANS_centers(world.GUs, seed)
            for _abs, _loc in zip(world.ABSs, kmeans_centers):
                _abs.pos[:2] = _loc
                _abs.pos[-1] = world.h_ABS
        else:
            for _abs in world.ABSs:
                _abs.pos[:2] = world.gen_1_2D_position(AVOID_COLLISION=False)
                _abs.pos[-1] = world.h_ABS
        # print(f'[env | init] initialization done.')

        #### reset CRs to 0.s ####
        world.CR_old = 0.
        world.CR_new = 0.


    ############## Getters ##############
    def get_states(self, world, POS_DIM=2):
        """
        Get the list of states for each agent (ABS).
        :return states: (n_ABS, 3*n_ABS),
        """
        assert isinstance(world, World)

        states = []
        id_encodings = get_onehot(world.n_ABS)

        locations = []
        for _abs in world.ABSs:
            assert isinstance(_abs, ABS)
            locations.append(_abs.pos[:POS_DIM])
        locations = np.array(locations).reshape(-1) / world.world_len
        assert locations.shape == (2 * world.n_ABS, )

        # add encoding to each
        for _encoding in id_encodings:
            state = np.concatenate([locations, _encoding], axis=0)
            states.append(state)

        states = np.array(states, dtype=np.float32)
        assert states.shape == (world.n_ABS, 3*world.n_ABS)
        return states

    def get_rewards(self, world):
        """
        Get the list of rewards for each agent (ABS).

        NOTE: based on total # of covered GUs

        :return rewards: shape=(n_ABS,)
        """
        assert isinstance(world, World)

        CR_delta = world.CR_new - world.CR_old
        if CR_delta > 0:
            reward = 1.
        elif CR_delta == 0:
            reward = -0.1
        else:
            reward = -1

        rewards = [reward for _ in range(world.n_ABS)]

        rewards = np.array(rewards, dtype=np.float32)
        assert rewards.shape == (world.n_ABS, )
        return rewards

    def get_costs(self, world):
        """
        Get the list of costs for each agent (ABS).

        NOTE: based on the # of GUs covered by the current agent

        :return costs: shape=(n_ABS,)
        """
        assert isinstance(world, World)

        costs = [0 for _ in range(world.n_ABS)]
        for _gu in world.GUs:
            assert isinstance(_gu, GU)
            if _gu.ON:
                if len(_gu.covered_by) > 0:
                    for _item in _gu.covered_by:
                        costs[_item.id] += 1.

        costs = np.array(costs, dtype=np.float32)
        assert costs.shape == (world.n_ABS,)
        return costs

    ############## Utils ##############
    def sample_actions(self, world, action_filters):
        """
        Sample a list of actions for ABSs (for epsilon greedy strategy).

        :return actions: shape=(n_ABS,)
        """
        assert isinstance(world, World)

        actions = []
        for i in range(world.n_ABS):
            while True:
                action = world.action_space.sample()
                if action_filters[i][action] == 1:
                    actions.append(action)
                    break

        actions = np.array(actions, dtype=np.int32)
        assert actions.shape == (world.n_ABS,)
        return actions

    def get_action_filters(self, world):
        """
        Get a list of actions filter
        :return action_filters: shape=(n_ABS, n_ABS)
        """
        assert isinstance(world, World)

        locations = []
        for _abs in world.ABSs:
            assert isinstance(_abs, ABS)
            locations.append(_abs.pos[:2])
        locations = np.array(locations) # (n_ABS, 2)

        one_step_distance = world.v_ABS * world.step_duration
        lower_threshold = 0. + one_step_distance
        upper_threshold = world.world_len - one_step_distance

        action_filters = np.ones((world.n_ABS, len(DIRECTIONs_3D)), dtype=np.int32) # (n_ABS, 5)

        for i in range(world.n_ABS):
            x, y = locations[i]
            if y >= upper_threshold:    # cannot go north
                action_filters[i][1] = 0
            if x >= upper_threshold:    # cannot go east
                action_filters[i][2] = 0
            if y <= lower_threshold:    # cannot go south
                action_filters[i][3] = 0
            if x <= lower_threshold:    # cannot go west
                action_filters[i][4] = 0
        return action_filters