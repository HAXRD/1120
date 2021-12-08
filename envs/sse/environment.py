# Copyright (c) 2021, Xu Chen, FUNLab, Xiamen University
# All rights reserved.

import gym
import numpy as np

from envs.sse.core import World, BM, GU, ABS, COLORs
from envs.sse.common import DIRECTIONs_2D, DIRECTIONs_3D
import envs.sse.rendering as rendering


class SiteSpecificEnv(gym.Env):
    """
    Site specific environment that compute LoS/NLoS directly according
    to building geography.
    """

    metadata = {
        "render.modes": ["human", "rgb_array"]
    }

    def __init__(self, args, world,
                 reset_world_callback=None,
                 get_P_GU_callback=None,
                 get_P_ABS_callback=None,
                 get_P_CGU_callback=None,
                 find_KMEANS_P_ABS_callback=None,
                 get_P_GU_with_augmentation_callback=None,
                 get_P_ABS_with_augmentation_callback=None,
                 get_P_CGU_with_augmentation_callback=None,

                 get_states_callback=None,
                 get_rewards_callback=None,
                 get_costs_callback=None,
                 sample_actions_callback=None,
                 get_action_filters_callback=None
                 ):

        assert isinstance(world, World)

        # store params
        self.scenario = args.scenario

        self.args = args
        self.world = world
        self.reset_world_callback = reset_world_callback

        self.get_P_GU_callback = get_P_GU_callback
        self.get_P_ABS_callback = get_P_ABS_callback
        self.get_P_CGU_callback = get_P_CGU_callback
        self.find_KMEANS_P_ABS_callback = find_KMEANS_P_ABS_callback
        self.get_P_GU_with_augmentation_callback = get_P_GU_with_augmentation_callback
        self.get_P_ABS_with_augmentation_callback = get_P_ABS_with_augmentation_callback
        self.get_P_CGU_with_augmentation_callback = get_P_CGU_with_augmentation_callback

        self.get_states_callback = get_states_callback
        self.get_rewards_callback = get_rewards_callback
        self.get_costs_callback = get_costs_callback
        self.sample_actions_callback = sample_actions_callback
        self.get_action_filters_callback = get_action_filters_callback

        # rendering
        self.cam_range = 1.2 * self.world.world_len
        self.viewer = None
        self._reset_render()

    ############## public methods ##############
    def reset(self):
        """
        Reset all GUs & ABSs.
        """
        # reset world
        self.reset_world_callback(self.world)
        # update coverage status
        self.world.update()

        # reset renderer
        self._reset_render()

    def walk(self):
        """
        Random walk GUs.
        """
        # set GUs' action
        for _gu in self.world.GUs:
            assert isinstance(_gu, GU)
            _gu.action[:2] = DIRECTIONs_2D[np.random.randint(len(DIRECTIONs_2D))]
        # perform the action
        self.world.walk()

    def step(self, action):
        """
        Given 'action', to move ABS.
        :param action:
            'pattern'-style: shape==(K, K)
            'precise'-style: shape==(n_ABS,)
        """
        if self.scenario == "pattern": # move ABSs to grids directly
            P_ABS = action
            K = self.world.K
            granularity = self.world.granularity
            l = 0
            for i in range(K):
                for j in range(K):
                    for _ in range(int(P_ABS[i, j])):
                        self.world.ABSs[l].pos[:2] = (np.array([i, j], dtype=np.float32) + 0.5) * granularity
                        l += 1
            self.world.update()
        elif self.scenario == 'precise': # set ABSs actions
            for _action, _abs in zip(action, self.world.ABSs):
                assert isinstance(_abs, ABS)
                _abs.action = DIRECTIONs_3D[_action]
            self.world.dispatch()

    def seed(self, seed=0):
        np.random.seed(seed)

    def render(self, mode='human'):
        """Render entities."""
        if mode == "non-display":
            return

        # create viewer
        if self.viewer == None:
            self.viewer = rendering.Viewer(500, 500)
        # create rendering geometries
        self.render_geoms = []
        self.render_geoms_xform = []
        for entity in self.world.entities:
            if isinstance(entity, BM):
                geom = rendering.make_square(entity.size)
                xform = rendering.Transform()
                geom.set_color(*entity.color, alpha=0.5)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)
            elif isinstance(entity, GU):
                geom = rendering.make_triangle(entity.size)
                xform = rendering.Transform()
                geom.set_color(*entity.color, alpha=0.5)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)
            elif isinstance(entity, ABS):
                geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()
                geom.set_color(*entity.color, alpha=0.5)
                geom.add_attr(xform)
                # NLoS radius
                geom_NLoS = rendering.make_circle(entity.R_2D_NLoS, filled=False)
                xform_NLoS = rendering.Transform()
                geom_NLoS.set_color(*COLORs['green'], alpha=.5)
                geom_NLoS.add_attr(xform_NLoS)
                # LoS radius
                geom_LoS = rendering.make_circle(entity.R_2D_LoS, filled=False)
                xform_LoS = rendering.Transform()
                geom_LoS.set_color(*COLORs['green'], alpha=.5)
                geom_LoS.add_attr(xform_LoS)

                self.render_geoms.append((geom, geom_NLoS, geom_LoS))
                self.render_geoms_xform.append((xform, xform_NLoS, xform_LoS))

        world_geom = rendering.make_square(self.world.world_len)
        world_xform = rendering.Transform()
        world_geom.set_color(*COLORs['grey'], 0.1)
        world_geom.add_attr(world_xform)

        # add geoms into viewer
        self.viewer.geoms = []
        for item in self.render_geoms:
            if isinstance(item, tuple):
                assert len(item) == 3
                for geom in item:
                    self.viewer.add_geom(geom)
            else:
                self.viewer.add_geom(item)
        self.viewer.add_geom(world_geom)

        # set initial position
        pos = np.ones(2) * self.world.world_len / 2
        self.viewer.set_bounds(pos[0] - self.cam_range/2, pos[0] + self.cam_range/2, pos[1] - self.cam_range/2, pos[1] + self.cam_range/2)
        for xform_item, entity in zip(self.render_geoms_xform, self.world.entities):
            if isinstance(entity, BM):
                xform_item.set_translation(*entity.pos[:2])
            elif isinstance(entity, ABS):
                for xform in xform_item:
                    xform.set_translation(*entity.pos[:2])
            elif isinstance(entity, GU):
                xform_item.set_translation(*entity.pos[:2])
        world_xform.set_translation(*pos)
        self.viewer.render()

    ############### private methods ###############
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    ############### getters ###############

    ### pattern only ###
    def get_P_GU(self):
        return self.get_P_GU_callback(self.world, self.args.normalize_pattern)

    def get_P_ABS(self):
        return self.get_P_ABS_callback(self.world)

    def get_P_CGU(self):
        return self.get_P_CGU_callback(self.world, self.args.normalize_pattern)

    def get_all_Ps(self):
        return (
            self.get_P_GU(),
            self.get_P_ABS(),
            self.get_P_CGU()
        )

    def get_P_GU_with_augmentation(self, abs_ids):
        return self.get_P_GU_with_augmentation_callback(self.world, abs_ids, self.args.normalize_pattern)

    def get_P_ABS_with_augmentation(self, abs_ids):
        return self.get_P_ABS_with_augmentation_callback(self.world, abs_ids)

    def get_P_CGU_with_augmentation(self, abs_ids):
        return self.get_P_CGU_with_augmentation_callback(self.world, abs_ids, self.args.normalize_pattern)

    def get_all_Ps_with_augmentation(self, abs_ids):
        return (
            self.get_P_GU_with_augmentation(abs_ids),
            self.get_P_ABS_with_augmentation(abs_ids),
            self.get_P_CGU_with_augmentation(abs_ids)
        )

    ### precise only ###
    def get_states(self):
        return self.get_states_callback(self.world)

    def get_rewards(self):
        return self.get_rewards_callback(self.world)

    def get_costs(self):
        return self.get_costs_callback(self.world)

    def sample_actions(self, action_filters):
        return self.sample_actions_callback(self.world, action_filters)

    def get_action_filters(self):
        return self.get_action_filters_callback(self.world)

    ############### utils methods ###############
    def find_KMEANS_P_ABS(self, seed=0):
        return self.find_KMEANS_P_ABS_callback(self.world, seed)
