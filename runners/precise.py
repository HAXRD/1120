# Copyright (c) 2021, Xu Chen, FUNLab, Xiamen University
# All rights reserved.

import os
import time
import numpy as np
import random
import logging
from pathlib import Path

from replays.precise.replay import Transition

class Runner(object):
    """
    Precise-style runner class. An implementation of 'DQN' algorithm.
    """
    def __init__(self, config):

        self.args = config["args"]
        self.run_dir = config["run_dir"]
        self.env = config["env"]
        self.device = config["device"]
        self.writer = config["writer"]

        """params"""
        ####### env params #######
        # shared
        self.world_len = self.args.world_len
        self.render = self.args.render
        self.n_step = self.args.n_step_explore + self.args.n_step_serve

        ####### precise only ########
        self.num_env_episodes = self.args.num_env_episodes
        self.num_episodes_per_trial = self.args.num_episodes_per_trial
        self.epsilon = self.args.epsilon

        self.n_warmup_episodes = self.args.n_warmup_episodes
        self.replay_size = self.args.replay_size
        self.batch_size = self.args.policy_batch_size

        """DQN"""
        from algorithms.policy import Policy
        self.policy = Policy(self.args, self.device)

        """Replay"""
        from replays.precise.replay import UniformReplay
        self.replay = UniformReplay(self.replay_size)

        """Logging"""
        self.logger = logging.getLogger("precise-runner")
        self.logger.setLevel(logging.DEBUG)

        # create file handler
        fh = logging.FileHandler(Path(self.run_dir) / 'log.log', mode="w")
        fh.setLevel(logging.DEBUG)
        self.logger.addHandler(fh)

        self.logger.info("Configuration completed.")

    def run(self,):
        """
        Main part of the algorithm.
        """

        """Specs"""
        num_env_episodes = self.num_env_episodes
        num_episodes_per_trial = self.num_episodes_per_trial

        n_step = self.n_step
        n_warmup_episodes = self.n_warmup_episodes
        batch_size = self.batch_size
        epsilon = self.epsilon

        """Start algorithm"""
        updates = 0
        total_numsteps = 0
        for _episode in range(num_env_episodes):

            # reset or walk
            if _episode % num_episodes_per_trial == 0: # TODO: might change this to reset env periodically
                self.env.reset()
            else:
                self.env.walk()
            self.env.render(self.render)

            # select & perform action to collect transitions
            step_CRs = []
            episode_reward = 0.

            """if _episode % 5 == 0:
                if _episode < n_warmup_episodes:
                    self.logger.info(f"[train | random | {_episode+1}/{num_env_episodes}] ")
                else:
                    self.logger.info(f"[train | policy | {_episode+1}/{num_env_episodes}]")"""

            for _step in range(n_step):

                # get states for all agents
                states = self.env.get_states()  # (n_ABS, 3*n_ABS)
                action_filters = self.env.get_action_filters()


                """
                if _episode < n_warmup_episodes:
                    actions = self.env.sample_actions(action_filters)
                else:
                    actions = self.policy.select_actions(states, action_filters)
                """

                if random.random() < epsilon:
                    actions = self.env.sample_actions(action_filters)
                else:
                    actions = self.policy.select_actions(states, action_filters)

                # perform actions
                self.env.step(actions)
                self.env.render(self.render)
                CR = self.env.world.n_covered_ON_GU / self.env.world.n_ON_GU
                step_CRs.append(CR)
                total_numsteps += 1

                # get <s, a, m, r, ns, nsaf>
                rewards = self.env.get_rewards()
                next_states = self.env.get_states()
                next_states_action_filters = self.env.get_action_filters()
                if _step == n_step - 1:
                    masks = np.array([0 for _ in range(self.env.world.n_ABS)])
                else:
                    masks = np.array([1 for _ in range(self.env.world.n_ABS)])

                episode_reward += np.mean(rewards)

                for _s, _a, _m, _r, _ns, _nsaf in zip(states, actions, masks, rewards, next_states, next_states_action_filters):
                    _data = _s, _a, _m, _r, _ns, _nsaf
                    self.replay.add(*_data)

                if len(self.replay) > batch_size:
                    updates += 1

                    transitions = self.replay.sample(batch_size)
                    batch = Transition(*zip(*transitions))

                    start = time.time()
                    loss = self.policy.update_parameters(batch, _episode)
                    end = time.time()
                    self.writer.add_scalar("loss", loss, updates)

                    print(f"[train | {_episode+1} | {end - start:.5f}s] loss: {loss}")

            mean_CR = np.mean(step_CRs)
            self.writer.add_scalar("metric/mean_CR", mean_CR, _episode)
            self.writer.add_scalar("metric/reward", episode_reward, _episode)
            self.logger.info(f"[train | {_episode} | policy] mean_CR: {mean_CR}, reward: {episode_reward}")
