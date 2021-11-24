# Copyright (c) 2021, Xu Chen, FUNLab, Xiamen University
# All rights reserved.

import os
import time
import numpy as np
import random
import torch
from pathlib import Path
from numpy.random import default_rng

def generate_1_sequence(size, MAX):
    rng = default_rng()
    sequence = rng.choice(MAX, size=size, replace=False)
    return sorted(sequence)

def mutate_1_sequence(base_sequence_idcs, K, L=1):
    """
    :param K: K x K pattern;
    :param L: number of outer rims to mutate.
    """

    return_sequence = []

    for _idx in base_sequence_idcs:
        # compute range to search for x axis and y axis
        # NOTE: the extra +1 for upper bound is for range() function
        x_range = [max(-L, (_idx // K) * K - _idx), min(L, (_idx // K + 1) * K - 1 - _idx) + 1]
        y_range = [max(-L, 0 - (_idx // K)), min(L, K - 1 - _idx // K) + 1]

        pool = [_idx + i + j * K for i in range(*x_range) for j in range(*y_range)]
        while True:
            sampled_idx = random.choice(pool)
            if sampled_idx not in return_sequence:
                return_sequence.append(sampled_idx)
                break
    return np.array(return_sequence, dtype=np.int32)

def get_nonzero_indices(sequence):
    """
    Return indices of nonzero values. Indices may be repeated if the value is > 1.
    :param sequence: 1d-array
    """
    sequence_ = sequence.astype(np.int32)
    nonzero_idcs = np.where(sequence_ != 0)[0]
    return_sequence = []
    for _idx in nonzero_idcs:
        for _ in range(sequence_[_idx]):
            return_sequence.append(_idx)

    return_sequence = np.array(return_sequence, dtype=np.int32)
    return return_sequence

def _t2n(x):
    return x.detach().cpu().numpy()

def _n2t(x, device=torch.device('cpu')):
    return torch.FloatTensor(x).to(device)


class Runner(object):
    """
    Pattern-style runner class. An implementation of 'Model-Based Planning' algorithm.
    """
    def __init__(self, runner_name, config):

        assert runner_name in ["TrainRunner", "EvalRunner"]

        self.runner_name = runner_name
        self.args = config["args"]
        self.run_dir = config["run_dir"]
        self.method_dir = config["method_dir"]
        self.env = config["env"]
        self.device = config["device"]
        self.writer = config["writer"]

        """params"""
        ####### env params #######
        # shared
        self.world_len = self.args.world_len
        self.render = self.args.render
        self.K = self.args.K
        if self.runner_name == "TrainRunner":
            self.top_k = self.args.n_step_explore + self.args.n_step_serve
        elif self.runner_name == "EvalRunner":
            self.top_k = self.args.n_step_explore

        ####### prepare params #######
        self.method = self.args.method

        ####### pattern only #######
        ## emulator φ params
        self.num_env_episodes = self.args.num_env_episodes
        self.num_episodes_per_trial = self.args.num_episodes_per_trial
        self.emulator_batch_size = self.args.emulator_batch_size
        self.emulator_val_batch_size = self.args.emulator_val_batch_size
        self.num_emulator_tolerance_epochs = self.args.num_emulator_tolerance_epochs
        self.emulator_train_repeats = self.args.emulator_train_repeats
        self.emulator_grad_clip_norm = self.args.emulator_grad_clip_norm

        self.emulator_val_loss = float('inf')
        self.emulator_ckpts_dir = os.path.join(self.method_dir, "emulator_ckpts")
        self.best_emulator_fpath = os.path.join(self.emulator_ckpts_dir, "best_emulator.pt")
        if not os.path.isdir(self.emulator_ckpts_dir):
            os.makedirs(self.emulator_ckpts_dir)

        ## planning methods
        self.planning_batch_size = self.args.planning_batch_size

        # naive-kmeans

        # mutation-kmeans
        self.num_mutation_seeds = self.args.num_mutation_seeds
        self.num_mutations_per_seed = self.args.num_mutations_per_seed
        self.L = self.args.L

        # map-elite

        ## replays
        self.file_size_limit = self.args.file_episode_limit
        self.emulator_replay_size = self.args.emulator_replay_size
        self.emulator_alpha = self.args.emulator_alpha
        self.emulator_beta = self.args.emulator_beta
        self.train_2_val_ratio = self.args.train_2_val_ratio

        ## interval
        self.log_interval = self.args.log_interval
        self.eval_interval = self.args.eval_interval

        ## eval
        self.use_eval = self.args.use_eval
        self.num_eval_episodes = self.args.num_eval_episodes

        ## preload
        self.use_preload = self.args.use_preload
        self.saved_once = False
        self.preload_dir = Path(os.path.join(self.method_dir, "emulator_replays"))
        # assert self.preload_dir.exists()

        """Replays for emulator φ: train:val = 10:1"""
        from replays.pattern.emulator import PrioritizedReplay
        from replays.pattern.emulator import UniformReplay
        self.emulator_replay = PrioritizedReplay(self.K, self.emulator_replay_size, self.emulator_alpha)
        self.emulator_val_replay = UniformReplay(self.K, self.emulator_replay_size // self.train_2_val_ratio)

        """Emulator φ"""
        from algorithms.emulator import Emulator
        self.emulator = Emulator(self.args, self.device)

        """Preload replays"""
        if self.use_preload:
            assert os.path.isdir(str(self.preload_dir))
            self.saved_once = True # if do preloading, there is no need to save to files again

            # TODO: preload replays and seeds
            self.emulator_replay.load_from_npz(self.preload_dir)
            self.emulator_val_replay.load_from_npz(self.preload_dir)

    def run(self, test_q):
        """
        Main part of the algorithm.
        """

        """Specs"""
        num_env_episodes = self.num_env_episodes
        num_episodes_per_trial = self.num_episodes_per_trial
        K = self.K
        top_k = self.top_k
        best_emulator_flag_counter = 0
        eval_episode = 0

        """Get base emulator φ_0"""
        self.emulator_load()

        """Start algorithm"""
        for _episode in range(num_env_episodes):

            start = time.time()

            # reset or walk
            if _episode % num_episodes_per_trial == 0:
                self.env.reset()
            else:
                self.env.walk()
            self.env.render(self.render)
            P_GU = self.env.get_P_GU()

            # plan with different methods
            if self.method == "naive-kmeans":
                pass
            elif self.method == "mutation-kmeans":
                planning_size, planning_P_ABSs = self.mutation_kmeans(self.num_mutation_seeds, self.num_mutations_per_seed)
            elif self.method == "map-elite":
                pass
            assert planning_P_ABSs.shape == (planning_size, K, K)

            # only do planning for "mutation-kmeans" or "map-elite"
            if self.method in ["mutation-kmeans", "map-elite"]:

                # preparation for planning
                repeated_P_GUs = np.repeat(np.expand_dims(P_GU, 0), planning_size, axis=0)
                assert repeated_P_GUs.shape == (planning_size, K, K)

                # use emulator to select top k transitions
                top_k_P_GUs, top_k_P_ABSs, top_k_P_rec_CGUs = self.plan(repeated_P_GUs, planning_P_ABSs) # (top_k, K, K)

            # interact with env
            top_k_P_CGUs = np.zeros((top_k, K, K), dtype=np.float32)
            for _idx, _P_ABS in enumerate(top_k_P_ABSs):
                self.env.step(_P_ABS)
                self.env.render(self.render)
                top_k_P_CGUs[_idx] = self.env.get_P_CGU()

            # store to replay φ
            for data in zip(top_k_P_GUs, top_k_P_ABSs, top_k_P_rec_CGUs, top_k_P_CGUs):
                if _episode % (1 + self.train_2_val_ratio) == 0 and not self.emulator_val_replay.is_full():
                    self.emulator_val_replay.add(data)
                else:
                    self.emulator_replay.add(data)

            # TODO:
            if not self.saved_once and self.emulator_replay.is_full() and self.emulator_val_replay.is_full():
                self.saved_once = True
                # TODO: save to files
                if not self.preload_dir.exists():
                    os.makedirs(str(self.preload_dir))
                # save emulator replay
                self.emulator_replay.save_to_npz(self.file_size_limit, self.preload_dir)
                self.emulator_val_replay.save_to_npz(self.file_size_limit, self.preload_dir)

            # train emulator
            if self.emulator_replay.is_full() and self.emulator_val_replay.is_full():

                emulator_loss, emulator_val_loss = self.train_emulator()
                if emulator_loss is not None:
                    if emulator_val_loss < self.emulator_val_loss:
                        best_emulator_flag_counter = 0

                        self.emulator_val_loss = emulator_val_loss
                        torch.save(self.emulator.model.state_dict(), self.best_emulator_fpath)
                        print(f"[train | emulator train] updating emulator ckpt, {_episode + 1}/{num_env_episodes}, loss {emulator_loss}, val loss {emulator_val_loss}.")
                    else:
                        best_emulator_flag_counter += 1

                        # save current episode ckpt
                        emulator_fpath = os.path.join(self.emulator_ckpts_dir, f"emulator-{_episode + 1}.pt")
                        torch.save(self.emulator.model.state_dict(), emulator_fpath)

                        print(f"[train | emulator train] not updating emulator ckpt, {_episode + 1}/{num_env_episodes}, loss {emulator_loss}, val loss {emulator_val_loss}. ")

                        if best_emulator_flag_counter >= self.num_emulator_tolerance_epochs:
                            print(f"[train | emulator train] overfitting.")
                            break
            else:
                emulator_loss, emulator_val_loss = [None for _ in range(2)]
                print(f"[train | emulator skipping] data still collecting, emulator replay {self.emulator_replay.size}/{self.emulator_replay.max_size}, emulator val replay {self.emulator_val_replay.size}/{self.emulator_val_replay.max_size}.")

            # log info
            if (_episode + 1) % self.log_interval == 0:
                end = time.time()
                print(f"[train | progress | {end - start}s] episode {_episode + 1}/{num_env_episodes}.")

                self.log_train(emulator_loss, emulator_val_loss, _episode)

            # eval
            if self.use_eval and self.emulator_replay.is_full() and self.emulator_val_replay.is_full():

                if eval_episode % self.eval_interval == 0:
                    test_q.put([_episode, self.best_emulator_fpath])

                eval_episode += 1

    def emulator_load(self, fpath=None):
        emulator_fpath = fpath if fpath is not None else os.path.join(self.run_dir, "base_emulator_ckpts", "best_base_emulator.pt")
        assert os.path.exists(emulator_fpath)
        assert os.path.isfile(emulator_fpath)

        emulator_state_dict = torch.load(emulator_fpath, map_location=self.device)
        self.emulator.model.load_state_dict(emulator_state_dict)
        print(f"[train | {self.runner_name} | emulator load] loaded emulator from '{emulator_fpath}'.")

    def mutation_kmeans(self, num_seeds, num_samples_per_seed):
        """
        Use different seeds to compute kmeans centers as P_ABSs.
        Then for each kmeans ABS, use nearby sampling to get
        new mutations.

        :return: (
            planning_size,
            planning_P_ABSs: shape=(planning_size, K, K)
        )
        """
        K = self.K
        L = self.L

        planning_P_ABS_idcs = set()

        for _seed in range(num_seeds):
            base_kmeans_P_ABS = self.env.find_KMEANS_P_ABS(_seed)
            base_kmeans_P_ABS_idx = tuple(sorted(get_nonzero_indices(base_kmeans_P_ABS.reshape(-1))))
            if base_kmeans_P_ABS_idx not in planning_P_ABS_idcs:
                planning_P_ABS_idcs.add(base_kmeans_P_ABS_idx)
                # use base pattern to sample
                for _ in range(num_samples_per_seed):
                    sampled_P_ABS_idx = tuple(sorted(mutate_1_sequence(base_kmeans_P_ABS_idx, K, L)))
                    if sampled_P_ABS_idx not in planning_P_ABS_idcs:
                        planning_P_ABS_idcs.add(sampled_P_ABS_idx)
        planning_size = len(planning_P_ABS_idcs)
        planning_P_ABSs = np.zeros((planning_size, K*K), dtype=np.float32)

        planning_P_ABS_idcs = list(planning_P_ABS_idcs)

        for _idx in range(planning_size):
            planning_P_ABSs[_idx][list(planning_P_ABS_idcs[_idx])] = 1.
        planning_P_ABSs = planning_P_ABSs.reshape(planning_size, K, K)

        return (
            planning_size,
            planning_P_ABSs
        )

    def plan(self, repeated_P_GUs, planning_P_ABSs):
        """
        Use emulator to predict `P_rec_CGU`, then select `top_k` transitions.

        :param repeated_P_GUs : (planning_size, K, K)
        :param planning_P_ABSs: (planning_size, K, K)
        :return: (
            top_k_P_GUs:
            top_k_P_ABSs:
            top_k_P_recs:
        )
        """
        K = self.K
        planning_size = repeated_P_GUs.shape[0]

        P_GUs  = torch.FloatTensor(repeated_P_GUs).unsqueeze(1).to(self.device)
        P_ABSs = torch.FloatTensor(planning_P_ABSs).unsqueeze(1).to(self.device)

        # feed in the emulator to get P_rec_CGUs
        if planning_size < self.planning_batch_size:
            P_rec_CGUs = self.emulator.model.predict(P_GUs, P_ABSs)
            y_hats = _t2n(P_rec_CGUs).squeeze(1) # (planning_size, K, K)
        else:
            batch_size = self.planning_batch_size
            P_GUs_chunks  = [P_GUs[i: i+batch_size] for i in range(0, len(P_GUs), batch_size)]
            P_ABSs_chunks = [P_ABSs[i: i+batch_size] for i in range(0, len(P_ABSs), batch_size)]
            y_hats_chunks = []

            for _P_GUs, _P_ABSs in zip(P_GUs_chunks, P_ABSs_chunks):
                _P_rec_CGUs = self.emulator.model.predict(_P_GUs, _P_ABSs)
                _y_hats = _t2n(_P_rec_CGUs).squeeze(1)
                y_hats_chunks.append(_y_hats)
            y_hats = np.concatenate(y_hats_chunks, axis=0) # (planning, K, K)
        assert y_hats.shape == (planning_size, K, K)

        P_rec_CGUs = y_hats
        P_rec_CRs = np.sum(P_rec_CGUs.reshape(planning_size, -1), axis=-1) / self.env.world.n_ON_GU # (planning_size,)
        top_k_idcs = np.argsort(-P_rec_CRs, axis=-1)[:self.top_k]

        top_k_P_GUs  = repeated_P_GUs[top_k_idcs]
        top_k_P_ABSs = planning_P_ABSs[top_k_idcs]
        top_k_P_rec_CGUs = P_rec_CGUs[top_k_idcs]

        return (
            top_k_P_GUs,
            top_k_P_ABSs,
            top_k_P_rec_CGUs
        )

    def train_emulator(self):
        """
        Train the emulator with sampled data from replay.
        """

        self.emulator.model.train()

        mean_loss = 0.
        for _ in range(self.emulator_train_repeats):
            sample = self.emulator_replay.sample(self.emulator_batch_size, self.emulator_beta)

            P_GUs  = _n2t(sample["P_GUs"], self.device)
            P_ABSs = _n2t(sample["P_ABSs"], self.device)
            P_CGUs = _n2t(sample["P_CGUs"], self.device)

            P_unmasked_rec_CGUs = self.emulator.model(P_GUs, P_ABSs)

            error = self.emulator.model.compute_errors(self.env.world.n_ON_GU, P_GUs, P_unmasked_rec_CGUs, P_CGUs)
            _uniform_loss = self.emulator.model.loss_function(P_unmasked_rec_CGUs, P_CGUs)
            _loss = torch.mean(_uniform_loss.new_tensor(sample["weights"]) * _uniform_loss)

            new_priorities = np.abs(error.cpu().numpy()) + 1.e-6
            self.emulator_replay.update_priorities(sample["indices"], new_priorities)

            self.emulator.optim.zero_grad()
            mean_loss += _loss.item()
            _loss.backward()
            if self.emulator_grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(self.emulator.model.parameters(), max_norm=0.5)
            self.emulator.optim.step()

        mean_loss /= self.emulator_train_repeats

        self.emulator.model.train()

        # validate emulator
        val_loss = 0.
        for _data in self.emulator_val_replay.data_loader(self.emulator_val_batch_size):
            P_GUs, P_ABSs, P_CGUs = _data["P_GUs"], _data["P_ABSs"], _data["P_CGUs"]
            P_GUs  = _n2t(P_GUs, self.device)
            P_ABSs = _n2t(P_ABSs, self.device)
            P_CGUs = _n2t(P_CGUs, self.device)

            P_rec_CGUs = self.emulator.model(P_GUs, P_ABSs)
            _uniform_loss = self.emulator.model.loss_function(P_rec_CGUs, P_CGUs)
            _loss = torch.mean(_uniform_loss)

            val_loss += _loss.item()

        val_loss /= self.emulator_val_replay.size

        return (
            mean_loss,
            val_loss
        )

    def log_train(self, emulator_loss, emulator_val_loss, _episode):
        if emulator_loss is not None:
            self.writer.add_scalar("emulator/train_loss", emulator_loss, _episode + 1)
            self.writer.add_scalar("emulator/val_loss", emulator_val_loss, _episode + 1)