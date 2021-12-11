# Copyright (c) 2021, Xu Chen, FUNLab, Xiamen University
# All rights reserved.

import os
import time
import math
import numpy as np
import random
import torch
import logging

from pathlib import Path
from numpy.random import default_rng
from pprint import pprint
from tqdm import tqdm
from scipy.spatial import distance

from common import binary_search

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
    def __init__(self, config):

        self.args = config["args"]
        self.run_dir = config["run_dir"]
        self.method_dir = config["method_dir"]
        self.env = config["env"]
        self.device = config["device"]

        """params"""
        ####### env params #######
        # shared
        self.world_len = self.args.world_len
        self.render = self.args.render
        self.K = self.args.K
        self.n_ABS = self.args.n_ABS
        self.top_k = self.args.n_step_explore

        ####### prepare params #######
        self.method = self.args.method

        ####### pattern only #######
        ## planning methods
        self.planning_batch_size = self.args.planning_batch_size

        # naive-kmeans

        # mutation-kmeans
        self.num_mutation_seeds = self.args.num_mutation_seeds
        self.num_mutations_per_seed = self.args.num_mutations_per_seed
        self.L = self.args.L

        # map-elites
        # custom adjusted bin size
        """
        mean bins:
            0~10,  bin_size = 10.
            10~20, bin_size = 1.
            20~30, bin_size = 0.5
            30~40, bin_size = 0.25
            40~50, bin_size = 0.5
            50~60, bin_size = 1.
            >60,   bin_size = inf
        stds bins:
            0~5,   bin_size = 0.5
            5~15,  bin_size = 0.25
            15~25, bin_size = 0.5
            25~35, bin_size = 1.
            35~45, bin_size = inf
        """
        def _gen_bins(start_val: float, end_val: float, bin_size=float('inf')):
            if bin_size == float('inf'):
                bin_size = end_val - start_val
            return [(start_val + bin_size * i) for i in range(int((end_val - start_val) / bin_size))]
        self.bin_means = _gen_bins(0, 10, 10.) + \
                         _gen_bins(10, 20, 1.) + \
                         _gen_bins(20, 30, 0.5) + \
                         _gen_bins(30, 40, 0.25) + \
                         _gen_bins(40, 50, 0.5) + \
                         _gen_bins(50, 60, 1.) + \
                         _gen_bins(60, 90)
        self.bin_stds = _gen_bins(0, 5, 0.5) + \
                        _gen_bins(5, 15, 0.25) + \
                        _gen_bins(15, 25, 0.5) + \
                        _gen_bins(25, 35, 1.) + \
                        _gen_bins(35, 45)

        self.ft_bins = [len(self.bin_means), len(self.bin_stds)]

        self.unique_populations = set()
        self.solutions = None
        self.performances = None

        """Emulator φ"""
        from algorithms.emulator import Emulator
        self.emulator = Emulator(self.args, self.device)

        """Logging"""
        self.logger = logging.getLogger("pattern-runner")
        self.logger.setLevel(logging.DEBUG)

        # create file handler
        fh = logging.FileHandler(Path(self.method_dir) / 'log.log', mode="w")
        fh.setLevel(logging.INFO)
        self.logger.addHandler(fh)

        self.logger.info("Configuration completed.")

    def emulator_load(self, fpath=None):
        emulator_fpath = fpath if fpath is not None else os.path.join(self.run_dir, "emulator_ckpts", "best_emulator.pt")
        assert os.path.exists(emulator_fpath)
        assert os.path.isfile(emulator_fpath)

        emulator_state_dict = torch.load(emulator_fpath, map_location=self.device)
        self.emulator.model.load_state_dict(emulator_state_dict)
        self.logger.info(f"[runner | emulator load] loaded emulator from '{emulator_fpath}'.")


    ############### naive-kmeans & mutation-kmeans ###############
    def mutation_kmeans_planning(self, top_k, P_GU, num_seeds, num_samples_per_seed):
        """
        Use different seeds to compute kmeans centers as P_ABSs.
        Then for each kmeans ABS, use nearby sampling to get new mutations.

        if `num_seeds`=top_k, `num_samples_per_seed`=0, the function degrades
        to naive_kmeans.

        :return: (
            top_k,
            return_planning_P_ABSs
        )
        """

        K = self.K
        L = self.L

        start = time.time()
        self.logger.info(f"[runner | {self.method}] start.")
        """Get planning P_ABSs"""
        unique_populations = set()

        for _seed in range(num_seeds):
            base_kmeans_P_ABS = self.env.find_KMEANS_P_ABS(_seed)
            base_kmeans_P_ABS_idx = tuple(sorted(get_nonzero_indices(base_kmeans_P_ABS.reshape(-1))))
            if base_kmeans_P_ABS_idx not in unique_populations:
                unique_populations.add(base_kmeans_P_ABS_idx)

                # use base pattern to sample
                for _ in range(num_samples_per_seed):
                    sampled_P_ABS_idx = tuple(sorted(mutate_1_sequence(base_kmeans_P_ABS_idx, K, L)))
                    if sampled_P_ABS_idx not in unique_populations:
                        unique_populations.add(sampled_P_ABS_idx)
        planning_size = len(unique_populations)
        planning_P_ABSs = np.zeros((planning_size, K * K), dtype=np.float32)

        planning_P_ABS_idcs = list(unique_populations)

        for _idx in range(planning_size):
            planning_P_ABSs[_idx][list(planning_P_ABS_idcs[_idx])] = 1.
        planning_P_ABSs = planning_P_ABSs.reshape(planning_size, K, K)

        if self.method == "mutation-kmeans":
            self.logger.info(f"[runner | {self.method}] generate {planning_size} different P_ABSs for planning.")

            # TODO: planning
            """Use planning to find believed top_k P_ABSs"""
            repeated_P_GUs = np.repeat(np.expand_dims(P_GU, 0), planning_size, axis=0)

            sorted_P_GUs, sorted_P_ABSs, sorted_P_rec_CGUs, sorted_rec_CRs = self.plan(repeated_P_GUs, planning_P_ABSs)
            return_planning_P_ABSs = sorted_P_ABSs
        else:
            return_planning_P_ABSs = planning_P_ABSs

        end = time.time()
        self.logger.info(f"[runner | {self.method} | {end - start}s] done.")

        top_k = min(top_k, return_planning_P_ABSs.shape[0])
        return (
            top_k,
            return_planning_P_ABSs
        )

    ############### map-elites ###############
    def _reinit_map(self):
        """
        Reinitialize MAP-Elites variables for different search instances.
        """
        self.unique_populations.clear()
        self.solutions = np.empty(self.ft_bins, dtype=object)
        self.performances = np.full(self.ft_bins, -np.inf, dtype=np.float32)

        self.logger.info(f"[runner | map-elites] reinitialized.")

    def _bootstrap(self, P_GU, n_individuals):
        """
        Randomly sample `n_individuals` to bootstrap.
        """

        K = self.K

        unique_populations = set()

        for _seed in range(n_individuals):
            kmeans_P_ABS = self.env.find_KMEANS_P_ABS(_seed)
            kmeans_P_ABS_idx = tuple(sorted(get_nonzero_indices(kmeans_P_ABS.reshape(-1))))
            if kmeans_P_ABS_idx not in self.unique_populations:
                self.unique_populations.add(kmeans_P_ABS_idx)
                unique_populations.add(kmeans_P_ABS_idx)

        planning_size = len(unique_populations)
        planning_P_ABSs = np.zeros((planning_size, K * K), dtype=np.float32)

        planning_P_ABS_idcs = list(unique_populations)

        for _idx in range(planning_size):
            planning_P_ABSs[_idx][list(planning_P_ABS_idcs[_idx])] = 1.
        planning_P_ABSs = planning_P_ABSs.reshape(planning_size, K, K)

        self.logger.info(f"[runner | map-elites] bootstrapping.")

        """Use planning"""
        repeated_P_GUs = np.repeat(np.expand_dims(P_GU, 0), planning_size, axis=0)

        sorted_P_GUs, sorted_P_ABSs, sorted_P_rec_CGUs, sorted_rec_CRs = self.plan(repeated_P_GUs, planning_P_ABSs)

        top_planning_size_P_ABSs = sorted_P_ABSs[:planning_size]
        top_planning_size_rec_CRs = sorted_rec_CRs[:planning_size]

        return (
            top_planning_size_P_ABSs,
            top_planning_size_rec_CRs
        )

    def _map_x_to_b(self, x):
        """
        Map x coordinates to feature space dimensions.
        :param x: (nparray) genotype of a solution.
        :return: (tuple) phenotype of the solution
        """

        K = self.K
        n_ABS = self.n_ABS

        # get P_ABSs' indices
        ABS_2D_coords = []
        for i in range(self.K):
            for j in range(self.K):
                for _ in range(int(x[i, j])):
                    ABS_2D_coords.append([float(i), float(j)])

        dists_matrix = distance.cdist(ABS_2D_coords, ABS_2D_coords, "euclidean").astype(np.float32)
        # process matrix
        tri_upper_no_diag = np.triu(dists_matrix, k=1)
        tri_upper_no_diag = tri_upper_no_diag.reshape(-1)
        dists = tri_upper_no_diag[np.abs(tri_upper_no_diag) > 1e-5]
        assert len(ABS_2D_coords) == n_ABS
        assert len(dists) == n_ABS * (n_ABS - 1) / 2

        mean = np.mean(dists)
        std = np.std(dists)

        self.logger.info(f"[runner | map-elites] mean {mean}, std{std}")

        i = binary_search(self.bin_means, mean)
        j = binary_search(self.bin_stds, std)

        # do check
        assert self.bin_means[i] <= mean
        if i < len(self.bin_means) - 1:
            assert mean < self.bin_means[i + 1]
        assert self.bin_stds[j] <= std
        if j < len(self.bin_stds) - 1:
            assert std < self.bin_stds[j + 1]

        return (
            i, j
        )

    def _place_in_mapelites(self, P_ABSs, CRs):
        """
        Update `self.solutions` and `self.performances`
        """
        self.logger.info("[runner | map-elites] update maps.")

        for _P_ABS, _CR in zip(P_ABSs, CRs):
            i, j = self._map_x_to_b(_P_ABS)
            if _CR >= self.performances[i, j]:
                self.performances[i, j] = _CR
                self.solutions[i][j] = _P_ABS
                self.logger.debug(f"[runner | map-elites] updated maps at {(i, j)} in feature space, new CR is {_CR}")


    def _random_selection(self, n_individuals):
        """
        Randomly select n_individuals from map.
        If the map is empty (at initial), randomly sample `n_individuals`
        to bootstrap.

        :return individuals: (list), a list of different P_ABSs.
        """

        candidates = np.argwhere(self.performances != -np.inf)
        if len(candidates) <= n_individuals:
            selected_candidates = candidates
        else:
            selected_candidates = random.sample(list(candidates), n_individuals)

        individuals = []
        for _candidate in selected_candidates:
            i, j = _candidate
            assert self.solutions[i][j] is not None
            individuals.append(self.solutions[i][j])

        self.logger.info(f"[runner | map-elites] randomly selected {len(individuals)} individuals.")

        return individuals

    def _mutation(self, P_GU, individuals):
        """
        Apply mutation to each individual (P_ABS).

        :return mutated_individuals: (list), a list of different P_ABSs
        """

        K = self.K
        L = self.L

        unique_populations = set()

        for _individual in individuals:
            assert _individual.shape == (K, K)
            P_ABS_idx = tuple(sorted(get_nonzero_indices(_individual.reshape(-1))))
            # try to do mutation
            tries = 0
            while tries < 5:
                mutated_P_ABS_idx = tuple(sorted(mutate_1_sequence(P_ABS_idx, K, L)))
                if mutated_P_ABS_idx not in unique_populations and \
                    mutated_P_ABS_idx not in self.unique_populations:
                    unique_populations.add(mutated_P_ABS_idx)
                    self.unique_populations.add(mutated_P_ABS_idx)
                    break

        planning_size = len(unique_populations)
        planning_P_ABSs = np.zeros((planning_size, K * K), dtype=np.float32)

        planning_P_ABSs_idcs = list(unique_populations)

        for _idx in range(planning_size):
            planning_P_ABSs[_idx][list(planning_P_ABSs_idcs[_idx])] = 1.
        planning_P_ABSs = planning_P_ABSs.reshape(planning_size, K, K)

        self.logger.info(f"[runner | map-elites] done mutations.")

        """Use planning"""
        repeated_P_GUs = np.repeat(np.expand_dims(P_GU, 0), planning_size, axis=0)

        sorted_P_GUs, sorted_P_ABSs, sorted_P_rec_CGUs, sorted_rec_CRs = self.plan(repeated_P_GUs, planning_P_ABSs)

        top_planning_size_P_ABSs = sorted_P_ABSs[:planning_size]
        top_planning_size_rec_CRs = sorted_rec_CRs[:planning_size]

        return (
            top_planning_size_P_ABSs,
            top_planning_size_rec_CRs
        )

    def _get_all_sorted_solutions(self):
        """
        Select top_k solutions from map
        Get all solutions sorted (by emulator) from map

        :return solutions: (list), a list of P_ABSs
        """

        candidates_coords = np.argwhere(self.performances != -np.inf)
        candidates_perfs, candidates_solus = [], []

        for _coord in candidates_coords:
            i, j = _coord
            candidates_perfs.append(self.performances[i, j])
            candidates_solus.append(self.solutions[i, j])
        candidates_perfs = np.array(candidates_perfs)
        candidates_solus = np.array(candidates_solus)

        sorted_indices = np.argsort(-candidates_perfs)
        solutions = candidates_solus[sorted_indices]

        self.logger.info(f"[runner | map-elites] found all {len(solutions)} solutions.")
        return solutions

    def map_elites(self, top_k, P_GU, iterations, n_sample_individuals):
        """
        Use MAP-Elites method to find top_k patterns.

        :return: (
            batch_size,
            top_k_P_ABSs
        )
        """

        start = time.time()
        self.logger.info(f"[runner | {self.method}] start.")
        """Start MAP-Elites"""
        # reinitialize mapelites
        self._reinit_map()

        # bootstrap
        P_ABSs, CRs = self._bootstrap(P_GU, n_sample_individuals)
        self._place_in_mapelites(P_ABSs, CRs)

        # tdqm: progress bar

        for i in range(iterations):
            self.logger.debug(f"[runner | map-elites | ITERATION {i}]")

            # get indices of random individuals from the map of elites
            individuals = self._random_selection(n_sample_individuals) # list of unique P_ABSs
            # mutation the individuals
            P_ABSs, CRs = self._mutation(P_GU, individuals) # list of unique P_ABSs

            # place the new individuals in the map of elites
            self._place_in_mapelites(P_ABSs, CRs)

        # select top_k best performed P_ABSs
        all_P_ABSs = self._get_all_sorted_solutions()

        end = time.time()
        self.logger.info(f"[runner | {self.method} | {end - start}s] MAP-Elites done.")

        top_k = min(top_k, all_P_ABSs.shape[0])
        return (
            top_k,
            all_P_ABSs
        )

    ############### planning ###############
    def plan(self, repeated_P_GUs, planning_P_ABSs):
        """
        Use emulator to predict `P_rec_CGU`, then sort transitions according
        to CRs.

        :param repeated_P_GUs : (planning_size, K, K)
        :param planning_P_ABSs: (planning_size, K, K)
        :return: (
            sorted_P_GUs,
            sorted_P_ABSs,
            sorted_P_rec_CGUs,
            sorted_rec_CRs
        )
        """
        K = self.K
        planning_size = repeated_P_GUs.shape[0]

        P_GUs = torch.FloatTensor(repeated_P_GUs).unsqueeze(1).to(self.device)
        P_ABSs = torch.FloatTensor(planning_P_ABSs).unsqueeze(1).to(self.device)

        # feed in the emulator to get P_rec_CGUs
        if planning_size < self.planning_batch_size:
            P_rec_CGUs = self.emulator.model.predict(P_GUs, P_ABSs)
            y_hats = _t2n(P_rec_CGUs).squeeze(1)
        else:
            batch_size = self.planning_batch_size
            P_GUs_chunks = [P_GUs[i: i+batch_size] for i in range(0, len(P_GUs), batch_size)]
            P_ABSs_chunks = [P_ABSs[i: i+batch_size] for i in range(0, len(P_ABSs), batch_size)]
            y_hats_chunks = []

            for _P_GUs, _P_ABSs in zip(P_GUs_chunks, P_ABSs_chunks):
                _P_rec_CGUs = self.emulator.model.predict(_P_GUs, _P_ABSs)
                _y_hats = _t2n(_P_rec_CGUs).squeeze(1)
                y_hats_chunks.append(_y_hats)
            y_hats = np.concatenate(y_hats_chunks, axis=0) # (planning_size, K, K)
        assert y_hats.shape == (planning_size, K, K)

        P_rec_CGUs = y_hats
        rec_CRs = np.sum(P_rec_CGUs.reshape(planning_size, -1), axis=-1) / self.env.world.n_ON_GU # (planning_size,)
        sorted_idcs = np.argsort(-rec_CRs, axis=-1)

        sorted_P_GUs  = repeated_P_GUs[sorted_idcs]
        sorted_P_ABSs = planning_P_ABSs[sorted_idcs]
        sorted_P_rec_CGUs = P_rec_CGUs[sorted_idcs]
        sorted_rec_CRs = rec_CRs[sorted_idcs]

        return (
            sorted_P_GUs,
            sorted_P_ABSs,
            sorted_P_rec_CGUs,
            sorted_rec_CRs
        )
