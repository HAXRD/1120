# Copyright (c) 2021, Xu Chen, FUNLab, Xiamen University
# All rights reserved.

import torch
from torch.utils.data import DataLoader

from replays.pattern.replay import UniformReplay

class Emulator:
    """
    Env emulator class. Used to assist planning.
    """
    def __init__(self, args, device=torch.device("cpu")):

        # general params
        self.args = args
        self.lr = args.emulator_lr
        self.K = args.K
        self.emulator_net_size = args.emulator_net_size

        # for SGD
        self.emulator_batch_size = args.emulator_batch_size
        self.emulator_grad_clip_norm = args.emulator_grad_clip_norm

        self.device = device
        if self.emulator_net_size == "default":
            from algorithms.unets.default import EmulatorAttentionUNet
        elif self.emulator_net_size == "original":
            from algorithms.unets.original import EmulatorAttentionUNet
        self.model = EmulatorAttentionUNet(2, 1).to(device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def SGD_compute(self, replay, UPDATE=False):
        """
        :param UPDATE: update parameters if True;
        """
        assert isinstance(replay, UniformReplay)

        # prepare DataLoader
        pin_memory = not (self.device == torch.device("cpu"))
        dataloader = DataLoader(replay, batch_size=self.emulator_batch_size, shuffle=UPDATE, pin_memory=pin_memory)

        total_loss = 0.
        for P_GUs, P_ABSs, P_CGUs in dataloader:
            bz = P_GUs.size()[0]

            P_GUs  = P_GUs.to(self.device)
            P_ABSs = P_ABSs.to(self.device)
            P_CGUs = P_CGUs.to(self.device)

            if UPDATE:
                self.model.train()
            else:
                self.model.eval()
            P_unmasked_rec_CGUs = self.model(P_GUs, P_ABSs)
            _uniform_loss = self.model.loss_function(P_unmasked_rec_CGUs, P_CGUs)
            loss = torch.mean(_uniform_loss)

            if UPDATE:
                self.optim.zero_grad()
                loss.backward()
                if self.emulator_grad_clip_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optim.step()
            self.model.eval()

            total_loss += loss.item() * bz

        return total_loss
