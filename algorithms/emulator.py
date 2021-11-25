# Copyright (c) 2021, Xu Chen, FUNLab, Xiamen University
# All rights reserved.

import torch

from replays.pattern.emulator import UniformReplay

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
        self.base_emulator_batch_size = args.base_emulator_batch_size

        self.device = device
        if self.emulator_net_size == "small":
            from algorithms.unets.emulator_net_small import EmulatorAttentionUNet
        elif self.emulator_net_size == "small_deeper":
            from algorithms.unets.emulator_net_small_deeper import EmulatorAttentionUNet
        else:
            from algorithms.unets.emulator_net import EmulatorAttentionUNet
        self.model = EmulatorAttentionUNet(2, 1).to(device)
        print(self.model)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def SGD_compute(self, replay, UPDATE=False):
        """
        :param UPDATE: update parameters if True
        """
        assert isinstance(replay, UniformReplay)
        total_loss = 0.
        for sample in replay.data_loader(self.base_emulator_batch_size):
            P_GUs  = torch.FloatTensor(sample["P_GUs"]).to(self.device)
            P_ABSs = torch.FloatTensor(sample["P_ABSs"]).to(self.device)
            P_CGUs = torch.FloatTensor(sample["P_CGUs"]).to(self.device)
            bz = P_GUs.size()[0]

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
                if self.args.base_emulator_grad_clip_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optim.step()
            self.model.eval()

            total_loss += loss.item() * bz
        total_loss /= replay.size
        return total_loss



