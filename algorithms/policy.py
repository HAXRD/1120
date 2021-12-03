# Copyright (c) 2021, Xu Chen, FUNLab, Xiamen University
# All rights reserved.

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from envs.sse.common import DIRECTIONs_3D

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1. - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

class Policy:
    """
    Parameter shared policy (DuelingDDQN).

    Use this as the baseline. This method (a vairation of DuelingDDQN) is used in
    'Three-Dimension Trajectory Design for Multi-UAV Wireless Network
    with Deep Reninforcement Learning'.

    The DuelingDDQN implementation is referring to: https://github.com/cyoon1729/deep-Q-networks
    """
    def __init__(self, args, device=torch.device("cpu")):

        # general args
        self.args = args
        self.lr = args.qnet_lr
        self.gamma = args.gamma # discount factor for reward
        self.tau = args.tau     # polyak update for model
        self.soft_update_period = args.soft_update_period
        self.lamb = 0.
        self.MbN = args.n_GU / args.n_ABS

        self.device = device

        # model
        input_dim = args.n_ABS * 3
        output_dim = len(DIRECTIONs_3D)
        hidden_size = args.hidden_size

        from algorithms.qnets.qnet import DuelingDQN
        self.model = DuelingDQN(input_dim, output_dim, hidden_size).to(self.device)
        self.target_model = DuelingDQN(input_dim, output_dim, hidden_size).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        hard_update(self.target_model, self.model)

    def select_actions(self, state, action_filter):
        """
        Revised batch style select action given state.
        :param state: (?, 3*n_ABS)
        """
        state = torch.FloatTensor(state).to(self.device)
        action_filter = torch.FloatTensor(action_filter).to(self.device)

        return self.model.select_action(state, action_filter)

    def update_parameters(self, batch, episode):
        """
        Update online model's parameters.
        """

        state, action, mask, reward, next_state, next_state_action_filter = batch

        state  = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).unsqueeze(1).to(self.device)
        mask   = torch.FloatTensor(mask).unsqueeze(1).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        next_state_action_filter = torch.FloatTensor(next_state_action_filter).to(self.device)

        # compute target
        with torch.no_grad():
            onlineQ_next = self.model(next_state)
            softmax_onlineQ_next = torch.softmax(onlineQ_next, dim=1)
            filtered_softmax_onlineQ_next = softmax_onlineQ_next * next_state_action_filter
            online_max_action = torch.argmax(filtered_softmax_onlineQ_next, dim=1, keepdim=True)

            targetQ_next = self.target_model(next_state)
            y = reward + self.gamma * mask * targetQ_next.gather(1, online_max_action.long())

        # compute loss
        loss = F.mse_loss(self.model(state).gather(1, action.long()), y)
        # update params
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        # polyak update
        if episode % self.soft_update_period == 0:
            soft_update(self.target_model, self.model, self.tau)

        return loss.item()