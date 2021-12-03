# Copyright (c) 2021, Xu Chen, FUNLab, Xiamen University
# All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class DuelingDQN(nn.Module):
    """
    A revised version of DuelingDQN.

    See blog post for summarized theories: https://towardsdatascience.com/dueling-deep-q-networks-81ffab672751
    Reference: https://github.com/cyoon1729/deep-Q-networks &
               https://github.com/guoxiangchen/dueling-DQN-pytorch
    """
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(DuelingDQN, self).__init__()

        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):

        feature = self.feature_layer(x)
        value = self.value_stream(feature)
        advantage = self.advantage_stream(feature)
        avg_advantage = torch.mean(advantage, dim=1, keepdim=True)

        qval = value + advantage - avg_advantage
        return qval

    def select_action(self, state, action_filter):

        with torch.no_grad():
            qval = self.forward(state)
            softmax_qval = torch.softmax(qval, dim=1)
            filtered_qval = softmax_qval * action_filter
            action = torch.argmax(filtered_qval, dim=1)
        return action.cpu().numpy()
