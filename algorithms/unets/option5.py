# Copyright (c) 2021, Xu Chen, FUNLab, Xiamen University
# All rights reserved.

# Reference code: https://github.com/LeeJunHyun/Image_Segmentation

import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.unets.common import ConvBlock, UpConv, AttentionBlock

class EmulatorAttentionUNet(nn.Module):
    """
    The framework is based on AttentionUNet.

    φ: [P_GUs.gt(0.).float, P_ABSs] --> P_unmasked_rec_CGUs
    """
    def __init__(self, ch_in=2, ch_out=1):
        super(EmulatorAttentionUNet, self).__init__()

        ch1, ch2, ch3, ch4, ch5 = 32, 32, 64, 64, 64

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.sigmoid = nn.Sigmoid()
        self.conv1 = ConvBlock(ch_in, ch1)
        self.conv2 = ConvBlock(ch1, ch2)
        self.conv3 = ConvBlock(ch2, ch3)
        self.conv4 = ConvBlock(ch3, ch4)
        # self.conv5 = ConvBlock(ch4, ch5)

        # self.up5 = UpConv(ch5, ch4)
        # self.att5 = AttentionBlock(ch4, ch4, ch4//2)
        # self.up_conv5 = ConvBlock(2*ch4, ch4)

        self.up4 = UpConv(ch4, ch3)
        self.att4 = AttentionBlock(ch3, ch3, ch3//2)
        self.up_conv4 = ConvBlock(2*ch3, ch3)

        self.up3 = UpConv(ch3, ch2)
        self.att3 = AttentionBlock(ch2, ch2, ch2//2)
        self.up_conv3 = ConvBlock(2*ch2, ch2)

        self.up2 = UpConv(ch2, ch1)
        self.att2 = AttentionBlock(ch1, ch1, ch1//2)
        self.up_conv2 = ConvBlock(2*ch1, ch1)

        self.conv_1x1 = nn.Conv2d(ch1, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, P_GUs, P_ABSs):
        """
        Feed foward func for emulator.
        :param P_GUs : (?, 1, K, K)
        :param P_ABSs: (?, 1, K, K)
        :return P_unmasked_rec_CGUs: (?, 1, K, K), not masked reconstructed P_CGUs.

        NOTE: usage of P_unmasked_rec_CGUs
        loss func: can be used to compute loss func directly
        whose 'target' is P_CGUs.gt(0.).float()
        """
        assert len(P_GUs.size()) == 4
        assert len(P_ABSs.size()) == 4
        # process input
        P_GUs1 = P_GUs.gt(0.).float()
        P_ABSs1 = P_ABSs.gt(0.).float()

        x = torch.cat([P_GUs1, P_ABSs1], dim=1)

        # encoding
        x1 = self.conv1(x)

        x2 = self.maxpool(x1)
        x2 = self.conv2(x2)

        x3 = self.maxpool(x2)
        x3 = self.conv3(x3)

        x4 = self.maxpool(x3)
        x4 = self.conv4(x4)

        # x5 = self.maxpool(x4)
        # x5 = self.conv5(x5)

        # # decoding + concat
        # d5 = self.up5(x5)
        # x4 = self.att5(g=d5, x=x4)
        # d5 = torch.cat((x4, x5), dim=1)
        # d5 = self.up_conv5(d5)

        # d4 = self.up4(d5)
        d4 = self.up4(x4)
        x3 = self.att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.up_conv4(d4)

        d3 = self.up3(d4)
        x2 = self.att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        x1 = self.att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.up_conv2(d2)

        d1 = self.conv_1x1(d2)

        P_unmasked_rec_CGUs = self.sigmoid(d1)
        return P_unmasked_rec_CGUs

    def loss_function(self, P_unmasked_rec_CGUs, P_CGUs):

        assert len(P_CGUs.size()) == 4
        # process input
        P_CGUs1 = P_CGUs.gt(0.).float()

        # compute loss
        criterion = torch.nn.BCELoss(reduction='none')
        loss = criterion(P_unmasked_rec_CGUs, P_CGUs1)
        loss = torch.mean(torch.flatten(loss, start_dim=1), dim=1)

        return loss

    def predict(self, P_GUs, P_ABSs):

        assert len(P_GUs.size()) == 4
        assert len(P_ABSs.size()) == 4

        # process input
        P_GUs1 = P_GUs.gt(0.).float()

        # use forward func to get P_unmasked_rec_CGUs
        P_unmasked_rec_CGUs = self.forward(P_GUs, P_ABSs)

        # 1. multiply by mask
        P_masked_rec_CGUs = P_unmasked_rec_CGUs * P_GUs1
        # 2. filter with threshold and multiply by P_GUs
        P_rec_CGUs = P_masked_rec_CGUs.gt(0.5).float() * P_GUs

        return P_rec_CGUs

    @torch.no_grad()
    def compute_errors(self, n_GU, P_GUs, P_unmasked_rec_CGUs, P_CGUs):
        """
        :param P_GUs: (?, 1, K, K)
        :param P_unmasked_rec_CGUs:
        :param P_CGUs: (?, 1, K, K)
        :return errors: (?, )
        """
        # process input
        P_GUs1 = P_GUs.gt(0.).float() # GUs mask

        # 1. multiply by mask
        P_masked_rec_CGUs = P_unmasked_rec_CGUs * P_GUs1
        # 2. filter with threshold and multiply by P_GUs
        P_rec_CGUs = P_masked_rec_CGUs.gt(0.5).float() * P_GUs

        # flatten
        P_CGUs_flatten = torch.flatten(P_CGUs, start_dim=1)
        P_rec_CGUs_flatten = torch.flatten(P_rec_CGUs, start_dim=1)

        abs_diff = torch.abs(P_rec_CGUs_flatten - P_CGUs_flatten)
        errors = torch.sum(abs_diff, dim=1) / n_GU

        return errors


