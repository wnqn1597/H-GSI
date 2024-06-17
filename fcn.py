# -*- coding: utf-8 -*-
import numpy as np
import torch.nn as nn
from safetensors.torch import load_model


class ResidualBlock(nn.Module):
    def __init__(self, ch, k, dil):
        super(ResidualBlock, self).__init__()
        assert dil * (k - 1) % 2 == 0
        padding = dil * (k - 1) // 2
        self.layer = nn.Sequential(
            nn.BatchNorm1d(ch),
            nn.ReLU(),
            nn.Conv1d(ch, ch, k, dilation=dil, padding=padding),
            nn.BatchNorm1d(ch),
            nn.ReLU(),
            nn.Conv1d(ch, ch, k, dilation=dil, padding=padding),
        )

    def forward(self, x):
        return self.layer(x) + x


class FCN(nn.Module):
    def __init__(self, emb, ch, K, DIL, num_labels=6):
        super(FCN, self).__init__()
        assert len(K) == len(DIL)
        self.num_layers = len(K)
        self.CL = 2 * np.sum(DIL * (K - 1))
        self.embeddings = nn.Embedding(4105, emb, padding_idx=1)
        self.pre_conv = nn.Conv1d(emb, ch, 1)
        self.pre_skip = nn.Conv1d(ch, ch, 1)
        self.residual_blocks = nn.ModuleList()
        self.skip_list = nn.ModuleDict()
        for i, (k, dil) in enumerate(zip(K, DIL)):
            self.residual_blocks.append(ResidualBlock(ch, k, dil))
            if ((i + 1) % 4 == 0) or ((i + 1) == len(K)):
                self.skip_list.add_module(str(i), nn.Conv1d(ch, ch, 1))
        self.out_conv = nn.Conv1d(ch, num_labels, 1)

    def forward(self, input_ids, labels=None):  # B x L
        x = self.embeddings(input_ids)
        x = x.permute(0, 2, 1)
        conv = self.pre_conv(x)
        skip = self.pre_skip(conv)
        for i in range(self.num_layers):
            conv = self.residual_blocks[i](conv)
            if ((i + 1) % 4 == 0) or ((i + 1) == self.num_layers):
                skip = skip + self.skip_list[str(i)](conv)
        logits = self.out_conv(skip).permute(0, 2, 1)  # B x L x 6
        return {"logits": logits}


def get_fcn():
    CH = 128
    K = np.array([11, 11, 11, 11, 11, 11, 11, 11, 21, 21, 21, 21])
    DIL = np.array([1, 1, 1, 1, 4, 4, 4, 4, 10, 10, 10, 10])
    model = FCN(32, CH, K, DIL)
    load_model(model, "models/fcn.safetensors")
    return model
