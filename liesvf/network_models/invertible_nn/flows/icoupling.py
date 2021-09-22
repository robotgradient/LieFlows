import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from liesvf.utils import get_jacobian

class IflowCouplingLayer(nn.Module):

    def __init__(self, d, intermediate_dim=64, swap=False, nonlinearity='ReLu'):
        nn.Module.__init__(self)
        self.d = d - (d // 2)
        self.swap = swap
        if nonlinearity=='ReLu':
            self.nonlinearity = nn.ReLU(inplace=True)
        elif nonlinearity=='Tanh':
            self.nonlinearity = nn.Tanh()

        self.net_s_t = nn.Sequential(
            nn.Linear(self.d, intermediate_dim),
            self.nonlinearity,
            nn.Linear(intermediate_dim, intermediate_dim),
            self.nonlinearity,
            nn.Linear(intermediate_dim, (d - self.d) * 2),
        )

    def forward(self, x, context=None):
        if self.swap:
            x = torch.cat([x[:, self.d:], x[:, :self.d]], 1)

        in_dim = self.d
        out_dim = x.shape[1] - self.d

        s_t = self.net_s_t(x[:, :in_dim])
        scale = torch.sigmoid(s_t[:, :out_dim]) +0.01
        shift = s_t[:, out_dim:]
        y1 = x[:, self.d:] * scale + shift

        y = torch.cat([x[:, :self.d], y1], 1) if not self.swap else torch.cat([y1, x[:, :self.d]], 1)

        return y

    def jacobian(self, inputs, context=None):
        return get_jacobian(self, inputs, inputs.size(-1))