import torch
import torch.nn as nn


class DynamicNet(nn.Module):
    def __init__(self, flowdynamics, steps=10, dim=2):
        super(DynamicNet, self).__init__()
        self.dt = 1./steps

        self.num_dims = dim
        self.flowdynamics = flowdynamics

        self.steps = steps

    def forward(self, x, context=None):
        for t in range(self.steps):
            dx = self.flowdynamics(x)
            x = x + dx*self.dt
        return x

