import torch.nn as nn

class DynamicSystem(nn.Module):
    def __init__(self, dim, device=None):
        super().__init__()
        self.dim = dim
        self.device = device

    def forward(self, x):
        pass

    def potential(self, x):
        pass
