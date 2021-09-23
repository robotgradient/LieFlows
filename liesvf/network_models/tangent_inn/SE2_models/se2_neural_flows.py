import torch
import torch.nn as nn
import math
from liesvf import network_models as models
from liesvf.utils import get_jacobian


class SE2PieceWiseFlows(nn.Module):
    def __init__(self, depth=7, hidden_units = 128, bins=40):
        super(SE2PieceWiseFlows, self).__init__()

        self.depth = depth
        self.hidden_units = hidden_units
        self.num_bins = bins

        self.dim = 3
        self.m_dim = 9

        self.layer = 'LinearSpline'
        self.main_map = models.DiffeomorphicNet(self.create_flow_sequence(), dim=self.dim)

        weights = torch.Tensor([3., 3., math.pi])
        weight_vect = torch.Tensor(1/weights)
        self.register_buffer('weight', weight_vect)


    def forward(self, x, context=None):
        y = torch.einsum('n,bn->bn',self.weight, x)
        z = self.main_map(y, jacobian=False)
        return z

    def pushforward(self, x, context=None):
        z = self.forward(x)
        J = get_jacobian(self, x, x.size(-1))
        return z, J

    def create_flow_sequence(self):
        chain = []
        for i in range(self.depth):
            chain.append(self.main_layer())
            chain.append(models.RandomPermutations(self.dim))
        chain.append(self.main_layer())
        return chain

    def main_layer(self):
        return models.LinearSplineLayer(num_bins=self.num_bins, features=self.dim, hidden_features=self.hidden_units)