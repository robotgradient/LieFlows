import torch
import torch.nn as nn
import math

from liesvf import network_models as models

from liesvf.utils import get_jacobian



class S2CouplingFlows(nn.Module):
    def __init__(self, depth=10, modeltype=1, hidden_units = 128, dim=2):
        super(S2CouplingFlows, self).__init__()

        self.depth = depth
        self.hidden_units = hidden_units

        self.dim = dim
        self.m_dim = 3

        if modeltype==1:
            self.layer = 'Coupling'
        elif modeltype==2:
            self.layer = 'iCoupling'

        self.main_map = models.DiffeomorphicNet(self.create_flow_sequence(), dim=self.dim)

    def forward(self, x, context=None):
        z = self.main_map(x, jacobian=False)
        return z

    def pushforward(self, x, context=None):
        z, J = self.main_map(x)
        return z, J

    def create_flow_sequence(self):
        chain = []
        for i in range(self.depth):
            chain.append(self.main_layer())
            #chain.append(models.RandomPermutations(self.dim))
        chain.append(self.main_layer())
        return chain

    def main_layer(self):
        if self.layer =='Coupling':
            mask = torch.arange(0, self.dim) % 2
            return models.CouplingLayer( num_inputs= self.dim, num_hidden=self.hidden_units, mask=mask)
        elif self.layer == 'iCoupling':
            return models.IflowCouplingLayer(d = self.dim, intermediate_dim=self.hidden_units)
