import torch
import torch.nn as nn
import math
from liesvf import network_models as models
from liesvf.utils import get_jacobian



class S2CouplingFlows(nn.Module):
    def __init__(self, depth=4, modeltype=1, hidden_units = 100, dim=2):
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
        z, J = self.main_map(x, jacobian=True)
        return z, J

    def create_flow_sequence(self):
        chain = []
        for i in range(self.depth):
            chain.append(self.main_layer(i))
        chain.append(self.main_layer())
        return chain

    def main_layer(self, i=0):
        if self.layer =='Coupling':
            #mask = torch.arange(0, self.dim) % 2
            if i%2:
                mask = torch.Tensor([1, 0])
            else:
                mask = torch.Tensor([0, 1])
            return models.CouplingLayer( num_inputs= self.dim, num_hidden=self.hidden_units, mask=mask)
        elif self.layer == 'iCoupling':
            return models.IflowCouplingLayer(d = self.dim, intermediate_dim=self.hidden_units)
