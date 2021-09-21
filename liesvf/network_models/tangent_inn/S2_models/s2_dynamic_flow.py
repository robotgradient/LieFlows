import torch
import torch.nn as nn
import math

from liesvf import network_models as models
from liesvf.utils import get_jacobian


class S2DynamicFlows(nn.Module):
    def __init__(self, steps=5, dt=0.01, hidden_units = 128):
        super(S2DynamicFlows, self).__init__()

        self.dt = dt
        self.steps = steps
        self.hidden_units = hidden_units

        self.dim = 2
        self.m_dim = 3

        self.flowdynamics = BoundedFlowDynamics()

        self.main_map = models.DynamicNet(flowdynamics=self.flowdynamics, steps=self.steps)

    def forward(self, x, context=None):
        z = self.main_map(x)
        return z

    def pushforward(self, x, context=None):
        z = self.forward(x)
        J = get_jacobian(self, x, x.size(-1))
        return z, J


class BoundedFlowDynamics(nn.Module):
    def __init__(self, in_features=2, out_features=2, units_per_dim=10):
        super(BoundedFlowDynamics, self).__init__()

        self.register_buffer('_pi', torch.Tensor([math.pi]))

        self.basis_func = models.gaussian

        #self.dynamics = models.RBF(in_features=in_features, out_features=out_features, units_per_dim=units_per_dim, basis_func=self.basis_func)
        self.dynamics = models.FCNN(in_dim= in_features, out_dim =out_features, hidden_dim= 512)
        #self.dynamics = models.RFFN(in_dim=in_features, out_dim=out_features, nfeat=10)

    def forward(self,x, context=None):
        dx = self.dynamics(x)

        r = x.pow(2).sum(-1).pow(0.5)/self._pi
        K = torch.clamp((1-r), min=0,max=1)
        dx = torch.einsum('b,bx->bx', K, dx)

        return dx



