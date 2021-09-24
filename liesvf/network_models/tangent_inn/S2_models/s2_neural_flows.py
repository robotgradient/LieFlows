import numpy as np
import torch
import torch.nn as nn
import math

from liesvf import network_models as models

from liesvf.utils import get_jacobian



class S2NeuralFlows(nn.Module):
    def __init__(self, depth=4, modeltype=0,  hidden_units = 256, dim=2, bins=50):
        super(S2NeuralFlows, self).__init__()

        self.depth = depth
        self.hidden_units = hidden_units
        self.num_bins = bins

        self.dim = dim
        self.m_dim = 3

        self.layer = 'LinearSpline'
        self.main_map = models.DiffeomorphicNet(self.create_flow_sequence(), dim=self.dim)
        self.c2b_map = Circle2Box()

    def forward(self, x, context=None):
        cx = self.c2b_map(x)
        z = self.main_map(cx, jacobian=False)
        return z

    def pushforward(self, x, context=None):
        z = self.forward(x)
        J = get_jacobian(self, x, x.size(-1))
        return z, J

    def create_flow_sequence(self):
        chain = []
        perm_i = np.linspace(0, self.dim-1, self.dim, dtype=int)
        perm_i = np.roll(perm_i, 1)
        for i in range(self.depth):
            chain.append(self.main_layer(i))
        chain.append(self.main_layer())
        return chain

    def main_layer(self, i=0):
        if i % 2:
            mask = [1, 0]
        else:
            mask = [0, 1]
        return models.LinearSplineLayer(num_bins=self.num_bins, features=self.dim,
                                        hidden_features=self.hidden_units, order=mask)



class Circle2Box(nn.Module):
    def __init__(self):
        super(Circle2Box, self).__init__()

        iot = math.sqrt(2) / 2
        self.register_buffer('_iot', torch.Tensor([iot]))
        self.register_buffer('_norm', torch.Tensor([math.pi]))
        self.register_buffer('_pi_2', torch.Tensor([math.pi/2]))


    def jacobian(self, inputs, context = None):
        return get_jacobian(self, inputs, inputs.size(-1))

    def forward(self, inputs, context=None):
        x = inputs/(self._norm)

        norm_x = torch.norm(x,dim=1)
        mask = norm_x<1.

        den = 1/ torch.sqrt(1- norm_x**2)
        y0 = torch.einsum('bd, b->bd',x, den)

        y1 = torch.atan(y0)/self._pi_2

        y_out = torch.einsum('b,bx->bx',mask,y1)
        y_out[y_out != y_out] = 0


        #y_out = inputs/(self._norm+0.1)
        return y_out


    def backwards(self, inputs):
        y0 = torch.tan(inputs * self._pi)

        norm_y = torch.norm(y0,dim=1)
        den = 1 / torch.sqrt(1 + norm_y ** 2)
        x = torch.einsum('bd, b->bd', y0, den)

        return x * self._norm

