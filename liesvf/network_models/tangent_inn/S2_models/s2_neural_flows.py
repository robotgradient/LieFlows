import numpy as np
import torch
import torch.nn as nn
import math

from liesvf import network_models as models
from liesvf.utils import get_jacobian


class S2NeuralFlows(nn.Module):
    def __init__(self, depth=4, modeltype=0,  hidden_units = 256, dim=2, bins=50, made='softmade'):
        super(S2NeuralFlows, self).__init__()

        self.depth = depth
        self.hidden_units = hidden_units
        self.num_bins = bins

        self.dim = dim
        self.m_dim = 3

        self.layer = 'LinearSpline'
        self.made_type = made
        self.main_map = models.DiffeomorphicNet(self.create_flow_sequence(), dim=self.dim)
        self.c2b_map = Circle2Box()

    def forward(self, x, context=None):
        z = self.main_map(x, jacobian=False)
        return z

    def pushforward(self, x, context=None):
        ## Map to square
        sx = self.c2b_map(x)

        # Euclidean Diffeomorphism
        z = self.main_map(sx, jacobian=False)

        J = get_jacobian(self, sx, sx.size(-1))
        ## J
        Jc2s = self.c2b_map.jacobian(sx)
        J = torch.einsum('bnm,bmk->bnk', Jc2s, J)

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
                                        hidden_features=self.hidden_units, order=mask,
                                        made= self.made_type)



class Circle2Box(nn.Module):
    def __init__(self):
        super(Circle2Box, self).__init__()

        iot = math.sqrt(2) / 2
        self.register_buffer('_iot', torch.Tensor([iot]))
        self.register_buffer('_norm', torch.Tensor([math.pi]))
        self.register_buffer('_pi_2', torch.Tensor([math.pi/2]))
        self.register_buffer('_pi', torch.Tensor([math.pi]))


    def jacobian(self, inputs, context = None):
        J_inv = self.jac_inv(inputs)
        return torch.inverse(J_inv)

    def jac_inv(self, inputs):
        s = inputs
        alpha = torch.sqrt(1. - s[:,[1,0]].pow(2) / 2)
        beta  =s/2* (-s[:,[1,0]]/alpha)

        dsxcx = alpha[:,0]
        dsxcy = beta[:,0]

        dsycx = beta[:,1]
        dsycy = alpha[:,1]

        J = torch.zeros(s.shape[0],2,2)
        J[:, 0, 0] = dsxcx
        J[:, 0, 1] = dsxcy
        J[:, 1, 0] = dsycx
        J[:, 1, 1] = dsycy
        return J * self._norm

    def forward(self, inputs, context=None):
        x = inputs/(self._norm)

        norm_x = torch.norm(x,dim=1)
        mask = norm_x<1.

        ## Circle to Square
        u = x[:, 0]
        v = x[:, 1]
        u_v = x[:,0].pow(2) - x[:,1].pow(2)
        sx = 0.5* torch.sqrt(2 + u_v + 2*u*math.sqrt(2)) - 0.5* torch.sqrt(2 + u_v - 2*u*math.sqrt(2))
        sy = 0.5* torch.sqrt(2 - u_v + 2*v*math.sqrt(2)) - 0.5* torch.sqrt(2 - u_v - 2*v*math.sqrt(2))
        y1 = torch.cat((sx[:,None], sy[:, None]),1)

        y_out = torch.einsum('b,bx->bx',mask,y1)
        y_out[y_out != y_out] = 0

        #y_out = inputs/(self._norm+0.1)
        return y_out


    def backwards(self, inputs):
        # y0 = torch.tan(inputs * self._pi)
        #
        # norm_y = torch.norm(y0,dim=1)
        # den = 1 / torch.sqrt(1 + norm_y ** 2)
        # x = torch.einsum('bd, b->bd', y0, den)
        y = inputs
        ## Square to Circle
        alpha = torch.sqrt(1. - y.pow(2)/2)
        x = y * alpha[:,[1,0]]

        return x * self._norm

