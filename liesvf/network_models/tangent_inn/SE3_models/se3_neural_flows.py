import torch
import torch.nn as nn
import math

from liesvf import network_models as models

from liesvf.utils import get_jacobian



class SE3NeuralFlows(nn.Module):
    def __init__(self, depth=7, hidden_units = 128, bins=40):
        super(SE3NeuralFlows, self).__init__()

        self.depth = depth
        self.hidden_units = hidden_units
        self.num_bins = bins

        self.dim = 6
        self.m_dim = 16

        self.layer = 'LinearSpline'

        self.main_map = models.DiffeomorphicNet(self.create_flow_sequence(), dim=self.dim)
        self.b2c_map = Sphere2Cube()

    def forward(self, x, context=None):
        y = x.clone()
        x_c = self.b2c_map(y[:, 3:])
        y[:, 3:] = x_c
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



class Sphere2Cube(nn.Module):
    def __init__(self):
        super(Sphere2Cube, self).__init__()
        iot = math.sqrt(2) / 2
        self.register_buffer('_iot', torch.Tensor([iot]))
        self.register_buffer('_norm', torch.Tensor([math.pi]))
        self.register_buffer('_pi_2', torch.Tensor([math.pi/2]))

    def jacobian(self, inputs, context=None):
        return get_jacobian(self, inputs, inputs.size(-1))

    def forward(self, inputs, context=None):
        x = inputs/self._norm

        norm_x = torch.norm(x,dim=1)
        den = 1/ torch.sqrt(1- norm_x**2)
        y0 = torch.einsum('bd, b->bd',x, den)

        y1 = torch.atan(y0)/self._pi_2
        return y1

    def backwards(self, inputs, context=None):
        y0 = torch.tan(inputs * self._pi_2)

        norm_y = torch.norm(y0,dim=1)
        den = 1 / torch.sqrt(1 + norm_y ** 2)
        x = torch.einsum('bd, b->bd', y0, den)
        return x * self._norm

