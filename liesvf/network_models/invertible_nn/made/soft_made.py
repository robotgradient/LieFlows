import torch
import torch.nn as nn
import torch.nn.functional as F

from .masked_layers import _get_input_degrees
from .masked_layers import MaskedLinear, MaskedFeedforwardBlock, MaskedResidualBlock, MaskedSoftFeedforwardBlock, MaskedClampedLinear
from liesvf.network_models.invertible_nn.nn import gaussian,quadratic,inverse_quadratic,inverse_multiquadric, linear
import numpy as np


class RbfMADE(nn.Module):
    """Implementation of MADE.

    It can use either feedforward blocks or residual blocks (default is residual).
    Optionally, it can use batch norm or dropout within blocks (default is no).
    """

    def __init__(self,
                 features,
                 hidden_features,
                 context_features=None,
                 num_blocks=1,
                 output_multiplier=1,
                 random_mask=False,
                 activation=F.relu,
                 sigma=0.45):
        super().__init__()



        # Initial layer.
        self.initial_layer = MaskedRBFLinear(
            in_degrees=_get_input_degrees(features),
            out_features=hidden_features,
            autoregressive_features=features,
            random_mask=random_mask,
            is_output=False)

        # Final layer.
        prev_out_degrees = self.initial_layer.degrees
        self.final_layer = MaskedLinear(
            in_degrees=prev_out_degrees,
            out_features=features * output_multiplier,
            autoregressive_features=features,
            random_mask=random_mask,
            is_output=True,
            bias = False
        )

        nn.init.zeros_(self.final_layer.weight.data)

    def forward(self, inputs, context=None):
        outputs = self.initial_layer(inputs)
        outputs = self.final_layer(outputs)
        return outputs



class MaskedRBFLinear(nn.Module):
    """A fixed linear module with a masked weight matrix and fixed parameters"""
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self,
                 in_degrees,
                 out_features,
                 autoregressive_features,
                 random_mask,
                 bias=True, is_output=False):
        super(MaskedRBFLinear, self).__init__()

        self.in_features = len(in_degrees)
        self.out_features = out_features

        centers = torch.rand(out_features, self.in_features)*2-1
        #self.register_buffer('centers', centers)

        self.centers = nn.Parameter(centers)
        self.log_sigmas = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()
        self.basis_func = gaussian


        mask, degrees = self._get_mask_and_degrees(
            in_degrees=in_degrees,
            out_features=out_features,
            autoregressive_features=autoregressive_features,
            random_mask=random_mask,
            is_output=is_output)
        self.register_buffer('mask', mask)
        self.register_buffer('degrees', degrees)

    def reset_parameters(self):
        nn.init.normal_(self.centers, 0, 1)
        nn.init.constant_(self.log_sigmas, 1.)

    @classmethod
    def _get_mask_and_degrees(cls,
                              in_degrees,
                              out_features,
                              autoregressive_features,
                              random_mask,
                              is_output):
        if is_output:
            out_degrees = np.tile(
                _get_input_degrees(autoregressive_features),
                out_features // autoregressive_features
            )
            out_degrees = _get_input_degrees(autoregressive_features).repeat(out_features // autoregressive_features)

            mask = (out_degrees[..., None] > in_degrees).float()

        else:
            if random_mask:
                min_in_degree = torch.min(in_degrees).item()
                min_in_degree = min(min_in_degree, autoregressive_features - 1)
                out_degrees = torch.randint(
                    low=min_in_degree,
                    high=autoregressive_features,
                    size=[out_features],
                    dtype=torch.long)
            else:
                max_ = max(1, autoregressive_features - 1)
                min_ = min(1, autoregressive_features - 1)
                out_degrees = torch.arange(out_features) % max_ + min_
            mask = (out_degrees[..., None] >= in_degrees).float()

        return mask, out_degrees

    def forward(self, input):
        size = (input.size(0), self.out_features, self.in_features)
        x = input.unsqueeze(1).expand(size)
        c = self.centers.unsqueeze(0).expand(size)
        dif = (x-c).pow(2)
        distances = torch.einsum('boi,oi->bo',dif,self.mask)
        distances = distances - self.log_sigmas
        return torch.exp(-distances)
