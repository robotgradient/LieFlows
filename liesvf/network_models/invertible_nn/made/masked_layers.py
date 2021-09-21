import torch.nn as nn
import torch
from torch.nn import functional as F, init

import numpy as np


def _get_input_degrees(in_features):
    """Returns the degrees an input to MADE should have."""
    return torch.arange(1, in_features + 1)


class MaskedLinear(nn.Linear):
    """A linear module with a masked weight matrix."""

    def __init__(self,
                 in_degrees,
                 out_features,
                 autoregressive_features,
                 random_mask,
                 is_output,
                 bias=True):
        super().__init__(
            in_features=len(in_degrees),
            out_features=out_features,
            bias=bias)
        mask, degrees = self._get_mask_and_degrees(
            in_degrees=in_degrees,
            out_features=out_features,
            autoregressive_features=autoregressive_features,
            random_mask=random_mask,
            is_output=is_output)
        self.register_buffer('mask', mask)
        self.register_buffer('degrees', degrees)

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

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)


class MaskedFeedforwardBlock(nn.Module):
    """A feedforward block based on a masked linear module.

    NOTE: In this implementation, the number of output features is taken to be equal to
    the number of input features.
    """

    def __init__(self,
                 in_degrees,
                 autoregressive_features,
                 context_features=None,
                 random_mask=False,
                 activation=F.relu,
                 dropout_probability=0.,
                 use_batch_norm=False):
        super().__init__()
        features = len(in_degrees)

        # Batch norm.
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(features, eps=1e-3)
        else:
            self.batch_norm = None

        if context_features is not None:
            raise NotImplementedError()

        # Masked linear.
        self.linear = MaskedLinear(
            in_degrees=in_degrees,
            out_features=features,
            autoregressive_features=autoregressive_features,
            random_mask=random_mask,
            is_output=False,
        )
        self.degrees = self.linear.degrees

        # Activation and dropout.
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_probability)

    def forward(self, inputs, context=None):
        if context is not None:
            raise NotImplementedError()

        if self.batch_norm:
            outputs = self.batch_norm(inputs)
        else:
            outputs = inputs
        outputs = self.linear(outputs)
        outputs = self.activation(outputs)
        outputs = self.dropout(outputs)
        return outputs


class MaskedSoftFeedforwardBlock(nn.Module):
    """A feedforward block based on a masked linear module.

    NOTE: In this implementation, the number of output features is taken to be equal to
    the number of input features.
    """

    def __init__(self,
                 in_degrees,
                 autoregressive_features,
                 context_features=None,
                 random_mask=False,
                 activation=torch.cos,
                 dropout_probability=0.,
                 use_batch_norm=False):
        super().__init__()
        features = len(in_degrees)

        # Batch norm.
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(features, eps=1e-3)
        else:
            self.batch_norm = None

        if context_features is not None:
            raise NotImplementedError()

        # Masked linear.
        self.linear = MaskedLinear(
            in_degrees=in_degrees,
            out_features=features,
            autoregressive_features=autoregressive_features,
            random_mask=random_mask,
            is_output=False,
        )
        self.degrees = self.linear.degrees

        # Activation and dropout.
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_probability)

    def forward(self, inputs, context=None):
        if context is not None:
            raise NotImplementedError()

        if self.batch_norm:
            outputs = self.batch_norm(inputs)
        else:
            outputs = inputs
        #outputs = self.linear(outputs)
        outputs = self.activation(outputs)
        return outputs


class MaskedResidualBlock(nn.Module):
    """A residual block containing masked linear modules."""

    def __init__(self,
                 in_degrees,
                 autoregressive_features,
                 context_features=None,
                 random_mask=False,
                 activation=F.relu,
                 dropout_probability=0.,
                 use_batch_norm=False,
                 zero_initialization=True):
        if random_mask:
            raise ValueError('Masked residual block can\'t be used with random masks.')
        super().__init__()
        features = len(in_degrees)

        if context_features is not None:
            self.context_layer = nn.Linear(context_features, features)

        # Batch norm.
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm_layers = nn.ModuleList([
                nn.BatchNorm1d(features, eps=1e-3)
                for _ in range(2)
            ])

        # Masked linear.
        linear_0 = MaskedLinear(
            in_degrees=in_degrees,
            out_features=features,
            autoregressive_features=autoregressive_features,
            random_mask=False,
            is_output=False,
        )
        linear_1 = MaskedLinear(
            in_degrees=linear_0.degrees,
            out_features=features,
            autoregressive_features=autoregressive_features,
            random_mask=False,
            is_output=False,
        )
        self.linear_layers = nn.ModuleList([linear_0, linear_1])
        self.degrees = linear_1.degrees
        if torch.all(self.degrees >= in_degrees).item() != 1:
            raise RuntimeError('In a masked residual block, the output degrees can\'t be'
                               ' less than the corresponding input degrees.')

        # Activation and dropout
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_probability)

        # Initialization.
        if zero_initialization:
            init.uniform_(self.linear_layers[-1].weight, a=-1e-3, b=1e-3)
            init.uniform_(self.linear_layers[-1].bias, a=-1e-3, b=1e-3)

    def forward(self, inputs, context=None):
        temps = inputs
        if self.use_batch_norm:
            temps = self.batch_norm_layers[0](temps)
        temps = self.activation(temps)
        temps = self.linear_layers[0](temps)
        if self.use_batch_norm:
            temps = self.batch_norm_layers[1](temps)
        temps = self.activation(temps)
        temps = self.dropout(temps)
        temps = self.linear_layers[1](temps)
        if context is not None:
            temps = F.glu(
                torch.cat((temps, self.context_layer(context)), dim=1),
                dim=1
            )
        return inputs + temps


class MaskedClampedLinear(nn.Module):
    """A fixed linear module with a masked weight matrix and fixed parameters"""
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self,
                 in_degrees,
                 out_features,
                 autoregressive_features,
                 random_mask,
                 weights, bias_values, bias=True, is_output=False):
        super(MaskedClampedLinear, self).__init__()

        self.in_features = in_degrees
        self.out_features = out_features
        self.register_buffer('weight', torch.Tensor(weights))
        if bias:
            self.register_buffer('bias', torch.Tensor(bias_values))

        mask, degrees = self._get_mask_and_degrees(
            in_degrees=in_degrees,
            out_features=out_features,
            autoregressive_features=autoregressive_features,
            random_mask=random_mask,
            is_output=is_output)
        self.register_buffer('mask', mask)
        self.register_buffer('degrees', degrees)

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

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)
