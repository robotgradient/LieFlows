"""Implementation of MADE."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .masked_layers import _get_input_degrees
from .masked_layers import MaskedLinear, MaskedFeedforwardBlock, MaskedResidualBlock, MaskedSoftFeedforwardBlock, MaskedClampedLinear


import numpy as np

class MADE(nn.Module):
    """Implementation of MADE.

    It can use either feedforward blocks or residual blocks (default is residual).
    Optionally, it can use batch norm or dropout within blocks (default is no).
    """

    def __init__(self,
                 features,
                 hidden_features,
                 context_features=None,
                 num_blocks=2,
                 output_multiplier=1,
                 use_residual_blocks=True,
                 random_mask=False,
                 activation=F.relu,
                 dropout_probability=0.,
                 use_batch_norm=False):
        if use_residual_blocks and random_mask:
            raise ValueError('Residual blocks can\'t be used with random masks.')
        super().__init__()

        # Initial layer.
        self.initial_layer = MaskedLinear(
            in_degrees=_get_input_degrees(features),
            out_features=hidden_features,
            autoregressive_features=features,
            random_mask=random_mask,
            is_output=False
        )

        if context_features is not None:
            self.context_layer = nn.Linear(context_features, hidden_features)

        # Residual blocks.
        blocks = []
        if use_residual_blocks:
            block_constructor = MaskedResidualBlock
        else:
            block_constructor = MaskedFeedforwardBlock

        prev_out_degrees = self.initial_layer.degrees
        for _ in range(num_blocks):
            blocks.append(
                block_constructor(
                    in_degrees=prev_out_degrees,
                    autoregressive_features=features,
                    context_features=context_features,
                    random_mask=random_mask,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=use_batch_norm,
                )
            )
            prev_out_degrees = blocks[-1].degrees
        self.blocks = nn.ModuleList(blocks)

        # Final layer.
        self.final_layer = MaskedLinear(
            in_degrees=prev_out_degrees,
            out_features=features * output_multiplier,
            autoregressive_features=features,
            random_mask=random_mask,
            is_output=True
        )

    def forward(self, inputs, context=None):
        outputs = self.initial_layer(inputs)
        if context is not None:
            outputs += self.context_layer(context)
        for block in self.blocks:
            outputs = block(outputs, context)
        outputs = self.final_layer(outputs)
        return outputs


class SoftMADE(nn.Module):
    """Implementation of MADE.

    It can use either feedforward blocks or residual blocks (default is residual).
    Optionally, it can use batch norm or dropout within blocks (default is no).
    """

    def __init__(self,
                 features,
                 hidden_features,
                 output_multiplier=1,
                 random_mask=False):
        super().__init__()

        ## Initial layer ##
        self.initial_layer = MaskedLinear(
            in_degrees=_get_input_degrees(features),
            out_features=hidden_features,
            autoregressive_features=features,
            random_mask=random_mask,
            bias=True,
            is_output=False)


        # self.initial_layer = nn.Linear(features, hidden_features)
        # self.final_layer = nn.Linear(hidden_features, features * output_multiplier)

        # Final layer ##
        prev_out_degrees = self.initial_layer.degrees
        self.final_layer = MaskedLinear(
            in_degrees=prev_out_degrees,
            out_features=features * output_multiplier,
            autoregressive_features=features,
            random_mask=random_mask,
            is_output=True,
            bias = True
        )

        nn.init.zeros_(self.initial_layer.weight.data)
        nn.init.zeros_(self.final_layer.weight.data)

    def forward(self, inputs, context=None):
        outputs = self.initial_layer(inputs)

        outputs = torch.softmax(-outputs**2, dim=1)
        #outputs = torch.exp(-outputs**2)

        outputs = self.final_layer(outputs)
        return outputs

