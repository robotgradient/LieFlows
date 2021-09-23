import random

import numpy as np
import torch
import torch.nn as nn

from liesvf.utils import get_jacobian
import torch.nn.functional as F

from liesvf.network_models.invertible_nn.splines import linear_spline, rational_quadratic_spline, unconstrained_rational_quadratic_spline
from liesvf.network_models.invertible_nn.made import MADE, SoftMADE, RbfMADE


DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3

class AutoregressiveTransform(nn.Module):
    """Transforms each input variable with an invertible elementwise transformation.

    The parameters of each invertible elementwise transformation can be functions of previous input
    variables, but they must not depend on the current or any following input variables.

    NOTE: Calculating the inverse transform is D times slower than calculating the
    forward transform, where D is the dimensionality of the input to the transform.
    """
    def __init__(self, autoregressive_net, order):
        super(AutoregressiveTransform, self).__init__()
        self.autoregressive_net = autoregressive_net
        self.order = order
        self.reverse_order = list(np.argsort(order))

    def forward(self, x, context=None):
        #x = x[:,self.order]
        autoregressive_params = self.autoregressive_net(x, context)
        y = self._elementwise_forward(x, autoregressive_params)
        #y = y[:,self.reverse_order]
        return y

    def backwards(self, x, context=None):
        num_inputs = np.prod(x.shape[1:])
        y = torch.zeros_like(x)
        for _ in range(num_inputs):
            autoregressive_params = self.autoregressive_net(y, context)
            y = self._elementwise_inverse(x, autoregressive_params)
        return y

    def jacobian(self, inputs, context=None):
        return get_jacobian(self, inputs, inputs.size(-1), context=context)

    def _output_dim_multiplier(self):
        raise NotImplementedError()

    def _elementwise_forward(self, inputs, autoregressive_params):
        raise NotImplementedError()

    def _elementwise_inverse(self, inputs, autoregressive_params):
        raise NotImplementedError()


class LinearSplineLayer(AutoregressiveTransform):
    def __init__(self,
                 num_bins,
                 features,
                 hidden_features,
                 context_features=None,
                 num_blocks=1,
                 random_mask=False,
                 min_x = -1.1,
                 max_x = 1.1):
        self.min_in = min_x
        self.max_in = max_x

        self.num_bins = num_bins
        self.features = features
        made = SoftMADE(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            output_multiplier=self._output_dim_multiplier(),
            random_mask=random_mask,
        )

        order = list(range(features))
        random.shuffle(order)
        super().__init__(made, order)

    def _output_dim_multiplier(self):
        return self.num_bins

    def _elementwise(self, inputs, autoregressive_params, inverse=False):
        batch_size = inputs.shape[0]

        unnormalized_pdf = autoregressive_params.view(batch_size,
                                                      self.features,
                                                      self._output_dim_multiplier())

        outputs = linear_spline(inputs=inputs, unnormalized_pdf=unnormalized_pdf,
                                           left=self.min_in , right=self.max_in , bottom=self.min_in , top=self.max_in,
                                inverse=inverse)

        return outputs

    def _elementwise_forward(self, inputs, autoregressive_params):
        return self._elementwise(inputs, autoregressive_params)

    def _elementwise_inverse(self, inputs, autoregressive_params):
        return self._elementwise(inputs, autoregressive_params, inverse=True)


class MaskedPiecewiseRationalQuadraticAutoregressiveTransform(AutoregressiveTransform):
    def __init__(self,
                 features,
                 hidden_features,
                 context_features=None,
                 num_bins=10,
                 tails=None,
                 tail_bound=1.,
                 num_blocks=2,
                 use_residual_blocks=True,
                 random_mask=False,
                 activation=F.relu,
                 dropout_probability=0.,
                 use_batch_norm=False,
                 min_bin_width=DEFAULT_MIN_BIN_WIDTH,
                 min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
                 min_derivative=DEFAULT_MIN_DERIVATIVE
                 ):
        self.num_bins = num_bins
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        self.tails = tails
        self.tail_bound = tail_bound

        autoregressive_net = MADE(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            output_multiplier=self._output_dim_multiplier(),
            random_mask=random_mask,
            activation=activation,
        )

        super().__init__(autoregressive_net)

    def _output_dim_multiplier(self):
        if self.tails == 'linear':
            return self.num_bins * 3 - 1
        elif self.tails is None:
            return self.num_bins * 3 + 1
        else:
            raise ValueError

    def _elementwise(self, inputs, autoregressive_params, inverse=False):
        batch_size, features = inputs.shape[0], inputs.shape[1]

        transform_params = autoregressive_params.view(
            batch_size,
            features,
            self._output_dim_multiplier()
        )

        unnormalized_widths = transform_params[...,:self.num_bins]
        unnormalized_heights = transform_params[...,self.num_bins:2*self.num_bins]
        unnormalized_derivatives = transform_params[...,2*self.num_bins:]

        if hasattr(self.autoregressive_net, 'hidden_features'):
            unnormalized_widths /= np.sqrt(self.autoregressive_net.hidden_features)
            unnormalized_heights /= np.sqrt(self.autoregressive_net.hidden_features)

        if self.tails is None:
            spline_fn = rational_quadratic_spline
            spline_kwargs = {}
        elif self.tails == 'linear':
            spline_fn = unconstrained_rational_quadratic_spline
            spline_kwargs = {
                'tails': self.tails,
                'tail_bound': self.tail_bound
            }
        else:
            raise ValueError

        outputs = spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            **spline_kwargs
        )

        return outputs

    def _elementwise_forward(self, inputs, autoregressive_params):
        return self._elementwise(inputs, autoregressive_params)

    def _elementwise_inverse(self, inputs, autoregressive_params):
        return self._elementwise(inputs, autoregressive_params, inverse=True)

