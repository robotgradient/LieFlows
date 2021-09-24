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
        x = x[:,self.order]
        autoregressive_params = self.autoregressive_net(x, context)
        y = self._elementwise_forward(x, autoregressive_params)
        y = y[:,self.reverse_order]
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
                 max_x = 1.1, order = None):
        if order is None:
            order = list(range(features))

        self.min_in = min_x
        self.max_in = max_x

        self.num_bins = num_bins
        self.features = features
        made = SoftMADE(
            features=features,
            hidden_features=hidden_features,
            output_multiplier=self._output_dim_multiplier(),
            random_mask=random_mask,
        )

        super().__init__(made, order)

    def _output_dim_multiplier(self):
        return self.num_bins

    def _elementwise(self, inputs, autoregressive_params, inverse=False):
        batch_size = inputs.shape[0]

        # unnormalized_pdf = autoregressive_params.view(batch_size,
        #                                               self.features,
        #                                               self._output_dim_multiplier())
        #
        unnormalized_pdf = autoregressive_params.view(batch_size,
                                                      self._output_dim_multiplier(),
                                                      self.features).transpose(-1,-2)

        outputs = linear_spline(inputs=inputs, unnormalized_pdf=unnormalized_pdf,
                                           left=self.min_in , right=self.max_in , bottom=self.min_in , top=self.max_in,
                                inverse=inverse)

        return outputs

    def _elementwise_forward(self, inputs, autoregressive_params):
        return self._elementwise(inputs, autoregressive_params)

    def _elementwise_inverse(self, inputs, autoregressive_params):
        return self._elementwise(inputs, autoregressive_params, inverse=True)