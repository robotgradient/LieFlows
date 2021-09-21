"""Implementations of permutation-like transforms."""

import torch
from iflow import is_positive_int
import torch.nn as nn

__all__ = ['RandomPermutations']


class Permutations(nn.Module):
    """Permutes inputs on a given dimension using a given permutation."""

    def __init__(self, permutation, dim=1):
        if permutation.ndimension() != 1:
            raise ValueError('Permutation must be a 1D tensor.')
        if not is_positive_int(dim):
            raise ValueError('dim must be a positive integer.')

        super().__init__()
        self._dim = dim
        self.register_buffer('_permutation', permutation)

        I = torch.eye(permutation.shape[0])
        perm_matrix = I.clone()
        for i in range(perm_matrix.shape[0]):
            perm_matrix[i,:] = I[permutation[i],:]
        self.register_buffer('_perm_matrix', perm_matrix)

    @property
    def _inverse_permutation(self):
        return torch.argsort(self._permutation)

    @staticmethod
    def _permute(inputs, permutation, dim):
        if dim >= inputs.ndimension():
            raise ValueError("No dimension {} in inputs.".format(dim))
        if inputs.shape[dim] != len(permutation):
            raise ValueError("Dimension {} in inputs must be of size {}."
                             .format(dim, len(permutation)))
        batch_size = inputs.shape[0]
        outputs = torch.index_select(inputs, dim, permutation)
        logabsdet = torch.zeros(batch_size)
        return outputs, logabsdet

    def forward(self, x, logpx=None, reverse=False, context=None):

        if not reverse:
            y, delta_logp = self._permute(x, self._permutation, self._dim)

        else:
            y, delta_logp = self._permute(x, self._inverse_permutation, self._dim)

        if logpx is None:
            return y
        else:
            return y, logpx

    def backwards(self, x):
        y, _ = self._permute(x, self._inverse_permutation, self._dim)
        return y

    def jacobian(self, x):
        return self._perm_matrix.repeat(x.shape[0],1,1)


class RandomPermutations(Permutations):
    """Permutes using a random, but fixed, permutation. Only works with 1D inputs."""

    def __init__(self, features, dim=1):
        if not is_positive_int(features):
            raise ValueError('Number of features must be a positive integer.')
        super().__init__(torch.randperm(features), dim)
