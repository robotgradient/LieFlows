"""Implementations of permutation-like transforms."""

import torch
from liesvf.utils.typechecks import is_positive_int
import torch.nn as nn

__all__ = ['RandomPermutations', 'Permutations']


class Permutations(nn.Module):
    """Permutes inputs on a given dimension using a given permutation."""

    def __init__(self, permutation, dim=1):
        if permutation.ndimension() != 1:
            raise ValueError('Permutation must be a 1D tensor.')
        if not is_positive_int(dim):
            raise ValueError('dim must be a positive integer.')

        super(Permutations, self).__init__()
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

        outputs = torch.index_select(inputs, dim, permutation)
        return outputs

    def forward(self, x, context=None):
        y = self._permute(x, self._permutation, self._dim)
        return y


    def jacobian(self, x, context=None):
        return self._perm_matrix.repeat(x.shape[0],1,1)


class RandomPermutations(Permutations):
    """Permutes using a random, but fixed, permutation. Only works with 1D inputs."""

    def __init__(self, features, dim=1):
        if not is_positive_int(features):
            raise ValueError('Number of features must be a positive integer.')
        super().__init__(torch.randperm(features), dim)


if __name__ == '__main__':
    perm_i = torch.Tensor([1,0]).to(torch.long)
    net = Permutations(perm_i, 1)
    x = torch.Tensor([[1,2],[4,5]])
    y = net(x)
    print(x)
    print(y)