import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class RFFN(nn.Module):
	def __init__(self, in_dim, out_dim, nfeat, sigma=0.45):
		super(RFFN, self).__init__()
		self.sigma = np.ones(in_dim) * sigma
		self.coeff = np.random.normal(0.0, 1.0, (nfeat, in_dim))
		self.coeff = self.coeff / self.sigma.reshape(1, len(self.sigma))
		self.offset = 2.0 * np.pi * np.random.rand(1, nfeat)

		self.network = nn.Sequential(
			nn.Linear(in_dim, nfeat, bias=True),
			Gaussian(),
			nn.Linear(nfeat, out_dim, bias=False))
		self.reset_parameters()

	def reset_parameters(self):
		nn.init.zeros_(self.network[-1].weight.data)

	def forward(self, x, context=None):
		return self.network(x)


class LinearClamped(nn.Module):
	'''
    Linear layer with user-specified parameters (not to be learrned!)
    '''

	__constants__ = ['bias', 'in_features', 'out_features']

	def __init__(self, in_features, out_features, weights, bias_values, bias=True):
		super(LinearClamped, self).__init__()
		self.in_features = in_features
		self.out_features = out_features

		self.register_buffer('weight', torch.Tensor(weights))
		if bias:
			self.register_buffer('bias', torch.Tensor(bias_values))

	def forward(self, input, context=None):
		if input.dim() == 1:
			return F.linear(input.view(1, -1), self.weight, self.bias)
		return F.linear(input, self.weight, self.bias)

	def extra_repr(self):
		return 'in_features={}, out_features={}, bias={}'.format(
			self.in_features, self.out_features, self.bias is not None
		)


class Cos(nn.Module):
	"""
    Applies the cosine element-wise function
    """

	def forward(self, inputs, context=None):
		return torch.cos(inputs)


class Gaussian(nn.Module):
	"""
    Applies the cosine element-wise function
    """
	def forward(self, inputs, context=None):
		return torch.exp(-inputs.pow(2))
