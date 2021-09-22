import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from liesvf.utils import get_jacobian

class CouplingLayer(nn.Module):
	""" An implementation of a coupling layer
	from RealNVP (https://arxiv.org/abs/1605.08803).
	"""

	def __init__(self, num_inputs, num_hidden, mask,
				 base_network='rffn', s_act='elu', t_act='elu', sigma=0.45):
		super(CouplingLayer, self).__init__()

		self.num_inputs = num_inputs
		#self.mask = mask
		self.register_buffer('mask', mask)

		if base_network == 'fcnn':
			self.scale_net = FCNN(in_dim=num_inputs, out_dim=num_inputs, hidden_dim=num_hidden, act=s_act)
			self.translate_net = FCNN(in_dim=num_inputs, out_dim=num_inputs, hidden_dim=num_hidden, act=t_act)
			print('Using neural network initialized with identity map!')

			nn.init.zeros_(self.translate_net.network[-1].weight.data)
			nn.init.zeros_(self.translate_net.network[-1].bias.data)

			nn.init.zeros_(self.scale_net.network[-1].weight.data)
			nn.init.zeros_(self.scale_net.network[-1].bias.data)

		elif base_network == 'rffn':
			print('Using random fouier feature with bandwidth = {}.'.format(sigma))
			self.scale_net = RFFN(in_dim=num_inputs, out_dim=num_inputs, nfeat=num_hidden, sigma=sigma)
			self.translate_net = RFFN(in_dim=num_inputs, out_dim=num_inputs, nfeat=num_hidden, sigma=sigma)

			print('Initializing coupling layers as identity!')
			nn.init.zeros_(self.translate_net.network[-1].weight.data)
			nn.init.zeros_(self.scale_net.network[-1].weight.data)
		else:
			raise TypeError('The network type has not been defined')

	def forward(self, inputs, mode='direct', context=None):
		mask = self.mask

		masked_inputs = inputs * mask
		# masked_inputs.requires_grad_(True)

		log_s = self.scale_net(masked_inputs) * (1 - mask)
		t = self.translate_net(masked_inputs) * (1 - mask)

		if mode == 'direct':
			s = torch.exp(log_s)
			return inputs * s + t
		else:
			s = torch.exp(-log_s)
			return (inputs - t) * s

	def jacobian(self, inputs, context=None):
		return get_jacobian(self, inputs, inputs.size(-1))


class RFFN(nn.Module):
    def __init__(self, in_dim, out_dim, nfeat, sigma=0.45):
        super(RFFN, self).__init__()
        self.sigma = np.ones(in_dim) * sigma
        self.coeff = np.random.normal(0.0, 1.0, (nfeat, in_dim))
        self.coeff = self.coeff / self.sigma.reshape(1, len(self.sigma))
        self.offset = 2.0 * np.pi * np.random.rand(1, nfeat)

        self.network = nn.Sequential(
            LinearClamped(in_dim, nfeat, self.coeff, self.offset),
            Cos(),
            nn.Linear(nfeat, out_dim, bias=False))

    def forward(self, x, context=None):
        return self.network(x)


class FCNN(nn.Module):
	'''
	2-layer fully connected neural network
	'''

	def __init__(self, in_dim, out_dim, hidden_dim, act='tanh'):
		super(FCNN, self).__init__()
		activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'leaky_relu': nn.LeakyReLU,
					   'elu': nn.ELU, 'prelu': nn.PReLU, 'softplus': nn.Softplus}

		act_func = activations[act]
		self.network = nn.Sequential(
			nn.Linear(in_dim, hidden_dim), act_func(),
			nn.Linear(hidden_dim, hidden_dim), act_func(),
			nn.Linear(hidden_dim, out_dim)
		)

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


