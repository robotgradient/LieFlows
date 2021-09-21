import numpy as np
import torch
from torch import autograd



def get_jacobian(net, x, output_dims, reshape_flag=True, context=None):
	if x.ndimension() == 1:
		n = 1
	else:
		n = x.size()[0]
	x_m = x.repeat(1, output_dims).view(-1, output_dims)
	if context is not None:
		context_dim = context.shape[1]
		c_m = context.repeat(1, output_dims).view(-1, context_dim)
	else:
		c_m = context


	x_m.requires_grad_(True)
	y_m = net(x_m, context = c_m)
	mask = torch.eye(output_dims).repeat(n, 1).to(x.device)
	# y.backward(mask)
	J = autograd.grad(y_m, x_m, mask, create_graph=True)[0]
	if reshape_flag:
		J = J.reshape(n, output_dims, output_dims)
	return J


## Get Hessian computes the Hessian of a certain fucntion and outputs the value with the shape (n, output_dim, input_dim, input_dim) ##
def get_jacobian_and_hessian(net, x, output_dims, reshape_flag=True, context=None):
	input_dim = x.shape[-1]
	if x.ndimension() == 1:
		n = 1
	else:
		n = x.size()[0]

	x_m = x.repeat(1, output_dims*input_dim).view(-1, output_dims)
	if context is not None:
		context_dim = context.shape[1]
		c_m = context.repeat(1, output_dims*input_dim).view(-1, context_dim)
	else:
		c_m = context

	x_m.requires_grad_(True)
	y_m = net(x_m, context=c_m)

	mask = torch.eye(output_dims).repeat(n*input_dim, 1).to(x.device)

	mask_H = torch.eye(input_dim).repeat_interleave(output_dims, dim=0).repeat(n,1)

	J = autograd.grad(y_m, x_m, mask, create_graph=True)[0]
	H = autograd.grad(J, x_m, mask_H, create_graph=True)[0]
	if reshape_flag:
		H = H.reshape(n, input_dim, output_dims, input_dim)
		H.transpose_(1,2)

		pick = np.arange(0,output_dims,1)
		pick_ext = np.tile(pick,n)
		init_n = np.arange(0,output_dims*input_dim*n, output_dims*input_dim)
		init_n = np.repeat(init_n,output_dims)
		pick = pick_ext + init_n
		J = J[pick,:]
		J = J.reshape(n, output_dims, input_dim)
	return J, H


def normalize_vector(q):
	qn = torch.norm(q, p=2, dim=1).detach()
	q = q.div(qn.repeat((q.shape[1],1)).T)
	return q

if __name__ == '__main__':
	def test_function(x):
		y0 = 20 * x[:, 0]**2 + x[:, 1]**2
		y1 = 5 * x[:, 0]**2 + 3 * x[:, 1]**2
		y2 = 7 * x[:, 0]*x[:, 1] + 90 * x[:, 1]**2

		y = torch.cat((y0[:, None], y1[:, None], y2[:, None]), 1)
		return y

	x = torch.ones(3,2)
	J, H = get_jacobian_and_hessian(test_function, x, 3)
	print(H)