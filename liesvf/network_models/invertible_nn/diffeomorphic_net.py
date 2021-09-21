import torch
import torch.nn as nn


class DiffeomorphicNet(nn.Sequential):
    def __init__(self,modules_list,  dim=2):
        self.num_dims = dim
        super(DiffeomorphicNet, self).__init__(*modules_list)

    def jacobian(self, inputs, mode='direct', context=None):
        '''
        Finds the product of all jacobians
        '''
        batch_size = inputs.size(0)
        J = torch.eye(self.num_dims, device=inputs.device).unsqueeze(0).repeat(batch_size, 1, 1)

        if mode == 'direct':
            for module in self._modules.values():
                J_module = module.jacobian(inputs, context)
                J = torch.matmul(J_module, J)
        else:
            for module in reversed(self._modules.values()):
                J_module = module.jacobian(inputs, context)
                J = torch.matmul(J_module, J)
        return J

    def forward(self, inputs, mode='direct', jacobian=True, context=None):
        """ Performs a forward or backward pass for flow modules.
        Args:
            inputs: a tuple of inputs and logdets
            mode: to run direct computation or inverse
        """
        assert mode in ['direct', 'inverse']
        batch_size = inputs.size(0)
        J = torch.eye(self.num_dims, device=inputs.device).unsqueeze(0).repeat(batch_size, 1, 1)

        if mode == 'direct' and jacobian==True:
            for module in self._modules.values():
                J_module = module.jacobian(inputs, context)
                J = torch.matmul(J_module, J)
                inputs = module(inputs, context)
                return inputs, J
        elif jacobian==True and mode != 'direct':
            for module in reversed(self._modules.values()):
                J_module = module.jacobian(inputs, context)
                J = torch.matmul(J_module, J)
                inputs = module(inputs, context)
                return inputs, J
        elif jacobian==False and mode == 'direct':
            for module in self._modules.values():
                inputs = module(inputs, context)
                return inputs
        else:
            for module in reversed(self._modules.values()):
                inputs = module(inputs, context)
                return inputs

    def backwards(self, inputs, context):
        for module in reversed(self._modules.values()):
            inputs = module.backwards(inputs, context)
        return inputs
