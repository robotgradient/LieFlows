import numpy as np
import torch
from liesvf.dynamic_systems.base_dynamic import DynamicSystem
from liesvf.network_models.invertible_nn import FCNN, RBF, gaussian


class SecondOrderLinearDynamics(DynamicSystem):
    def __init__(self, dim, device=None):
        super().__init__(dim, device)
        self.dim = dim
        k = 1.
        b = np.sqrt(4*k)
        I = torch.eye(dim)
        K = -k*torch.eye(dim)
        B = -b*torch.eye(dim)
        A = torch.zeros(dim*2,dim*2)
        A[:dim, dim:] = I
        A[dim:, :dim] = K
        A[dim:, dim:] = B
        self.register_buffer('A', A)

    def forward(self, x, p):
        xp = torch.cat((x,p),1)
        dxp =  torch.einsum('nm,bm->bn', self.A, xp)
        return dxp[:, :self.dim], dxp[:, self.dim:]


class ConditionedScaledSecondOrderLinearDynamics(DynamicSystem):
    def __init__(self, dim, context_dim, device=None):
        super().__init__(dim, device)

        self.context_dim = context_dim

        self.eps = 1e-12
        self.log_vel_scalar = FCNN(dim+context_dim, 1, 100, act='leaky_relu')
        self.vel_scalar = lambda x: torch.exp(self.log_vel_scalar(x)) + self.eps

        self.x_des_fn = FCNN(context_dim, dim, 100, act='tanh')

        self.log_var = FCNN(dim, dim, 100, act='leaky_relu')

    def compute_A(self, x, p, context):
        x_context = torch.cat((x,context),1)
        k =  self.vel_scalar(x_context)
        b = torch.sqrt(4*k)
        I = torch.eye(self.dim)
        I = I[None, ...].repeat(x.shape[0], 1, 1)

        K = torch.einsum('b,jk->bjk', -k.squeeze(dim=1), torch.eye(self.dim))
        B = torch.einsum('b,jk->bjk', -b.squeeze(dim=1), torch.eye(self.dim))
        A = torch.zeros(x.shape[0], self.dim*2,self.dim*2)
        A[:, :self.dim, self.dim:] = I
        A[:, self.dim:, :self.dim] = K
        A[:, self.dim:, self.dim:] = B
        return A

    def forward(self, x, p, context = None):
        ## Compute Desired Target Pose ##
        if context is not None:
            xdes = self.x_des_fn(context)
        else:
            xdes = torch.zeros_like(x)

        x = x - xdes
        A = self.compute_A(x, p, context)
        xp = torch.cat((x, p), 1)
        dxp = torch.einsum('bmn,bn->bm', A, xp)
        return dxp[:,:self.dim], dxp[:,self.dim:]

    def get_target(self, context):
        ## Compute Desired Target Pose ##
        xdes = self.x_des_fn(context)
        return xdes


class ConditionedScaledSecondOrderFiberDynamics(DynamicSystem):
    def __init__(self, dim, context_dim, device=None):
        super().__init__(dim, device)

        self.context_dim = context_dim
        self.context_features_dim = 280
        self.context_features_fn = RBF(context_dim, self.context_features_dim, basis_func = gaussian)
        #self.context_features_fn = RFFN(context_dim, self.context_features_dim, nfeat = 100)
        #self.context_features_fn = FCNN(context_dim, self.context_features_dim, 128, act='leaky_relu')


        self.eps = 1e-12
        self.log_vel_scalar = FCNN(self.context_features_dim+dim, 1, 128, act='leaky_relu')
        self.vel_scalar = lambda x: torch.exp(self.log_vel_scalar(x)) + self.eps

        ## Only For Training ##
        self.log_vel_scalar2 = FCNN(self.context_features_dim + dim, 1, 128, act='leaky_relu')
        self.vel_scalar2 = lambda x: torch.exp(self.log_vel_scalar2(x)) + self.eps

        self.x_des_fn  = FCNN(self.context_features_dim, dim, 128, act='tanh')
        self.dx_des_fn = FCNN(self.context_features_dim + dim, dim, 512, act='tanh')

        #self.fiber_dyn = RBF(dim*2+context_dim, dim-1, basis_func = gaussian)#256, act='leaky_relu')
        #self.fiber_dyn.reset_parameters()

        self.fiber_dyn = FCNN(dim*2+self.context_features_dim, dim-1, 512, act='leaky_relu')
        self.fiber_1_dyn = FCNN(dim+self.context_features_dim, dim, 512, act='tanh')

        self.log_var = FCNN(dim, dim, 128, act='leaky_relu')

    def compute_A(self, x, p, context):
        x_context = torch.cat((x,context),1)
        k =  self.vel_scalar(x_context)

        b = torch.sqrt(4*k)
        I = torch.eye(self.dim)
        I = I[None, ...].repeat(x.shape[0], 1, 1)

        K = torch.einsum('b,jk->bjk', -k.squeeze(dim=1), torch.eye(self.dim))
        B = torch.einsum('b,jk->bjk', -b.squeeze(dim=1), torch.eye(self.dim))
        A = torch.zeros(x.shape[0], self.dim*2,self.dim*2)
        A[:, :self.dim, self.dim:] = I
        A[:, self.dim:, :self.dim] = K
        A[:, self.dim:, self.dim:] = B
        return A

    def first_order(self, x, context=None):
        ## Compute Desired Target Pose ##
        if context is not None:
            xdes = self.x_des_fn(context)
        else:
            xdes = torch.zeros_like(x)
        x = x - xdes

        ## Compute Modified goal ##
        fiber_input = torch.cat((x, context), 1)
        out = self.fiber_1_dyn(fiber_input)
        dx_fiber = out[:,:-1]
        sc = torch.tanh(torch.exp(out[:,-1])) + self.eps

        er_x = torch.tanh(torch.norm(x[:,:1], dim=1))

        dx_base = x[:,:1]
        dx_base = torch.einsum('b, bn->bn', sc*(1-er_x), dx_base)

        dx_fiber = torch.einsum('b, bn->bn',(1-er_x), x[:,1:]) + torch.einsum('b, bn->bn',er_x, dx_fiber)

        dx = torch.cat((dx_base, dx_fiber),1)
        return dx#torch.einsum('b, bn->bn', sc, normalize_vector(dx))

    def velocity_prediction(self, x, er_x, context=None):
        input = torch.cat((x, context), 1)
        ## Compute Modified goal ##
        fiber_input = torch.cat((x, context), 1)
        out = self.fiber_1_dyn(fiber_input)
        dx_fiber = out[:, :-1]
        sc = torch.exp(out[:, -1]) + self.eps

        dx_base = x[:, :1]
        dx_base = torch.einsum('b, bn->bn', sc * (1 - er_x), dx_base)
        dx_fiber = torch.einsum('b, bn->bn', (1 - er_x), x[:, 1:]) + torch.einsum('b, bn->bn', er_x, dx_fiber)
        dx = torch.cat((dx_base, dx_fiber), 1)
        return dx

    def forward(self, x, p, context = None, additional_info=False):
        ## Compute Context Features ##
        context = self.context_features_fn(context)

        ## Compute Desired Target Pose ##
        if context is not None:
            xdes = self.x_des_fn(context)
        else:
            xdes = torch.zeros_like(x)

        delta_x = x - xdes
        A = self.compute_A(delta_x, p, context)
        xp = torch.cat((delta_x, p), 1)
        dxp = torch.einsum('bmn,bn->bm', A, xp)

        ## Compute First Order nonlinearity ##
        dx_des = self.first_order(x, context)

        # delta_p = p-dx_des
        # delta_p[:,:1] = p[:,:1]
        # delta_p = p
        # delta_x = dx_des
        # delta_x[:,1:] = dx_des[:,1:] + delta_x[:,1:]


        xp2 = torch.cat((dx_des, p), 1)
        ddx_PD = torch.einsum('bmn,bn->bm', A, xp2)[:,self.dim:]


        fiber_input = torch.cat((x,p,context),1)
        ddx_forward = 2*torch.tanh(self.fiber_dyn(fiber_input))
        er_x = torch.tanh(torch.norm(delta_x[:,:1], dim=1))

        # dx_des_pred = self.velocity_prediction(delta_x, context=context)
        # gain = (p - dx_des_pred)/5

        # ddx_PD += - gain

        # ddx_fiber = torch.einsum('b, bn->bn',(1 - er_x),  dxp[:,self.dim:])+ torch.einsum('b, bn->bn',er_x, ddx_PD)
        # #ddx_base = dxp[:, self.dim:self.dim + 1]
        # #ddx = torch.cat((ddx_base, ddx_fiber),1)
        # ddx = ddx_fiber
        ddx_fiber = ddx_PD[:,1:] + torch.einsum('b, bn->bn',er_x, ddx_forward)
        ddx_base  = ddx_PD[:,:1]
        ddx = torch.cat((ddx_base, ddx_fiber), 1)


        if additional_info:
            return p, ddx, [1]
        else:
            return p, ddx

    def train(self, x, p, context = None):
        dx, dp = self.forward(x, p, context)

        context = self.context_features_fn(context)

        xdes = self.x_des_fn(context)
        x = x - xdes

        x_context = torch.cat((x,context),1)
        k =  self.vel_scalar2(x_context)
        dx_pred = -torch.einsum('bm,bn->bn',k,x)
        return dx, dp, dx_pred

    def forward2(self, x, p, context = None, additional_info=False):
        ## Compute Context Features ##
        context = self.context_features_fn(context)

        ## Compute Desired Target Pose ##
        if context is not None:
            xdes = self.x_des_fn(context)
        else:
            xdes = torch.zeros_like(x)
        delta_x = x - xdes

        ## dx prediction ##
        er_x = torch.tanh(torch.norm(x[:, :1], dim=1))
        dx_des_pred = self.velocity_prediction(delta_x, er_x=er_x, context=context)

        ddx = -(p - dx_des_pred) - delta_x
        if additional_info:
            return p, ddx, [dx_des_pred]
        else:
            return p, ddx


    def get_target(self, context):
        ## Compute Context Features ##
        context = self.context_features_fn(context)
        xdes = self.x_des_fn(context)
        return xdes


if __name__ == '__main__':
    lin_dyn = SecondOrderLinearDynamics(dim=2)

    x = torch.ones(1,2)
    trj = lin_dyn.gen_traj(x, dt= 0.01, n_samples=100)
    print(trj.shape)