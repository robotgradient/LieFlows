import torch
from liesvf.dynamic_systems.base_dynamic import DynamicSystem
import torch.nn.functional as F

from liesvf.network_models.invertible_nn import FCNN



class LinearDynamics(DynamicSystem):
    def __init__(self, dim, device=None):
        super().__init__(dim, device)

    def forward(self, x):
        return F.normalize(x)

    def potential(self, x):
        return torch.norm(x, p=2, dim=1)

    def gen_traj(self, x, dt, n_samples):
        trj = x.clone()
        for i in range(n_samples):
            v = self.forward(x)
            x  = x - v*dt
            trj = torch.cat([trj, x], 0)
        return trj


class ScaledLinearDynamics(DynamicSystem):
    def __init__(self, dim, device=None):
        super().__init__(dim, device)

        self.eps = 1e-12
        self.log_vel_scalar = FCNN(dim, 1, 100, act='leaky_relu')
        self.vel_scalar = lambda x: torch.exp(torch.tanh(self.log_vel_scalar(x))) + self.eps
        #self.vel_scalar = lambda x: torch.sigmoid(self.log_vel_scalar(x))*5. + self.eps

        self.log_var = FCNN(dim, dim, 100, act='leaky_relu')

    def forward(self, x):
        sc = self.vel_scalar(x)
        return sc * F.normalize(x)

    def density(self, x):
        sc = self.vel_scalar(x)
        lvar = self.log_var(x) + self.eps
        var = torch.exp(lvar)
        Sigma = torch.diag(var)
        mu = sc * F.normalize(x)
        return mu, Sigma

    def potential(self, x):
        return torch.norm(x, p=2, dim=1)

    def gen_traj(self, x, dt, n_samples):
        trj = x.clone()
        for i in range(n_samples):
            v = self.forward(x)
            x  = x - v*dt
            trj = torch.cat([trj, x], 0)
        return trj


if __name__ == '__main__':
    lin_dyn = LinearDynamics(dim=2)

    x = torch.ones(1,2)
    trj = lin_dyn.gen_traj(x, dt= 0.01, n_samples=100)
    print(trj.shape)