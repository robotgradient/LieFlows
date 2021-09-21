import torch
import torch.nn as nn

class MainManifoldModel():
    def __init__(self, device, bijective_map, dynamics, manifold='S2_models', X_origin = torch.eye(4)):
        self.device = device
        self.bijective_map = bijective_map
        self.dynamics = dynamics
        self.manifold = manifold
        self.H_origin = X_origin
        self.iflow = ManifoldSVF(manifold= manifold, bijective_map=bijective_map, dynamics=dynamics).to(device)

    def get_msvf(self):
        return self.iflow


class ManifoldSVF(nn.Module):
    '''
    Manifold Stable Vector Fields. Our proposed model is composed of three elements: An invertible map to the latent tangent space,
    the latent stable dynamics and the Maps to the manifolds
    '''
    def __init__(self, manifold, bijective_map, dynamics):
        super().__init__()

        self.manifold = manifold
        self.bijective_map = bijective_map
        self.dynamics = dynamics

    def forward(self,x, already_tangent=True):

        if already_tangent:
            x_hat = x

        z_hat, J = self.bijective_map.pushforward(x_hat)
        dz = -self.dynamics(z_hat)
        J_inv = torch.inverse(J)

        dx = torch.bmm(J_inv, dz.unsqueeze(2)).squeeze()
        return dx

    def pushforward(self, x, already_tangent=True):
        if already_tangent:
            x_hat = x
        z_hat, _ = self.bijective_map.pushforward(x_hat)
        return z_hat

    def density(self,x):
        y = self.c2b(x)
        J = self.c2b.jacobian(y)

        z_hat, J_hat = self.flow(y)
        J_hat = torch.matmul(J_hat, J)

        dz_hat, Sigma_z = self.latent_dynamics.density(z_hat)
        dz_hat = -dz_hat

        J_hat_inv = torch.inverse(J_hat)

        dx_hat = torch.bmm(J_hat_inv, dz_hat.unsqueeze(2)).squeeze()
        dx_hat = dx_hat
        if self.scale_vel:
            dx_hat = self.vel_scalar(x)*dx_hat

        return dx_hat

    def generate_trajectory(self, x0, dt=0.001, n_samples=1000):
        y = torch.clone(x0)
        xc0= self.sphere2cube(x0[:,3:])
        y[:,3:] = xc0

        z0, _ = self.flow(y)
        trj_z = self.latent_dynamics.gen_traj(z0, dt=dt, n_samples=n_samples)
        trj_xc = self.flow.backwards(trj_z)


        trj_xc[:,3:]  = self.sphere2cube.backwards(trj_xc[:,3:])
        return trj_xc


