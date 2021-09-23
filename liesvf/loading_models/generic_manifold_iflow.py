import torch
import torch.nn as nn

class MainManifoldModel():
    def __init__(self, device, bijective_map, dynamics, manifold='S2_models', H_origin = torch.eye(4)):
        self.device = device
        self.bijective_map = bijective_map
        self.dynamics = dynamics
        self.manifold = manifold
        self.H_origin = H_origin
        self.iflow = ManifoldSVF(manifold= manifold, bijective_map=bijective_map, dynamics=dynamics, H_origin=H_origin).to(device)

    def get_msvf(self):
        return self.iflow


class ManifoldSVF(nn.Module):
    '''
    Manifold Stable Vector Fields. Our proposed model is composed of three elements: An invertible map to the latent tangent space,
    the latent stable dynamics and the Maps to the manifolds
    '''
    def __init__(self, manifold, bijective_map, dynamics, H_origin=torch.eye(3)):
        super().__init__()

        self.manifold = manifold
        self.bijective_map = bijective_map
        self.dynamics = dynamics
        self.register_buffer('H_origin',H_origin)

    def forward(self,x, already_tangent=True):

        if already_tangent and x.shape[-1]==self.manifold.dim:
            x_hat = x
        else:
            if x.dim()< len(self.manifold.manifold_shape)+1:
                x = x.unsqueeze(0)
            x_hat = self.manifold.LogMap(x, self.H_origin)

        z_hat, J = self.bijective_map.pushforward(x_hat)
        dz = -self.dynamics(z_hat)
        J_inv = torch.inverse(J)
        dx = torch.einsum('bqx, bx->bq',J_inv, dz)
        #dx = -x_hat

        if already_tangent and x.shape[-1]==self.manifold.dim:
            dx = dx.squeeze()
        else:
            dx = self.manifold.Pullback(dx, self.H_origin).squeeze()

        return dx

    def pushforward(self, x, already_tangent=True):
        if already_tangent:
            x_hat = x
        z_hat, _ = self.bijective_map.pushforward(x_hat)
        return z_hat

    def generate_trj(self, x0, dt=0.01, T=1000):
        y = torch.clone(x0)

        trj_out = y[None, ...]
        for t in range(T):
            dx = self.forward(y)
            y1 = y + dx*dt
            trj_out = torch.cat((trj_out, y1[None,...]),0)
            y = y1
        return trj_out


