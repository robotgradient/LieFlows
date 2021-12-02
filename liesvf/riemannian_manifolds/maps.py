import torch
from liesvf.riemannian_manifolds.liegroups.torch import SE3, S2_V2, SE2
from liesvf.utils.geometry import invert_H

import math


class SE3Map():
    def __init__(self):
        self.dim = 6
        self.manifold_dim = 16
        self.manifold_shape = [4,4]

    def LogMap(self, H, H_origin):
        H_origin_inv = invert_H(H_origin)
        oH = H_origin_inv @ H

        x_hat = SE3.from_matrix(oH).log()
        if x_hat.dim()==1:
            x_hat = x_hat.unsqueeze(0)
        ## Exclude elements out of the coordinate chart
        x_hat = self._set_outliers(x_hat)
        return x_hat

    def Pullback(self, dx, H_origin):
        A = SE3.from_matrix(H_origin)
        Adj_lw = A.adjoint()
        dX = torch.einsum('qx,bx->bq', Adj_lw, dx)
        ## Set 0 velocity in points out of the coordinate chart
        dX[self.mask, 3:] = torch.zeros_like(dX[self.mask, 3:]) + torch.randn(self.mask.sum(), 3)*0.1
        return dX

    def _set_outliers(self, x):
        r = x[:,3:].pow(2).sum(-1).pow(.5)
        mask_r = r>=math.pi
        x[mask_r, 3:] = torch.zeros_like(x[mask_r, 3:])
        self.mask = mask_r
        return x


class SE2Map():
    def __init__(self):
        self.dim = 3
        self.manifold_dim = 9
        self.manifold_shape = [3,3]

    def LogMap(self, H, H_origin):
        H_origin_inv = invert_H(H_origin)
        oH = H_origin_inv @ H

        x_hat = SE2.from_matrix(oH).log()
        if x_hat.dim() == 1:
            x_hat = x_hat.unsqueeze(0)
        ## Exclude elements out of the coordinate chart
        x_hat = self._set_outliers(x_hat)
        return x_hat

    def Pullback(self, dx, H_origin):
        A = SE2.from_matrix(H_origin)
        Adj_lw = A.adjoint()
        dX = torch.einsum('qx,bx->bq', Adj_lw, dx)
        ## Set 0 velocity in points out of the coordinate chart
        dX[self.mask, -1] = torch.zeros_like(dX[self.mask, -1])
        return dX

    def _set_outliers(self, x):
        r = x[:,-1]
        mask_r = r>=math.pi
        x[mask_r,-1] = torch.zeros_like(x[mask_r,-1])
        self.mask = mask_r
        return x


class S2Map():
    def __init__(self):
        self.dim = 2
        self.manifold_dim = 3
        self.manifold_shape = [1]
        self.s2 = S2_V2()

    def LogMap(self, H, H_origin):
        x = H
        x_origin = H_origin

        self.s2.from_xyz(x=x[:, 0], y=x[:, 1], z=x[:, 2])
        pw = self.s2.log()
        ## Exclude elements out of Coordinate chart
        pw = self._set_outliers(pw)
        return pw

    def Pullback(self, dx, H_origin):
        J = self.s2.jacobian()
        dX = torch.einsum('bqx,bx->bq',J, dx)
        ## Set 0 velocity in excluded points
        dX[self.mask,...] = torch.zeros_like(dX[self.mask,...])
        return dX

    def _set_outliers(self, x):
        r = x.pow(2).sum(-1).pow(.5)
        mask_r = r>=math.pi
        x[mask_r,:] = torch.zeros_like(x[mask_r,:])
        self.mask = mask_r
        return x


