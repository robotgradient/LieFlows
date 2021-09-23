import torch
from liesvf.riemannian_manifolds.liegroups.torch import SE3, S2_V2
from liesvf.utils.geometry import invert_H


class SE3Map():
    def __init__(self):
        self.dim = 6
        self.manifold_dim = 16
        self.manifold_shape = [2,2]


    def LogMap(self, H, H_origin):
        H_origin_inv = invert_H(H_origin)
        oH = H_origin_inv @ H

        x_hat = SE3.from_matrix(oH).log()
        if x_hat.dim()==1:
            x_hat = x_hat.unsqueeze(0)
        return x_hat

    def Pullback(self, dx, H_origin):
        A = SE3.from_matrix(H_origin)
        Adj_lw = A.adjoint()
        return torch.einsum('qx,bx->bq',Adj_lw, dx)


class SE2Map():
    def __init__(self):
        self.dim = 3
        self.manifold_dim = 9

    def LogMap(self, H, H_origin):
        print("To be code")

    def Pullback(self, dx, H_origin):
        print("To be code")


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
        return pw

    def Pullback(self, dx, H_origin):
        J = self.s2.jacobian()
        dX = torch.einsum('bqx,bx->bq',J, dx)
        return dX

