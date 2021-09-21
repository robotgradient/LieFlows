import numpy as np
import torch


class S2():
    def __init__(self):
        """Homogeneous transformation matrix in :math:`S(2)`"""
        dim = 2
        dof = 3
        self.u = None
        self.v = None

        self.x = None
        self.y = None
        self.z = None

    def from_tangent(self, u, v):
        self.u = u
        self.v = v

    def from_xyz(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def exp(self):
        self.x = torch.cos(self.v)*torch.sin(self.u)
        self.y = torch.sin(self.v)*torch.sin(self.u)
        self.z = torch.cos(self.u)

        return torch.Tensor([self.x, self.y, self.z]).T

    def log(self):
        d = torch.sqrt(self.x**2 + self.y**2)
        self.u = torch.atan2(d,self.z)
        self.v = torch.atan2(self.y,self.x)

        return torch.cat((self.u[:,None], self.v[:,None]),1)

    def log_2(self):
        uv = self.log()
        return self.polar_to_cartesian(uv[:,0], uv[:,1])

    def jac_xyz2uv(self):
        r2 = self.x**2 + self.y**2 + self.z**2
        d = torch.sqrt(self.x**2 + self.y**2)

        dudx = self.x*self.z/(d*r2)
        dudy = self.y*self.z/(d*r2)
        dudz = d/r2

        dvdx = -self.y/d**2
        dvdy = self.x/d**2
        dvdz = torch.zeros_like(self.z)

        J = torch.zeros((self.x.shape[0], 2, 3)).to(self.x)
        J[:,0,0] = dudx
        J[:,0,1] = dudy
        J[:,0,2] = dudz
        J[:,1,0] = dvdx
        J[:,1,1] = dvdy
        J[:,1,2] = dvdz
        return J

    def jac_uv2xy(self):
        dxdu = torch.cos(self.v)
        dxdv = -self.u*torch.sin(self.v)

        dydu = torch.sin(self.v)
        dydv = self.u*torch.cos(self.v)

        J = torch.zeros((self.u.shape[0], 2, 2)).to(self.x)
        J[:, 0, 0] = dxdu
        J[:, 0, 1] = dxdv
        J[:, 1, 0] = dydu
        J[:, 1, 1] = dydv
        return J


    def jacobian(self,u,v):
        su = torch.sin(u)
        sv = torch.sin(v)
        cu = torch.cos(u)
        cv = torch.cos(v)

        dxdu = cu*cv
        dydu = cu*sv
        dzdu = -su

        dxdv = -su*sv
        dydv = su*cv
        dzdv = 0.

        J = torch.zeros((u.shape[0],3,2))

        J[:,0,0] = dxdu
        J[:,1,0] = dydu
        J[:,2,0] = dzdu

        J[:,0,1] = dxdv
        J[:,1,1] = dydv
        J[:,2,1] = dzdv
        return J

    def polar_to_cartesian(self,p, theta):
        x = p*torch.cos(theta)
        y = p*torch.sin(theta)

        return torch.cat((x[:,None], y[:,None]),1)


    def cartesian_to_polar(self, x, y):
        p = torch.sqrt(x**2 + y**2)
        theta = torch.arctan2(y,x)
        return torch.Tensor([p, theta]).T

    def jac_p2x(self, x, y):
        p = torch.sqrt(x**2 + y**2)
        dpdx = x/p
        dpdy = y/p
        dtdx = -y/p**2
        dtdy = x/p**2

        J = torch.zeros((x.shape[0], 2, 2))
        J[:, 0, 0] = dpdx
        J[:, 1, 0] = dtdx
        J[:, 0, 1] = dpdy
        J[:, 1, 1] = dtdy
        return J







