import numpy as np

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
        self.x = np.cos(self.v)*np.sin(self.u)
        self.y = np.sin(self.v)*np.sin(self.u)
        self.z = np.cos(self.u)

        # self.x = np.cos(self.v)*np.sin(self.u)
        # self.y = np.sin(self.v)
        # self.z = np.cos(self.u)*np.cos(self.v)

        return np.array([self.x, self.y, self.z]).T

    def log(self):
        # d = np.sqrt(self.x**2 + self.y**2)
        # self.u = np.arctan2(d,self.z)
        # self.v = np.arctan2(self.y,self.x)

        d = np.sqrt(self.x**2 + self.z**2)
        self.u = np.arctan2(self.x,self.z)
        self.v = np.arctan2(self.y,d)

        return np.array([self.u, self.v]).T

    def jacobian(self,u,v):
        su = np.sin(u)
        sv = np.sin(v)
        cu = np.cos(u)
        cv = np.cos(v)

        #v theta u phi
        # dxdr = cv*su
        # dydr = sv*su
        # dzdr = cu

        # dxdu = cu*cv
        # dydu = cu*sv
        # dzdu = -su
        #
        # dxdv = -su*sv
        # dydv = su*cv
        # dzdv = 0.

        dxdu = cv*cu
        dydu = 0
        dzdu = -su*cv

        dxdv = -su*sv
        dydv = cv
        dzdv = -cu*sv



        J = np.zeros((u.shape[0],3,2))
        # J[:, 0, 0] = dxdr
        # J[:, 1, 0] = dydr
        # J[:, 2, 0] = dzdr

        J[:,0,0] = dxdu
        J[:,1,0] = dydu
        J[:,2,0] = dzdu

        J[:,0,1] = dxdv
        J[:,1,1] = dydv
        J[:,2,1] = dzdv
        return J

    def normal_jac(self,x,y,z):
        J = np.array([2*x,2*y,2*z]).T
        return J

    def tangent_jac(self,x,y,z):
        d = np.sqrt(x**2+y**2) #+ 10**(-16)

        dtdx = np.divide(x*z, d, out=np.zeros_like(x), where=d>0.001)
        dtdy = np.divide(y*z, d, out=np.zeros_like(x), where=d>0.001)
        dtdz = -d

        dpdx = np.divide(-y, d**2, out=np.zeros_like(x), where=d>0.001)
        dpdy = np.divide(x, d**2, out=np.zeros_like(x), where=d>0.001)
        dpdz = np.zeros(z.shape[0])

        Jt = np.array([dtdx, dtdy, dtdz]).T
        Jp = np.array([dpdx, dpdy, dpdz]).T
        J = np.concatenate((Jt[...,None],Jp[...,None]),axis=2)
        return J



class S2_angle():
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
        self.x = np.cos(self.v)*np.sin(self.u)
        self.y = np.sin(self.v)*np.sin(self.u)
        self.z = np.cos(self.u)

        return np.array([self.x, self.y, self.z]).T

    def log(self):
        d = np.sqrt(self.x**2 + self.y**2)
        self.u = np.arctan2(d,self.z)
        self.v = np.arctan2(self.y,self.x)

        return np.array([self.u, self.v]).T

    def jacobian(self,u,v):
        su = np.sin(u)
        sv = np.sin(v)
        cu = np.cos(u)
        cv = np.cos(v)

        dxdu = cu*cv
        dydu = cu*sv
        dzdu = -su

        dxdv = -su*sv
        dydv = su*cv
        dzdv = 0.

        J = np.zeros((u.shape[0],3,2))

        J[:,0,0] = dxdu
        J[:,1,0] = dydu
        J[:,2,0] = dzdu

        J[:,0,1] = dxdv
        J[:,1,1] = dydv
        J[:,2,1] = dzdv
        return J

    def polar_to_cartesian(self,p, theta):
        x = p*np.cos(theta)
        y = p*np.sin(theta)
        return np.array([x, y]).T

    def cartesian_to_polar(self, x, y):
        p = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y,x)
        return np.array([p, theta]).T

    def jac_p2x(self, x, y):
        p = np.sqrt(x**2 + y**2)
        dpdx = x/p
        dpdy = y/p
        dtdx = -y/p**2
        dtdy = x/p**2

        J = np.zeros((x.shape[0], 2, 2))
        J[:, 0, 0] = dpdx
        J[:, 1, 0] = dtdx
        J[:, 0, 1] = dpdy
        J[:, 1, 1] = dtdy
        return J







