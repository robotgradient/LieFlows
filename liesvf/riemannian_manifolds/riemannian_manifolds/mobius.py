import numpy as np

class Mobius():
    def __init__(self):
        dim = 2
        dof = 3
        ## Generate Points ##
        s = np.linspace(0.0, 4 * np.pi, 100)
        t = np.linspace(-1.0, 1.0, 50)


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
        u = np.copy(self.u)
        v = np.copy(self.v)

        u = np.arctan2(np.sin(2*u),np.cos(2*u))/2
        _u = np.arctan2(np.sin(self.u),np.cos(self.u))
        self.loop = np.abs(_u)>np.pi/2

        c_x = np.cos(2 * u)
        c_y = np.sin(2 * u)
        c_z = np.zeros_like(u)

        r_x = np.cos(u) * np.cos(2 * u)
        r_y = np.cos(u) * np.sin(2 * u)
        r_z = np.sin(u)

        v = np.sign(np.cos(_u))*v

        x = c_x + v/2 * r_x
        y = c_y + v/2 * r_y
        z = c_z + v/2 * r_z

        self.x = x
        self.y = y
        self.z = z

        return np.array([self.x, self.y, self.z]).T

    def log(self):
        x = self.x
        y = self.y
        z = self.z

        u = np.arctan2(y, x) / 2

        # d_xy = np.sqrt(x ** 2 + y ** 2)
        # w = d_xy - 1
        # u = np.arctan2(z, w)

        w = np.sqrt(x ** 2 + y ** 2) - 1

        v = 2*np.sqrt(w ** 2 + z ** 2)

        v_sign = np.sign(z) * np.sign(np.sin(u))
        v = v_sign* v

        self.u = u
        self.v = v
        return np.array([self.u, self.v]).T

    def inv_jacobian(self,u,v):
        _a = (1 + 0.5 * v * np.cos(u))
        _dadu = -1/2*v*np.sin(u)
        _dadv = 0.5 * np.cos(u)

        _b = v/2
        _dbdv = 1/2

        c2u = np.cos(2*u)
        s2u = np.sin(2*u)

        r_z = np.sin(u)
        z = v/2 * r_z

        dxdu = _dadu*c2u - _a*2*s2u
        dydu = _dadu*s2u + _a*2*c2u
        dzdu = _b*2*np.cos(u)

        dxdv = _dadv*c2u
        dydv = _dadv*s2u
        dzdv = _dbdv*np.sin(u)

        J = np.zeros((u.shape[0],3,2))

        J[:,0,0] = dxdu
        J[:,1,0] = dydu
        J[:,2,0] = dzdu

        J[:,0,1] = dxdv
        J[:,1,1] = dydv
        J[:,2,1] = dzdv
        return J

    def jacobian(self, x, y, z):

        d_xy = np.sqrt(x**2 + y**2)
        w = np.sqrt(x**2 + y**2)-1
        # u = np.arctan2(z,w)

        u = np.arctan2(y, x) / 2
        v_sign = np.sign(z) * np.sign(np.sin(u))

        d = x **2 + y **2
        sqrt_d = np.sqrt(d)
        den = np.sqrt((sqrt_d-1)**2 + z**2)

        # dudx = -z/(w**2+z**2) * x/d_xy
        # dudy = -z/(w**2+z**2) * y/d_xy
        # dudz = w/(w**2+z**2)

        dudx = 2*x/d
        dudy = -2*y/d
        dudz = w/(w**2+z**2)


        dvdx = v_sign*2*(np.sqrt(2)*x*(sqrt_d-1))/(sqrt_d*den)
        dvdy = v_sign*2*(np.sqrt(2)*y*(sqrt_d-1))/(sqrt_d*den)
        dvdz = v_sign*2*(np.sqrt(2)*z)/den

        J = np.zeros((x.shape[0], 2, 3))

        J[:, 0, 0] = dudx
        J[:, 0, 1] = dudy
        J[:, 0, 2] = dudz

        J[:, 1, 0] = dvdx
        J[:, 1, 1] = dvdy
        J[:, 1, 2] = dvdz
        return J


