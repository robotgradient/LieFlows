import numpy as np
import pyvista as pv

u = np.linspace(np.pi/2 -0.4 ,np.pi/2, 100)[None, :]
v = np.linspace(-1, 1, 400)[:, None]

### uv to xyz ###
c_x = np.cos(2*u)
c_y = np.sin(2*u)
c_z = np.zeros_like(u)

r_x = np.cos(u)*np.cos(2*u)
r_y = np.cos(u)*np.sin(2*u)
r_z = np.sin(u)

x = c_x + v/2*r_x
y = c_y + v/2*r_y
z = c_z + v/2*r_z

### xyz to uv ###

ur = np.arctan2(y,x)/2

w = np.sqrt(x**2 + y**2) - 1

vr = 2*np.sqrt(w**2 + z**2)

v_sign = np.sign(z)*np.sign(np.sin(ur))
vr = v_sign*vr


print(np.allclose(ur[0,:],u))
print(np.allclose(vr[:,1],v))


P = np.stack([x, y, z], axis=-1)
xyz = np.reshape(P,(-1,3))

N = np.cross(np.gradient(P, axis=1), np.gradient(P, axis=0))
N /= np.sqrt(np.sum(N ** 2, axis=-1))[:, :, None]


PLOT = False
if PLOT:
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    n = 100

    ax.scatter(P[:,:,0], P[:,:,1], P[:,:,2],c = 'r')


    plt.show()


mobius_points = pv.PolyData(xyz)

mobius_points.plot(point_size=30, color='tan')

surf = mobius_points.delaunay_2d(alpha=0.05, offset=.0010)
surf.plot()