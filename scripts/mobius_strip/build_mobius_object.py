import numpy as np
import pyvista as pv

alpha = 0.2
angle = 0.4
n_total = 100
n2_total = 20
n_piece0 = int(n_total/np.pi *0.4)
n_piece1 = n_piece0
n_piece2 = n_total - n_piece1 - n_piece0

## Mobius 1 ##
u = np.linspace(np.pi/2+0.0001, np.pi/2 + angle, n_piece0)[None, :]
v = np.linspace(-1, 1, n2_total)[:, None]

def to_xyz(u, v):
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
    P = np.stack([x, y, z], axis=-1)
    xyz = np.reshape(P,(-1,3))
    return xyz

xyz = to_xyz(u, v)

mobius_points_0 = pv.PolyData(xyz)
mobius_0 = mobius_points_0.delaunay_2d(alpha=alpha, offset=.0010)
#mobius_0.plot()


## Mobius 1 ##
u = np.linspace(np.pi/2-angle, np.pi/2, n_piece1)[None, :]
v = np.linspace(-1, 1, n2_total)[:, None]
xyz = to_xyz(u, v)

mobius_points_1 = pv.PolyData(xyz)
mobius_1 = mobius_points_1.delaunay_2d(alpha=alpha, offset=.0010)
#mobius_1.plot()


## Mobius 2 ##
u = np.linspace(-np.pi/2+angle , np.pi/2-angle, n_piece2)[None, :]
v = np.linspace(-1, 1, n2_total)[:, None]
xyz = to_xyz(u, v)

mobius_points_2 = pv.PolyData(xyz)
mobius_2 = mobius_points_2.delaunay_2d(alpha=alpha, offset=.0010)
#mobius_2.plot()


mobius = mobius_0 + mobius_1 + mobius_2
mobius.save('mobius.vtk')


pv.set_plot_theme("document")
# p = pv.Plotter()
p = pv.Plotter()

p.disable_shadows()
silhouette = dict(
    color='black',
    line_width=6.,
    decimate=0.,
    feature_angle=True,
)
p.add_mesh(mobius, smooth_shading=True, ambient=0.7, opacity=1.)


p.show()


