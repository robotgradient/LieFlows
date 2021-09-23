import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from liesvf.riemannian_manifolds.riemannian_manifolds import Mobius

## Load Mobius Dataset ##
mobius = pv.PolyData('mobius.vtk')
xyz = np.array(mobius.points)

## Test Mobius Map works well xyz -> uv -> xyz
mobius_map = Mobius()
mobius_map.from_xyz(xyz[:,0], xyz[:,1], xyz[:,2])
uv = mobius_map.log()

mobius_map.from_tangent(uv[:,0], uv[:,1])
xyz_r = mobius_map.exp()
print(np.allclose(xyz, xyz_r))


## Create simple stable dynamics
def svf(x):
    dx = -10.1*x
    #dx[:,0] = np.ones_like(dx[:,0])

    #dx[:,1] = np.zeros_like(dx[:,1])
    norm = np.sqrt(np.sum(dx**2,axis=1))

    thrs = 0.05
    norm_thrs = norm>thrs
    norm = norm_thrs*norm/thrs + (1- norm_thrs)*np.ones_like(norm)
    norm = np.tile(norm,(2,1)).T
    return dx/norm

n_samples = uv.shape[0]
uv_2pi = np.copy(uv)

uv_2pi[:, 0] = uv_2pi[:,0] + np.pi
uv_2pi[:,0] = np.arctan2(np.sin(uv_2pi[:, 0]),np.cos(uv_2pi[:, 0]))



uv_ext = np.concatenate((uv, uv_2pi), axis=0)
uv_ext_J = np.concatenate((uv, uv), axis=0)


duv = svf(uv_ext)
Jinv = mobius_map.inv_jacobian(uv_ext_J[:,0], uv_ext_J[:,1])
dxyz = np.einsum('bnm,bm->bn', Jinv, duv)


PLOT = True
if PLOT:
    ##########################################
    pv.set_plot_theme("document")
    p = pv.Plotter()

    p.disable_shadows()
    silhouette = dict(
        color='black',
        line_width=6.,
        decimate=None,
        feature_angle=True,
    )
    p.add_mesh(mobius, color='white', smooth_shading=True, silhouette=silhouette,  ambient=0.7, opacity=1.)

    # Compute the normals in-place and use them to warp the globe
    n_points = mobius.points.shape[0]
    mobius.compute_normals(inplace=True)  # this activates the normals as well
    mobius.vectors = 0.1*np.array(mobius.point_normals)
    ones = np.ones(mobius.points.shape[0])


    mobius2 = mobius.warp_by_vector(factor= .3)
    mobius2.vectors = dxyz[:n_points,:]
    p.add_mesh(mobius2.arrows,color='black', lighting=False,)

    mobius3 = mobius.warp_by_vector(factor= -.3)
    mobius3.vectors = dxyz[n_points:,:]
    p.add_mesh(mobius3.arrows,color='black', lighting=False,)

    p.show()



x_lat = np.array([[-np.pi/2-0.1, 0.8]])
trj = np.copy(x_lat)

T = 1000
dt = 0.1
for t in range(T):
    v = svf(x_lat)
    x1 = x_lat + v*dt
    trj = np.concatenate((trj,x1),0)
    x_lat = x1


plt.plot(trj[:,0],trj[:,1])
plt.show()



mobius_map.from_tangent(trj[:,0], trj[:,1])
trj_xyz = mobius_map.exp()


## Plot Trajectories as a ball ##
def point_object_from_points(points):
    """Given an array of points, make a line set"""
    poly = pv.PolyData()
    poly.points = points
    cells = np.full((len(points)-1, 3), 2, dtype=np.int_)
    cells[:, 1] = np.arange(0, len(points)-1, dtype=np.int_)
    cells[:, 2] = np.arange(1, len(points), dtype=np.int_)
    poly.lines = cells
    return poly


## Check the element in the mobius strip closest to the ball ##
mobius_points = np.array(mobius.points)[:,None,:]
normals = mobius.point_normals
trj_xyz_ext = trj_xyz[None,:,:]
d = trj_xyz_ext - mobius_points
eucl_d = np.sum(d**2,axis=-1)
closest = np.argmin(eucl_d, axis=0)

normals_2 = normals[closest,:]
trj_side = mobius_map.loop

one_side = np.where(trj_side)[0]
other_side = np.where(1 - trj_side)[0]

line_oneside = point_object_from_points(trj_xyz[one_side,:])
line_otherside = point_object_from_points(trj_xyz[other_side,:])


#line_oneside.point_normals = normals_2[one_side,:]
line_oneside.vectors = 0.1 * np.array(normals_2[one_side,:])
line_otherside.vectors = 0.1 * np.array(normals_2[other_side,:])

#line_otherside.point_normals = normals_2[other_side,:]

line_oneside.warp_by_vector(factor=.3, inplace=True)
line_otherside.warp_by_vector(factor=-.3, inplace=True)

#line_otherside.warp_by_vector(factor=-.3, inplace=True)


line = line_oneside + line_otherside

##########################################

pv.set_plot_theme("document")
p = pv.Plotter()

p.disable_shadows()
silhouette = dict(
    color='black',
    line_width=6.,
    decimate=None,
    feature_angle=True,
)
p.add_mesh(mobius, color='white', smooth_shading=True, silhouette=silhouette,  ambient=0.7, opacity=1.)


# create many spheres from the point cloud
sphere = pv.Sphere(radius=0.02)
pc = line.glyph(scale=False, geom=sphere)

p.add_mesh(pc, show_scalar_bar=False)


p.show()
