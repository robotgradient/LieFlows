import numpy as np
import pyvista as pv
from liesvf.riemannian_manifolds.riemannian_manifolds import Mobius

mobius_map = Mobius()

def visualize_vector_field_mobius(p, mobius, dynamics):
    xyz = np.array(mobius.points)
    mobius_map.from_xyz(xyz[:, 0], xyz[:, 1], xyz[:, 2])
    uv = mobius_map.log()

    n_samples = uv.shape[0]
    uv_2pi = np.copy(uv)

    uv_2pi[:, 0] = uv_2pi[:, 0] + np.pi
    uv_2pi[:, 0] = np.arctan2(np.sin(uv_2pi[:, 0]), np.cos(uv_2pi[:, 0]))

    uv_ext = np.concatenate((uv, uv_2pi), axis=0)
    uv_ext_J = np.concatenate((uv, uv), axis=0)

    duv = dynamics(uv_ext)
    Jinv = mobius_map.inv_jacobian(uv_ext_J[:, 0], uv_ext_J[:, 1])
    dxyz = np.einsum('bnm,bm->bn', Jinv, duv)

    # Compute the normals in-place and use them to warp the globe
    n_points = mobius.points.shape[0]
    mobius.compute_normals(inplace=True)  # this activates the normals as well
    mobius.vectors = 0.1 * np.array(mobius.point_normals)


    mobius2 = mobius.warp_by_vector(factor=.3)
    mobius2.vectors = dxyz[:n_points, :]
    p.add_mesh(mobius2.arrows, color='black', lighting=False, )

    mobius3 = mobius.warp_by_vector(factor=-.3)
    mobius3.vectors = dxyz[n_points:, :]
    p.add_mesh(mobius3.arrows, color='black', lighting=False, )


## Plot Trajectories as a ball ##
def point_object_from_points(points):
    """Given an array of points, make a line set"""
    poly = pv.PolyData()
    poly.points = points
    # cells = np.full((len(points) - 1, 3), 2, dtype=np.int_)
    # cells[:, 1] = np.arange(0, len(points) - 1, dtype=np.int_)
    # cells[:, 2] = np.arange(1, len(points), dtype=np.int_)
    # poly.lines = cells
    return poly

def visualize_latent_points_on_mobius(p, mobius, uv):
    N = uv.shape[0]
    mobius_map.from_tangent(uv[:, 0], uv[:, 1])
    xyz = mobius_map.exp()

    ## Check the element in the mobius strip closest to the ball ##
    mobius_points = np.array(mobius.points)[:, None, :]
    xyz_ext = xyz[None, :, :]
    d = xyz_ext - mobius_points
    eucl_d = np.sum(d ** 2, axis=-1)
    closest = np.argmin(eucl_d, axis=0)

    ## Assign closest normal ##
    normals = mobius.point_normals
    point_normals = normals[closest, :]
    points_side = mobius_map.loop

    ## Move Points to manifold surface ##
    one_side = np.where(points_side)[0]
    other_side = np.where(1 - points_side)[0]

    points_one_side = point_object_from_points(xyz[one_side, :])
    points_other_side = point_object_from_points(xyz[other_side, :])

    points_one_side.vectors = 0.1 * np.array(-point_normals[one_side, :])
    points_other_side.vectors = 0.1 * np.array(point_normals[other_side, :])

    points_one_side.warp_by_vector(factor=1., inplace=True)
    points_other_side.warp_by_vector(factor=1., inplace=True)

    points = points_one_side + points_other_side
    #points = point_object_from_points(xyz)

    colors = np.arange(N)
    colors = np.concatenate((colors[one_side], colors[other_side]),0)

    points['values'] = colors

    # create many spheres from the point cloud
    sphere = pv.Sphere(radius=0.02)
    values = np.arange(N)
    #values = np.random.randn(N)
    pc = points.glyph(scale=False, geom=sphere, indices = values)

    p.add_mesh(pc, show_scalar_bar=False, scalars = pc['values'])








