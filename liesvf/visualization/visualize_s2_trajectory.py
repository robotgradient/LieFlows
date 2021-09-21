import numpy as np

from liesvf.utils import to_torch, to_numpy

import matplotlib.pyplot as plt
import pyvista as pv

from liesvf.riemannian_manifolds.liegroups.numpy.s2 import S2, S2_angle


def plot_sphere(ax):
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color="r")


def visualize_s2_tangent(traj, ax=None, n_samples=20, fig_number=1):
    if ax is None:
        fig = plt.figure()
        ax = fig.gca(projection='3d')

    plot_sphere(ax)

    indexes = np.floor(np.linspace(0, traj.shape[0]-1, n_samples)).astype(int)
    trj_plot = traj[:,:]

    s2= S2()
    s2.from_tangent(u=trj_plot[:,0], v=trj_plot[:,1])
    xyz = s2.exp()

    ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2], color="g", s=100)


def visualize_sphere(p):
    p.disable_shadows()
    sphere = pv.Sphere(radius=1.)
    silhouette = dict(
        color='black',
        line_width=6.0,
        decimate=0.,
        feature_angle=True,
    )
    p.add_mesh(sphere, color="tan", smooth_shading=True, silhouette=silhouette, ambient=0.7, opacity=0.8)
    #p.add_mesh(sphere, color='#FFFFFF', smooth_shading=True, silhouette=silhouette, ambient=0.7)

    return sphere


def visualize_s2_vector_field(p, policy, sphere=pv.Sphere(radius=1.), torch=False, device=None):

    p_xyz = np.array(sphere.points.tolist())

    s2 = S2()
    s2.from_xyz(x=p_xyz[:,0],y=p_xyz[:,1], z=p_xyz[:,2])
    uv = s2.log()

    # s2_r = S2_models()
    # s2_r.from_tangent(uv[:,0],uv[:,1])
    # xyz_r = s2_r.exp()
    # print(np.allclose(xyz_r, p_xyz))
    if torch:
        uv = to_torch(uv, device)

    vel_uv = policy(uv)
    if torch:
        vel_uv = to_numpy(vel_uv)
    norm_vel = np.linalg.norm(vel_uv,axis=0)
    mean_norm = np.mean(norm_vel)
    #vel_uv = vel_uv/mean_norm*20.

    J = s2.jacobian(u=uv[:,0], v=uv[:,1])
    vel_xyz = np.einsum('bmn,bn->bm',J ,vel_uv)
    sphere.vectors = vel_xyz * 0.1

    p.add_mesh(sphere.arrows, lighting=False,)


def visualize_S2_vector_field(p, policy, sphere=pv.Sphere(radius=1.), torch=True, device=None):

    xyz = np.array(sphere.points.tolist())

    if torch:
        xyz = to_torch(xyz, device)

    vel_xyz = policy(xyz)
    if torch:
        vel_xyz = to_numpy(vel_xyz)


    norm_vel = np.linalg.norm(vel_xyz,axis=1)
    norm_vel_ext = np.repeat(norm_vel[:,None],3,axis=1)
    v_clip =1.
    clip_norm = np.clip(norm_vel_ext,0,v_clip)/v_clip
    #mean_norm = np.mean(norm_vel)
    vel_xyz = vel_xyz/norm_vel_ext*clip_norm


    sphere.vectors = vel_xyz * 0.1

    p.add_mesh(sphere.arrows, color='black')


    #p.add_mesh(sphere.arrows, scalars='GlyphScale', lighting=False,)



def visualize_s2_angle_vector_field(p, policy, sphere=pv.Sphere(radius=1.), torch=False, device=None):

    p_xyz = np.array(sphere.points.tolist())

    s2 = S2_angle()
    s2.from_xyz(x=p_xyz[:,0],y=p_xyz[:,1], z=p_xyz[:,2])
    uv = s2.log()

    xy = s2.polar_to_cartesian(uv[:,0], uv[:,1])
    Jp2x = s2.jac_p2x(xy[:,0],xy[:,1])

    if torch:
        xy = to_torch(xy, device)
    vel_xy = policy(xy)
    if torch:
        vel_xy = to_numpy(vel_xy)

    norm_vel = np.linalg.norm(vel_xy,axis=1)
    norm_vel_ext = np.repeat(norm_vel[:,None],2,axis=1)
    v_clip =1.
    clip_norm = np.clip(norm_vel_ext,0,v_clip)/v_clip
    #mean_norm = np.mean(norm_vel)
    vel_xy = vel_xy/norm_vel_ext*clip_norm


    vel_uv = np.einsum('bnm,bm->bn', Jp2x, vel_xy)


    J = s2.jacobian(u=uv[:,0], v=uv[:,1])
    vel_xyz = np.einsum('bmn,bn->bm',J ,vel_uv)
    sphere.vectors = vel_xyz * 0.1

    #p.add_mesh(sphere.arrows, scalars='GlyphScale', lighting=True)
    p.add_mesh(sphere.arrows, color='black')



def visualize_s2_tangent_trajectories(p, trj, color='r'):

    T = trj.shape[0]

    s2 = S2_angle()
    xy = trj
    uv = s2.cartesian_to_polar(xy[:, 0], xy[:, 1])
    s2.from_tangent(uv[:, 0], uv[:, 1])
    trj_xyz = s2.exp()

    # generate same spline with 400 interpolation points
    spline = pv.Spline(trj_xyz, np.clip(T,0,400))

    # plot without scalars
    p.add_mesh(spline, line_width=8, color=color)



def visualize_s2_tangent_trajectories2(p, trj, color='r'):

    T = trj.shape[0]

    s2_r = S2()
    s2_r.from_tangent(trj[:,0],trj[:,1])
    trj_xyz = s2_r.exp()

    # generate same spline with 400 interpolation points
    spline = pv.Spline(trj_xyz, np.clip(T,0,400))

    # plot without scalars
    p.add_mesh(spline, line_width=8, color=color)


if __name__ == "__main__":
    def pi(x):
        vx = np.zeros((x.shape[0],2))
        #x[:, 0] = x[:,1]
        #vx[:,1] = np.zeros(x.shape[0])
        vx[:,1] = x[:,1]#np.ones(x.shape[0])
        return -x#np.ones((x.shape[0],2))

    import torch
    device = torch.device('cpu')

    pv.set_plot_theme("document")

    p = pv.Plotter(window_size=[2024, 2024], off_screen=True)
    p.camera_position = 'yz'
    p.camera.position = (7., 0., 0.)
    p.camera.roll += 10
    p.camera.azimuth += 60
    p.camera.elevation += 50


    sphere = visualize_sphere(p)
    # visualize_s2_vector_field(p=p, policy=pi)
    visualize_s2_angle_vector_field(p=p, policy=pi)


    T = 1000
    x = np.linspace(0.,1.,T)
    trj = np.zeros((T,3))
    trj[:,0] = x
    #visualize_s2_tangent_trajectories(p=p, trj=trj)

    p.show(screenshot='extended_experiment.png')

    #p.show()