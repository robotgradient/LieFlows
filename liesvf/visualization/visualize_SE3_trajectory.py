import numpy as np
import matplotlib.pyplot as plt
from itertools import product, combinations
from math import acos, atan2, cos, pi, sin
from liesvf.riemannian_manifolds.liegroups.numpy import SE3


def R_axis_angle(axis, angle):
    """Generate the rotation matrix from the axis-angle notation.
    Conversion equations
    ====================
    From Wikipedia (http://en.wikipedia.org/wiki/Rotation_matrix), the conversion is given by::
        c = cos(angle); s = sin(angle); C = 1-c
        xs = x*s;   ys = y*s;   zs = z*s
        xC = x*C;   yC = y*C;   zC = z*C
        xyC = x*yC; yzC = y*zC; zxC = z*xC
        [ x*xC+c   xyC-zs   zxC+ys ]
        [ xyC+zs   y*yC+c   yzC-xs ]
        [ zxC-ys   yzC+xs   z*zC+c ]
    @param matrix:  The 3x3 rotation matrix to update.
    @type matrix:   3x3 numpy array
    @param axis:    The 3D rotation axis.
    @type axis:     numpy array, len 3
    @param angle:   The rotation angle.
    @type angle:    float
    """

    # Trig factors.
    ca = cos(angle)
    sa = sin(angle)
    C = 1 - ca

    # Depack the axis.
    x, y, z = axis

    # Multiplications (to remove duplicate calculations).
    xs = x*sa
    ys = y*sa
    zs = z*sa
    xC = x*C
    yC = y*C
    zC = z*C
    xyC = x*yC
    yzC = y*zC
    zxC = z*xC

    matrix = np.zeros((3,3))
    # Update the rotation matrix.
    matrix[0, 0] = x*xC + ca
    matrix[0, 1] = xyC - zs
    matrix[0, 2] = zxC + ys
    matrix[1, 0] = xyC + zs
    matrix[1, 1] = y*yC + ca
    matrix[1, 2] = yzC - xs
    matrix[2, 0] = zxC - ys
    matrix[2, 1] = yzC + xs
    matrix[2, 2] = z*zC + ca
    return matrix

def draw_cube(ax):
    #draw cube
    r = [-1, 1]
    for s, e in combinations(np.array(list(product(r,r,r))), 2):
        if np.sum(np.abs(s-e)) == r[1]-r[0]:
            ax.plot3D(*zip(s,e), color="b")

def visualize_SE3_traj(traj, ax=None, n_samples = 20, alpha =0.1 ):

    #draw_cube(ax)

    positions = np.linspace(-1, 1., n_samples)
    indexes = np.floor(np.linspace(0, traj.shape[0]-1, n_samples))
    for i in range(n_samples):
        H = traj[int(indexes[i]),...]
        plot_tf_frame(H, ax, alpha=alpha)

def visualize_se3_traj(traj, ax=None, n_samples = 20,colors=0, H_origin = np.eye(4), alpha = 0.5):

    draw_cube(ax)

    H = np.eye(4)
    positions = np.linspace(-1, 1., n_samples)
    indexes = np.floor(np.linspace(0, traj.shape[0]-1, n_samples))
    for i in range(n_samples):
        x = traj[int(indexes[i]),:]
        H = SE3.exp(x).as_matrix()
        H = np.matmul(H_origin,H)

        plot_tf_frame(H, ax, alpha=alpha, colors=colors)

def plot_tf_frame(H, ax, alpha = 0.05, colors=0):
    x0 = H[:-1, -1]

    if colors==0:
        c = ['red', 'green', 'blue']
    else:
        c = ['pink', 'olive', 'cyan']


    for i in range(3):
        v_x = 0.2 * H[:-1, i] + H[:-1, -1]
        x = np.concatenate((x0[None,:],v_x[None,:]),0)

        ax.plot(x[:,0], x[:,1], x[:,2], color= c[i], linewidth=4., alpha=alpha)


if __name__ == '__main__':

    x = np.random.randn(100,6)

    fig = plt.figure(figsize=(20, 20))
    ax = fig.gca(projection='3d')
    visualize_se3_traj(x, ax=ax)
    plt.show()
