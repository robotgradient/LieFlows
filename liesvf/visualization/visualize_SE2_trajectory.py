import torch
import numpy as np
from liesvf.utils.generic import to_numpy, to_torch
import matplotlib.pyplot as plt

from itertools import product, combinations
from math import acos, atan2, cos, pi, sin

from liesvf.riemannian_manifolds.liegroups.numpy import SE2


def plot_SE2_frame(H, ax, alpha = 0.3, colors=0):
    x0 = H[:-1, -1]

    if colors==0:
        c = ['red', 'green']
    else:
        c = ['pink', 'olive']

    for i in range(2):
        v_x = 0.2 * H[:-1, i] + H[:-1, -1]
        x = np.concatenate((x0[None,:],v_x[None,:]),0)

        ax.plot(x[:,0], x[:,1], color= c[i], linewidth=4., alpha=alpha)


def visualize_SE2_traj(traj, ax=None, n_samples = 60,):

    positions = np.linspace(-1, 1., n_samples)
    indexes = np.floor(np.linspace(0, traj.shape[0]-1, n_samples))
    for i in range(n_samples):
        H = traj[int(indexes[i]),...]
        plot_SE2_frame(H, ax)


def visualize_se2_traj(traj, ax=None, n_samples = 20,colors=0):

    H = np.eye(3)
    positions = np.linspace(-1, 1., n_samples)
    indexes = np.floor(np.linspace(0, traj.shape[0]-1, n_samples))
    for i in range(n_samples):
        x = traj[int(indexes[i]),:]
        H = SE2.exp(x).as_matrix()
        plot_SE2_frame(H, ax, alpha=0.5, colors=colors)

