import numpy as np
import os
import torch
import matplotlib.pyplot as plt

from liesvf.utils import R_2_axis_angle

from liesvf.visualization import visualize_SE3_traj
from liesvf.dataset.generic_dataset import VDataset
from liesvf.riemannian_manifolds.liegroups.numpy import SE3


directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..','data')) + '/POURING_dataset/'


class POURING_SE3():
    def __init__(self, n_trjs =1000 ,device = torch.device('cpu'), PLOT=False):
        ## Define Variables and Load trajectories ##
        self.type = type
        self.dim = 6
        self.dt = .01

        trj_filename = 'Pouring.npy'
        trajs_np = np.load(os.path.join(directory, trj_filename),allow_pickle=True)

        self.trajs_real=[]
        for i in range(trajs_np.shape[0]):
            if i < n_trjs:
                self.trajs_real.append(trajs_np[i])
        self.n_trajs = len(self.trajs_real)
        self.n_dims = trajs_np[0].shape[-1]

        self.rot_trajs = []
        for trj in self.trajs_real:
            trj_rot = xyzrpy_2_rot(trj)
            self.rot_trajs.append(trj_rot)

        ##### Compute the goal SE(3) pose #####
        ### We compute the goal SE(3) by computing the mean of the goal poses in the Tangent Space ###
        ### SE(3) -> LOG -> T_SE(3) -> Mean -> SE(3) ### In order to do it properly, it is better if I iterate
        goal_H = self.compute_se3_center(self.rot_trajs)
        self.goal_H = goal_H
        self.norm_rot_trajs = []
        for trj in self.rot_trajs:
            n_trj = np.zeros((0,4,4))
            for t in range(trj.shape[0]):
                H = trj[t,...]
                H2 = np.matmul(np.linalg.inv(goal_H), H)
                n_trj = np.concatenate((n_trj, H2[None,...]),0)
            self.norm_rot_trajs.append(n_trj)

        if PLOT:
            fig = plt.figure(figsize=(20, 20), num=1)
            ax = fig.gca(projection='3d')
            for i in range(30):
                visualize_SE3_traj(self.norm_rot_trajs[i][:,:], ax=ax)
            plt.show()
            plt.pause(3.)

        ###### ROT TRAJECTORY to AXIS-ANGLE representation #########
        ## The tangent space is represented as x = [X, Y, Z, Wx, Wy, Wz] ##
        self.main_trajs = []
        for trj in self.norm_rot_trajs:
            trj_ax_angle = self.rot_traj_2_ax_angle_traj(trj)
            self.main_trajs.append(trj_ax_angle)

        if PLOT:
            fig, axs = plt.subplots(6)
            fig.suptitle('Vertically stacked subplots')
            for trj in self.main_trajs:
                for i in range(6):
                   axs[i].plot(trj[:,i])
            plt.show()

        self.min_max = self.get_max_and_min(self.main_trajs)
        self.dataset = VDataset(trajs=self.main_trajs, dt=self.dt)

    def compute_se3_center(self, rot_trajectories):
        ## get last element array ##
        goals = np.zeros((0,4,4))
        for trj in rot_trajectories:
            goals = np.concatenate((goals,trj[-1:,...]),0)

        mean = np.eye(4)
        opt_steps = 10
        for i in range(opt_steps):
            axis_array = np.zeros((0,6))
            for t in range(goals.shape[0]):
                g = goals[t, ...]
                g_I = np.matmul(np.linalg.inv(mean), g)
                axis = SE3.from_matrix(g_I).log()
                axis_array = np.concatenate((axis_array,axis[None,:]),0)

            mean_tangent = np.mean(axis_array,0)
            mean_new = SE3.exp(mean_tangent).as_matrix()
            mean = np.matmul(mean, mean_new)
        return mean

    def get_max_and_min(self, trajs):
        max_values = np.ones(6)*-99999999
        min_values = np.ones(6)*99999999
        for trj in trajs:
            max_i = np.max(trj,0)
            min_i = np.min(trj,0)
            for i in range(6):
                if max_values[i]< max_i[i]:
                    max_values[i] = max_i[i]
                if min_values[i] > min_i[i]:
                    min_values[i] = min_i[i]
        return [min_values, max_values]

    def rot_traj_2_ax_angle_traj(self, rot_traj):
        ax_angle_trj = np.zeros((0, 6))
        for t in range(rot_traj.shape[0]):
            vt = SE3.from_matrix(rot_traj[t, ...]).log()
            ax_angle_trj = np.concatenate((ax_angle_trj, vt[None, :]), 0)
        return ax_angle_trj


def xyzrpy_2_rot(trj):
    rot_trj = np.zeros((0,4,4))
    for i in range(trj.shape[0]):
        R = eul2rot(trj[i,3:])
        X = trj[i,:3]
        H = np.eye(4)
        H[:3,:3] = R
        H[:3,-1] = X
        rot_trj = np.concatenate((rot_trj,H[None,...]),0)
    return rot_trj


def eul2rot(theta) :

    R = np.array([[np.cos(theta[1])*np.cos(theta[2]),       np.sin(theta[0])*np.sin(theta[1])*np.cos(theta[2]) - np.sin(theta[2])*np.cos(theta[0]),      np.sin(theta[1])*np.cos(theta[0])*np.cos(theta[2]) + np.sin(theta[0])*np.sin(theta[2])],
                  [np.sin(theta[2])*np.cos(theta[1]),       np.sin(theta[0])*np.sin(theta[1])*np.sin(theta[2]) + np.cos(theta[0])*np.cos(theta[2]),      np.sin(theta[1])*np.sin(theta[2])*np.cos(theta[0]) - np.sin(theta[0])*np.cos(theta[2])],
                  [-np.sin(theta[1]),                        np.sin(theta[0])*np.cos(theta[1]),                                                           np.cos(theta[0])*np.cos(theta[1])]])

    return R


def rot_traj_2_ax_angle_traj(rot_traj):
    ax_angle_trj = np.zeros((0,6))
    for t in range(rot_traj.shape[0]):
        axis, angle = R_2_axis_angle(rot_traj[t,:3,:3])
        xyz = rot_traj[t,:3,-1]
        w = axis*angle
        vector_so3 = np.concatenate((xyz, w),0)
        ax_angle_trj = np.concatenate((ax_angle_trj, vector_so3[None,:]),0)
    return ax_angle_trj


if __name__ == "__main__":
    pouring_data = POURING_SE3(device=None, PLOT=True)
    print(pouring_data)
