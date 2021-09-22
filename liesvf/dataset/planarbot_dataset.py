import numpy as np
import os
import matplotlib.pyplot as plt

from liesvf.visualization import visualize_SE2_traj
from liesvf.dataset.generic_dataset import VDataset
from liesvf.riemannian_manifolds.liegroups.numpy import SE2

import os
import json
import numpy as np


directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..','data')) + '/PLANARBOT01_dataset/'
filename = 'qtrjs_500.json'
dirname = os.path.abspath(os.path.dirname(__file__ + '/../../../'))
dirname = os.path.join(dirname, 'data', 'PLANARBOT01_dataset')
file = os.path.join(dirname, filename)


class PLANARBOT_SE2():
    def __init__(self, datapercentage=1., PLOT = False):
        ## Define Variables and Load trajectories ##
        self.type = type
        self.dim = 3
        self.dt = .01


        self.trajs_real = []
        self.v_trajs_real = []
        self.times_real = []

        T = 1000
        for i in range(T):
            if PLOT:
                print(i)

            name = 'xyz_se2_trjs_{}.json'.format(i)
            file = os.path.join(dirname, name)

            with open(file, 'r') as json_file:
                data = json.load(json_file)

            for trj in data['trajectories']:
                c = np.random.rand()
                if c<datapercentage:
                    x_trj = np.asarray(trj['positions'])
                    t_trj  = np.asarray(trj['times'])

                    self.trajs_real.append(x_trj)
                    self.times_real.append(t_trj)


        self.rot_trajs = []
        for trj in self.trajs_real:
            self.rot_trajs.append(trj)

        ##### Compute the goal SE(2) pose #####
        goal_H = self.compute_se2_center(self.rot_trajs)

        self.goal_H = goal_H
        self.norm_rot_trajs = []
        for trj in self.rot_trajs:
            n_trj = np.zeros((0,3,3))
            for t in range(trj.shape[0]):
                H = trj[t,...]
                H2 = np.matmul(np.linalg.inv(goal_H), H)
                n_trj = np.concatenate((n_trj, H2[None,...]),0)
            self.norm_rot_trajs.append(n_trj)

        if PLOT:
            fig = plt.figure(figsize=(20, 20), num=1)
            ax = fig.gca()
            for i in range(10):
                visualize_SE2_traj(self.norm_rot_trajs[i][:,:], ax=ax)
            plt.show()
            plt.pause(3.)

        ###### ROT TRAJECTORY to AXIS-ANGLE representation #########
        ## The tangent space is represented as x = [X, Y, Wz] ##
        self.ax_angle_trajs = []
        for trj in self.norm_rot_trajs:
            trj_ax_angle = self.rot_traj_2_ax_angle_traj(trj)
            self.ax_angle_trajs.append(trj_ax_angle)

        self.main_trajs, self.ind_list = self.eliminate_outlier_trjs(self.ax_angle_trajs)
        self.n_trajs = len(self.main_trajs)
        if PLOT:
            fig, axs = plt.subplots(3)
            fig.suptitle('Vertically stacked subplots')
            for trj in self.main_trajs:
                for i in range(3):
                   axs[i].plot(trj[:,i])
            plt.show()

        self.min_max = self.get_max_and_min(self.main_trajs)
        gain = np.max(np.abs(self.min_max), 0)
        gain[2] = np.pi
        self.gain = gain

        self.dataset = VDataset(trajs=self.main_trajs, v_trj=None, dt=self.dt)

    def compute_se2_center(self, rot_trajectories):
        ## get last element array ##
        goals = np.zeros((0,3,3))
        for trj in rot_trajectories:
            goals = np.concatenate((goals,trj[-1:,...]),0)

        mean = np.eye(3)
        opt_steps = 10
        for i in range(opt_steps):
            axis_array = np.zeros((0,3))
            for t in range(goals.shape[0]):
                g = goals[t, ...]
                g_I = np.matmul(np.linalg.inv(mean), g)
                axis = SE2.from_matrix(g_I).log()
                axis_array = np.concatenate((axis_array,axis[None,:]),0)

            mean_tangent = np.mean(axis_array,0)
            mean_new = SE2.exp(mean_tangent).as_matrix()
            mean = np.matmul(mean, mean_new)
        return mean

    def get_max_and_min(self, trajs):
        dim = trajs[0].shape[-1]

        max_values = np.ones(dim)*-99999999
        min_values = np.ones(dim)*99999999
        for trj in trajs:
            max_i = np.max(trj,0)
            min_i = np.min(trj,0)
            for i in range(dim):
                if max_values[i]< max_i[i]:
                    max_values[i] = max_i[i]
                if min_values[i] > min_i[i]:
                    min_values[i] = min_i[i]
        return [min_values, max_values]

    def rot_traj_2_ax_angle_traj(self, rot_traj, dim = 3):
        ax_angle_trj = np.zeros((0, 3))
        for t in range(rot_traj.shape[0]):
            vt = SE2.from_matrix(rot_traj[t, ...]).log()
            ax_angle_trj = np.concatenate((ax_angle_trj, vt[None, :]), 0)
        return ax_angle_trj

    def eliminate_outlier_trjs(self, trjs):
        trjs_clean = []
        ind_list = []
        c = 0
        for trj in trjs:
            save = True
            x0  = trj[0,:]
            for t in range(1,trj.shape[0]):
                x1 = trj[t,:]
                delta = x1 - x0
                if np.max(np.abs(delta))>np.pi:
                    save=False
                    break
            if save:
                trjs_clean.append(trj)
                ind_list.append(c)
            c +=1
        return trjs_clean, ind_list

    def transform_velocity_trj(self, Hcenter, v_trj):
        Adj = SE2.from_matrix(np.linalg.inv(Hcenter)).adjoint()
        n_v_trj = np.matmul(Adj,v_trj.T).T
        return n_v_trj


class PLANARBOT_Q():
    def __init__(self, datapercentage=1., PLOT = False):

        ## Define Variables and Load trajectories ##
        self.type = type
        self.dim = 5
        self.dt = .01

        self.trajs_real = []
        self.v_trajs_real = []
        self.times_real = []

        T = 998
        for i in range(T):
            if PLOT:
                print(i)

            # name = 'xyz_{}.json'.format(i)
            # file = os.path.join(dirname, 'q_smooth_2', name)
            name = 'q_trjs_{}.json'.format(i)
            file = os.path.join(dirname, 'q_new', name)


            with open(file, 'r') as json_file:
                data = json.load(json_file)

            for trj in data['trajectories']:
                c = np.random.rand()
                if c<datapercentage:
                    x_trj = np.asarray(trj['q_positions'])
                    self.trajs_real.append(x_trj)


        ##### Compute the goal Q pose #####
        goal_H = self.trajs_real[0][-1,:]

        self.goal_H = goal_H
        self.norm_rot_trajs = []
        for trj in self.trajs_real:
            n_trj = trj - self.goal_H
            self.norm_rot_trajs.append(n_trj)

        gain = np.ones(5)*4.
        self.gain = gain

        ## Get Min and Max ##
        # temp_trj = np.zeros((0,self.dim))
        # for trj in self.norm_rot_trajs:
        #     temp_trj = np.concatenate((temp_trj, trj),0)
        #
        # print(temp_trj)
        self.min = np.array([-4.38530029, -1.40307733, -1.36645868, -1.41845268, -5.19781929])
        self.max = np.array([2.93735432, 5.25511116, 5.36307186, 5.97583948, 4.20908085])


        if PLOT:
            fig, axs = plt.subplots(self.dim)
            fig.suptitle('Vertically stacked subplots')
            for trj in self.norm_rot_trajs:
                for i in range(self.dim):
                   axs[i].plot(trj[:,i])
            plt.show()

        self.dataset = VDataset(trajs=self.norm_rot_trajs, v_trj=None, dt=self.dt)


if __name__ == "__main__":
    planarbot = PLANARBOT_SE2(datapercentage=0.5, PLOT=True)
    planar_data_q = PLANARBOT_Q(datapercentage=1., PLOT=True)








