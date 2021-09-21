import numpy as np
import os
import torch
from scipy.signal import savgol_filter

import matplotlib.pyplot as plt



class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, trajs, device, steps=20):
        'Initialization'
        dim = trajs[0].shape[1]

        self.x = []
        self.x_n = np.zeros((0, dim))
        for i in range(steps):
            tr_i_all = np.zeros((0,dim))
            for tr_i in  trajs:
                _trj = tr_i[i:i-steps,:]
                tr_i_all = np.concatenate((tr_i_all, _trj), 0)
                self.x_n = np.concatenate((self.x_n, tr_i[-1:,:]),0)
            self.x.append(tr_i_all)

        self.x = torch.from_numpy(np.array(self.x)).float().to(device)
        self.x_n = torch.from_numpy(np.array(self.x_n)).float().to(device)

        self.len_n = self.x_n.shape[0]
        self.len = self.x.shape[1]
        self.steps_length = steps
        self.step = steps - 1

    def __len__(self):
        'Denotes the total number of samples'
        return self.len

    def set_step(self, step=None):
        if step is None:
            self.step = np.random.randint(1, self.steps_length-1)

    def __getitem__(self, index):
        'Generates one sample of data'

        X = self.x[0, index, :]
        X_1 = self.x[self.step, index, :]

        index = np.random.randint(self.len_n)
        X_N = self.x_n[index, :]

        return X, [X_1, int(self.step), X_N, index]


class CycleDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, trajs, device, trajs_phase, steps=20):
        'Initialization'
        dim = trajs[0].shape[1]

        self.x = []
        self.x_n = np.zeros((0, dim))
        for i in range(steps):
            tr_i_all = np.zeros((0,dim))
            for tr_i in  trajs:
                _trj = tr_i[i:i-steps,:]
                tr_i_all = np.concatenate((tr_i_all, _trj), 0)
                self.x_n = np.concatenate((self.x_n, tr_i[-1:,:]),0)
            self.x.append(tr_i_all)

        self.x = torch.from_numpy(np.array(self.x)).float().to(device)
        self.x_n = torch.from_numpy(np.array(self.x_n)).float().to(device)

        ## Phase ordering ##
        trp_all = np.zeros((0))
        for trp_i in  trajs_phase:
            _trjp = trp_i[:-steps]
            trp_all = np.concatenate((trp_all, _trjp), 0)
        self.trp_all = torch.from_numpy(trp_all).float().to(device)

        self.len_n = self.x_n.shape[0]
        self.len = self.x.shape[1]
        self.steps_length = steps
        self.step = steps - 1

    def __len__(self):
        'Denotes the total number of samples'
        return self.len

    def set_step(self, step=None):
        if step is None:
            self.step = np.random.randint(1, self.steps_length-1)

    def __getitem__(self, index):
        'Generates one sample of data'

        X = self.x[0, index, :]
        X_1 = self.x[self.step, index, :]
        phase = self.trp_all[index]

        return X, [X_1, int(self.step), phase]


class VDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, trajs, v_trj=None, dt=0.01):
        'Initialization'
        dim = trajs[0].shape[1]
        dt = dt

        self.x = []
        self.dx = []

        if v_trj is None:
            for tr_i in  trajs:
                num_pts = tr_i.shape[0]
                demo_smooth = np.zeros_like(tr_i)
                window_size = int(2 * (25. * num_pts / 150 // 2) + 1)
                for j in range(dim):
                    try:
                        if window_size>3:
                            poly = 3
                        else:
                            poly = window_size-1
                        demo_smooth[:, j] = savgol_filter(tr_i[:, j], window_size, poly)
                    except:
                        print('fail')

                demo_vel = np.diff(demo_smooth, axis=0) / dt
                self.x.append(demo_smooth[:-1,:])
                self.dx.append(demo_vel)
        else:
            self.x = trajs
            self.dx = v_trj

        self.x_data = np.zeros((0, dim))
        self.dx_data = np.zeros((0, dim))
        for i in range(len(self.x)):
            self.x_data = np.concatenate((self.x_data, self.x[i]),0)
            self.dx_data = np.concatenate((self.dx_data, self.dx[i]),0)

        self.x_data = torch.from_numpy(np.array(self.x_data)).float()
        self.dx_data = torch.from_numpy(np.array(self.dx_data)).float()

        self.len = self.x_data.shape[0]

    def __len__(self):
        'Denotes the total number of samples'
        return self.len

    def __getitem__(self, index):
        'Generates one sample of data'
        x = self.x_data[index,:]
        dx = self.dx_data[index, :]
        return x,dx


class ContextualizedDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, trajs, contexts, device, steps=20):
        'Initialization'
        dim = trajs[0][0].shape[1]

        self.n_conditions = len(contexts)
        self.c_n = np.zeros((0, 3))
        self.x = []
        self.x_n = np.zeros((0, dim))
        self.x_n_ref = np.zeros((0, dim))

        for i in range(steps):
            self.c = np.zeros((0, 3))
            tr_i_all = np.zeros((0, dim))

            for i in range(len(trajs)):
                ref_trjs  = trajs[0]
                trj_list = trajs[i]
                context = contexts[i]
                for tr_i in trj_list:
                    _trj = tr_i[i:i - steps, :]
                    tr_i_all = np.concatenate((tr_i_all, _trj), 0)

                    c_all = np.concatenate((_trj.shape[0] * [context[None, :]]), 0)
                    self.c = np.concatenate((self.c, c_all), 0)

                    self.x_n = np.concatenate((self.x_n, tr_i[-1:, :]), 0)

                    tr_size = len(ref_trjs)
                    index_ref = np.random.randint(tr_size)
                    tr_ref = ref_trjs[index_ref]
                    self.x_n_ref = np.concatenate((self.x_n_ref, tr_ref[-1:, :]), 0)

                    self.c_n = np.concatenate((self.c_n, context[None, :]), 0)

            self.x.append(tr_i_all)


        self.x = torch.from_numpy(np.array(self.x)).float().to(device)
        self.x_n = torch.from_numpy(np.array(self.x_n)).float().to(device)
        self.x_n_ref = torch.from_numpy(np.array(self.x_n_ref)).float().to(device)

        self.c = torch.from_numpy(self.c).float().to(device)
        self.c_n = torch.from_numpy(self.c_n).float().to(device)

        self.len_n = self.x_n.shape[0]
        self.len = self.x.shape[1]
        self.steps_length = steps
        self.step = steps - 1

    def __len__(self):
        'Denotes the total number of samples'
        return self.len

    def set_step(self, step=None):
        if step is None:
            self.step = np.random.randint(1, self.steps_length - 1)

    def __getitem__(self, index):
        'Generates one sample of data'

        X = self.x[0, index, :]
        X_1 = self.x[self.step, index, :]
        C = self.c[index, :]

        index = np.random.randint(self.len_n)
        X_N = self.x_n[index, :]
        C_N = self.c_n[index, :]
        X_REF_N = self.x_n_ref[index,:]

        return X, [X_1, int(self.step), X_N, C, C_N, X_REF_N]



class CADataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, trajs, v_trajs, c_trajs, a_trajs=None, dt=0.01):
        'Initialization'
        dim = trajs[0].shape[1]
        c_dim = c_trajs[0].shape[1]
        dt = dt

        self.x = []
        self.dx = []
        self.ddx = []
        self.cx = []
        self.stable_x = []

        for tr_i in trajs:
            goal = tr_i[-1,:]
            goal_trji = np.tile(goal[None,:],(tr_i.shape[0],1))
            self.stable_x.append(goal_trji)

        if a_trajs is None:
            for tr_i, vtr_i, c_tri in  zip(trajs, v_trajs, c_trajs):
                num_pts = tr_i.shape[0]
                demo_smooth = np.zeros_like(tr_i)
                window_size = int(2 * (25. * num_pts / 150 // 2) + 1)
                for j in range(dim):
                    try:
                        if window_size>3:
                            poly = 3
                        else:
                            poly = window_size-1
                        demo_smooth[:, j] =  savgol_filter(vtr_i[:, j], window_size, poly)
                    except:
                        print('fail')

                demo_vel = np.diff(demo_smooth, axis=0) / dt
                self.x.append(tr_i[:-1,:])
                self.dx.append(vtr_i[:-1,:])
                self.ddx.append(demo_vel)
                self.cx.append(c_tri)
        else:
            self.x = trajs
            self.dx = v_trajs
            self.ddx = a_trajs
            self.cx = c_trajs

        self.x_data = np.zeros((0, dim))
        self.dx_data = np.zeros((0, dim))
        self.ddx_data = np.zeros((0, dim))
        self.cx_data = np.zeros((0, c_dim))
        self.stab_data = np.zeros((0, dim))

        for i in range(len(self.x)):
            self.x_data = np.concatenate((self.x_data, self.x[i]),0)
            self.dx_data = np.concatenate((self.dx_data, self.dx[i]),0)
            self.ddx_data = np.concatenate((self.ddx_data, self.ddx[i]),0)
            self.cx_data = np.concatenate((self.cx_data, self.cx[i]),0)
            self.stab_data = np.concatenate((self.stab_data, self.stable_x[i]),0)


        self.x_data = torch.from_numpy(np.array(self.x_data)).float()
        self.dx_data = torch.from_numpy(np.array(self.dx_data)).float()
        self.ddx_data = torch.from_numpy(np.array(self.ddx_data)).float()
        self.cx_data = torch.from_numpy(np.array(self.cx_data)).float()
        self.stab_data = torch.from_numpy(np.array(self.stab_data)).float()


        self.len = self.x_data.shape[0]

    def __len__(self):
        'Denotes the total number of samples'
        return self.len

    def __getitem__(self, index):
        'Generates one sample of data'
        x = self.x_data[index,:]
        dx = self.dx_data[index, :]
        ddx = self.ddx_data[index, :]
        cx = self.cx_data[index, :]
        stabx = self.stab_data[index, :]
        return x, dx, ddx, cx, stabx



class StructuredCADataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, trajs, v_trajs, c_trajs, a_trajs=None, dt=0.01, H=10):
        'Initialization'
        ## Planning Horizon ##
        index_list = []
        self.H = H

        dim = trajs[0].shape[1]
        c_dim = c_trajs[0].shape[1]
        dt = dt

        self.x = []
        self.dx = []
        self.ddx = []
        self.cx = []
        self.stable_x = []

        for tr_i in trajs:
            goal = tr_i[-1,:]
            goal_trji = np.tile(goal[None,:],(tr_i.shape[0],1))
            self.stable_x.append(goal_trji)

        if a_trajs is None:
            for tr_i, vtr_i, c_tri in  zip(trajs, v_trajs, c_trajs):
                goal = tr_i[-1,:]
                goal_ext = np.tile(goal,(H,1))
                tr_i = np.concatenate((tr_i, goal_ext),0)

                v_goal = vtr_i[-1,:]
                v_goal_ext = np.tile(v_goal,(H,1))
                vtr_i = np.concatenate((vtr_i, v_goal_ext),0)

                c_goal = c_tri[-1, :]
                c_goal_ext = np.tile(c_goal, (H, 1))
                c_tri = np.concatenate((c_tri, c_goal_ext), 0)

                num_pts = tr_i.shape[0]
                demo_smooth = np.zeros_like(tr_i)
                window_size = int(2 * (25. * num_pts / 150 // 2) + 1)
                for j in range(dim):
                    try:
                        if window_size>3:
                            poly = 3
                        else:
                            poly = window_size-1
                        demo_smooth[:, j] =  savgol_filter(vtr_i[:, j], window_size, poly)
                    except:
                        print('fail')

                demo_vel = np.diff(demo_smooth, axis=0) / dt
                self.x.append(tr_i[:-1,:])
                self.dx.append(vtr_i[:-1,:])
                self.ddx.append(demo_vel)
                self.cx.append(c_tri)
        else:
            self.x = trajs
            self.dx = v_trajs
            self.ddx = a_trajs
            self.cx = c_trajs

        ## Build Index ##
        self.indexes = torch.zeros(0, 2).int()
        for idx, trj in enumerate(self.x):
            TRJ_T = torch.arange(0,trj.shape[0]-self.H,1)
            TRJ_I = torch.tensor([idx]).repeat(TRJ_T.shape[0])

            trj_c = torch.cat((TRJ_T[:,None], TRJ_I[:,None]), 1).int()
            self.indexes = torch.cat([self.indexes, trj_c], 0)


        # self.x_data = np.zeros((0, dim))
        # self.dx_data = np.zeros((0, dim))
        # self.ddx_data = np.zeros((0, dim))
        # self.cx_data = np.zeros((0, c_dim))
        # self.stab_data = np.zeros((0, dim))
        #
        # for i in range(len(self.x)):
        #     self.x_data = np.concatenate((self.x_data, self.x[i]),0)
        #     self.dx_data = np.concatenate((self.dx_data, self.dx[i]),0)
        #     self.ddx_data = np.concatenate((self.ddx_data, self.ddx[i]),0)
        #     self.cx_data = np.concatenate((self.cx_data, self.cx[i]),0)
        #     self.stab_data = np.concatenate((self.stab_data, self.stable_x[i]),0)
        #
        #
        # self.x_data = torch.from_numpy(np.array(self.x_data)).float()
        # self.dx_data = torch.from_numpy(np.array(self.dx_data)).float()
        # self.ddx_data = torch.from_numpy(np.array(self.ddx_data)).float()
        # self.cx_data = torch.from_numpy(np.array(self.cx_data)).float()
        # self.stab_data = torch.from_numpy(np.array(self.stab_data)).float()
        #

        self.len = self.indexes.shape[0]

    def __len__(self):
        'Denotes the total number of samples'
        return self.len

    def __getitem__(self, index):
        'Generates one sample of data'
        trj_idx = self.indexes[index,1]
        T_idx = self.indexes[index,0]


        x = torch.from_numpy(self.x[trj_idx][T_idx,:]).float()
        dx = torch.from_numpy(self.dx[trj_idx][T_idx,:]).float()
        ddx = torch.from_numpy(self.ddx[trj_idx][T_idx,:]).float()
        cx = torch.from_numpy(self.cx[trj_idx][T_idx,:]).float()
        stabx = torch.from_numpy(self.stable_x[trj_idx][T_idx,:]).float()

        x_1_T = torch.from_numpy(self.x[trj_idx][T_idx+1:T_idx+self.H+1, :]).float()

        # x = self.x_data[index,:]
        # dx = self.dx_data[index, :]
        # ddx = self.ddx_data[index, :]
        # cx = self.cx_data[index, :]
        # stabx = self.stab_data[index, :]
        return x, dx, ddx, cx, stabx, x_1_T

