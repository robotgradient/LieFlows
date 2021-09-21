import os, sys, time
import json
import numpy as np
import scipy.io as spio
import torch
from liesvf.dataset.generic_dataset import VDataset
from liesvf import visualization as vis

directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..','data')) + '/LASA_dataset/'
directory_s2 = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..','data')) + '/LASA_S2_dataset/'


class V_S2LASA():
    def __init__(self, filename, PLOT=False):
        ## Define Variables and Load trajectories ##
        self.filename = filename
        self.dim = 2
        self.dt = 0.01

        mat = spio.loadmat(directory + filename + '.mat', squeeze_me=True)
        self.trajs_real=[]
        for demo_i in mat['demos']:
            x = demo_i[0]
            y = demo_i[1]
            tr_i = np.stack((x,y))
            self.trajs_real.append(tr_i.T)
        self.trajs_np = np.asarray(self.trajs_real)

        self.n_trajs = self.trajs_np.shape[0]
        self.trj_length = self.trajs_np.shape[1]

        ## Normalize trajectories ##
        self.trajs_normal, self.min_trjs, self.max_trjs, self.goal = self.normalize(self.trajs_np)
        self.init_position = self.trajs_normal[:,0,:]

        ## Build Train Dataset
        r = np.linalg.norm(self.trajs_normal, axis=-1)
        r_max = np.max(r)
        self.alpha = (np.pi-0.1)/r_max
        self.alpha = 1.
        self.train_data = []
        for i in range(self.trajs_normal.shape[0]):
            self.train_data.append(self.alpha*self.trajs_normal[i, ...])
        self.dataset = VDataset(trajs=self.train_data, dt = self.dt)

        ## VISUALIZE ##
        if PLOT:
            import pyvista as pv
            pv.set_plot_theme("document")
            p = pv.Plotter()
            sphere = vis.visualize_sphere(p)
            vis.visualize_s2_tangent_trajectories(p,self.train_data[0])
            def policy(x):
                return -x
            vis.visualize_s2_vector_field(p,policy)
            p.show()
            pv.close_all()


    def normalize(self, trjs):
        min_trjs = np.min(trjs)
        max_trjs = np.max(trjs)

        trjs_normalized = (trjs - min_trjs)/(max_trjs - min_trjs)
        goal = np.mean(trjs_normalized[:,-1,:], 0)
        trjs_normalized_centered = trjs_normalized - goal
        return trjs_normalized_centered, min_trjs, max_trjs, goal

    def unormalize(self, trjs):
        trjs_uncentered = trjs + self.goal
        trjs_unnormalized_uncentered = trjs_uncentered * ((self.max_trjs - self.min_trjs)) + self.min_trjs
        return trjs_unnormalized_uncentered


class S2_LASA():
    def __init__(self, filename, PLOT=False):
        ## Define Variables and Load trajectories ##
        self.filename = filename
        self.dim = 3
        self.dt = 0.01

        load_file = os.path.join(directory_s2, self.filename+'.txt')
        with open(load_file) as json_file:
            data = json.load(json_file)

            xyz = data['xyz']
            dxyz = data['d_xyz']
            tangent_x = data['x']
            tangent_dx  = data['dx']

        self.xyz = np.array(xyz)
        self.dxyz = np.array(dxyz)
        self.tangent_x = data['x']
        self.tangent_dx = data['dx']

        self.goal = np.array([0.,0.,1.])


        ## Build Train Dataset
        self.train_data = []
        self.d_train_data = []
        for i in range(self.xyz.shape[0]):
            self.train_data.append(self.xyz[i, ...])
            self.d_train_data.append(self.dxyz[i,...])

        self.dataset = VDataset(trajs=self.train_data, v_trj=self.d_train_data, dt = self.dt)

        ## VISUALIZE ##
        if PLOT:
            import pyvista as pv
            pv.set_plot_theme("document")
            p = pv.Plotter()
            for trj, v_trj in zip(self.train_data, self.d_train_data):

                T = trj.shape[0]
                spline = pv.Spline(trj, np.clip(T, 0, 400))
                p.add_mesh(spline, line_width=4, color="r")

                grid = pv.PolyData()
                grid.points = trj
                grid.vectors = v_trj/100.
                p.add_mesh(grid.arrows, lighting=True)
            p.show()
            pv.close_all()


if __name__ == "__main__":
    ##### S2_models ####
    filename = 'CShape'
    device = torch.device('cpu')
    lasa = V_S2LASA(filename, PLOT=True)
    print(lasa)

    trj = lasa.trajs_normal[0,:,:]

    ##### S2_models ####
    filename = 'S2_Spoon'
    device = torch.device('cpu')
    lasa = S2_LASA(filename, PLOT=True)
    print(lasa)
