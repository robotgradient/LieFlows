import os, sys, time
import numpy as np
import torch
import torch.optim as optim
from liesvf.dataset import s2_lasa_dataset
from torch.utils.data import DataLoader

from liesvf import dynamic_systems, riemannian_manifolds, loading_models
from liesvf.network_models.tangent_inn import S2_models
from liesvf.trainers import goto_train, fix_center

import matplotlib.pyplot as plt
from liesvf import visualization as vis

percentage = .99
## optimization ##
lr = 0.001
weight_decay = 0.000001
#weight_decay = 0.01
## training variables ##
nr_epochs = 40000

######### GPU/ CPU #############
device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
plot_resolution = 0.01


if __name__ == '__main__':
    filename = 'Sshape'
    data = s2_lasa_dataset.V_S2LASA(filename)
    dim = data.dim
    batch_size =100
    params = {'batch_size': batch_size, 'shuffle': True}
    dataloader = DataLoader(data.dataset, **params)

    ######### Model #########
    manifold = riemannian_manifolds.S2()
    dynamics = dynamic_systems.ScaledLinearDynamics(dim = 2)
    bijective_mapping = S2_models.S2DynamicFlows()

    model = loading_models.MainManifoldModel(device=device, bijective_map = bijective_mapping, dynamics = dynamics, manifold= manifold)
    msvf = model.get_msvf()

    ########## Optimization ################
    params = list(msvf.parameters())
    optimizer = optim.Adamax(params, lr = lr, weight_decay= weight_decay)
    #######################################
    for i in range(nr_epochs):
        ## Training ##
        for local_x, local_y in dataloader:

            local_x = local_x.to(device)
            local_x.requires_grad = True
            local_y = local_y.to(device)

            optimizer.zero_grad()
            loss = goto_train(msvf, local_x, local_y) + 10*fix_center(msvf, dim=dim, device=device)
            loss.backward(retain_graph=True)
            optimizer.step()

        ## Validation ##
        if i%60 == 0:
            print(loss)

            PLOT_2D = True
            if PLOT_2D:
                #fig = plt.figure(1)
                plt.clf()
                min_max = [[-np.pi/2, -np.pi/2],[np.pi/2, np.pi/2]]
                vis.visualize_vector_field(msvf, device, min_max=min_max)

                trj = data.train_data[0]
                plt.plot(trj[:,0], trj[:,1])

                plt.ion()
                plt.show()
                plt.pause(0.001)

            # PLOT_S2= False
            # if PLOT_S2:
            #     pv.set_plot_theme("document")
            #     #p = pv.Plotter(off_screen=True)
            #     p = pv.Plotter()
            #
            #     sphere = vis.visualize_sphere(p)
            #     vis.visualize_s2_tangent_trajectories(p,data.train_data[0])
            #
            #     vis.visualize_s2_angle_vector_field(p, iflow, torch=True, device=device)
            #
            #     p.show()
                # p.show(screenshot='plot_{}.png'.format(i))
                # plt.imshow(p.image)
                # plt.show()


        ## Save Model ##
        # if i%10==0:
        #     dirname = os.path.abspath(os.path.dirname(__file__+'/../../../../'))
        #     folder = 'models/s2_models'
        #     dirname = os.path.join(dirname, folder)
        #     save_file = '00_s2_lasa_model_{}.pth'.format(i)
        #     filename = os.path.join(dirname, save_file)
        #     s2_model.save_model(filename)