import os, sys, time
import numpy as np
import torch
import torch.optim as optim
from liesvf.dataset import s2_lasa_dataset
from torch.utils.data import DataLoader

from liesvf import dynamic_systems, riemannian_manifolds, loading_models
from liesvf.network_models.tangent_inn.S2_models import baselines
from liesvf.trainers import goto_train, fix_center

import matplotlib.pyplot as plt
from liesvf import visualization as vis

percentage = .99
## optimization ##
lr = 0.0001
weight_decay = 0.1
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
    dynamics = dynamic_systems.ScaledLinearDynamics(dim=2)

    bijective_mapping = baselines.S2CouplingFlows()

    model = loading_models.MainManifoldModel(device=device, bijective_map=bijective_mapping, dynamics=dynamics, manifold=manifold)
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
                min_max = [[-np.pi, -np.pi],[np.pi, np.pi]]
                vis.visualize_vector_field(msvf, device, min_max=min_max)

                trj = data.train_data[0]
                plt.plot(trj[:,0], trj[:,1])

                plt.ion()
                plt.show()
                plt.pause(0.001)