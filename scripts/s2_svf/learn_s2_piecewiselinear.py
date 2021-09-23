import os
import numpy as np
import torch
import torch.optim as optim
from liesvf.dataset import s2_lasa_dataset
from liesvf import dynamic_systems, riemannian_manifolds, loading_models
from liesvf.network_models.tangent_inn import S2_models
from liesvf.trainers import goto_train, fix_center
import matplotlib.pyplot as plt
from liesvf import visualization as vis
from torch.utils.tensorboard import SummaryWriter
from liesvf.trainers import regression_trainer


######### GPU/ CPU #############
device = 'cpu'#torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')

## Training parameters ##
percentage = .99
lr = 0.001
batch_size = 64
weight_decay = 0.00001
nr_epochs = 40000
clip_gradient=True
clip_value_grad=0.1

## Logger & Visualization parameters ##
letter = 'Sshape'
log_dir = 'runs/piecewise_s2'
dirname = os.path.abspath(os.path.dirname(__file__))
model_save_file = letter + '_piecewise_s2.pth'
model_save_file = os.path.join(dirname, model_save_file)

if __name__ == '__main__':
    data = s2_lasa_dataset.V_S2LASA(letter)
    dim = data.dim

    ######### Model #########
    manifold = riemannian_manifolds.S2Map()
    dynamics = dynamic_systems.ScaledLinearDynamics(dim = dim)
    bijective_mapping = S2_models.S2NeuralFlows()

    model = loading_models.MainManifoldModel(device=device, bijective_map = bijective_mapping, dynamics = dynamics, manifold= manifold)
    msvf = model.get_msvf()

    ########## Optimization ################
    params = list(msvf.parameters())
    optimizer = optim.Adamax(params, lr = lr, weight_decay= weight_decay)

    ## Prepare Training ##
    def loss_fn(model,x,y):
        return goto_train(model, x, y) + fix_center(msvf, dim=dim, device=device)

    def visualization_fn(model):
        plt.clf()
        min_max = [[-np.pi, -np.pi], [np.pi, np.pi]]
        vis.visualize_vector_field(model, device, min_max=min_max)

        for i in range(len(data.train_data)):
            trj = data.train_data[i]
            plt.plot(trj[:, 0], trj[:, 1])

        plt.ion()
        plt.show()
        plt.pause(0.001)

    logger = SummaryWriter(log_dir=log_dir)

    msvf, loss = regression_trainer(model=msvf, loss_fn = loss_fn, optimizer=optimizer, dataset= data.dataset, n_epochs=nr_epochs,
                       batch_size=batch_size, device=device, vis_fn=visualization_fn, vis_freq=100, logger= logger, model_save_file=None)

    logger.close()