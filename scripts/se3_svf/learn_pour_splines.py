import os, sys, time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from liesvf.dataset import pouring_dataset

from liesvf import loading_models, riemannian_manifolds, dynamic_systems
from liesvf.trainers import goto_train, fix_center, regression_trainer
from liesvf.network_models.tangent_inn import SE3_models

from liesvf import visualization as vis
from liesvf.utils import to_torch

######### GPU/ CPU #############
device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')

## Training Parameters ##
data_percentage = .1
batch_size = 100
lr = 0.0001
weight_decay = 0.00001
nr_epochs = 40000
clip_gradient=True
clip_value_grad=0.1

## Logger & Visualization parameters ##
visualized_trajs = 2
log_dir = 'runs/neural_spline_1'

dirname = os.path.abspath(os.path.dirname(__file__))
model_save_file = 'neural_se3.pth'
model_save_file = os.path.join(dirname, model_save_file)

if __name__ == '__main__':
    data = pouring_dataset.POURING_SE3()
    dim = data.dim

    ######### Model #########
    manifold = riemannian_manifolds.SE3Map()
    dynamics = dynamic_systems.ScaledLinearDynamics(dim = dim)
    bijective_mapping = SE3_models.SE3NeuralFlows()
    H_origin = to_torch(data.goal_H, device)

    model = loading_models.MainManifoldModel(device=device, bijective_map = bijective_mapping, dynamics = dynamics,
                                             manifold= manifold, H_origin=H_origin)
    msvf = model.get_msvf()

    ########## Optimization ################
    params = list(msvf.parameters())
    optimizer = optim.Adamax(params, lr = lr, weight_decay= weight_decay)
    #######################################

    ## Prepare Training ##
    def loss_fn(model,x,y):
        return goto_train(model, x, y) + fix_center(msvf, dim=dim, device=device)

    def visualization_fn(model):
        n_trjs = len(data.main_trajs)
        idx = np.random.randint(0, n_trjs, visualized_trajs)
        val_trjs = [data.main_trajs[i] for i in idx]
        vis.visualize_trajectories(val_trajs=val_trjs, vector_field=model, device=device)

    logger = SummaryWriter(log_dir=log_dir)

    msvf, loss = regression_trainer(model=msvf, loss_fn = loss_fn, optimizer=optimizer, dataset= data.dataset, n_epochs=nr_epochs,
                       batch_size=batch_size, device=device, vis_fn=None, vis_freq=30, logger= logger, model_save_file=model_save_file)

    logger.close()