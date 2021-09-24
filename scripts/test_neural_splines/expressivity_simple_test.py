import torch
import torch.nn as nn
from liesvf import network_models as models
import numpy as np
import torch.optim as optim
from liesvf.dataset import SimpleDataset
from liesvf.trainers import regression_trainer, goto_train
from liesvf.utils import to_numpy
import matplotlib.pyplot as plt


## Parameters ###
percentage = .99
lr = 0.1
batch_size = 64
weight_decay = 0.00000000000000000000001
nr_epochs = 40000
clip_gradient=True
clip_value_grad=0.1
device = 'cpu'


## Data ##
n_points = 100
line = np.linspace(-0.8,0.8,n_points)
x = np.zeros((n_points,2))
x[:,1] = line
y = np.zeros((n_points,2))
y[:,0] = line
y[:,1] = line
dataset = SimpleDataset(y, x, device)


## Model ##
class Model(nn.Module):
    def __init__(self, depth=3, modeltype=0,  hidden_units = 60, dim=2, bins=10):
        super(Model, self).__init__()

        self.depth = depth
        self.hidden_units = hidden_units
        self.num_bins = bins

        self.dim = dim
        self.m_dim = 3

        self.layer = 'LinearSpline'
        self.main_map = models.DiffeomorphicNet(self.create_flow_sequence(), dim=self.dim)


    def forward(self, x, context=None):
        z = self.main_map(x, jacobian=False)
        return z

    def pushforward(self, x, context=None):
        z, J = self.main_map(x, jacobian=True)
        return z, J

    def create_flow_sequence(self):
        chain = []
        perm_i = np.linspace(0, self.dim-1, self.dim, dtype=int)
        for i in range(self.depth):
            perm_i = np.roll(perm_i,1)
            chain.append(self.main_layer())
            chain.append(models.Permutations(permutation=torch.Tensor(perm_i).to(torch.long)))
        chain.append(self.main_layer())
        return chain

    def main_layer(self):
        return models.LinearSplineLayer(num_bins=self.num_bins, features=self.dim, hidden_features=self.hidden_units)

model = Model()


## Train ##
if __name__ == '__main__':
    ########## Optimization ################
    params = list(model.parameters())
    optimizer = optim.Adamax(params, lr = lr, weight_decay= weight_decay)

    ## Prepare Training ##
    def loss_fn(model,x,y):
        return goto_train(model, x, y)

    def vis_fn(model):
        plt.clf()

        input = dataset.x
        y_target = dataset.y
        y_pred = model(input)

        x = to_numpy(input)
        yt = to_numpy(y_target)
        yp = to_numpy(y_pred)

        plt.plot(x[:,0], x[:,1], 'y')
        plt.plot(yt[:,0], y[:,1], 'r')
        plt.plot(yp[:,0], yp[:,1], 'g')
        plt.ion()
        plt.show()
        plt.pause(0.0001)



    msvf, loss = regression_trainer(model=model, loss_fn = loss_fn, optimizer=optimizer, dataset= dataset, n_epochs=nr_epochs,
                       batch_size=batch_size, device=device, vis_fn=vis_fn, vis_freq=100, logger= None, model_save_file=None)



