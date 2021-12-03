import math
import os
import torch
import numpy as np

from liesvf import loading_models, riemannian_manifolds, dynamic_systems
from liesvf.network_models.tangent_inn import S2_models
from liesvf.utils import load_torch_file, to_torch,to_numpy
from liesvf import visualization as vis



## Device: CPU/GPU ##
device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
device = 'cpu'

## Testing parameters ##
## trained models: DynamicFlows/Neural Splines/MADE
letter = 'NShape'
MODEL = 'DynamicFlows'# 'MADE' # 'Neural Splines'

if __name__ == '__main__':
    ## Load Vector Field Model ##
    dirname = os.path.abspath(os.path.dirname(__file__)+'/../')

    if MODEL=='DynamicFlows':
        load_file = letter + '_dynamic_s2.pth'
    elif MODEL=='Neural Splines':
        load_file = letter + '_piecewise_s2.pth'
    elif MODEL=='MADE':
        load_file = letter + '_made_piecewise_s2.pth'


    load_file = os.path.join(dirname, load_file)
    load_params = load_torch_file(load_file)


    manifold = riemannian_manifolds.S2Map()
    dynamics = dynamic_systems.ScaledLinearDynamics(dim = 2)

    if MODEL=='DynamicFlows':
        bijective_mapping = S2_models.S2DynamicFlows()
    elif MODEL=='Neural Splines':
        bijective_mapping = S2_models.S2NeuralFlows()
    elif MODEL=='MADE':
        bijective_mapping = S2_models.S2NeuralFlows(made='made')

    model = loading_models.MainManifoldModel(device=device, bijective_map = bijective_mapping, dynamics = dynamics, manifold= manifold)
    msvf = model.get_msvf()
    msvf.load_state_dict(load_params['params'])

    ## Visualize Vector Field ##
    import pyvista as pv
    from liesvf.dataset import s2_lasa_dataset

    pv.set_plot_theme("document")
    p = pv.Plotter()

    sphere = vis.visualize_sphere(p)

    p_xyz = torch.Tensor(sphere.points.tolist()).to(device)
    dx = msvf(p_xyz)
    dx_norm = dx.pow(2).sum(-1).pow(.5)
    dx = to_numpy(dx)
    dx_norm = to_numpy(dx_norm)
    dx = dx/dx_norm[:,None]
    #alpha_clip = np.clip(dx_norm, 0, 1)
    #dx = dx*alpha_clip[:,None]

    sphere.vectors = dx*0.1
    p.add_mesh(sphere.arrows, color='black')

    ## Trajectories From Data ##
    data = s2_lasa_dataset.V_S2LASA(letter)

    for i in range(len(data.train_data)):
        trj = data.train_data[i]
        vis.visualize_s2_tangent_trajectories(p,trj)

    ## Generate Trajectories
    N = 100
    r0 = torch.ones(N)*(math.pi - 0.1)
    theta0 = torch.linspace(-math.pi, math.pi, N)
    x0 = r0*torch.cos(theta0)
    y0 = r0*torch.sin(theta0)
    xy = torch.cat((x0[:, None], y0[:,None]),1)
    trjs = msvf.generate_trj(x0=xy, T=200)
    trjs = to_numpy(trjs)
    for i in range(trjs.shape[1]):
        trj = trjs[:, i, :]
        vis.visualize_s2_tangent_trajectories(p,trj, color='b')


    p.show()

