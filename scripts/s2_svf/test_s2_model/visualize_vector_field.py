import os
import torch
from liesvf import loading_models, riemannian_manifolds, dynamic_systems
from liesvf.network_models.tangent_inn import S2_models
from liesvf.utils import load_torch_file, to_torch,to_numpy
from liesvf import visualization as vis



## Device: CPU/GPU ##
device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')

## Testing parameters ##
## trained models: DynamicFlows/Neural Splines
letter = 'NShape'
MODEL = 'DynamicFlows'


if __name__ == '__main__':
    ## Load Vector Field Model ##
    dirname = os.path.abspath(os.path.dirname(__file__)+'/../')

    load_file = letter + '_dynamic_s2.pth'
    load_file = os.path.join(dirname, load_file)
    load_params = load_torch_file(load_file)


    manifold = riemannian_manifolds.S2Map()
    dynamics = dynamic_systems.ScaledLinearDynamics(dim = 2)
    bijective_mapping = S2_models.S2DynamicFlows()

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
    dx = to_numpy(dx)

    sphere.vectors = dx * 0.05
    p.add_mesh(sphere.arrows, color='black')

    ## Trajectories From Data ##
    data = s2_lasa_dataset.V_S2LASA(letter)

    for i in range(len(data.train_data)):
        trj = data.train_data[i]
        vis.visualize_s2_tangent_trajectories(p,trj)

    p.show()

