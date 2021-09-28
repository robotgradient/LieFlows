import os
import torch
from liesvf import loading_models, riemannian_manifolds, dynamic_systems
from liesvf.network_models.tangent_inn import SE3_models
from liesvf.utils import load_torch_file
from liesvf.robot_testers import se3_evaluation

## Device: CPU/GPU ##
device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')

## Testing parameters ##
## trained models: DynamicFlows/Neural Splines
MODEL = 'Neural Splines'


if __name__ == '__main__':
    ## Load Vector Field Model ##
    dirname = os.path.abspath(os.path.dirname(__file__)+'/../')
    if MODEL=='Neural Splines':
        load_file = 'neural_se3.pth'
    elif MODEL =='DynamicFlows':
        load_file = 'dynamic_se3.pth'
    load_file = os.path.join(dirname, load_file)
    load_params = load_torch_file(load_file)
    if MODEL=='Neural Splines':
        manifold = riemannian_manifolds.SE3Map()
        dynamics = dynamic_systems.ScaledLinearDynamics(dim = 6)
        bijective_mapping = SE3_models.SE3NeuralFlows()

        model = loading_models.MainManifoldModel(device=device, bijective_map = bijective_mapping, dynamics = dynamics, manifold= manifold)
        msvf = model.get_msvf()
        msvf.load_state_dict(load_params['params'])
    elif MODEL=='DynamicFlows':
        manifold = riemannian_manifolds.SE3Map()
        dynamics = dynamic_systems.ScaledLinearDynamics(dim = 6)
        bijective_mapping = SE3_models.SE3DynamicFlows()

        model = loading_models.MainManifoldModel(device=device, bijective_map = bijective_mapping, dynamics = dynamics, manifold= manifold)
        msvf = model.get_msvf()
        msvf.load_state_dict(load_params['params'])

    ## Run Policy in Test Environment ##
    se3_evaluation(msvf, device)




