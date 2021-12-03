import os
import torch
from liesvf import loading_models, riemannian_manifolds, dynamic_systems
from liesvf.network_models.tangent_inn import SE2_models
from liesvf.utils import load_torch_file
from liesvf.robot_testers import se2_evaluation

## Device: CPU/GPU ##
device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')

## Testing parameters ##
## trained models: DynamicFlows/PieceWise
MODEL = 'DynamicFlows'

if __name__ == '__main__':
    ## Load Vector Field Model ##
    dirname = os.path.abspath(os.path.dirname(__file__)+'/../')
    if MODEL=='PieceWise':
        load_file = 'piecewise_se2.pth'
    elif MODEL =='DynamicFlows':
        load_file = 'dynamic_se2.pth'
    load_file = os.path.join(dirname, load_file)
    load_params = load_torch_file(load_file)

    if MODEL=='DynamicFlows':
        manifold = riemannian_manifolds.SE2Map()
        dynamics = dynamic_systems.ScaledLinearDynamics(dim = 3)
        bijective_mapping = SE2_models.SE2DynamicFlows()

        model = loading_models.MainManifoldModel(device=device, bijective_map = bijective_mapping, dynamics = dynamics,
                                                 manifold= manifold, H_origin=torch.eye(3))
        msvf = model.get_msvf()
        msvf.load_state_dict(load_params['params'])

    elif MODEL=='PieceWise':
        manifold = riemannian_manifolds.SE2Map()
        dynamics = dynamic_systems.ScaledLinearDynamics(dim = 3)
        bijective_mapping = SE2_models.SE2PieceWiseFlows()

        model = loading_models.MainManifoldModel(device=device, bijective_map = bijective_mapping, dynamics = dynamics,
                                                 manifold= manifold, H_origin=torch.eye(3))
        msvf = model.get_msvf()
        msvf.load_state_dict(load_params['params'])

    ## Run Policy in Test Environment ##
    se2_evaluation(msvf, device)




