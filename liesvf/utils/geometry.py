import numpy as np
import torch

def invert_H(H):

    if isinstance(H, np.ndarray):
        Hout = np.eye(4)

    elif isinstance(H, torch.Tensor):
        Hout = torch.eye(4).to(H)

    Hout[:3,:3] = H[:3,:3].T
    Hout[:3,-1] = -H[:3,:3].T@H[:3,-1]
    return Hout

