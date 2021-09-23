import numpy as np
import torch

def invert_H(H):
    dim = H.shape[0]
    if isinstance(H, np.ndarray):
        Hout = np.eye(dim)

    elif isinstance(H, torch.Tensor):
        Hout = torch.eye(dim).to(H)

    Hout[:-1,:-1] = H[:-1,:-1].T
    Hout[:-1,-1] = -H[:-1, :-1].T@H[:-1,-1]
    return Hout

