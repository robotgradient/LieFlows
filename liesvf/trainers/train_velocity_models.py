import torch

def goto_train(flow, x, y):
    ## Separate Data ##
    dx = y
    ## Evolve dynamics backwards ##
    dx_pred = flow(x)
    #### Complete Loss is composed between the stable loss and the trajectory loss
    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(dx_pred,dx)
    return loss


def fix_center(flow, dim = 2, device='cpu', x=None):
    if x is None:
        x = torch.zeros(1, dim).float()
    x.requires_grad = True
    z = torch.zeros(1, dim).float()
    x = x.to(device)
    z = z.to(device)
    z_pred = flow.pushforward(x)
    #### Complete Loss is composed between the stable loss and the trajectory loss
    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(z_pred, z)
    return loss