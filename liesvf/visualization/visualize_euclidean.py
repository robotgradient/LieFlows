import numpy as np
import torch


def generate_second_order_traj(dynamics, x0, dx0, dt=0.01, context=None, steps=100):
    x_trj = x0.clone()
    dx_trj = dx0.clone()
    ddx_trj = torch.zeros(1, x0.shape[1])
    for i in range(steps):
        v, a = dynamics(x0, dx0, context=context)
        dx1 = dx0 + a * dt
        x1 = x0 + v * dt + 0.5 * a * dt ** 2

        x1 = x1.detach()
        dx1 = dx1.detach()
        a = a.detach()

        x_trj = torch.cat([x_trj, x1], 0)
        dx_trj = torch.cat([dx_trj, dx1], 0)
        ddx_trj = torch.cat([ddx_trj, a], 0)

        x0 = x1
        dx0 = dx1

    return x_trj, dx_trj, ddx_trj



