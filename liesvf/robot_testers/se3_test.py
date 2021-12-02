import numpy as np
import pybullet as p
import os
import time
from tqdm import tqdm

import matplotlib.pyplot as plt

from liesvf.environments import DariasHandSimple
from liesvf.kinematics import DarIASArm
from liesvf.riemannian_manifolds.liegroups.numpy import SE3

import torch
from liesvf.utils import makedirs
from liesvf.utils import to_torch,to_numpy
from liesvf.utils.geometry import invert_H

q_inits = []
q_inits.append(np.array([0.2466, 1.234, 1.971, -1.0153, -1.81158, 0.20280, -1.17]))
q_inits.append(np.array([-0.6033, 1.202, 1.920, -1.69, -1.86, 0.078, -1.19]))
q_inits.append(np.array([0.447, 1.116, 2.01, -1.161, -1.910, 0.403, -1.209]))
q_inits.append(np.array([-0.3618, 1.089, 2.524, -1.584, -1.646, 0.085, -1.209]))
q_inits.append(np.array([0.50484, 1.3375860, 1.6416, -1.1276, -1.81, 0.108, -1.208]))
q_inits.append(np.array([-0.360, 1.231, 1.346, -2.0469, -1.845, 0.104, -1.207]))
q_inits.append(np.array([0.645, 1.444, 1.256, -0.7008, -1.824, 0.1562, -1.207]))
q_inits.append(np.array([0.6385, 1.439, 1.252, -1.152, -1.80, 0.160, -1.207]))
q_inits.append(np.array([0.1602, 1.564, 1.147, -1.163, -1.8047, 0.195, -1.2192]))
q_inits.append(np.array([0.2523, 1.634, 1.091, -1.440, -1.803, 0.1945, -1.219]))

class Controller():
    def __init__(self, policy, device, dt=1 / 240., dtype='float64', q_home = None):
        self.dt = dt
        self.dtype = dtype
        self.device = device

        self.link_id = 'R_endeffector_link'

        self.darias = DarIASArm()
        self.policy = policy

        self.H_origin = to_numpy(self.policy.H_origin)

        self.H_origin_inv = invert_H(self.H_origin)

        self.q_home = q_home

    def get_action(self, state):
        ## SE3 ##
        ## Action is applied in Position, so you might integrate your acceleration ##
        joint_poses = state[0, :7]
        joint_vels = state[0, 7:]

        self.darias.update_kindyn(joint_poses)
        x_ee_p = self.darias.link_fk(self.link_id)
        Htw = np.eye(4)
        Htw[:3, :3] = x_ee_p[1]
        Htw[:3, -1] = x_ee_p[0]
        J_ee_w = self.darias.link_worldJ(self.link_id)
        J_pinv = np.linalg.pinv(J_ee_w)
        ###################

        ### SE3 error ###
        Htl = np.matmul(self.H_origin_inv, Htw)
        Xe = SE3.from_matrix(Htl, normalize=True)
        error = Xe.log()
        error = np.linalg.norm(error)
        #################

        ## SE3 policy ##
        Htw = to_torch(Htw, device=self.device)
        dxw= self.policy(Htw)
        dxw = to_numpy(dxw)
        ################

        ## Map velocity to configuration Space ##
        dq = J_pinv @ dxw

        ## Add Null Space Control ##
        q_home = np.array([-0.361, 1.08, 2.52, -1.58, -1.64, 0.08, -1.20])
        dq_home = - (joint_poses - q_home)
        I = np.eye(7)
        J = self.darias.link_localJ(self.link_id)
        JTpinv = np.linalg.pinv(J.T)
        IJJ = (I - np.matmul(J.T, JTpinv))
        dq_null =  np.matmul(IJJ, dq_home)
        dq += dq_null
        ############################

        joint_poses, joint_vels = self.step(joint_poses, joint_vels, dq)
        return joint_poses, joint_vels, [error]

    def step(self, joint_poses, joint_vels, joint_accs):
        norm_v = np.linalg.norm(joint_accs) + 0.0001
        joint_accs = joint_accs / norm_v * np.tanh(norm_v / 2) * 2
        joint_poses = joint_poses + joint_accs * self.dt
        joint_vels = joint_accs
        return joint_poses, joint_vels


def se3_evaluation(policy, device):
    p.connect(p.GUI_SERVER, 1234,
              options='--background_color_red=1. --background_color_green=1. --background_color_blue=1.')
    p.resetDebugVisualizerCamera(2.0, 127.2, -49.4, [0.04, 0.06, 0.31])
    time_step = 1 / 250.
    env = DariasHandSimple(time_step=time_step)

    ## Controller ##
    controller = Controller(policy = policy, device=device)
    ################

    n_trials = 10
    horizon = 1500
    c = 0
    s = 0
    succeded = 0
    for itr in range(n_trials):

        print('Iteration: {}'.format(itr))
        ############################
        q0 = q_inits[itr]
        q0 = np.random.randn(7)
        state = env.reset(q0)
        p.addUserDebugLine([0., 0., -0.189], [1.5, 0., -0.189], [1., 0., 0.])

        for i in tqdm(range(horizon)):
            init = time.time()

            #### Get Control Action (Position Control)####
            a = controller.get_action(state)
            if a[2][0] < 0.01:
                succeded += 1
                break

            state, reward, done, success = env.step(a[:2])
            #############################

            end = time.time()
            time.sleep(np.clip(time_step - (end - init), 0, time_step))

    p.disconnect()

    print('Succeded Cases: ', succeded)


if __name__ == '__main__':
    import torch.nn as nn
    ## Device ##
    device = 'cpu'

    ## Policy ##
    class SillyPolicy(nn.Module):
        def __init__(self):
            super(SillyPolicy, self).__init__()
            self.H_origin = torch.eye(4)

        def forward(self, x):
            return torch.zeros(6)
    random_policy = SillyPolicy()

    ## Test Policy Model ##
    se3_evaluation(policy=random_policy, device=device)