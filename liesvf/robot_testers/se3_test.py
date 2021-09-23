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
        state = env.reset()
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