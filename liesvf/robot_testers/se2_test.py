import numpy as np
import pybullet as p
import os
import time
from tqdm import tqdm
from liesvf.environments import PlanarBot

from liesvf.kinematics import PlanarBotKinematics
from liesvf.riemannian_manifolds.liegroups.numpy import SE2

import torch
from liesvf.utils import makedirs
from liesvf.utils import to_torch,to_numpy
from liesvf.utils.geometry import invert_H


class Controller():
    def __init__(self, policy, device, dt=1 / 240., dtype='float64', q_home = None):
        self.dt = dt
        self.dtype = dtype
        self.device = device

        self.link_id = 'link_ee'

        ## Load Planarbot  ##
        self.robot = PlanarBotKinematics()

        ## SE2 Policy ##
        self.policy = policy

        self.H_origin = to_numpy(self.policy.H_origin)
        self.H_origin_inv = invert_H(self.H_origin)

        self.q_home = q_home

    def get_action(self, state):
        ## SE3 ##
        ## Action is applied in Position, so you might integrate your acceleration ##
        joint_poses = state[0][0,:]
        joint_vels = state[1][0,:]

        self.robot.update_kindyn(joint_poses)
        x_ee_p = self.robot.link_fk(self.link_id)
        Htw = np.eye(3)
        Htw[:-1, :-1] = x_ee_p[1][:-1,:-1]
        Htw[:-1, -1] = x_ee_p[0][:-1]
        ###################

        ### SE2 error ###
        Htl = np.matmul(self.H_origin_inv, Htw)
        Xe = SE2.from_matrix(Htl, normalize=True)
        error = Xe.log()
        error = np.linalg.norm(error)
        #################

        ## SE2 policy ##
        Htw = to_torch(Htw, device=self.device)
        dxw= self.policy(Htw, already_tangent=False)
        dxw = to_numpy(dxw)
        ################

        ## Map velocity to configuration Space ##
        J = self.robot.link_worldJ('link_ee')[[0, 1, -1], :]
        J_pinv = np.linalg.pinv(J)
        dq = J_pinv @ dxw

        ## Add Null Space Control ##
        q_home = np.array([-1.39494436, 1.07762038, 0.95932704, 0.30511624, -0.93979831])
        dq_home = - (joint_poses - q_home)
        I = np.eye(5)
        J = self.robot.link_localJ(self.link_id)[[0, 1, -1], :]
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


def se2_evaluation(policy, device):
    p.connect(p.GUI_SERVER, 1234,
              options='--background_color_red=1. --background_color_green=1. --background_color_blue=1.')
    p.resetDebugVisualizerCamera(1.2, -90, -89.9, [0.24, 0.0, 0.31])

    time_step = 1 / 100.

    env = PlanarBot(time_step=time_step)

    ## Controller ##
    controller = Controller(policy = policy, device=device)
    ################

    q_goal = np.array([-1.39494436, 1.07762038, 0.95932704, 0.30511624, -0.93979831])
    ################

    #######
    collisions = 0.
    successed = 0.
    #######

    n_trials = 100
    horizon = 1000
    for itr in range(n_trials):
        print('Iteration: {}'.format(itr))
        print('Total collisions: {}'.format(collisions))
        print('Total Success: {}'.format(successed))


        state = env.reset(q0=q_goal, std=.5)


        p.addUserDebugLine([0., 0., -0.189], [1.5, 0., -0.189], [1., 0., 0.])

        for i in tqdm(range(horizon)):
            init = time.time()

            #### Get Control Action (Position Control)####
            a = controller.get_action(state)

            state, reward, success = env.step(a)
            #############################
            if reward:
                collisions += 1
                break

            if a[2][0]<0.1:
                successed +=1
                break

            # if i % 4 == 0:
            #     img = p.getCameraImage(1920, 980, renderer=p.ER_BULLET_HARDWARE_OPENGL)
            #
            #     file = 'image_{}_{}.png'.format(itr+10, i)
            #     save_file = os.path.join(img_dir, file)
            #     plt.imsave(save_file, img[2])

            end = time.time()
            time.sleep(np.clip(time_step - (end - init), 0, time_step))

    p.disconnect()

if __name__ == '__main__':
    import torch.nn as nn
    ## Device ##
    device = 'cpu'

    ## Policy ##
    class SillyPolicy(nn.Module):
        def __init__(self):
            super(SillyPolicy, self).__init__()
            self.H_origin = torch.eye(3)

        def forward(self, x, already_tangent=None):
            return torch.zeros(3)
    random_policy = SillyPolicy()

    ## Test Policy Model ##
    se2_evaluation(policy=random_policy, device=device)