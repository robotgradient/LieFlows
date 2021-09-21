import os
from liesvf.kinematics import Robot
import numpy as np
from liesvf.riemannian_manifolds.liegroups.numpy import SE2

import torch
from liesvf.utils import to_torch, to_numpy

dirname = os.path.abspath(os.path.dirname(__file__ + '/../../../'))
dirname = os.path.join(dirname, 'robots', 'planar_bot')
robot_dir = os.path.join(dirname, 'robot')
urdf_filename = os.path.join(robot_dir, 'robot.urdf')
link_names = ['link_1', 'link_2', 'link_3', 'link_4', 'link_5', 'link_ee']
robot = Robot(urdf_file=urdf_filename, link_name_list=link_names)

def se2flow_q_policy(q, flow, H_origin = np.eye(3), H_origin_inv=np.eye(3), device = torch.device('cpu')):

    q = q[0,:]
    robot.update_kindyn(q)
    xyz, rot = robot.link_fk('link_ee')

    H = np.eye(3)
    H[:2, :2] = rot[:2, :2]
    H[:2, -1] = xyz[:2]

    ## SE2 ##
    Htl = np.matmul(H_origin_inv, H)
    Xe = SE2.from_matrix(Htl)
    xe = Xe.log()

    xe = to_torch(xe[None,:], device)
    #print('current state: {}'.format(xe))
    ve = flow(xe)
    if ve.dim()==2:
        ve = ve[0,:]
    ve = to_numpy(ve)

    A = SE2.from_matrix(H_origin)
    Adj_lw = A.adjoint()
    ve_w = np.matmul(Adj_lw, ve)

    J = robot.link_worldJ('link_ee')[[0, 1, -1], :]
    J_pinv = np.linalg.pinv(J)
    dq = J_pinv @ ve_w
    return dq[None,:] , [to_numpy(xe) , ve]


def qflow_policy(q, flow, q_origin = np.zeros(5), device = torch.device('cpu')):
    q = q[0,:]
    q = q - q_origin

    q = to_torch(q[None,:], device)
    vq = flow(q)
    vq = to_numpy(vq)

    return vq[None,:] , [to_numpy(q) , vq]


def se2_trj_from_q(q_trj):
    H_trj = np.zeros((0, 3, 3))
    for t in range(q_trj.shape[0]):
        qt = q_trj[t, :]
        robot.update_kindyn(qt)
        xyz, rot = robot.link_fk('link_ee')
        H = np.eye(3)
        H[:2, :2] = rot[:2, :2]
        H[:2, -1] = xyz[:2]

        H_trj = np.concatenate((H_trj, H[None, ...]), 0)
    return H_trj


def q_to_SE2(q):
    robot.update_kindyn(q)
    xyz, rot = robot.link_fk('link_ee')

    H = np.eye(3)
    H[:2, :2] = rot[:2, :2]
    H[:2, -1] = xyz[:2]
    return H


