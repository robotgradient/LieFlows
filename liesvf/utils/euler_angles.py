import torch
import math
import numpy as np


def rot2eul(R):
    beta = -torch.arcsin(R[2,0])
    alpha = torch.atan2(R[2,1]/torch.cos(beta),R[2,2]/torch.cos(beta))
    gamma = torch.atan2(R[1,0]/torch.cos(beta),R[0,0]/torch.cos(beta))
    return torch.tensor((alpha, beta, gamma)).to(R)


def eul2rot(theta) :

    R = torch.tensor([[torch.cos(theta[1])*torch.cos(theta[2]),       torch.sin(theta[0])*torch.sin(theta[1])*torch.cos(theta[2]) - torch.sin(theta[2])*torch.cos(theta[0]),      torch.sin(theta[1])*torch.cos(theta[0])*torch.cos(theta[2]) + torch.sin(theta[0])*torch.sin(theta[2])],
                  [torch.sin(theta[2])*torch.cos(theta[1]),           torch.sin(theta[0])*torch.sin(theta[1])*torch.sin(theta[2]) + torch.cos(theta[0])*torch.cos(theta[2]),      torch.sin(theta[1])*torch.sin(theta[2])*torch.cos(theta[0]) - torch.sin(theta[0])*torch.cos(theta[2])],
                  [-torch.sin(theta[1]),                              torch.sin(theta[0])*torch.cos(theta[1]),                                                                    torch.cos(theta[0])*torch.cos(theta[1])]])

    return R.to(theta)

def eul_rot_batch(theta):
    r11 = torch.cos(theta[:,1])*torch.cos(theta[:,2])
    r12 = torch.sin(theta[:,0])*torch.sin(theta[:,1])*torch.cos(theta[:,2]) - torch.sin(theta[:,2])*torch.cos(theta[:,0])
    r13 = torch.sin(theta[:,1])*torch.cos(theta[:,0])*torch.cos(theta[:,2]) + torch.sin(theta[:,0])*torch.sin(theta[:,2])
    r21 = torch.sin(theta[:,2])*torch.cos(theta[:,1])
    r22 = torch.sin(theta[:,0])*torch.sin(theta[:,1])*torch.sin(theta[:,2]) + torch.cos(theta[:,0])*torch.cos(theta[:,2])
    r23 = torch.sin(theta[:,1])*torch.sin(theta[:,2])*torch.cos(theta[:,0]) - torch.sin(theta[:,0])*torch.cos(theta[:,2])
    r31 = -torch.sin(theta[:,1])
    r32 = torch.sin(theta[:,0])*torch.cos(theta[:,1])
    r33 = torch.cos(theta[:,0])*torch.cos(theta[:,1])

    R = torch.zeros(theta.shape[0],3,3).to(theta)
    R[:, 0, 0] = r11
    R[:, 0, 1] = r12
    R[:, 0, 2] = r13
    R[:, 1, 0] = r21
    R[:, 1, 1] = r22
    R[:, 1, 2] = r23
    R[:, 2, 0] = r31
    R[:, 2, 1] = r32
    R[:, 2, 2] = r33
    return R

def rot2quat(R):
    qw = torch.sqrt(1+ R[0,0] + R[1,1] + R[2,2])/2.
    qx = (R[2,1] - R[1,2])/(4*qw)
    qy = (R[0,2] - R[2,0])/(4*qw)
    qz = (R[1,0] - R[0,1])/(4*qw)
    return torch.tensor([qw,qx,qy,qz]).to(R)


def rot2quat_np(R):
    qw = np.sqrt(1+ R[0,0] + R[1,1] + R[2,2])/2.
    qx = (R[2,1] - R[1,2])/(4*qw)
    qy = (R[0,2] - R[2,0])/(4*qw)
    qz = (R[1,0] - R[0,1])/(4*qw)
    return np.array([qw,qx,qy,qz])


############################################################
def eul2rot_np(theta) :

    R = np.array([[np.cos(theta[1])*np.cos(theta[2]),       np.sin(theta[0])*np.sin(theta[1])*np.cos(theta[2]) - np.sin(theta[2])*np.cos(theta[0]),      np.sin(theta[1])*np.cos(theta[0])*np.cos(theta[2]) + np.sin(theta[0])*np.sin(theta[2])],
                  [np.sin(theta[2])*np.cos(theta[1]),           np.sin(theta[0])*np.sin(theta[1])*np.sin(theta[2]) + np.cos(theta[0])*np.cos(theta[2]),      np.sin(theta[1])*np.sin(theta[2])*np.cos(theta[0]) - np.sin(theta[0])*np.cos(theta[2])],
                  [-np.sin(theta[1]),                              np.sin(theta[0])*np.cos(theta[1]),                                                                    np.cos(theta[0])*np.cos(theta[1])]])

    return R


def rot2eul_np(R):
    beta = -np.arcsin(R[2,0])
    alpha = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta))
    gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta))
    return np.array((alpha, beta, gamma))


def euler_to_quaternion(roll, pitch, yaw):
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(
        yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(
        yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(
        yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(
        yaw / 2)

    return [qx, qy, qz, qw]


def quaternion_to_euler(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return [yaw, pitch, roll]

############################################################
from math import  atan2, cos, sin
from numpy import float64, hypot, zeros


def R_2_axis_angle(matrix):
    # Axes.
    axis = zeros(3, float64)
    axis[0] = matrix[2,1] - matrix[1,2]
    axis[1] = matrix[0,2] - matrix[2,0]
    axis[2] = matrix[1,0] - matrix[0,1]

    # Angle.
    r = hypot(axis[0], hypot(axis[1], axis[2]))
    t = matrix[0,0] + matrix[1,1] + matrix[2,2]
    theta = atan2(r, t-1)

    # Normalise the axis.
    axis = axis / r

    # Return the data.
    return axis, theta


def axis_angle_2_R(axis, angle):
    # Trig factors.
    ca = cos(angle)
    sa = sin(angle)
    C = 1 - ca

    # Depack the axis.
    x, y, z = axis

    # Multiplications (to remove duplicate calculations).
    xs = x*sa
    ys = y*sa
    zs = z*sa
    xC = x*C
    yC = y*C
    zC = z*C
    xyC = x*yC
    yzC = y*zC
    zxC = z*xC

    matrix = np.zeros((3,3))
    # Update the rotation matrix.
    matrix[0, 0] = x*xC + ca
    matrix[0, 1] = xyC - zs
    matrix[0, 2] = zxC + ys
    matrix[1, 0] = xyC + zs
    matrix[1, 1] = y*yC + ca
    matrix[1, 2] = yzC - xs
    matrix[2, 0] = zxC - ys
    matrix[2, 1] = yzC + xs
    matrix[2, 2] = z*zC + ca
    return matrix

def SE3_H_2_vector(matrix):
    R = matrix[:3,:3]
    axis, angle = R_2_axis_angle(R)
    xyz = matrix[:3,-1]
    w = axis*angle
    return np.concatenate((xyz, w),0)

def SE3_vector_2_H(v):
    xyz = v[:3]
    w = v[3:]
    angle = np.linalg.norm(w)
    axis = w/angle
    R = axis_angle_2_R(axis, angle)
    H = np.eye(4)
    H[:3,:3] = R
    H[:3,-1] = xyz
    return H


