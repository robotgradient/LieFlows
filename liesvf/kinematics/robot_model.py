import pinocchio as pin
import os

import numpy as np
from numpy.linalg import norm, solve

eps = 1e-4
IT_MAX = 1000
DT = 1e-1
damp = 1e-12

class Robot(object):
    def __init__(self, urdf_file, link_name_list):
        self.model = pin.buildModelFromUrdf(urdf_file)
        self.data = self.model.createData()

        self.links_names = link_name_list
        self.links_ids = [
            self.model.getFrameId(link_name)
            for link_name in self.links_names
        ]

    def update_kindyn(self, q):
        pin.computeJointJacobians(self.model, self.data, q)
        pin.framesForwardKinematics(self.model, self.data, q)

    def links_fk(self, rotation=False):
        if rotation:
            return [
                [np.asarray(self.data.oMf[link_id].translation).reshape(-1).tolist(),
                np.asarray(self.data.oMf[link_id].rotation)]
                for link_id in self.links_ids
            ]
            # return [
            #     [np.asarray(self.data.oMf[link_id].translation).reshape(-1).tolist(),
            #      rot2eul_np(np.asarray(self.data.oMf[link_id].rotation))]
            #     for link_id in self.links_ids
            # ]
        else:
            return [
                np.asarray(self.data.oMf[link_id].translation).reshape(-1).tolist()
                for link_id in self.links_ids
            ]

    def links_J(self):
        J_list = []
        for link_id in self.links_ids:
            Ji = pin.getFrameJacobian(
                self.model,
                self.data,
                link_id,
                pin.ReferenceFrame.WORLD,
            )
            J_list.append(Ji)
        return J_list

    def link_fk(self, name_idx):
        return [
            np.asarray(self.data.oMf[self.model.getFrameId(name_idx)].translation).reshape(-1).tolist(),
            np.asarray(self.data.oMf[self.model.getFrameId(name_idx)].rotation)
        ]

    def link_J(self, name_idx):
        return pin.getFrameJacobian(
                self.model,
                self.data,
                self.model.getFrameId(name_idx),
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )

    def link_localJ(self, name_idx):
        return pin.getFrameJacobian(
                self.model,
                self.data,
                self.model.getFrameId(name_idx),
                pin.ReferenceFrame.LOCAL
            )

    def link_worldJ(self, name_idx):
        return pin.getFrameJacobian(
                self.model,
                self.data,
                self.model.getFrameId(name_idx),
                pin.ReferenceFrame.WORLD
            )

    def ik(self, name_idx, q, Hdes):
        oMdes = pin.SE3(Hdes[:-1,:-1], Hdes[:-1,-1])

        name_idx = self.model.getFrameId(name_idx)
        i = 0
        while True:
            pin.forwardKinematics(self.model, self.data, q)
            dMf = oMdes.actInv(self.data.oMf[name_idx])
            err = pin.log(dMf).vector
            if norm(err) < eps:
                success = True
                break
            if i >= IT_MAX:
                success = False
                break
            J = pin.computeJointJacobian(self.model, self.data, q, name_idx)
            v = - J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))
            q = pin.integrate(self.model, q, v * DT)
            if not i % 10:
                print('%d: error = %s' % (i, err.T))
            i += 1
        return q


if __name__ == '__main__':
    base_dir = os.path.abspath(os.path.dirname(__file__) + '../../..')
    robot_dir = os.path.join(base_dir, 'robots/darias_description/robots')
    urdf_filename = os.path.join(robot_dir, 'darias_clean.urdf')

    link_names = ['R_1_link', 'R_2_link', 'R_3_link']

    robot = Robot(urdf_file=urdf_filename, link_name_list=link_names)
    q = np.random.rand(7)
    robot.update_kindyn(q=q)

    print(robot)




