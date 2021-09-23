import os
import pybullet as p
import numpy as np
import time
from gym import spaces
from liesvf.environments.joint_slider import Debug_Joint_Slider

def solve_euler(q,dq,dt):
    return q+dq*dt
    
class PlanarBot():
    """
    Darias Simple Environment. Environment to evaluate the quality of prior composition for obstacle avoidance + Goto Target
    State dim 14; 7 q / 7 dq;
    Action dim (7,2); 7 q_des/ 7 dq_des

    """
    def __init__(self, time_step=1/240., slider = False, self_collision=False):


        self.qlimits = [[-2.96, 2.96], [-2.06, 2.06], [-2.06, 2.06], [-2.06, 2.06], [-2.06, 2.06]]
        self.JOINT_ID = [1,3,5,7,9]

        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(time_step)
        p.setGravity(0, 0, 0)

        basename = os.path.dirname(os.path.abspath(__file__ + '/../../'))
        self.robot_file = os.path.join(basename, "robots/planar_bot/urdf/4_link_planar_bot.urdf")
        #self.robot_file = os.path.join(basename, "robots/planar_bot/urdf/se2_planar_bot.urdf")

        self.robot = p.loadURDF(self.robot_file, flags=p.URDF_USE_SELF_COLLISION)


        ## Set Joint Limits ##
        self.collisions = self_collision

        if self.collisions:
            for i, joint_id in enumerate(self.JOINT_ID):
                p.changeDynamics(self.robot, joint_id, jointLowerLimit = self.qlimits[i][0], jointUpperLimit = self.qlimits[i][1])

            for i in range(2,10):
                p.setCollisionFilterPair(self.robot, i, self.robot, i + 1, 0)
                p.setCollisionFilterPair(self.robot, i, self.robot, i - 1, 0)
                p.setCollisionFilterPair(self.robot, i, self.robot, i - 2, 0)
                p.setCollisionFilterPair(self.robot, i, self.robot, i - 3, 0)
                p.setCollisionFilterPair(self.robot, i, self.robot, i - 4, 0)

        self.obstacle_file = os.path.join(basename, "robots/planar_bot/urdf/wall_with_hole.urdf")
        self.obstacle = p.loadURDF(self.obstacle_file)


        ##########################################
        self.qlimits = [[-2.96, 2.96], [-2.09, 2.09], [-2.96, 2.96], [-2.96, 2.96], [-2.96, 2.96]]
        self.JOINT_ID = [1,3,5,7,9]


        # self.qlimits = [[-2.96, 2.96], [-2.09, 2.09], [-2.96, 2.96]]
        # self.JOINT_ID = [1,2,3]

        ## Debugger ##
        self.SLIDER_ON = False
        self.q_slider = Debug_Joint_Slider(limits=self.qlimits, p=p, JOINT_ID=self.JOINT_ID, robot=self.robot)
        p.addUserDebugLine([0., 0., -0.189], [1.5, 0., -0.189], [1., 0., 0.])
        ##########################################
        while(slider):
            time.sleep(0.01)
            self.q_slider.read()
            self.q_slider.set()
            self._compute_reward()
            p.stepSimulation()


        joint_num = p.getNumJoints(self.robot)
        print("joint_num ", joint_num)
        ##############################################################


        ############## Robot Initial State #################
        self.q_home = [0., 0.659, - 1.5890, 1.4079, - 0.716]
        self.q_goal = np.array([0., 0.659, - 1.5890, 1.4079, - 0.716])

        self.qlimits = [[-2.96, 2.96], [-2.09, 2.09], [-2.96, 2.96], [-2.09, 2.09], [-2.96, 2.96], [-2.09, 2.09],
                   [-2.96, 2.96]]

        ## Observation-Action Space ##
        self.action_space = spaces.Box(-10, 10, shape=(10,), dtype='float32')
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(10,), dtype='float32')

        self.max_obs_dist_init = 0.01
        ## Threshold ##
        self.break_threshold = 0.08
        self.success_threshold = 0.02


    @property
    def dt(self):
        return 1/self.fq * self.n_substeps

    def reset(self, q0=None, std=None):
            # Initialize Pybullet Environment
            if std is None:
                std = 1.

            while True:
                if q0 is None:
                    for i, q_i in enumerate(self.q_home):
                        q_i = q_i + std*np.random.randn()
                        p.resetJointState(self.robot, self.JOINT_ID[i], q_i)
                    p.stepSimulation()
                else:
                    for i, q_i in enumerate(q0):
                        q_i = q_i + std*np.random.randn()
                        p.resetJointState(self.robot, self.JOINT_ID[i], q_i)
                    p.stepSimulation()

                ## check validity ##
                collision = self._compute_reward()
                if collision == 0 :
                    break

            ## Joints ##
            q = np.array([[p.getJointState(self.robot, self.JOINT_ID[0])[0], p.getJointState(self.robot, self.JOINT_ID[1])[0], p.getJointState(self.robot, self.JOINT_ID[2])[0],
                           p.getJointState(self.robot, self.JOINT_ID[3])[0], p.getJointState(self.robot, self.JOINT_ID[4])[0]]])
            
            # AL: fixed the dimensions for the returned state
            return [q, np.zeros_like(q)]

    def step(self, action):
        a_p = action[0]
        a_v = action[1]

        for i, q_i in enumerate(a_p):
            p.resetJointState(self.robot, self.JOINT_ID[i], q_i)


        self.q = np.array([[p.getJointState(self.robot, self.JOINT_ID[0])[0], p.getJointState(self.robot, self.JOINT_ID[1])[0], p.getJointState(self.robot, self.JOINT_ID[2])[0],
                           p.getJointState(self.robot, self.JOINT_ID[3])[0], p.getJointState(self.robot, self.JOINT_ID[4])[0]]])
        self.dq = a_v[None,:]

        obs = [self.q, self.dq]

        r = self._compute_reward()

        success = self._check_success()

        return obs, r, success

    def _compute_reward(self):
        ## Check Collision to the wall ##
        closest_points = p.getClosestPoints(self.robot, self.obstacle, distance = 0.001 )


        #closest_points_robot = p.getContactPoints(self.robot, self.robot)
        if self.collisions:
            closest_points_robot = p.getClosestPoints(self.robot, self.robot, distance = 0.000001 )
        else:
            closest_points_robot = []

        if len(closest_points)>0 or len(closest_points_robot)>15:
            return 1
        else:
            return 0

    def _check_success(self):
        ## Check if target is achieved ##
        return 0



if __name__ == '__main__':
    p.connect(p.GUI_SERVER, 1234,
              options='--background_color_red=1. --background_color_green=1. --background_color_blue=1.')
    p.resetDebugVisualizerCamera(2.2, 55.6, -47.4, [0.04, 0.06, 0.31])

    planar_bot = PlanarBot(slider=True)


