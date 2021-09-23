import os
import pybullet as p
import numpy as np
from gym import error, spaces


def add_object(obs_state, p, color=[1.0, 0.0, 0.0, 1], size=0.05, object='Table', table_size=[0.5, 0.5, 0.5],
               radius=0.1, height=0.1):
    ### DRAW OBSTACLE ###
    obst_radius = size
    obst_collision = -1

    if object == 'Sphere':
        obst_visual = p.createVisualShape(p.GEOM_SPHERE, radius=obst_radius, length=0.005,
                                          rgbaColor=color)
    elif object == 'Table':
        obst_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=table_size, length=0.005,
                                          rgbaColor=color)
    elif object == 'cylinder':
        obst_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height,
                                          rgbaColor=color)

    obst = p.createMultiBody(baseMass=0.1,
                             baseCollisionShapeIndex=obst_collision,
                             baseVisualShapeIndex=obst_visual,
                             basePosition=obs_state)
    return obst


def solve_euler(q, dq, dt):
    return q + dq * dt


class DariasHandSimple():
    """
    Darias Simple Environment. Environment to evaluate the quality of prior composition for obstacle avoidance + Goto Target
    State dim 14; 7 q / 7 dq;
    Action dim (7,2); 7 q_des/ 7 dq_des

    """

    def __init__(self, reward_type=0, time_step=1 / 240.):
        """
        Constructor.

        """
        ######################## DARIAS ##################################
        self.EE_link = "R_endeffector_link"
        self.name_base_link = "world"
        self.JOINT_ID = [2, 3, 4, 5, 6, 7, 8]
        self.link_names = ["R_4_link", "R_3_link", "R_5_link", "R_2_link", self.EE_link]
        self.darias_right_arm_dict = {
            "R_1_link": [0],
            "R_2_link": [0, 1],
            "R_3_link": [0, 1, 2],
            "R_4_link": [0, 1, 2, 3],
            "R_5_link": [0, 1, 2, 3, 4],
            "R_6_link": [0, 1, 2, 3, 4, 5],
            "R_endeffector_link": [0, 1, 2, 3, 4, 5, 6],
        }

        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(time_step)
        p.setGravity(0, 0, 0)

        basename = os.path.dirname(os.path.abspath(__file__ + '/../../'))
        self.robot_file = os.path.join(basename, "robots/darias_description/robots/darias_with_hand_clean.urdf")

        self.robot = p.loadURDF(self.robot_file,
                                flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT | p.URDF_USE_INERTIA_FROM_FILE)
        joint_num = p.getNumJoints(self.robot)
        print("joint_num ", joint_num)
        ##############################################################

        ##  ADD TABLE ##
        table_pose = [0.5737, 0.0, 0.4]
        add_object(table_pose, p, color=[0.0, 0.0, 0.0, 1.])

        pot_pose = [0.6337, -0.05, .85]
        add_object(pot_pose, p, object='cylinder', color=[0.0, 0.0, 1.0, 1.])

        ############## Robot Initial State #################
        self.q_home = [0., 0.659, - 1.5890, 1.4079, - 0.716, - 0.967, 1.1528]
        self.qlimits = [[-2.96, 2.96], [-2.09, 2.09], [-2.96, 2.96], [-2.09, 2.09], [-2.96, 2.96], [-2.09, 2.09],
                        [-2.96, 2.96]]

        ## Observation-Action Space ##
        self.action_space = spaces.Box(-10, 10, shape=(7,), dtype='float32')
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(18,), dtype='float32')

        self.max_obs_dist_init = 0.01

        self.reward_type = reward_type

        ## VARIABLES ##
        self.DYNAMICS_ON = False

        ## Threshold ##
        self.break_threshold = 0.08
        self.success_threshold = 0.02

    @property
    def dt(self):
        return 1 / self.fq * self.n_substeps

    def reset(self, q0=None):
        # Initialize Pybullet Environment
        if q0 is None:
            for i, q_i in enumerate(self.q_home):
                q_i = q_i + np.random.randn() * 0.5
                p.resetJointState(self.robot, self.JOINT_ID[i], q_i)
            p.stepSimulation()
        else:
            for i, q_i in enumerate(q0):
                p.resetJointState(self.robot, self.JOINT_ID[i], q_i)
            p.stepSimulation()

        # AL: make sure the initial configuration is not in self-collision
        closest_points = p.getClosestPoints(self.robot, self.robot, self.max_obs_dist_init)
        n_links = p.getNumJoints(self.robot) - 1
        # if len(closest_points) > 3 * n_links - 2:
        #     print('WARNING: invalid initialization: in self-collision, please regenerate an initial configuration!')
        #     return np.nan * np.ones(7)

        q = np.array(
            [[p.getJointState(self.robot, 2)[0], p.getJointState(self.robot, 3)[0], p.getJointState(self.robot, 4)[0],
              p.getJointState(self.robot, 5)[0], p.getJointState(self.robot, 6)[0], p.getJointState(self.robot, 7)[0],
              p.getJointState(self.robot, 8)[0]]])

        # AL: fixed the dimensions for the returned state
        return np.concatenate((q, np.zeros_like(q)), 1)

    def step(self, action):
        a_p = action[0]
        a_v = action[1]

        for i, q_i in enumerate(a_p):
            if self.DYNAMICS_ON:
                p.setJointMotorControl2(self.robot, self.JOINT_ID[i], p.POSITION_CONTROL, targetPosition=q_i,
                                        targetVelocity=a_v[i],
                                        force=p.getJointInfo(self.robot, self.JOINT_ID[i])[10])
            else:
                p.resetJointState(self.robot, self.JOINT_ID[i], q_i)

        self.q = np.array(
            [[p.getJointState(self.robot, 2)[0], p.getJointState(self.robot, 3)[0], p.getJointState(self.robot, 4)[0],
              p.getJointState(self.robot, 5)[0], p.getJointState(self.robot, 6)[0], p.getJointState(self.robot, 7)[0],
              p.getJointState(self.robot, 8)[0]]])

        if self.DYNAMICS_ON:
            self.dq = np.array(
                [[p.getJointState(self.robot, 2)[1], p.getJointState(self.robot, 3)[1],
                  p.getJointState(self.robot, 4)[1],
                  p.getJointState(self.robot, 5)[1], p.getJointState(self.robot, 6)[1],
                  p.getJointState(self.robot, 7)[1],
                  p.getJointState(self.robot, 8)[1]]])
        else:
            self.dq = a_v[None, :]
        obs = np.concatenate((self.q, self.dq), 1)

        r = self._compute_reward()

        done = self._check_done(r)
        success = self._check_success(r)

        return obs, r, done, success

    def _compute_reward(self):
        return 0

    def _check_done(self, r):
        return False

    def _check_success(self, r):
        success = False
        return success


