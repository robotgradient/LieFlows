from liesvf.kinematics.robot_model import Robot
import os

base_dir = os.path.abspath(os.path.dirname(__file__) + '../../..')
robot_dir = os.path.join(base_dir, 'robots/darias_description/robots')
urdf_filename_r = os.path.join(robot_dir, 'darias_clean.urdf')
urdf_filename_l = os.path.join(robot_dir, 'darias_left_arm.urdf')

link_names_r = ['R_1_link', 'R_2_link', 'R_3_link', 'R_4_link','R_5_link','R_6_link', 'R_endeffector_link']
link_names_l = ['L_1_link', 'L_2_link', 'L_3_link', 'L_4_link','L_5_link','L_6_link', 'L_endeffector_link']

class DarIASArm(Robot):
    def __init__(self,arm='right'):
        if arm =='right':
            self.arm = arm
            self.link_names = link_names_r
            self.urdf_filename = urdf_filename_r
        elif arm == 'left':
            self.arm = arm
            self.link_names = link_names_l
            self.urdf_filename = urdf_filename_l

        super(DarIASArm, self).__init__(urdf_file=self.urdf_filename, link_name_list=self.link_names)


urdf_l_filename = os.path.join(robot_dir, 'darias_left.urdf')
link_l_names = ['L_1_link', 'L_2_link', 'L_3_link', 'L_4_link','L_5_link','L_6_link', 'L_endeffector_link']

class LeftDarIASArm(Robot):
    def __init__(self):
        super(LeftDarIASArm, self).__init__(urdf_file=urdf_l_filename, link_name_list=link_l_names)

if __name__ == '__main__':
    import pybullet as p

    time_step = 0.001
    p.connect(p.GUI_SERVER, 1234,
              options='--background_color_red=1. --background_color_green=1. --background_color_blue=1.')
    p.resetDebugVisualizerCamera(2.2, 55.6, -47.4, [0.04, 0.06, 0.31])
    p.setPhysicsEngineParameter(numSolverIterations=150)
    p.setTimeStep(time_step)
    p.setGravity(0, 0, 0)

    base_dir = os.path.abspath(os.path.dirname(__file__) + '../../..')
    robot_dir = os.path.join(base_dir, 'robots/darias_description/robots')
    urdf_filename = os.path.join(robot_dir, 'darias_left_arm.urdf')
    robot_file = urdf_filename
    robot = p.loadURDF(robot_file, flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)

    print('check')

