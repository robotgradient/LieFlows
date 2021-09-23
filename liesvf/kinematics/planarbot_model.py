from liesvf.kinematics.robot_model import Robot
import os

dirname = os.path.abspath(os.path.dirname(__file__ + '/../../../'))
dirname = os.path.join(dirname, 'data', 'PLANARBOT01_dataset')
robot_dir = os.path.join(dirname, 'robot')
urdf_filename = os.path.join(robot_dir, 'robot.urdf')
link_names = ['link_1', 'link_2', 'link_3', 'link_4', 'link_5', 'link_ee']

class PlanarBotKinematics(Robot):
    def __init__(self):
        self.link_names = link_names
        self.urdf_filename = urdf_filename
        super(PlanarBotKinematics, self).__init__(urdf_file=self.urdf_filename, link_name_list=self.link_names)


if __name__ == '__main__':
    import pybullet as p

    time_step = 0.001
    p.connect(p.GUI_SERVER, 1234,
              options='--background_color_red=1. --background_color_green=1. --background_color_blue=1.')
    p.resetDebugVisualizerCamera(2.2, 55.6, -47.4, [0.04, 0.06, 0.31])
    p.setPhysicsEngineParameter(numSolverIterations=150)
    p.setTimeStep(time_step)
    p.setGravity(0, 0, 0)

    planarbot = PlanarBotKinematics()
    print('check')

