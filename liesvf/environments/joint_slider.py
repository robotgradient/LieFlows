
class Debug_Joint_Slider():
    def __init__(self, limits, p, JOINT_ID, robot):
        self.q_ids = []
        self.p = p
        self.JOINT_ID = JOINT_ID
        self.robot = robot

        for i, limit in enumerate(limits):
            self.q_ids.append(p.addUserDebugParameter(paramName='Joint'+str(i), rangeMin=limit[0], rangeMax=limit[1], startValue=0))

    def read(self):
        q = []
        for q_id in self.q_ids:
            q.append(self.p.readUserDebugParameter(q_id))
        return q

    def set(self):
        q = self.read()
        for i, q_i in enumerate(q):
            self.p.resetJointState(self.robot, self.JOINT_ID[i], q_i)
