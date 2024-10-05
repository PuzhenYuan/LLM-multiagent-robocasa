import numpy as np

class PIDController:
    """
    PID controller class
    """
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = None
        self.previous_error = None

    def compute(self, current_value=None, target_value=None, error=None):
        assert (current_value is not None and target_value is not None) or error is not None
        if error is None:
            error = target_value - current_value
        self.integral = error if self.integral is None else self.integral + error
        derivative = 0 if self.previous_error is None else (error - self.previous_error)
        self.previous_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative
    
    def reset(self):
        self.integral = None
        self.previous_error = None

# global PID controller parameters, TODO: further tune these parameters
kp_base_pos = 5.0
ki_base_pos = 0.01
kd_base_pos = 0.01

kp_base_ori = 1.5
ki_base_ori = 0.05
kd_base_ori = 0

kp_eef_pos = 2.0
ki_eef_pos = 0.02
kd_eef_pos = 0

kp_eef_axisangle = 1.0
ki_eef_axisangle = 0.01
kd_eef_axisangle = 0

# global pid controllers
pid_base_pos_ctlr = PIDController(kp=kp_base_pos, ki=ki_base_pos, kd=kd_base_pos)
pid_base_ori_ctlr = PIDController(kp=kp_base_ori, ki=ki_base_ori, kd=kd_base_ori)
pid_eef_pos_ctlr = PIDController(kp=kp_eef_pos, ki=ki_eef_pos, kd=kd_eef_pos)
pid_eef_axisangle_ctlr = PIDController(kp=kp_eef_axisangle, ki=ki_eef_axisangle, kd=kd_eef_axisangle)


def map_action(action, base_ori=None):
    """
    Map action from world frame to robot body frame.
    The action can be 3-dim for eef pos and axisangle control or 2-dim for base pos control.
    
    Args:
        action (np.array): action vector in world frame
        base_ori (np.array): base orientation in world frame

    Returns:
        np.array: action vector in robot body frame
    """
    if not isinstance(action, np.ndarray):
        action = np.array([action])

    if action.shape[0] == 1:
        return action
    
    # base pos control
    elif action.shape[0] == 2:
        Fx = action[0]
        Fy = action[1]
        fx = Fx * np.cos(base_ori) + Fy * np.sin(base_ori)
        fy = -Fx * np.sin(base_ori) + Fy * np.cos(base_ori)
        return np.array([fx, fy])
    
    # eef pos and axisangle control
    elif action.shape[0] == 3:
        Fx = action[0]
        Fy = action[1]
        Fz = action[2]
        fx = Fx * np.cos(base_ori) + Fy * np.sin(base_ori)
        fy = -Fx * np.sin(base_ori) + Fy * np.cos(base_ori)
        return np.array([fx, fy, Fz])
    
    else:
        raise ValueError

last_grasp0 = False
last_grasp1 = False # TODO: more elegant way to handle this?

def create_action(eef_pos=None, eef_axisangle=None, base_pos=None, base_ori=None, base_height=None, grasp=None, id=0):
    """
    Create action vector for single mobile robot control.

    Args:
        eef_pos (np.array, optional): eef pos action. Defaults to None.
        eef_axisangle (np.array, optional): eef axisangle action. Defaults to None.
        base_pos (np.array, optional): base pos action. Defaults to None.
        base_ori (np.array, optional): base ori action. Defaults to None.
        grasp (bool, optional): whether to grasp. Defaults to None.

    Returns:
        np.array: action vector
    """
    
    eef_pos = np.zeros(3) if eef_pos is None else eef_pos
    eef_axisangle = np.zeros(3) if eef_axisangle is None else eef_axisangle
    base_pos = np.zeros(2) if base_pos is None else base_pos
    base_ori = np.zeros(1) if base_ori is None else base_ori
    
    if base_height == None:
        base_height_action = np.array([0])
    elif base_height == "up":
        base_height_action = np.array([1])
    elif base_height == "down":
        base_height_action = np.array([-1])
    
    global last_grasp0
    global last_grasp1
    
    if grasp == None:
        if id == 0:
            grasp_action = np.array([1]) if last_grasp0 else np.array([-1])
        elif id == 1:
            grasp_action = np.array([1]) if last_grasp1 else np.array([-1])
        else:
            raise ValueError("create action id should be 0 or 1")
    else:
        assert grasp == True or grasp == False
        if id == 0:
            last_grasp0 = grasp
        elif id == 1:
            last_grasp1 = grasp
        else:
            raise ValueError("create action id should be 0 or 1")
        grasp_action = np.array([1]) if grasp else np.array([-1])
        
    action = np.concatenate((eef_pos, eef_axisangle, grasp_action, base_pos, base_ori, base_height_action, np.array([-1])), axis=0)

    assert action.shape[0] == 12
    return action


def reset_controller():
    """
    Reset all pid controllers.
    """
    pid_base_pos_ctlr.reset()
    pid_base_ori_ctlr.reset()
    pid_eef_pos_ctlr.reset()
    pid_eef_axisangle_ctlr.reset()


if __name__ == '__main__':
    print(create_action())
    print(create_action(grasp=True))
    print(create_action())
    print(create_action(grasp=False))
    print(create_action())
    from copy import deepcopy
    pid_base_pos_ctlr0 = deepcopy(pid_base_pos_ctlr)
    pid_base_pos_ctlr1 = deepcopy(pid_base_pos_ctlr)
    print(pid_base_pos_ctlr)
    print(pid_base_pos_ctlr0)
    print(pid_base_pos_ctlr1)