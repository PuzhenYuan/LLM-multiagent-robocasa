from robocasa.environments.kitchen.kitchen import *


class PIDController:
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

class PnPCounterToCounter(Kitchen):
    def __init__(self, *args, **kwargs):
        # kwargs["layout_ids"] = -2 # no island
        
        # controller specific parameters
        self.task_stage = 0 # used in planning
        self.init1_eef_pos = None
        self.init2_eef_pos = None
        self.init_eef_mat = None
        
        # PID controller parameters
        kp_base_pos = 4.0
        ki_base_pos = 0.01
        kd_base_pos = 0.01
        
        kp_base_ori = 1.5
        ki_base_ori = 0.05
        kd_base_ori = 0
        
        kp_eef_pos = 1.0
        ki_eef_pos = 0.01
        kd_eef_pos = 0
        
        kp_eef_axisangle = 1.0
        ki_eef_axisangle = 0.01
        kd_eef_axisangle = 0

        # create pid controllers
        self.pid_base_pos = PIDController(kp=kp_base_pos, ki=ki_base_pos, kd=kd_base_pos)
        self.pid_base_ori = PIDController(kp=kp_base_ori, ki=ki_base_ori, kd=kd_base_ori)
        self.pid_eef_pos = PIDController(kp=kp_eef_pos, ki=ki_eef_pos, kd=kd_eef_pos)
        self.pid_eef_axisangle = PIDController(kp=kp_eef_axisangle, ki=ki_eef_axisangle, kd=kd_eef_axisangle)
        
        super().__init__(*args, **kwargs)

    def _reset_internal(self):
        super()._reset_internal()
        
        # reset controller specific parameters
        self.task_stage = 0
        self.init1_eef_pos = None
        self.init2_eef_pos = None
        self.init_eef_mat = None
        self.pid_base_pos.reset()
        self.pid_base_ori.reset()
        self.pid_eef_pos.reset()
        self.pid_eef_axisangle.reset()

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.counter = self.register_fixture_ref("counter", dict(id=FixtureType.COUNTER, ref=self.sink, size=(0.45, 0.55)))
        self.init_robot_base_pos = self.counter
    
    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang(obj_name="vegetable")
        ep_meta["lang"] = f"pick the {obj_lang} from the counter and place it in the plate"
        return ep_meta
    
    def _get_obj_cfgs(self):
        cfgs = []
        
        cfgs.append(dict(
            name="vegetable",
            obj_groups=("vegetable"),
            graspable=True,
            placement=dict(
                fixture=self.counter,
                sample_region_kwargs=dict(
                    ref=self.sink,
                    loc="left_right",
                    top_size=(0.45, 0.55)
                ),
                size=(0.60, 0.30),
                pos=(0.0, -1.0),
                offset=(0.0, 0.10),
            ),
        ))
        
        cfgs.append(dict( # near sink
            name="container",
            obj_groups=("plate"),
            graspable=False,
            placement=dict(
                fixture=self.counter,
                sample_region_kwargs=dict(
                    ref=self.sink,
                    loc="left_right",
                    top_size=(0.45, 0.55)
                ),
                size=(0.35, 0.45),
                pos=("ref", -1.0),
            ),
        ))
        
        return cfgs
    
    def _check_success(self):
        obj_in_plate = OU.check_obj_in_receptacle(self, obj_name="vegetable", receptacle_name="container")
        gripper_obj_far = OU.gripper_obj_far(self, obj_name="vegetable")
        return obj_in_plate and gripper_obj_far
    
    def get_control(self, obs):
        
        vegetable = self.objects["vegetable"] # rot
        vegetable.pos = obs['vegetable_pos'] # pos
        plate = self.objects["container"] # rot
        plate.pos = obs["container_pos"] # pos

        # move to the pick position, single stage movement
        if self.task_stage == 0:
            
            if self.init_eef_mat is None:
                self.init_eef_mat = T.quat2mat(obs['robot0_eef_quat'])
            
            base_pos = obs['robot0_base_pos'][:2]
            base_ori = T.mat2euler(T.quat2mat(obs['robot0_base_quat']))[2]
            target_pos, target_ori = self.compute_robot_base_placement_pose(vegetable)
            target_pos = target_pos[:2]
            target_ori = target_ori[2]
            
            action = self.pid_base_pos.compute(current_value=base_pos, target_value=target_pos) # ground coordinates
            Fx = action[0]
            Fy = action[1]
            fx = Fx * np.cos(base_ori) + Fy * np.sin(base_ori) # robot coordinates
            fy = -Fx * np.sin(base_ori) + Fy * np.cos(base_ori)
            tz = 0 # self.pid_base_ori.compute(current_value=base_ori, target_value=target_ori)
            action = np.array([0, 0, 0, 0, 0, 0, -1, fx, fy, tz, 0, -1])
            
            if np.linalg.norm(base_pos - target_pos) < 0.05:
                self.task_stage += 1
                self.pid_base_pos.reset()
                self.pid_base_ori.reset()
        
        # move the gripper to the object position
        elif self.task_stage == 1:
            
            if self.init1_eef_pos is None:
                self.init1_eef_pos = obs['robot0_eef_pos']
            
            # position control
            base_ori = T.mat2euler(T.quat2mat(obs['robot0_base_quat']))[2]
            eef_pos = obs['robot0_eef_pos']
            target_pos = obs['vegetable_pos']
            
            action_pos = self.pid_eef_pos.compute(current_value=eef_pos, target_value=target_pos) # ground coordinates
            Fx = action_pos[0]
            Fy = action_pos[1]
            Fz = action_pos[2]
            fx = Fx * np.cos(base_ori) + Fy * np.sin(base_ori) # robot coordinates
            fy = -Fx * np.sin(base_ori) + Fy * np.cos(base_ori)
            action_pos = np.array([fx, fy, Fz])
            
            # orientation control
            eef_wrt_world_mat = T.quat2mat(obs['robot0_eef_quat'])
            obj_wrt_world_mat = T.quat2mat(obs['vegetable_quat'])
            obj_wrt_eef_mat = np.array([[1, 0, 0],[0, -1, 0],[0, 0, -1]]) @ np.array([[0, 1, 0],[-1, 0, 0],[0, 0, 1]]) # desired
            target_eef_wrt_world_mat = obj_wrt_world_mat @ obj_wrt_eef_mat.T
            error_mat = target_eef_wrt_world_mat @ eef_wrt_world_mat.T
            error_axisangle = T.quat2axisangle(T.mat2quat(error_mat))
            
            action_axisangle = self.pid_eef_axisangle.compute(error=error_axisangle)
            Tx = action_axisangle[0]
            Ty = action_axisangle[1]
            Tz = action_axisangle[2]
            tx = Tx * np.cos(base_ori) + Ty * np.sin(base_ori)
            ty = -Tx * np.sin(base_ori) + Ty * np.cos(base_ori)
            action_axisangle = np.array([tx, ty, Tz])
            # action_axisangle = np.array([0, 0, 0])
            action = np.concatenate((action_pos, action_axisangle, np.array([-1, 0, 0, 0, 0, -1])), axis=0)
            
            if np.linalg.norm(eef_pos - target_pos) < 0.01:
                self.task_stage += 1
                self.pid_eef_pos.reset()
                self.pid_eef_axisangle.reset()
                self.timmer = 10
        
        # close the gripper until contact
        elif self.task_stage == 2:
            action = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1])
            obj_contact = self.check_contact(vegetable, self.robots[0].gripper["right"])
            if obj_contact:
                self.timmer -= 1
            if self.timmer == 0:
                self.task_stage += 1
        
        # reset arm position using control sequence
        elif self.task_stage == 3:
            
            # position control
            base_ori = T.mat2euler(T.quat2mat(obs['robot0_base_quat']))[2]
            eef_pos = obs['robot0_eef_pos']
            target_pos = self.init1_eef_pos
            
            action_pos = self.pid_eef_pos.compute(current_value=eef_pos, target_value=target_pos) # ground coordinates
            Fx = action_pos[0]
            Fy = action_pos[1]
            Fz = action_pos[2]
            fx = Fx * np.cos(base_ori) + Fy * np.sin(base_ori) # robot coordinates
            fy = -Fx * np.sin(base_ori) + Fy * np.cos(base_ori)
            action_pos = np.array([fx, fy, Fz])
            
            # orientation control
            eef_wrt_world_mat = T.quat2mat(obs['robot0_eef_quat'])
            target_eef_wrt_world_mat = self.init_eef_mat
            error_mat = target_eef_wrt_world_mat @ eef_wrt_world_mat.T
            error_axisangle = T.quat2axisangle(T.mat2quat(error_mat))
            
            action_axisangle = self.pid_eef_axisangle.compute(error=error_axisangle)
            Tx = action_axisangle[0]
            Ty = action_axisangle[1]
            Tz = action_axisangle[2]
            tx = Tx * np.cos(base_ori) + Ty * np.sin(base_ori)
            ty = -Tx * np.sin(base_ori) + Ty * np.cos(base_ori)
            action_axisangle = np.array([tx, ty, Tz])
            
            action = np.concatenate((action_pos, action_axisangle, np.array([1, 0, 0, 0, 0, -1])), axis=0)
            
            if np.linalg.norm(eef_pos - target_pos) < 0.01:
                self.task_stage += 1
                self.pid_eef_pos.reset()
                self.pid_eef_axisangle.reset()
        
        # move to the place position, single stage movement
        elif self.task_stage == 4:
            
            base_pos = obs['robot0_base_pos'][:2]
            base_ori = T.mat2euler(T.quat2mat(obs['robot0_base_quat']))[2]
            target_pos, target_ori = self.compute_robot_base_placement_pose(plate)
            target_pos = target_pos[:2]
            target_ori = target_ori[2]
            
            action = self.pid_base_pos.compute(current_value=base_pos, target_value=target_pos) # ground coordinates
            Fx = action[0]
            Fy = action[1]
            fx = Fx * np.cos(base_ori) + Fy * np.sin(base_ori) # robot coordinates
            fy = -Fx * np.sin(base_ori) + Fy * np.cos(base_ori)
            tz = 0 # self.pid_base_ori.compute(current_value=base_ori, target_value=target_ori)
            action = np.array([0, 0, 0, 0, 0, 0, 1, fx, fy, tz, 0, 1])
            
            if np.linalg.norm(base_pos - target_pos) < 0.05:
                self.task_stage += 1
                self.pid_base_pos.reset()
                self.pid_base_ori.reset()
        
        # move the gripper to the object position
        elif self.task_stage == 5:
            
            if self.init2_eef_pos is None:
                self.init2_eef_pos = obs['robot0_eef_pos']
            
            base_ori = T.mat2euler(T.quat2mat(obs['robot0_base_quat']))[2]
            eef_pos = obs['robot0_eef_pos']
            target_pos = obs['container_pos'] + np.array([0, 0, 0.05])
            
            # position control
            action_pos = self.pid_eef_pos.compute(current_value=eef_pos, target_value=target_pos) # ground coordinates
            Fx = action_pos[0]
            Fy = action_pos[1]
            Fz = action_pos[2]
            fx = Fx * np.cos(base_ori) + Fy * np.sin(base_ori) # robot coordinates
            fy = -Fx * np.sin(base_ori) + Fy * np.cos(base_ori)
            action_pos = np.array([fx, fy, Fz])
            
            action_axisangle = np.array([0, 0, 0])
            action = np.concatenate((action_pos, action_axisangle, np.array([1, 0, 0, 0, 0, -1])), axis=0)
            
            if np.linalg.norm(eef_pos - target_pos) < 0.01:
                self.task_stage += 1
                self.pid_eef_pos.reset()
                self.pid_eef_axisangle.reset()
                self.timmer = 10
        
        # open the gripper until not contact
        elif self.task_stage == 6:
            action = np.array([0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1])
            obj_contact = self.check_contact(plate, self.robots[0].gripper["right"])
            if not obj_contact:
                self.timmer -= 1
            if self.timmer == 0:
                self.task_stage += 1
        
        # reset arm position using control sequence
        elif self.task_stage == 7:
            
            base_ori = T.mat2euler(T.quat2mat(obs['robot0_base_quat']))[2]
            eef_pos = obs['robot0_eef_pos']
            target_pos = self.init2_eef_pos
            
            action_pos = self.pid_eef_pos.compute(current_value=eef_pos, target_value=target_pos) # ground coordinates
            Fx = action_pos[0]
            Fy = action_pos[1]
            Fz = action_pos[2]
            fx = Fx * np.cos(base_ori) + Fy * np.sin(base_ori) # robot coordinates
            fy = -Fx * np.sin(base_ori) + Fy * np.cos(base_ori)
            action_pos = np.array([fx, fy, Fz])
            
            # orientation control
            eef_wrt_world_mat = T.quat2mat(obs['robot0_eef_quat'])
            target_eef_wrt_world_mat = self.init_eef_mat
            error_mat = target_eef_wrt_world_mat @ eef_wrt_world_mat.T
            error_axisangle = T.quat2axisangle(T.mat2quat(error_mat))
            
            action_axisangle = self.pid_eef_axisangle.compute(error=error_axisangle)
            Tx = action_axisangle[0]
            Ty = action_axisangle[1]
            Tz = action_axisangle[2]
            tx = Tx * np.cos(base_ori) + Ty * np.sin(base_ori)
            ty = -Tx * np.sin(base_ori) + Ty * np.cos(base_ori)
            action_axisangle = np.array([tx, ty, Tz])
            
            action = np.concatenate((action_pos, action_axisangle, np.array([-1, 0, 0, 0, 0, -1])), axis=0)
            
            if np.linalg.norm(eef_pos - target_pos) < 0.01:
                self.task_stage += 1
                self.pid_eef_pos.reset()
                self.pid_eef_axisangle.reset()
        
        return action
    