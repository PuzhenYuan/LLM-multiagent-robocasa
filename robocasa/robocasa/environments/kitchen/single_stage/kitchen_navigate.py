from robocasa.environments.kitchen.kitchen import *
import robocasa.macros_private as macros # use private macros to debug

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = None
        self.previous_error = None

    def compute(self, current_value, target_value):
        error = target_value - current_value
        self.integral = error if self.integral is None else self.integral + error
        derivative = 0 if self.previous_error is None else (error - self.previous_error)
        self.previous_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative
    
    def reset(self):
        self.integral = None
        self.previous_error = None

class NavigateKitchen(Kitchen):
    def __init__(self, *args, **kwargs):
        kwargs["layout_ids"] = -2 # no island
        super().__init__(*args, **kwargs)
        
        self.task_stage = 0 # used in planning
        self.init_pos = None
        self.middle1_pos = None
        self.middle2_pos = None
        
        # PID controller parameters
        kp_pos = 3.0
        ki_pos = 0
        kd_pos = 0
        kp_ori = 1.5
        ki_ori = 0.1
        kd_ori = 0

        # create pid controllers
        self.pid_x = PIDController(kp=kp_pos, ki=ki_pos, kd=kd_pos)
        self.pid_y = PIDController(kp=kp_pos, ki=ki_pos, kd=kd_pos)
        self.pid_ori = PIDController(kp=kp_ori, ki=ki_ori, kd=kd_ori)
    
    def _reset_internal(self):
        super()._reset_internal()
        self.task_stage = 0
        self.init_pos = None
        self.middle1_pos = None
        self.middle2_pos = None
        self.pid_x.reset() if hasattr(self, "pid_x") else None
        self.pid_y.reset() if hasattr(self, "pid_y") else None
        self.pid_ori.reset() if hasattr(self, "pid_ori") else None

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        if "src_fixture" in self.fixture_refs:
            self.src_fixture = self.fixture_refs["src_fixture"]
            self.target_fixture = self.fixture_refs["target_fixture"]
        else:
            fixtures = list(self.fixtures.values())
            valid_src_fixture_classes = [
                "CoffeeMachine", "Toaster", "Stove", "Stovetop", "SingleCabinet", "HingeCabinet", "OpenCabinet", "Drawer",
                "Microwave", "Sink", "Hood", "Oven", "Fridge", "Dishwasher",
            ]
            while True:
                self.src_fixture = self.rng.choice(fixtures)
                fxtr_class = type(self.src_fixture).__name__
                if fxtr_class not in valid_src_fixture_classes:
                    continue
                break
            
            fxtr_classes = [type(fxtr).__name__ for fxtr in fixtures]
            valid_target_fxtr_classes = [
                cls for cls in fxtr_classes if fxtr_classes.count(cls) == 1 and cls in [
                    "CoffeeMachine", "Toaster", "Stove", "Stovetop", "OpenCabinet",
                    "Microwave", "Sink", "Hood", "Oven", "Fridge", "Dishwasher",
                ]
            ]
                
            while True:
                self.target_fixture = self.rng.choice(fixtures)
                fxtr_class = type(self.target_fixture).__name__
                if self.target_fixture == self.src_fixture or fxtr_class not in valid_target_fxtr_classes:
                    continue
                if fxtr_class == "Accessory":
                    continue
                # don't sample closeby fixtures
                if OU.fixture_pairwise_dist(self.src_fixture, self.target_fixture) <= 1.0:
                    continue
                break

            self.fixture_refs["src_fixture"] = self.src_fixture
            self.fixture_refs["target_fixture"] = self.target_fixture
        
        self.target_pos, self.target_ori = self.compute_robot_base_placement_pose(self.target_fixture)
        
        self.init_robot_base_pos = self.src_fixture

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = f"navigate to the {self.target_fixture.nat_lang}"
        return ep_meta
    
    def _check_success(self):
        robot_id = self.sim.model.body_name2id("base0_base")
        base_pos = np.array(self.sim.data.body_xpos[robot_id])
        pos_check = np.linalg.norm(self.target_pos[:2] - base_pos[:2]) <= 0.20
        base_ori = T.mat2euler(np.array(self.sim.data.body_xmat[robot_id]).reshape((3, 3)))
        ori_check = np.cos(self.target_ori[2] - base_ori[2]) >= 0.98

        return pos_check and ori_check
    
    def get_action(self, obs):
        # get base position and orientation
        base_pos = obs['robot0_base_pos'][:2]
        base_ori = T.mat2euler(T.quat2mat(obs['robot0_base_quat']))[2]

        # get target position and orientation
        target_pos = self.target_pos[:2]
        target_ori = self.target_ori[2]
        
        # calculate middle point 1 and middle point 2
        if self.init_pos is None:
            self.init_pos = base_pos
        if self.middle1_pos is None:
            self.middle1_pos = np.array([1.5, -1.5])
            self.middle1_pos[0] = base_pos[0] - 0.25 * np.cos(base_ori)
            self.middle1_pos[1] = base_pos[1] - 0.25 * np.sin(base_ori)
        if self.middle2_pos is None:
            self.middle2_pos = np.array([1.5, -1.5])
            self.middle2_pos[0] = target_pos[0] - 0.25 * np.cos(target_ori)
            self.middle2_pos[1] = target_pos[1] - 0.25 * np.sin(target_ori)
        
        # move to middle position 1
        if self.task_stage == 0:
            Fx = self.pid_x.compute(current_value=base_pos[0], target_value=self.middle1_pos[0]) # ground coordinates
            Fy = self.pid_y.compute(current_value=base_pos[1], target_value=self.middle1_pos[1])
            fx = Fx * np.cos(base_ori) + Fy * np.sin(base_ori) # robot coordinates
            fy = -Fx * np.sin(base_ori) + Fy * np.cos(base_ori)
            tz = 0
            if np.linalg.norm(self.middle1_pos - base_pos) <= 0.10:
                self.task_stage = 1
                self.pid_x.reset()
                self.pid_y.reset()
        
        # then turn the orientation
        if self.task_stage == 1:
            fx = 0
            fy = 0
            tz = self.pid_ori.compute(current_value=base_ori, target_value=target_ori)
            if np.cos(target_ori - base_ori) >= 0.998:
                self.task_stage = 2
                self.pid_ori.reset()
        
        # then move to middle position 2
        if self.task_stage == 2:
            Fx = self.pid_x.compute(current_value=base_pos[0], target_value=self.middle2_pos[0]) # ground coordinates
            Fy = self.pid_y.compute(current_value=base_pos[1], target_value=self.middle2_pos[1])
            fx = Fx * np.cos(base_ori) + Fy * np.sin(base_ori) # robot coordinates
            fy = -Fx * np.sin(base_ori) + Fy * np.cos(base_ori)
            tz = 0
            if np.linalg.norm(self.middle2_pos - base_pos) <= 0.10:
                self.task_stage = 3
                self.pid_x.reset()
                self.pid_y.reset()
            
        # finally move to target position
        if self.task_stage == 3:
            Fx = self.pid_x.compute(current_value=base_pos[0], target_value=target_pos[0]) # ground coordinates
            Fy = self.pid_y.compute(current_value=base_pos[1], target_value=target_pos[1])
            fx = Fx * np.cos(base_ori) + Fy * np.sin(base_ori) # robot coordinates
            fy = -Fx * np.sin(base_ori) + Fy * np.cos(base_ori)
            tz = 0 # self.pid_ori.compute(current_value=base_ori, target_value=target_ori)

        # construct action array
        action = np.array([0, 0, 0, 0, 0, 0, -1, fx, fy, tz, 0, -1])
        
        if macros.VERBOSE:
            print('base pos', base_pos)
            print('base ori', base_ori)
            print('init pos', self.init_pos)
            print('middle1 pos', self.middle1_pos)
            print('middle2 pos', self.middle2_pos)
            print('target pos', target_pos)
            print('target ori', target_ori)
            print('task stage', self.task_stage)
        
        return action

        