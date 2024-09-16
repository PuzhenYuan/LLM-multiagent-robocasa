from robocasa.environments.kitchen.kitchen import *
import robocasa.macros_private as macros # use private macros to debug

# for controller usage
import robocasa.utils.control_utils as CU


class NavigateKitchen(Kitchen):
    def __init__(self, *args, **kwargs):
        kwargs["layout_ids"] = -2 # no island
        
        # controller specific parameters
        self.task_stage = 0 # used in planning
        self.init_pos = None
        self.middle1_pos = None
        self.middle2_pos = None
        CU.reset_controller()
        
        super().__init__(*args, **kwargs)
    
    def _reset_internal(self):
        super()._reset_internal()
        
        # reset controller specific parameters
        self.task_stage = 0
        self.init_pos = None
        self.middle1_pos = None
        self.middle2_pos = None
        CU.reset_controller()

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
    
    def get_control(self, obs):
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
            action = CU.pid_base_pos_ctlr.compute(current_value=base_pos, target_value=self.middle1_pos)
            action = CU.map_action(action, base_ori) # 2-dim
            action = CU.create_action(base_pos=action)
            if np.linalg.norm(self.middle1_pos - base_pos) <= 0.05:
                self.task_stage = 1
                CU.pid_base_pos_ctlr.reset()
        
        # then turn the orientation
        elif self.task_stage == 1:
            tz = CU.pid_base_ori_ctlr.compute(current_value=base_ori, target_value=target_ori)
            action = CU.map_action(tz, base_ori) # 1-dim
            action = CU.create_action(base_ori=action)
            if np.cos(target_ori - base_ori) >= 0.998:
                self.task_stage += 1
                CU.pid_base_ori_ctlr.reset()
        
        # then move to middle position 2
        elif self.task_stage == 2:
            action = CU.pid_base_pos_ctlr.compute(current_value=base_pos, target_value=self.middle2_pos) # ground coordinates
            action = CU.map_action(action, base_ori) # 2-dim
            action = CU.create_action(base_pos=action)
            if np.linalg.norm(self.middle2_pos - base_pos) <= 0.05:
                self.task_stage += 1
                CU.pid_base_pos_ctlr.reset()
            
        # finally move to target position
        elif self.task_stage == 3:
            action = CU.pid_base_pos_ctlr.compute(current_value=base_pos, target_value=target_pos) # ground coordinates
            action = CU.map_action(action, base_ori) # 2-dim
            action = CU.create_action(base_pos=action)
        
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

        