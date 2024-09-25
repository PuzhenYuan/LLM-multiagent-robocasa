from robocasa.environments.kitchen.kitchen import *
import robocasa.macros_private as macros # use private macros to debug

# for controller usage
import robocasa.utils.control_utils as CU


class PnPCounterToCounter(Kitchen):
    def __init__(self, *args, **kwargs):
        # kwargs["layout_ids"] = -2 # no island
        
        # controller specific parameters
        self.task_stage = 0 # used in planning
        self.init1_eef_pos = None
        self.init2_eef_pos = None
        self.init_eef_mat = None
        CU.reset_controller()
        
        super().__init__(*args, **kwargs)

    def _reset_internal(self):
        super()._reset_internal()
        
        # reset controller specific parameters
        self.task_stage = 0
        self.init1_eef_pos = None
        self.init2_eef_pos = None
        self.init_eef_mat = None
        CU.reset_controller()

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
            
            action = CU.pid_base_pos_ctlr.compute(current_value=base_pos, target_value=target_pos) # ground coordinates
            action = CU.map_action(action, base_ori) # 2-dim
            action = CU.create_action(base_pos=action, grasp=False)
            
            if np.linalg.norm(base_pos - target_pos) < 0.05:
                self.task_stage += 1
                CU.pid_base_pos_ctlr.reset()
        
        # move the gripper to the object position
        elif self.task_stage == 1:
            
            if self.init1_eef_pos is None:
                self.init1_eef_pos = obs['robot0_eef_pos']
            
            # position control
            base_ori = T.mat2euler(T.quat2mat(obs['robot0_base_quat']))[2]
            eef_pos = obs['robot0_eef_pos']
            target_pos = obs['vegetable_pos']
            
            action_pos = CU.pid_eef_pos_ctlr.compute(current_value=eef_pos, target_value=target_pos) # ground coordinates
            action_pos = CU.map_action(action_pos, base_ori) # 3-dim
            
            # orientation control
            eef_wrt_world_mat = T.quat2mat(obs['robot0_eef_quat'])
            obj_wrt_world_mat = T.quat2mat(obs['vegetable_quat']) 
            # close to np.array([[0, 1, 0],[-1, 0, 0],[0, 0, 1]]), so only extract z axis rotation
            theta_z = np.arctan2(obj_wrt_world_mat[1, 0], obj_wrt_world_mat[0, 0])
            desired_obj_wrt_world_mat = np.array(
                [
                    [np.cos(theta_z), -np.sin(theta_z), 0],
                    [np.sin(theta_z), np.cos(theta_z), 0],
                    [0, 0, 1]
                ]
            )
            target_obj_wrt_eef_mat = np.array([[1, 0, 0],[0, -1, 0],[0, 0, -1]]) @ np.array([[0, 1, 0],[-1, 0, 0],[0, 0, 1]])
            target_eef_wrt_world_mat = desired_obj_wrt_world_mat @ target_obj_wrt_eef_mat.T
            error_mat = target_eef_wrt_world_mat @ eef_wrt_world_mat.T
            error_axisangle = T.quat2axisangle(T.mat2quat(error_mat))
            
            action_axisangle = CU.pid_eef_axisangle_ctlr.compute(error=error_axisangle)
            action_axisangle = CU.map_action(action_axisangle, base_ori) # 3-dim

            action = CU.create_action(eef_pos=action_pos, eef_axisangle=action_axisangle, grasp=False)
            
            if np.linalg.norm(eef_pos - target_pos) < 0.01:
                self.task_stage += 1
                CU.pid_eef_pos_ctlr.reset()
                CU.pid_eef_axisangle_ctlr.reset()
                self.timmer = 10
        
        # close the gripper until contact
        elif self.task_stage == 2:
            action = CU.create_action(grasp=True)
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
            
            action_pos = CU.pid_eef_pos_ctlr.compute(current_value=eef_pos, target_value=target_pos) # ground coordinates
            action_pos = CU.map_action(action_pos, base_ori) # 3-dim
            
            # orientation control
            eef_wrt_world_mat = T.quat2mat(obs['robot0_eef_quat'])
            target_eef_wrt_world_mat = self.init_eef_mat
            error_mat = target_eef_wrt_world_mat @ eef_wrt_world_mat.T
            error_axisangle = T.quat2axisangle(T.mat2quat(error_mat))
            
            action_axisangle = CU.pid_eef_axisangle_ctlr.compute(error=error_axisangle)
            action_axisangle = CU.map_action(action_axisangle, base_ori) # 3-dim
            
            action = CU.create_action(eef_pos=action_pos, eef_axisangle=action_axisangle, grasp=True)
            
            if np.linalg.norm(eef_pos - target_pos) < 0.01:
                self.task_stage += 1
                CU.pid_eef_pos_ctlr.reset()
                CU.pid_eef_axisangle_ctlr.reset()
        
        # move to the place position, single stage movement
        elif self.task_stage == 4:
            
            base_pos = obs['robot0_base_pos'][:2]
            base_ori = T.mat2euler(T.quat2mat(obs['robot0_base_quat']))[2]
            target_pos, target_ori = self.compute_robot_base_placement_pose(plate)
            target_pos = target_pos[:2]
            target_ori = target_ori[2]
            
            action = CU.pid_base_pos_ctlr.compute(current_value=base_pos, target_value=target_pos) # ground coordinates
            action = CU.map_action(action, base_ori) # 2-dim
            action = CU.create_action(base_pos=action, grasp=True)
            
            if np.linalg.norm(base_pos - target_pos) < 0.05:
                self.task_stage += 1
                CU.pid_base_pos_ctlr.reset()
        
        # move the gripper to the object position
        elif self.task_stage == 5:
            
            if self.init2_eef_pos is None:
                self.init2_eef_pos = obs['robot0_eef_pos']
            
            # position control
            base_ori = T.mat2euler(T.quat2mat(obs['robot0_base_quat']))[2]
            eef_pos = obs['robot0_eef_pos']
            target_pos = obs['container_pos'] + np.array([0, 0, 0.05])
            
            action_pos = CU.pid_eef_pos_ctlr.compute(current_value=eef_pos, target_value=target_pos) # ground coordinates
            action_pos = CU.map_action(action_pos, base_ori) # 3-dim
            
            action = CU.create_action(eef_pos=action_pos, grasp=True)
            
            if np.linalg.norm(eef_pos - target_pos) < 0.01:
                self.task_stage += 1
                CU.pid_eef_pos_ctlr.reset()
                self.timmer = 10
        
        # open the gripper until not contact
        elif self.task_stage == 6:
            action = CU.create_action(grasp=False)
            obj_contact = self.check_contact(plate, self.robots[0].gripper["right"])
            if not obj_contact:
                self.timmer -= 1
            if self.timmer == 0:
                self.task_stage += 1
        
        # reset arm position using control sequence
        elif self.task_stage == 7:
            
            # position control
            base_ori = T.mat2euler(T.quat2mat(obs['robot0_base_quat']))[2]
            eef_pos = obs['robot0_eef_pos']
            target_pos = self.init2_eef_pos

            action_pos = CU.pid_eef_pos_ctlr.compute(current_value=eef_pos, target_value=target_pos) # ground coordinates
            action_pos = CU.map_action(action_pos, base_ori) # 3-dim
            
            # orientation control
            eef_wrt_world_mat = T.quat2mat(obs['robot0_eef_quat'])
            target_eef_wrt_world_mat = self.init_eef_mat
            error_mat = target_eef_wrt_world_mat @ eef_wrt_world_mat.T
            error_axisangle = T.quat2axisangle(T.mat2quat(error_mat))
            
            action_axisangle = CU.pid_eef_axisangle_ctlr.compute(error=error_axisangle)
            action_axisangle = CU.map_action(action_axisangle, base_ori) # 3-dim
            
            action = CU.create_action(eef_pos=action_pos, eef_axisangle=action_axisangle, grasp=False)
            
            if np.linalg.norm(eef_pos - target_pos) < 0.01:
                self.task_stage += 1
                CU.pid_eef_pos_ctlr.reset()
                CU.pid_eef_axisangle_ctlr.reset()
        
        return action
    