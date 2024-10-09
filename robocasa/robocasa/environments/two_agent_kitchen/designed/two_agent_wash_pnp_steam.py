from robocasa.environments.two_agent_kitchen.two_agent_kitchen import *
import robocasa.macros_private as macros # use private macros to debug
import random

class TwoAgentWashPnPSteam(TwoAgentKitchen):
    """
    Two agent kitchen environment for the task of first wash vegetables and then heat it in the microwave.
    """
    EXCLUDE_LAYOUTS = [] 
    def __init__(self, *args, **kwargs):
        
        kwargs["layout_ids"] = 10 # 10 is designed for this non navigation task
        # kwargs["style_ids"] = 10
        # kwargs["style_ids"] = random.choice([0, 2, 3, 4, 7, 8, 11]) # for big sink, obj_x_percent: 0.63
        kwargs["style_ids"] = random.choice([1, 5, 6, 9, 10]) # for small sink, obj_x_percent: 0.69
        super().__init__( *args, **kwargs)

        self.vegetables_washed = False
        self.washed_time = 0
        
        self.task_has_succeeded = False
        self.task0_has_succeeded = False
        self.task1_has_succeeded = False
        self.task2_has_succeeded = False
        self.task3_has_succeeded = False
        self.task4_has_succeeded = False
        self.task5_has_succeeded = False
        self.task6_has_succeeded = False
        self.task7_has_succeeded = False

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.counter = self.register_fixture_ref("counter", dict(id=FixtureType.COUNTER, ref=self.sink, size=(0.45, 0.55)))
        self.microwave = self.register_fixture_ref("microwave", dict(id=FixtureType.MICROWAVE),)
        self.distr_counter = self.register_fixture_ref("distr_counter", dict(id=FixtureType.COUNTER, ref=self.microwave),)
        self.init_robot_base_pos = []
        
        self.init_robot_base_pos.append(self.sink)
        self.init_robot_base_pos.append(self.microwave)

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang(obj_name="vegetable")
        # cont_lang = self.get_obj_lang(obj_name="container")
        ep_meta["lang"] = f"agent0 pick the {obj_lang} from the counter and place it in the sink, " + \
                        f"agent0 turn on the sink faucet, " + \
                        f"agent0 turn off the sink faucet, " + \
                        f"agent0 pick the {obj_lang} from the sink and place it on the the plate located on the counter, " + \
                        f"agent1 open the microwave door, " + \
                        f"agent1 pick the {obj_lang} from the counter and place it in the microwave, " + \
                        f"agent1 close the microwave door, " + \
                        f"agent1 press the start button on the microwave"
        return ep_meta
    
    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        self.vegetables_washed = False
        self.washed_time = 0
        super()._reset_internal()
        self.sink.set_handle_state(mode="off", env=self, rng=self.rng)
        self.microwave.set_door_state(min=0.0, max=0.0, env=self, rng=self.rng) # door is closed initially

    def _get_obj_cfgs(self):
        cfgs = []
        
        cfgs.append(dict( # near sink
            name="container1",
            obj_groups=("plate"),
            graspable=False,
            placement=dict(
                fixture=self.counter,
                sample_region_kwargs=dict(
                    ref=self.sink,
                    loc="right",
                    top_size=(0.45, 0.55)
                ),
                size=(0.35, 0.45),
                pos=("ref", -1.0),
            ),
        ))
        
        # cfgs.append(dict( # near microwave
        #     name="container2",
        #     obj_groups=("plate"),
        #     graspable=False,
        #     placement=dict(
        #         fixture=self.distr_counter,
        #         sample_region_kwargs=dict(
        #             ref=self.microwave,
        #             loc="left",
        #             top_size=(0.45, 0.55)
        #         ),
        #         size=(0.35, 0.45),
        #         pos=("ref", -1.0),
        #     ),
        # ))
        
        cfgs.append(dict( # in microwave
            name="container2",
            obj_groups=("plate"),
            heatable=True,
            placement=dict(
                fixture=self.microwave,
                size=(0.05, 0.05),
                ensure_object_boundary_in_range=False,
            ),
        ))
        
        cfgs.append(dict(
            name="vegetable",
            obj_groups="carrot", # "vegetable"
            graspable=True,
            washable=True,
            microwavable=True,
            heatable=True,
            placement=dict(
                fixture=self.counter,
                sample_region_kwargs=dict(
                    ref=self.sink,
                    loc="left"
                ),
                size=(0.30, 0.30),
                pos=("ref", -1.0),
                # try_to_place_in="container", # not container1
            ),
        ))

        return cfgs

    def _check_success(self):
        
        obj = self.objects["vegetable"]
        container1 = self.objects["container1"] # near sink
        # container2 = self.objects["container2"] # near microwave
        container2 = self.objects["container2"] # in microwave
        
        vegetables_in_sink = OU.obj_inside_of(self, f"vegetable", self.sink) 
        water_on = self.sink.get_handle_state(env=self)["water_on"]
        if vegetables_in_sink and water_on:
            self.washed_time += 1
            self.vegetables_washed = self.washed_time > 10
        else:
            self.washed_time = 0
        
        task0_success = vegetables_in_sink
        task1_success = self.task0_has_succeeded and water_on and self.vegetables_washed
        task2_success = self.task1_has_succeeded and not water_on
        
        vegetables_in_place = OU.check_obj_in_receptacle(self, "vegetable", "container1")
        
        task3_success = self.task2_has_succeeded and vegetables_in_place
        
        door_state = self.microwave.get_door_state(env=self)
        microwave_door_open = True
        for joint_p in door_state.values():
            if joint_p < 0.9:
                microwave_door_open = False
                break
        
        # task4_success = self.task3_has_succeeded and microwave_door_open
        task4_success = microwave_door_open
        
        obj_container_contact = self.check_contact(obj, container2)
        container_micro_contact = self.check_contact(container2, self.microwave)
        gripper_obj_far = OU.gripper_obj_far(self, obj_name="vegetable")
        
        task5_success = self.task4_has_succeeded and obj_container_contact and container_micro_contact and gripper_obj_far
        
        microwave_door_close = True
        for joint_p in door_state.values():
            if joint_p > 0.05:
                microwave_door_close = False
                break
        
        task6_success = self.task5_has_succeeded and microwave_door_close
        
        microwave_turned_on = self.microwave.get_state()["turned_on"]
        gripper_button_far = self.microwave.gripper_button_far(self, button="start_button")
        
        task7_success = self.task6_has_succeeded and microwave_turned_on and gripper_button_far
        
        task_success = self.task7_has_succeeded
        
        # for debug use
        if macros.VERBOSE:
            if self.task0_has_succeeded == False and task0_success == True:
                print("Task 0 is success")
            if self.task1_has_succeeded == False and task1_success == True:
                print("Task 1 is success")
            if self.task2_has_succeeded == False and task2_success == True:
                print("Task 2 is success")
            if self.task3_has_succeeded == False and task3_success == True:
                print("Task 3 is success")
            if self.task4_has_succeeded == False and task4_success == True:
                print("Task 4 is success")
            if self.task5_has_succeeded == False and task5_success == True:
                print("Task 5 is success")
            if self.task6_has_succeeded == False and task6_success == True:
                print("Task 6 is success")
            if self.task7_has_succeeded == False and task7_success == True:
                print("Task 7 is success")
            if self.task_has_succeeded == False and task_success == True:
                print("All tasks are success")
        
        self.task_has_succeeded = self.task_has_succeeded or task_success
        self.task0_has_succeeded = self.task0_has_succeeded or task0_success
        self.task1_has_succeeded = self.task1_has_succeeded or task1_success
        self.task2_has_succeeded = self.task2_has_succeeded or task2_success
        self.task3_has_succeeded = self.task3_has_succeeded or task3_success
        self.task4_has_succeeded = self.task4_has_succeeded or task4_success
        self.task5_has_succeeded = self.task5_has_succeeded or task5_success
        self.task6_has_succeeded = self.task6_has_succeeded or task6_success
        self.task7_has_succeeded = self.task7_has_succeeded or task7_success
        
        return {
            'task': self.task_has_succeeded, 
            'task0': self.task0_has_succeeded, 
            'task1': self.task1_has_succeeded, 
            'task2': self.task2_has_succeeded, 
            'task3': self.task3_has_succeeded, 
            'task4': self.task4_has_succeeded, 
            'task5': self.task5_has_succeeded, 
            'task6': self.task6_has_succeeded, 
            'task7': self.task7_has_succeeded
        }