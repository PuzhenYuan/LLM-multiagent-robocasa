from robocasa.environments.kitchen.kitchen import *


class OpenMicrowavePnP(Kitchen):
    EXCLUDE_LAYOUTS = [8]
    def __init__(
        self, 
        obj_groups="vegetable", 
        exclude_obj_groups=None,
        *args, 
        **kwargs
    ):
        self.obj_groups = obj_groups
        self.exclude_obj_groups = exclude_obj_groups
        super().__init__(*args, **kwargs)
    
    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.microwave = self.register_fixture_ref(
            "microwave", dict(id=FixtureType.MICROWAVE),
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.microwave),
        )
        self.distr_counter = self.register_fixture_ref(
            "distr_counter", dict(id=FixtureType.COUNTER, ref=self.microwave),
        )
        self.init_robot_base_pos = self.microwave
    
    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        self.microwave.set_door_state(min=0.0, max=0.1, env=self, rng=self.rng) # door is closed initially

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang()
        ep_meta["lang"] = f"open the microwave door, pick the {obj_lang} from the counter and place it in the microwave"
        return ep_meta
    
    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(dict(
            name="obj",
            obj_groups=self.obj_groups,
            exclude_obj_groups=self.exclude_obj_groups,
            graspable=True, microwavable=True,
            placement=dict(
                fixture=self.counter,
                sample_region_kwargs=dict(
                    ref=self.microwave,
                ),
                size=(0.30, 0.30),
                pos=("ref", -1.0),
                try_to_place_in="container",
            ),
        ))
        cfgs.append(dict(
            name="container",
            obj_groups=("plate"),
            placement=dict(
                fixture=self.microwave,
                size=(0.05, 0.05),
                ensure_object_boundary_in_range=False,
            ),
        ))

        # distractors
        cfgs.append(dict(
            name="distr_counter",
            obj_groups="all",
            placement=dict(
                fixture=self.distr_counter,
                sample_region_kwargs=dict(
                    ref=self.microwave,
                ),
                size=(0.30, 0.30),
                pos=("ref", 1.0),
            ),
        ))

        return cfgs

    def _check_success(self):
        obj = self.objects["obj"]
        container = self.objects["container"]
        door_state = self.microwave.get_door_state(env=self)
        
        task0_success = True
        for joint_p in door_state.values():
            if joint_p < 0.90:
                task0_success = False
                break

        obj_container_contact = self.check_contact(obj, container)
        container_micro_contact = self.check_contact(container, self.microwave)
        gripper_obj_far = OU.gripper_obj_far(self)
        task1_success =  obj_container_contact and container_micro_contact and gripper_obj_far # return a list maybe work

        task_success = task0_success and task1_success
        return {'task': task_success, 'task0': task0_success, 'task1': task1_success}