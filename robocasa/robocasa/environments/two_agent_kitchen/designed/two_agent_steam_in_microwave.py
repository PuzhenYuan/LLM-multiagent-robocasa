from robocasa.environments.two_agent_kitchen.two_agent_kitchen import *


class TwoAgentSteamInMicrowave(TwoAgentKitchen):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.microwave = self.register_fixture_ref("microwave", dict(id=FixtureType.MICROWAVE))
        self.counter = self.register_fixture_ref("counter", dict(id=FixtureType.COUNTER, ref=self.sink))
        self.distr_counter = self.register_fixture_ref("distr_counter", dict(id=FixtureType.COUNTER, ref=self.microwave))
        self.init_robot_base_pos = []
        self.init_robot_base_pos.append(self.microwave)
        self.init_robot_base_pos.append(self.sink)

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        vegetable_name = self.get_obj_lang("vegetable")
        ep_meta["lang"] = f"Pick the {vegetable_name} from the sink and place it in the bowl. Then pick the bowl and place it in the microwave. Then close the microwave door and press the start button."
        
        return ep_meta

    def _reset_internal(self):
        super()._reset_internal()
        self.sink.set_handle_state(mode="off", env=self, rng=self.rng)
        self.microwave.set_door_state(min=0.90, max=1.0, env=self, rng=self.rng)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(dict(
            name="bowl",
            obj_groups="bowl",
            placement=dict(
                fixture=self.counter,
                sample_region_kwargs=dict(
                    ref=self.sink,
                    loc="left_right",
                ),
                size=(0.35, 0.40),
                pos=("ref", -1.0),
            ),
        ))

        cfgs.append(dict(
            name="vegetable",
            obj_groups="vegetable",
            graspable=True,
            washable=True,
            placement=dict(
                fixture=self.sink,
                size=(0.3, 0.2),
                pos=(0.0, 1.0),
            ),
        ))

        # distractors
        # cfgs.append(dict(
        #     name="distr_counter_0",
        #     obj_groups="all",
        #     placement=dict(
        #         fixture=self.distr_counter,
        #         sample_region_kwargs=dict(
        #             ref=self.microwave,
        #         ),
        #         size=(0.50, 0.50),
        #         pos=("ref", -1.0),
        #         offset=(0.0, 0.40),
        #     ),
        # ))

        # cfgs.append(dict(
        #     name="distr_counter_1",
        #     obj_groups="all",
        #     placement=dict(
        #         fixture=self.distr_counter,
        #         sample_region_kwargs=dict(
        #             ref=self.microwave,
        #             loc="left_right"
        #         ),
        #         size=(0.50, 0.50),
        #         pos=("ref", -1.0),
        #     ),
        # ))
        
        cfgs.append(dict(
            name="container1",
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
        
        cfgs.append(dict(
            name="container2",
            obj_groups=("plate"),
            graspable=False,
            placement=dict(
                fixture=self.distr_counter,
                sample_region_kwargs=dict(
                    ref=self.microwave,
                    loc="left_right",
                    top_size=(0.45, 0.55)
                ),
                size=(0.35, 0.45),
                pos=("ref", -1.0),
            ),
        ))

        return cfgs

    def _check_success(self):
        vegetable_in_bowl = OU.check_obj_in_receptacle(self, "vegetable", "bowl")
        bowl_in_microwave = OU.obj_inside_of(self, "bowl", self.microwave)

        door_state = self.microwave.get_door_state(env=self)
        door_closed = True
        for joint_p in door_state.values():
            if joint_p > 0.05:
                door_closed = False
                break

        button_pressed = self.microwave.get_state()["turned_on"]

        return vegetable_in_bowl and bowl_in_microwave and door_closed and button_pressed