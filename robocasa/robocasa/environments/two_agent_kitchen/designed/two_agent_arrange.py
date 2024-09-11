from robocasa.environments.two_agent_kitchen.two_agent_kitchen import *


class TwoAgentArrange(TwoAgentKitchen):
    """
    Two agent kitchen environment for the task of arranging objects.
    """

    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs)
        layout = 7 # "G-shaped"
        style = 1
        self.layout_and_style_ids = [[layout, style]]

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.counter = self.register_fixture_ref("counter", dict(id=FixtureType.COUNTER, ref=self.sink, size=(0.45, 0.55)))
        self.init_robot_base_pos = []
        self.init_robot_base_pos.append(self.sink)
        self.init_robot_base_pos.append(self.counter)

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = f"agent1 pick the can from the counter and place them on the cutting board, \
                        agent0 pick the can from the cutting board and place them in the sink"
        return ep_meta
    
    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

    def _get_obj_cfgs(self):
        cfgs = []        
        cfgs.append(dict(
            name="cutting_board",
            obj_groups="cutting_board",
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
            name="item1",
            obj_groups="can",
            graspable=True,
            placement=dict(
                fixture=self.counter,
                size=(0.40, 0.40),
                pos=(0.0, 0.0),
            ),
        ))

        return cfgs    

    def _check_success(self):
        # item1_cutting_board_contact = OU.check_obj_in_receptacle(self, "item1", "cutting_board")
        gripper_obj_far = OU.gripper_obj_far(self, obj_name="cutting_board")
        obj_in_sink = OU.obj_inside_of(self, "item1", self.sink)
        return obj_in_sink and gripper_obj_far