from copy import deepcopy
import numpy as np
import random
import math

from robosuite.utils.mjcf_utils import array_to_string as a2s, string_to_array as s2a, find_elements, xml_path_completion, new_geom
from robocasa.models.objects.fixtures.fixture import ProcGenFixture
from robocasa.utils.object_utils import get_rel_transform, get_pos_after_rel_offset, get_fixture_to_point_rel_offset
from robocasa.models.objects.fixtures.fixture import get_texture_name_from_file

import robocasa

SIDES = ["left", "right", "front", "back"]

class Counter(ProcGenFixture):
    def __init__(
        self,
        name="counter",
        size=(0.72, 0.60, 0.60),
        overhang=0,

        # top
        top_texture=None,
        top_thickness=0.03,
        half_top=[False, False],  # for aligning corner counters

        # base, can use both
        base_texture=None,
        base_color=None,
        base_opening=[False, False],

        # to add bottom row cabinets inside
        # [back, front]
        hollow=[False, True],

        # for sinks, cooktops
        interior_obj=None,
        obj_y_percent=0.5,
        obj_x_percent=0.5,
        
        *args,
        **kwargs,
    ):
        self.has_opening = (interior_obj is not None)
        if self.has_opening:
            xml = "fixtures/counters/counter_with_opening"
        else:
            xml = "fixtures/counters/counter"
        self.interior_obj = None

        self.size = size
        self.th = top_thickness
        self.overhang = overhang
        self.half_top = half_top

        super().__init__(
            xml=xml,
            name=name,
            *args,
            **kwargs,
        )

        assert len(hollow) == 2
        self.hollow = hollow

        if len(base_opening) != 2 or sum(base_opening) > 1:
            raise ValueError("Invalid value for `base_opening`:", base_opening)
        self.base_opening = base_opening
        if sum(self.base_opening) == 1:
            # overwrites hollow parameter
            self.hollow = [False, False]

        if interior_obj is None:
            self._make_counter()
        else:
            if type(interior_obj) == str:
                raise ValueError("Sink must be initialized before CounterSink")
            self.interior_obj = interior_obj
            self.obj_x_percent = obj_x_percent
            self.obj_y_percent = obj_y_percent

            top_padding = self._place_interior_obj()
            self._make_counter_with_opening(top_padding)

        # set top texture
        self._set_texture(top_texture, base_texture, base_color)

        # set sites
        x, y, z = np.array(self.size) / 2
        self.set_bounds_sites({
            "ext_p0": [-x, -y + self.overhang, -z],
            "ext_px": [x, -y + self.overhang , -z],
            "ext_py": [-x, y, -z],
            "ext_pz": [-x, -y + self.overhang, z],
        })

    def _set_texture(self, top_texture, base_texture, base_color):
        # set top and bottom textures
        self.top_texture = xml_path_completion(top_texture, root=robocasa.models.assets_root)
        self.base_texture = xml_path_completion(base_texture, root=robocasa.models.assets_root)
        
        # set top texture and materials
        texture = find_elements(
            self.root, tags="texture", 
            attribs={"name": "tex_top_2d"},
            return_first=True
        )
        tex_name = get_texture_name_from_file(self.top_texture) + "_2d"
        texture.set("name", tex_name)
        texture.set("file", self.top_texture)
        material = find_elements(
            self.root, tags="material", 
            attribs={"name": "{}_counter_top".format(self.name)},
            return_first=True
        )
        material.set("texture", tex_name)
        
        texture = find_elements(
            self.root, tags="texture", 
            attribs={"name": "tex_base"},
            return_first=True
        )
        tex_name = get_texture_name_from_file(self.base_texture)
        texture.set("name", tex_name)
        texture.set("file", self.base_texture)
        material = find_elements(
            self.root, tags="material", 
            attribs={"name": "{}_counter_base".format(self.name)},
            return_first=True
        )
        material.set("texture", tex_name)

        # need to look at this later, not sure why
        prefix = self.naming_prefix if self.name != "counter" else ""
        
        # set base color
        if base_color is not None:
            self.base_color = base_color
            base_material = find_elements(
                self.root, "material", 
                attribs={"name": prefix + "counter_base"}, 
                return_first=True
            )
            
            if len(self.base_color) == 3:
                self.base_color.append(1)
            base_material.set("rgba", a2s(self.base_color))

    def _get_counter_geoms(self):
        """
        searches for geoms corresponding to each of the four components of the counter.
        """

        geoms = dict()
        for side in SIDES:
            geoms["base" + '_' + side] = list()
        
        if self.has_opening:
            for side in SIDES:
                geoms["top" + '_' + side] = list()
        else:
            geoms["top"] = list()

        for geom_name in geoms.keys():
            for postfix in ["", "_visual"]:
                g = find_elements(
                    root=self._obj,
                    tags="geom",
                    attribs={"name": "{}_{}{}".format(self.name, geom_name, postfix)},
                    return_first=True
                )
                geoms[geom_name].append(g)
        return geoms

    def _place_interior_obj(self):
        """
        calculates and sets the position of the sink, 
        calculates and returns the sizes of padding around the sink.

        x_percent/y_percent specifies at what percent of the fixture's entire width/depth
        the center of the sink should be placed.
        """

        x_percent, y_percent = self.obj_x_percent, self.obj_y_percent

        # remove overhang from consideration for placement
        top_size = [self.size[0], self.size[1] - self.overhang, self.size[2]] # remove the overhang

        # respect boundaires: limit range of x_percent and y_percent so interior object doesn't overflow
        gap = 0.02
        max_x_percent = (top_size[0] - gap - self.interior_obj.width / 2) / top_size[0]
        x_percent = np.clip(x_percent, 1 - max_x_percent, max_x_percent)
        max_y_percent = (top_size[1] - gap - self.interior_obj.depth / 2) / top_size[1]
        y_percent = np.clip(y_percent, 1 - max_y_percent, max_y_percent)

        # calculate and set the position of sink
        interior_origin = [
            self.pos[0] + (x_percent - 0.50) * top_size[0],
            self.pos[1] + self.overhang / 2 + (y_percent - 0.50) * top_size[1],
            self.pos[2] + top_size[2] / 2 - self.interior_obj.height / 2,
        ]

        self.interior_obj.set_origin(interior_origin)

        # calculate the size of padding around the sink
        left_pad = x_percent * top_size[0] - self.interior_obj.width / 2
        right_pad = (1 - x_percent) * top_size[0] - self.interior_obj.width / 2
        front_pad = y_percent * top_size[1] - self.interior_obj.depth / 2 + self.overhang # add the overhang to the front
        back_pad = (1 - y_percent) * top_size[1] - self.interior_obj.depth / 2

        return [left_pad, right_pad, front_pad, back_pad]

    def _get_chunks(self, pos, size, chunk_size=0.5):
        top_x_sizes = np.array(math.ceil(size[0]/chunk_size) * [chunk_size])
        top_x_sizes[-1] = size[0] - np.sum(top_x_sizes[:-1])

        chunk_sizes = [np.array([x_size, size[1], size[2]]) for x_size in top_x_sizes]
        left_pos = pos - np.array([size[0]/2, 0, 0])
        chunk_positions = []
        for i in range(len(top_x_sizes)):
            chunk_positions.append(np.array([
                left_pos[0] + np.sum(top_x_sizes[:max(0, i)]) + top_x_sizes[i] / 2,
                pos[1],
                pos[2],
            ]))
        return chunk_positions, chunk_sizes
    
    def _make_counter(self):
        w, d, h = np.array(self.size)
        th = self.th

        geoms = dict()

        # get the base geoms
        for side in SIDES:
            geoms["base" + '_' + side] = list()
        for geom_name in geoms.keys():
            for postfix in ["", "_visual"]:
                g = find_elements(
                    root=self._obj,
                    tags="geom",
                    attribs={"name": "{}_{}{}".format(self.name, geom_name, postfix)},
                    return_first=True
                )
                geoms[geom_name].append(g)

        assert sum(self.half_top) < 2
        if sum(self.half_top) == 0:
            pos = np.array([0, 0, h / 2 - th / 2])
            size = np.array([w, d, th])
        else:
            # used for aligning corner counters
            size = np.array([w / 2, d, th])
            if (self.half_top[1]):
                # keep right half
                pos = np.array([w / 4, 0, h / 2 - th / 2])
            else:
                # keep left half
                pos = np.array([-w / 4, 0, h / 2 - th / 2])

        half_size = size / 2

        geom_name = self._name + "_top"
        g_vis = new_geom(
            name=geom_name + "_visual",
            type="box",
            size=half_size,
            pos=pos,
            group=1,
            material=self._name + "_counter_top",
            density=10,
            conaffinity=0,
            contype=0,
            mass=1e-8,
        )
        self._obj.append(g_vis)
        # manually update visual geoms registry
        self._visual_geoms.append("top_visual")

        chunk_positions, chunk_sizes = self._get_chunks(pos, size, chunk_size=0.5)
        for i in range(len(chunk_sizes)):
            g = new_geom(
                name=geom_name + "_{}".format(i),
                type="box",
                size=chunk_sizes[i] / 2,
                pos=chunk_positions[i],
                group=0,
                density=10,
                rgba="0.5 0 0 1",
            )
            self._obj.append(g)
            # manually update contact geoms registry
            self._contact_geoms.append("top_{}".format(i))

        base_size, base_pos = self._get_base_dimensions()

        for side in SIDES:
            for elem in geoms["base_{}".format(side)]:
                if side == "front" and self.hollow[1]:
                    self._remove_element(elem)
                elif side == "back" and self.hollow[0]:
                    self._remove_element(elem)
                # houses bottom row cabinets
                elif sum(self.base_opening) == 0 and sum(self.hollow) > 0:
                    self._remove_element(elem)
                else:
                    elem.set("pos", a2s(base_pos[side]))
                    elem.set("size", a2s(base_size[side]))
    
    def _make_counter_with_opening(self, padding):
        """
        calculates and set the size and position of each component in the counter

        coordinate system change may or may not be necessary. currently all other fixtures
        use Mujoco's center-based coordinate system
        """
        w, d, h = self.size
        th = self.th
        side_th = 0.2 if sum(self.base_opening) == 1 else 0.0002
        left_pad, right_pad, front_pad, back_pad = padding

        if front_pad < self.overhang:
            raise ValueError("Overhang value is too large, must be lower " \
                             "than front padding ({:.2f})".format(front_pad))

        # calculate the (full) size of each component
        top_size = dict(
            back=[self.interior_obj.width, back_pad, h - th],
            front=[self.interior_obj.width, front_pad - self.overhang, h - th],
            left=[left_pad, d - self.overhang, h - th],
            right=[right_pad, d - self.overhang, h - th],
        )

        # all coordinates are the bottom-left corners
        # the origin is the bottom-left corner of the entire fixture
        top_pos = dict(
            back=[left_pad, front_pad + self.interior_obj.depth, -th],
            front=[left_pad, self.overhang, - th],
            left=[0, self.overhang, -th],
            right=[left_pad + self.interior_obj.width, self.overhang, - th],
        )

        base_size = dict(
            back=[w, 0.0002, h - th],
            front=[w, 0.0002, h - th],
            left=[side_th, d - self.overhang, h - th],
            right=[side_th, d - self.overhang, h - th],
        )
        base_pos = dict(
            back=[0.0, d, -th],
            front=[0.0, -d + 2 * self.overhang, -th],
            left=[-w + side_th, self.overhang, -th],
            right=[w - side_th, self.overhang, -th],
        )

        if self.base_opening[0]:
            # move panel to middle of the island
            base_pos["front"] = [0, 0, -th]
        elif self.base_opening[1]:
            base_pos["back"] = [0, 0, -th]

        geoms = dict()
        # get the base geoms
        for side in SIDES:
            geoms["base" + '_' + side] = list()
        for geom_name in geoms.keys():
            for postfix in ["", "_visual"]:
                g = find_elements(
                    root=self._obj,
                    tags="geom",
                    attribs={"name": "{}_{}{}".format(self.name, geom_name, postfix)},
                    return_first=True
                )
                geoms[geom_name].append(g)
        
        for side in SIDES:
            # convert coordinate of bottom-left corner to coordinate of center
            # the origin is now the center of the entire fixture
            top_pos[side][0] = (top_pos[side][0] + top_size[side][0] / 2 - w / 2)
            top_pos[side][1] = (top_pos[side][1] + top_size[side][1] / 2 - d / 2)
            top_pos[side][2] /= 2

            base_pos[side] = np.array(base_pos[side]) / 2

            for elem in geoms["base_{}".format(side)]:
                if side == "front" and self.hollow[1]:
                    self._remove_element(elem)
                elif side == "back" and self.hollow[0]:
                    self._remove_element(elem)
                elif sum(self.base_opening) == 0 and sum(self.hollow) > 0:
                    self._remove_element(elem)
                else:
                    elem.set("pos", a2s(base_pos[side]))
                    # divide sizes by 2 according to mujoco convention
                    elem.set("size", a2s(np.array(base_size[side]) / 2))

            # for elem in geoms["top_{}".format(side)]:
            pos = np.array(top_pos[side])
            size = np.array(top_size[side])
            pos[2] = (h - th) / 2
            size[2] = th

            # account for overhang
            if side != "back":
                size[1] += self.overhang
                pos[1] -= self.overhang / 2

            geom_name = self._name + "_top_{}".format(side)

            half_size = size / 2
            
            g_vis = new_geom(
                name=geom_name + "_visual",
                type="box",
                size=half_size,
                pos=pos,
                group=1,
                material=self._name + "_counter_top",
                density=10,
                conaffinity=0,
                contype=0,
                mass=1e-8,
            )
            self._obj.append(g_vis)
            # manually update visual geoms registry
            self._visual_geoms.append("top_{}_visual".format(side))

            chunk_positions, chunk_sizes = self._get_chunks(pos, size, chunk_size=0.5)
            for i in range(len(chunk_sizes)):
                g = new_geom(
                    name=geom_name + "_{}".format(i),
                    type="box",
                    size=chunk_sizes[i] / 2,
                    pos=chunk_positions[i],
                    group=0,
                    density=10,
                    rgba="0.5 0 0 1",
                    # rgba=a2s(np.concatenate((np.random.uniform(size=3), [1]))),
                )
                self._obj.append(g)
                self._contact_geoms.append("top_{}_{}".format(side, i))
    
    def _get_base_dimensions(self):
        # divide everything by 2 per mujoco convention
        x, y, z = np.array(self.size) / 2
        overhang = self.overhang / 2
        th = self.th / 2
        side_th = 0.1 if sum(self.base_opening) == 1 else 0.0001

        base_size = dict(
            back=[x-side_th*2, th, z - th],
            front=[x-side_th*2,  th, z - th],
            left=[side_th, y - overhang, z - th],
            right=[side_th, y - overhang, z - th],
        )
        base_pos = dict(
            back=[0, y - th, -th],
            front=[0, -y + 2 * overhang + th, -th],
            left=[-x + side_th, overhang, -th],
            right=[x - side_th, overhang, -th],
        )

        if self.base_opening[0]:
            # move panel to middle of the island
            base_pos["front"] = [0, 0, -th]
        elif self.base_opening[1]:
            base_pos["back"] = [0, 0, -th]

        return base_size, base_pos

    # to overwrite Fixture class default
    @property
    def width(self):
        return self.size[0]
    
    @property
    def depth(self):
        return self.size[1]
    
    @property
    def height(self):
        return self.size[2]

    def set_pos(self, pos):
        super().set_pos(pos)
        # we have to set the postion of the interior object as well
        if self.interior_obj is not None:
            self._place_interior_obj()

    def get_reset_regions(self, env, ref=None, loc="nn", top_size=(0.4, 0.4) ):
        """
        returns dictionary of reset regions, each region defined as offsets and size
        """
        all_geoms = []
        for (k, v) in self._get_counter_geoms().items():
            if not k.startswith("top"):
                continue
            geom = v[-1]
            top_pos = s2a(geom.get("pos"))
            this_top_size = s2a(geom.get("size")) * 2
            # make sure region is sufficiently large
            if this_top_size[0] >= top_size[0] and this_top_size[1] >= top_size[1]:
                all_geoms.append(geom)

        # # randomize order of geoms
        # random.shuffle(valid_top_geoms)

        reset_regions = {}

        if ref is None:
            geom_i = 0
            for g in all_geoms:
                top_pos = s2a(g.get("pos"))
                top_half_size = s2a(g.get("size"))
                offset = (top_pos[0], top_pos[1], self.size[2] / 2)
                size = (top_half_size[0] * 2, top_half_size[1] * 2)
                reset_regions[f"geom_{geom_i}"] = dict(size=size, offset=offset)
                geom_i += 1
        else:
            ref_fixture = env.get_fixture(ref)

            ### find an appropriate geom to sample ###
            fixture_to_geom_offsets = []
            for g in all_geoms:
                g_pos = get_pos_after_rel_offset(self, s2a(g.get("pos")))
                rel_offset = get_fixture_to_point_rel_offset(ref_fixture, g_pos)
                fixture_to_geom_offsets.append(rel_offset)
            
            valid_geoms = []

            if loc == "nn":
                dists = [np.linalg.norm(offset) for offset in fixture_to_geom_offsets]
                chosen_top = all_geoms[np.argmin(dists)]
                valid_geoms.append(chosen_top)
            elif loc == "right":
                for offset, g in zip(fixture_to_geom_offsets, all_geoms):
                    if offset[0] > 0.30:
                        valid_geoms.append(g)
            elif loc == "left":
                for offset, g in zip(fixture_to_geom_offsets, all_geoms):
                    if offset[0] < -0.30:
                        valid_geoms.append(g)
            elif loc == "left_right":
                for offset, g in zip(fixture_to_geom_offsets, all_geoms):
                    if np.abs(offset[0]) > 0.30:
                        valid_geoms.append(g)
            elif loc == "any":
                valid_geoms = all_geoms
            else:
                raise ValueError
            
            geom_i = 0
            for g in valid_geoms:
                top_pos = s2a(g.get("pos"))
                top_half_size = s2a(g.get("size"))
                offset = [top_pos[0], top_pos[1], self.size[2] / 2]
                size = [top_half_size[0] * 2, top_half_size[1] * 2]

                min_x = top_pos[0] - top_half_size[0]
                max_x = top_pos[0] + top_half_size[0]

                ref_pos, _ = get_rel_transform(self, ref_fixture)
                if min_x <= ref_pos[0] <= max_x:
                    # restrict sample region to be below fixture
                    offset[0] = ref_pos[0]
                    size[0] = min(ref_pos[0] - min_x, max_x - ref_pos[0]) * 2

                reset_regions[f"geom_{geom_i}"] = dict(size=size, offset=offset)
                geom_i += 1
        return reset_regions
