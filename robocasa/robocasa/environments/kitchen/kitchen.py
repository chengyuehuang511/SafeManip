import os
import xml.etree.ElementTree as ET
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import (
    array_to_string,
    find_elements,
    string_to_array,
)
from robosuite.models.robots.robot_model import REGISTERED_ROBOTS
from robosuite.utils.observables import Observable, sensor
from robosuite.environments.base import EnvMeta
from collections import defaultdict
from scipy.spatial.transform import Rotation

from robosuite.models.robots import PandaOmron

import robocasa
import robocasa.macros as macros
import robocasa.utils.camera_utils as CamUtils
import robocasa.utils.env_utils as EnvUtils
import robocasa.models.scenes.scene_registry as SceneRegistry
from robocasa.models.scenes import KitchenArena
from robocasa.models.fixtures import *
import robocasa.models.fixtures.fixture_utils as FixtureUtils
from robocasa.models.objects.kitchen_object_utils import (
    sample_kitchen_object,
)
from robocasa.utils.texture_swap import (
    get_random_textures,
    replace_cab_textures,
    replace_counter_top_texture,
    replace_floor_texture,
    replace_wall_texture,
)
from robocasa.utils.config_utils import refactor_composite_controller_config
from robocasa.utils.errors import PlacementError
from robocasa.models.objects.kitchen_objects import OBJ_GROUPS, OBJ_CATEGORIES
from robocasa.models.fixtures.fixture_utils import fixture_is_type
from robocasa.models.fixtures import FixtureType
from robocasa.environments.kitchen.predicates import (
    build_predicate_snapshot,
    build_predicate_static_spec,
)
import re


REGISTERED_KITCHEN_ENVS = {}
SLIDING_INTERIOR_FIXTURES = [
    FixtureType.DRAWER,
    FixtureType.DISHWASHER,
    FixtureType.OVEN,
    FixtureType.TOASTER_OVEN,
    FixtureType.FRIDGE,
]


def register_kitchen_env(target_class):
    REGISTERED_KITCHEN_ENVS[target_class.__name__] = target_class


class KitchenEnvMeta(EnvMeta):
    """Metaclass for registering robocasa environments"""

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)
        if cls.__name__ not in [
            "MG_Robocasa_Env",
            "PickPlace",
            "ManipulateDoor",
            "OpenDoor",
            "CloseDoor",
        ]:
            register_kitchen_env(cls)
        return cls


class Kitchen(ManipulationEnv, metaclass=KitchenEnvMeta):
    """
    Initialized a Base Kitchen environment.

    Args:
        robots: Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)

        env_configuration (str): Specifies how to position the robot(s) within the environment. Default is "default",
            which should be interpreted accordingly by any subclasses.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        base_types (None or str or list of str): type of base, used to instantiate base models from base factory.
            Default is "default", which is the default base associated with the robot(s) the 'robots' specification.
            None results in no base, and any other (valid) model overrides the default base. Should either be
            single str if same base type is to be used for all robots or else it should be a list of the same
            length as "robots" param

        gripper_types (None or str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        placement_initializer (ObjectPositionSampler): if provided, will be used to place objects on every reset,
            else a UniformRandomSampler is used by default.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the abase of
            simulation time that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        renderer (str): Specifies which renderer to use.

        renderer_config (dict): dictionary for the renderer configurations

        init_robot_base_ref (str): name of the fixture to place the near. If None, will randomly select a fixture.

        seed (int): environment seed. Default is None, where environment is unseeded, ie. random

        layout_and_style_ids (list of list of int): list of layout and style ids to use for the kitchen.

        layout_ids ((list of) LayoutType or int or dict):  layout id(s) to use for the kitchen. -1 and None specify all layouts
            -2 specifies layouts not involving islands/wall stacks, -3 specifies layouts involving islands/wall stacks,
            -4 specifies layouts with dining areas.

        enable_fixtures (list of str): a list of fixtures to enable in the scene (some fixtures are disabled by default)

        update_fxtr_cfg_dict (dict): a dictionary which maps fixture names to dictionaries to update the fixture config specified in the layout yaml.

        style_ids ((list of) StyleType or int or dict): style id(s) to use for the kitchen. -1 and None specify all styles.

        generative_textures (str): if set to "100p", will use AI generated textures

        obj_registries (tuple of str): tuple containing the object registries to use for sampling objects.
            can contain "objaverse" and/or "aigen" to sample objects from objaverse, AI generated, or both.

        obj_instance_split (str): string for specifying a custom set of object instances to use. "train" specifies
            all but the last 4 object instances (or the first half - whichever is larger), "test" specifies the
            rest, and None specifies all.

        use_distractors (bool): if True, will add distractor objects to the scene

        translucent_robot (bool): if True, will make the robot appear translucent during rendering

        randomize_cameras (bool): if True, will add gaussian noise to the position and rotation of the
            wrist and agentview cameras

        clutter_mode (int): sets clutter level. default is 0.
    """

    EXCLUDE_LAYOUTS = []

    EXCLUDE_STYLES = []

    PROBLEMATIC_BLENDER_LID_STYLES = [
        13,
        29,
        31,
    ]

    OVEN_EXCLUDED_LAYOUTS = [
        1,
        3,
        5,
        6,
        8,
        10,
        11,
        13,
        14,
        16,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        30,
        32,
        33,
        36,
        38,
        40,
        41,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
    ]

    DOUBLE_CAB_EXCLUDED_LAYOUTS = [32, 41, 59]

    DINING_COUNTER_EXCLUDED_LAYOUTS = [
        1,
        3,
        5,
        6,
        18,
        20,
        36,
        39,
        40,
        43,
        47,
        50,
        52,
    ]

    ISLAND_EXCLUDED_LAYOUTS = [
        1,
        3,
        5,
        6,
        8,
        9,
        10,
        11,
        13,
        18,
        19,
        22,
        27,
        30,
        36,
        40,
        43,
        46,
        47,
        52,
        53,
        60,
    ]

    # freezer is inaccessible for french_door and bottom_freezer
    FREEZER_EXCLUDED_LAYOUTS = [
        1,
        3,
        4,
        5,
        6,
        8,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        19,
        20,
        22,
        23,
        25,
        26,
        27,
        28,
        30,
        31,
        33,
        34,
        35,
        36,
        37,
        39,
        40,
        42,
        46,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        58,
        59,
        60,
    ]

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise="default",
        use_camera_obs=True,
        use_object_obs=True,  # currently unused variable
        reward_scale=1.0,  # currently unused variable
        reward_shaping=False,  # currently unused variables
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="robot0_agentview_center",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        renderer="mjviewer",
        renderer_config=None,
        init_robot_base_ref=None,
        seed=None,
        layout_and_style_ids=None,
        layout_ids=None,
        style_ids=None,
        enable_fixtures=None,
        generative_textures=None,
        obj_registries=(
            "objaverse",
            "lightwheel",
        ),
        obj_instance_split=None,
        use_distractors=False,
        translucent_robot=False,
        randomize_cameras=False,
        robot_spawn_deviation_pos_x=0.15,
        robot_spawn_deviation_pos_y=0.05,
        robot_spawn_deviation_rot=0.0,
        clutter_mode=0,
        update_fxtr_cfg_dict=None,
        use_cotraining_cameras=False,
        use_novel_instructions=False,
    ):
        self.init_robot_base_ref = init_robot_base_ref

        self.robot_spawn_deviation_pos_x = robot_spawn_deviation_pos_x
        self.robot_spawn_deviation_pos_y = robot_spawn_deviation_pos_y
        self.robot_spawn_deviation_rot = robot_spawn_deviation_rot

        # object placement initializer
        self.placement_initializer = placement_initializer
        self.obj_registries = obj_registries
        self.obj_instance_split = obj_instance_split

        if layout_and_style_ids is not None:
            assert (
                layout_ids is None and style_ids is None
            ), "layout_ids and style_ids must both be set to None if layout_and_style_ids is set"
            if isinstance(layout_and_style_ids, str):
                if layout_and_style_ids == "5x5":
                    self.layout_and_style_ids = EnvUtils.KITCHEN_SCENES_5X5
                elif layout_and_style_ids == "5x1":
                    self.layout_and_style_ids = EnvUtils.KITCHEN_SCENES_5X1
            else:
                self.layout_and_style_ids = layout_and_style_ids
        else:
            layout_ids = SceneRegistry.unpack_layout_ids(layout_ids)
            style_ids = SceneRegistry.unpack_style_ids(style_ids)
            self.layout_and_style_ids = [(l, s) for l in layout_ids for s in style_ids]

        # remove excluded layouts and styles
        self.layout_and_style_ids = [
            (l, s)
            for (l, s) in self.layout_and_style_ids
            if l not in self.EXCLUDE_LAYOUTS and s not in self.EXCLUDE_STYLES
        ]

        self.enable_fixtures = enable_fixtures
        self.update_fxtr_cfg_dict = update_fxtr_cfg_dict
        self.clutter_mode = clutter_mode
        assert generative_textures in [None, False, "100p"]
        self.generative_textures = generative_textures

        self.use_distractors = use_distractors
        self.translucent_robot = translucent_robot
        self.randomize_cameras = randomize_cameras

        if isinstance(robots, str):
            robots = [robots]

        # backward compatibility: rename all robots that were previously called PandaMobile -> PandaOmron
        for i in range(len(robots)):
            if robots[i] == "PandaMobile":
                robots[i] = "PandaOmron"
        assert len(robots) == 1

        # intialize cameras
        self.use_cotraining_cameras = use_cotraining_cameras
        self._cam_configs = CamUtils.get_robot_cam_configs(
            robots[0], use_cotraining_cameras=self.use_cotraining_cameras
        )

        # set up currently unused variables (used in robosuite)
        self.use_object_obs = use_object_obs
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        if controller_configs is not None:
            # detect if using stale controller configs (before robosuite v1.5.1) and update to new convention
            arms = REGISTERED_ROBOTS[robots[0]].arms
            controller_configs = refactor_composite_controller_config(
                controller_configs, robots[0], arms
            )
            if robots[0] == "PandaOmron":
                if "composite_controller_specific_configs" not in controller_configs:
                    controller_configs["composite_controller_specific_configs"] = {}
                controller_configs["composite_controller_specific_configs"][
                    "body_part_ordering"
                ] = ["right", "right_gripper", "base", "torso"]
                controller_configs["body_parts"]["base"][
                    "type"
                ] = "JOINT_VELOCITY_LEGACY"

        # if not hasattr(self, "_reset_internal_end_callbacks"):
        #     self._reset_internal_end_callbacks = []
        # self._reset_internal_end_callbacks.append(lambda: EnvUtils.set_robot_state(self))

        self.use_novel_instructions = use_novel_instructions
        if self.use_novel_instructions:
            self._load_novel_instructions()
        else:
            self.novel_instructions = None

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types=base_types,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=True,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=True,
            load_model_on_init=False,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            renderer=renderer,
            renderer_config=renderer_config,
            seed=seed,
        )

    def _load_novel_instructions(self):
        """
        Loads novel instructions for the environment.
        """
        # load novel instructions from csv file
        import csv

        novel_instructions_path = os.path.join(
            robocasa.__path__[0],
            "models",
            "assets",
            "novel_instructions",
            "task_instruction_variants.csv",
        )
        self.novel_instructions = None
        task_name = self.__class__.__name__
        with open(novel_instructions_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] == task_name:
                    # column 0 is task name, column 1 is the original instruction, rest are new instructions
                    self.novel_instructions = list(row[2:])
                    break

        assert (
            self.novel_instructions is not None
        ), f"Novel instructions for task {task_name} not found in {novel_instructions_path}"

    def _setup_model(self):
        """
        helper function called by _load_model to setup the mjcf model
        """
        for robot in self.robots:
            if isinstance(robot.robot_model, PandaOmron):
                robot.init_qpos = (
                    -0.01612974,
                    -1.03446714,
                    -0.02397936,
                    -2.27550888,
                    0.03932365,
                    1.51639493,
                    0.69615947,
                )
                robot.init_torso_qpos = np.array([0.0])

        for robot in self.robots:
            if isinstance(robot.robot_model, PandaOmron):
                robot.init_qpos = (
                    -0.01612974,
                    -1.03446714,
                    -0.02397936,
                    -2.27550888,
                    0.03932365,
                    1.51639493,
                    0.69615947,
                )
                robot.init_torso_qpos = np.array([0.0])

        # determine sample layout and style
        if "layout_id" in self._ep_meta and "style_id" in self._ep_meta:
            self.layout_id = self._ep_meta["layout_id"]
            self.style_id = self._ep_meta["style_id"]
        else:
            layout_id, style_id = self.rng.choice(self.layout_and_style_ids)
            self.layout_id = layout_id
            self.style_id = style_id

        if macros.VERBOSE:
            print(
                "layout: {}, style: {}".format(
                    self.layout_id,
                    "custom" if isinstance(self.style_id, dict) else self.style_id,
                )
            )

        # to be set later inside edit_model_xml function
        self._curr_gen_fixtures = self._ep_meta.get("gen_textures")

        # setup scene
        self.mujoco_arena = KitchenArena(
            layout_id=self.layout_id,
            style_id=self.style_id,
            rng=self.rng,
            enable_fixtures=self.enable_fixtures,
            clutter_mode=self.clutter_mode,
            update_fxtr_cfg_dict=self.update_fxtr_cfg_dict,
        )
        # Arena always gets set to zero origin
        self.mujoco_arena.set_origin([0, 0, 0])
        CamUtils.set_cameras(self)  # setup cameras

        # setup rendering for this layout
        if self.renderer == "mjviewer":
            camera_config = CamUtils.LAYOUT_CAMS.get(
                self.layout_id, CamUtils.DEFAULT_LAYOUT_CAM
            )
            self.renderer_config = {"cam_config": camera_config}

        # setup fixtures
        self.fixture_cfgs = self.mujoco_arena.get_fixture_cfgs()
        self.fixtures = {cfg["name"]: cfg["model"] for cfg in self.fixture_cfgs}

        # setup scene, robots, objects
        self.model = ManipulationTask(
            mujoco_arena=self.mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=list(self.fixtures.values()),
            enable_multiccd=True,
            enable_sleeping_islands=False,
        )

    def _load_model(self, attempt_num=1):
        """
        Loads an xml model, puts it in self.model
        """
        if attempt_num >= 50:
            raise RuntimeError(
                "Ran _load_model() 50 times but could not initialize task!"
            )

        super()._load_model()

        self._setup_model()

        if self.init_robot_base_ref is not None:
            for i in range(50):  # keep searching for valid environment
                init_fixture = self.get_fixture(self.init_robot_base_ref)
                if init_fixture is not None:
                    break
                self._setup_model()

        MAX_FIXTURE_PLACEMENT_ATTEMPTS = 3
        self.fxtr_placements = {}

        """
        step 1: identify all fixtures that involve auxiliary items
        these have their own placement logic
        """
        # identify base-aux pairs
        paired_fixtures = defaultdict(dict)

        def strip_group_suffix(name):
            if "_auxiliary_" in name:
                prefix, _, group_suffix = name.partition("_auxiliary_")
                return prefix
            return name

        for fxtr_name in self.fixtures:
            matched = False

            # First pass: check against known base fixture names
            for base, aux in Fixture.BASE_TO_AUXILIARY_FIXTURES.items():
                if fxtr_name.startswith(base) and "_auxiliary" not in fxtr_name:
                    paired_fixtures[fxtr_name]["base"] = fxtr_name
                    matched = True
                elif fxtr_name.startswith(base) and "_auxiliary" in fxtr_name:
                    canonical_base = strip_group_suffix(fxtr_name)
                    paired_fixtures[canonical_base]["aux"] = fxtr_name
                    matched = True

            # Second pass: reverse lookup based on known auxiliary names
            if not matched:
                for aux in Fixture.BASE_TO_AUXILIARY_FIXTURES.values():
                    if fxtr_name.startswith(aux):
                        canonical_base = strip_group_suffix(fxtr_name)
                        paired_fixtures[canonical_base]["aux"] = fxtr_name

        paired_fixtures = {
            k: v for k, v in paired_fixtures.items() if "base" in v and "aux" in v
        }
        paired_names = {
            v
            for pair in paired_fixtures.values()
            for v in pair.values()
            if v in self.fixtures
        }

        # sample aux fixture pairs first
        for _, pair in paired_fixtures.items():
            base_name = pair.get("base")
            aux_name = pair.get("aux")

            if not base_name or not aux_name:
                continue

            cfg_base = next(
                cfg for cfg in self.fixture_cfgs if cfg["name"] == base_name
            )
            cfg_aux = next(cfg for cfg in self.fixture_cfgs if cfg["name"] == aux_name)

            cfg_pair = [cfg_base, cfg_aux]
            sampler_pair = EnvUtils.get_single_fixture_sampler(self, cfg_pair)

            success = False
            for _ in range(MAX_FIXTURE_PLACEMENT_ATTEMPTS * 2):
                try:
                    placement_pair = sampler_pair.sample(
                        placed_objects=self.fxtr_placements
                    )
                    self.fxtr_placements.update(placement_pair)

                    success = True
                    break

                except PlacementError as e:
                    if macros.VERBOSE:
                        print(f"Placement error for auxiliary pair fixture: {e}")
                    # retry both objects if one fails
                    for name in [base_name, aux_name]:
                        if name in self.fxtr_placements:
                            del self.fxtr_placements[name]

            if not success:
                if macros.VERBOSE:
                    print(
                        f"Could not place {fxtr_name}. Trying again with self._load_model()"
                    )
                self._destroy_sim()
                self._load_model(attempt_num=attempt_num + 1)
                return

        """
        step 2: placement logic for all other fixtures

        """
        for fxtr_obj in self.fixtures.values():

            fxtr_name = fxtr_obj.name
            if fxtr_name in paired_names:
                continue

            fxtr_cfg = next(
                cfg for cfg in self.fixture_cfgs if cfg["name"] == fxtr_name
            )
            sampler = EnvUtils.get_single_fixture_sampler(self, fxtr_cfg)

            success = False
            for _ in range(MAX_FIXTURE_PLACEMENT_ATTEMPTS):
                try:
                    placement = sampler.sample(placed_objects=self.fxtr_placements)
                    self.fxtr_placements.update(placement)
                    success = True
                    break
                except PlacementError as e:
                    if macros.VERBOSE:
                        print(f"Placement error for fixture: {e}")
                    continue

            if not success:
                if macros.VERBOSE:
                    print(
                        f"Could not place {fxtr_name}. Trying again with self._load_model()"
                    )
                self._destroy_sim()
                self._load_model(attempt_num=attempt_num + 1)
                return

        # apply placements
        for obj_pos, obj_quat, obj in self.fxtr_placements.values():
            assert isinstance(obj, Fixture)
            obj.set_pos(obj_pos)
            # hacky code to set orientation
            obj.set_euler(T.mat2euler(T.quat2mat(T.convert_quat(obj_quat, "xyzw"))))

        # setup internal references related to fixtures
        self._setup_kitchen_references()

        # create and place objects
        self._create_objects()

        # setup object locations
        try:
            self.placement_initializer = EnvUtils._get_placement_initializer(
                self, self.object_cfgs
            )
        except PlacementError as e:
            if macros.VERBOSE:
                print(
                    "Could not create placement initializer for objects. Trying again with self._load_model()"
                )
            self._destroy_sim()
            self._load_model(attempt_num=attempt_num + 1)
            return
        object_placements = None
        for attempt in range(1):
            try:
                object_placements = self.placement_initializer.sample(
                    placed_objects=self.fxtr_placements
                )
            except PlacementError as e:
                if macros.VERBOSE:
                    print("Placement error for objects")
                continue
            break
        if object_placements is None:
            if macros.VERBOSE:
                print("Could not place objects. Trying again with self._load_model()")
            self._destroy_sim()
            self._load_model(attempt_num=attempt_num + 1)
            return

        self.object_placements = object_placements

        (
            self.init_robot_base_pos_anchor,
            self.init_robot_base_ori_anchor,
        ) = EnvUtils.init_robot_base_pose(self)

        robot_model = self.robots[0].robot_model
        # set the robot way out of the scene at the start, it will be placed correctly later
        robot_model.set_base_xpos([10.0, 10.0, self.init_robot_base_pos_anchor[2]])
        robot_model.set_base_ori(self.init_robot_base_ori_anchor)

        self.robot_geom_ids = None

    def _create_objects(self):
        """
        Creates and places objects in the kitchen environment.
        Helper function called by _create_objects()
        """
        # add objects
        self.objects = {}
        if "object_cfgs" in self._ep_meta:
            self.object_cfgs = self._ep_meta["object_cfgs"]
            for obj_num, cfg in enumerate(self.object_cfgs):
                if "name" not in cfg:
                    cfg["name"] = "obj_{}".format(obj_num + 1)
                model, info = EnvUtils.create_obj(self, cfg)
                cfg["info"] = info
                self.objects[model.name] = model
                self.model.merge_objects([model])
        else:
            self.object_cfgs = self._get_obj_cfgs()
            all_obj_cfgs = []
            for obj_num, cfg in enumerate(self.object_cfgs):
                cfg["type"] = "object"
                if "name" not in cfg:
                    cfg["name"] = "obj_{}".format(obj_num + 1)
                model, info = EnvUtils.create_obj(self, cfg)
                cfg["info"] = info
                self.objects[model.name] = model
                self.model.merge_objects([model])

                try_to_place_in = cfg["placement"].get("try_to_place_in", None)
                object_ref = cfg["placement"].get("object", None)

                if try_to_place_in and (
                    "in_container" in cfg["info"]["groups_containing_sampled_obj"]
                ):
                    container_cfg = {
                        "name": cfg["name"] + "_container",
                        "obj_groups": cfg["placement"].get("try_to_place_in"),
                        "placement": deepcopy(cfg["placement"]),
                        "type": "object",
                    }

                    init_robot_here = cfg.get("init_robot_here", False)
                    if init_robot_here is True:
                        cfg["init_robot_here"] = False
                        container_cfg["init_robot_here"] = True

                    try_to_place_in_kwargs = cfg["placement"].get(
                        "try_to_place_in_kwargs", None
                    )
                    if try_to_place_in_kwargs is not None:
                        for k, v in try_to_place_in_kwargs.items():
                            container_cfg[k] = v

                    # add in the new object to the model
                    all_obj_cfgs.append(container_cfg)
                    model, info = EnvUtils.create_obj(self, container_cfg)
                    container_cfg["info"] = info
                    self.objects[model.name] = model
                    self.model.merge_objects([model])

                    # modify object config to lie inside of container
                    cfg["placement"] = dict(
                        size=(0.01, 0.01),
                        ensure_object_boundary_in_range=False,
                        sample_args=dict(
                            reference=container_cfg["name"],
                        ),
                    )
                elif (
                    object_ref
                    and "in_container" in info["groups_containing_sampled_obj"]
                ):
                    parent = object_ref
                    if parent in self.objects:
                        container_name = parent
                        container_obj = self.objects[parent]
                        container_size = container_obj.size
                        smaller_dim = min(container_size[0], container_size[1])
                        sampling_size = (smaller_dim * 0.5, smaller_dim * 0.5)
                        cfg["placement"] = {
                            "size": sampling_size,
                            "ensure_object_boundary_in_range": False,
                            "sample_args": {"reference": container_name},
                        }

                # append the config for this object
                all_obj_cfgs.append(cfg)

                # trigger only if the task enabled it and the base category declares an auxiliary_obj
                aux_enable = cfg.get("auxiliary_obj_enable", False)
                obj_group = cfg.get("obj_groups", "")
                auxiliary_obj_group = None

                if aux_enable and obj_group in OBJ_CATEGORIES:
                    for registry_name, obj_cat in OBJ_CATEGORIES[obj_group].items():
                        if hasattr(obj_cat, "auxiliary_obj") and obj_cat.auxiliary_obj:
                            auxiliary_obj_group = obj_cat.auxiliary_obj
                            break

                if aux_enable and auxiliary_obj_group:
                    aux_obj_cfg = {}
                    aux_obj_cfg["name"] = cfg["name"] + "_auxiliary"
                    aux_obj_cfg["type"] = "object"
                    direct_aux_mjcf_path = self._get_aux_obj_instance(
                        cfg.get("info", {}).get("mjcf_path")
                    )
                    aux_obj_cfg["obj_groups"] = (
                        direct_aux_mjcf_path or auxiliary_obj_group
                    )
                    if "object_scale" in cfg:
                        aux_obj_cfg["object_scale"] = cfg["object_scale"]
                    specified_aux_placement = cfg.get("auxiliary_obj_placement")
                    if specified_aux_placement is not None:
                        aux_obj_cfg["placement"] = specified_aux_placement
                    else:
                        aux_obj_cfg["placement"] = dict(
                            anchor_to=cfg["name"],
                            ensure_object_boundary_in_range=False,
                            ensure_valid_placement=False,
                        )

                    all_obj_cfgs.append(aux_obj_cfg)
                    model, info = EnvUtils.create_obj(self, aux_obj_cfg)
                    aux_obj_cfg["info"] = info
                    self.objects[model.name] = model
                    self.model.merge_objects([model])

            # prepend the new object configs in
            self.object_cfgs = all_obj_cfgs

            # # remove objects that didn't get created
            # self.object_cfgs = [cfg for cfg in self.object_cfgs if "model" in cfg]

    def _get_aux_obj_instance(self, base_mjcf_path):
        """
        Given a base object's mjcf_path, try to resolve a direct auxiliary
        object instance path

        Returns a string path to the auxiliary object's model.xml if found,
        otherwise None.
        """
        if not isinstance(base_mjcf_path, str):
            return None
        base_dir = os.path.dirname(base_mjcf_path)
        for subdir in os.listdir(base_dir):
            subdir_path = os.path.join(base_dir, subdir)
            if os.path.isdir(subdir_path):
                candidate = os.path.join(subdir_path, "model.xml")
                if os.path.exists(candidate):
                    return candidate
        return None

    def _setup_kitchen_references(self):
        """
        setup fixtures (and their references). this function is called within load_model function for kitchens
        """
        serialized_refs = self._ep_meta.get("fixture_refs", {})
        # unserialize refs
        self.fixture_refs = {
            k: self.get_fixture(v) for (k, v) in serialized_refs.items()
        }

    def _reset_observables(self):
        if self.hard_reset:
            self._observables = self._setup_observables()

    def _update_sliding_fxtr_obj_placement(self):
        initial_state_copy = self.sim.get_state()
        # update the dynamics so that any sliding joints/regions that were
        # moved during reset internal are reflected in sim state
        self.sim.forward()

        # cache offsets to object placements to be re-used for objects who's
        # placement depends on the moved object
        sliding_fixture_objs = {}
        for obj_i, cfg in enumerate(self.object_cfgs):
            obj_name = cfg["name"]
            ref = cfg["placement"].get("sample_args", {}).get("reference")
            # check if current object's placement depends on an object who's
            # placement has been updated
            if ref and ref in sliding_fixture_objs:
                pos_offset = sliding_fixture_objs[
                    cfg["placement"]["sample_args"]["reference"]
                ]

                old_obj_pos, old_obj_quat, obj = self.object_placements[obj_name]
                new_obj_pos = tuple(np.array(old_obj_pos) + pos_offset)
                self.object_placements[obj_name] = (
                    new_obj_pos,
                    old_obj_quat,
                    obj,
                )
            # check if objects placement should be update if it was placed
            # inside a fixture with a sliding joint
            elif any(
                [
                    fixture_is_type(cfg["placement"].get("fixture", -1), fxtr_type)
                    for fxtr_type in SLIDING_INTERIOR_FIXTURES
                ]
            ):
                sliding_fixture = cfg["placement"]["fixture"]
                reset_region_name = cfg["reset_region"]["name"]
                reset_region_curr_pos = self.sim.data.get_geom_xpos(
                    f"{sliding_fixture.naming_prefix}reg_{reset_region_name}"
                )
                # original position of reset region relative to fixture
                reset_region_orig_pos = s2a(
                    sliding_fixture._regions[reset_region_name]["elem"].get(
                        "pos", "0 0 0"
                    )
                )
                # original position of reset region in absolute coordinates
                reset_region_orig_pos = OU.get_pos_after_rel_offset(
                    sliding_fixture, reset_region_orig_pos
                )
                # change in position of reset region after dynamics updated
                reset_region_offset = reset_region_curr_pos - reset_region_orig_pos
                sliding_fixture_objs[obj_name] = reset_region_offset

                old_obj_pos, old_obj_quat, obj = self.object_placements[obj_name]
                new_obj_pos = tuple(np.array(old_obj_pos) + reset_region_offset)
                # updated position
                self.object_placements[obj_name] = (
                    new_obj_pos,
                    old_obj_quat,
                    obj,
                )

                # update visualization of sampling regions according to changes
                # to sliding joint
                if macros.SHOW_SITES:
                    for region in ["reset_region_outer", "reset_region_inner"]:
                        xml_site = find_elements(
                            self.model.worldbody,
                            "site",
                            dict(name=f"{region}_{obj_name}"),
                            return_first=True,
                        )
                        prev_pos = s2a(xml_site.get("pos", "0 0 0"))
                        new_pos = prev_pos + reset_region_offset
                        xml_site.set(
                            "pos",
                            array_to_string(new_pos),
                        )
        # reset sim state to before the most recent forward call to maintain
        # consistent physics
        self.sim.set_state(initial_state_copy)

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # set up the scene (fixtures, variables, etc)
        self._setup_scene()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset and self.placement_initializer is not None:
            # use pre-computed object placements
            object_placements = self.object_placements
            self._update_sliding_fxtr_obj_placement()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(
                    obj.joints[0],
                    np.concatenate([np.array(obj_pos), np.array(obj_quat)]),
                )

        # set the robot here
        if "init_robot_base_pos" in self._ep_meta:
            self.init_robot_base_pos = self._ep_meta["init_robot_base_pos"]
            self.init_robot_base_ori = self._ep_meta["init_robot_base_ori"]
            EnvUtils.set_robot_to_position(self, self.init_robot_base_pos)
            self.sim.forward()
        else:
            robot_pos = EnvUtils.set_robot_base(
                env=self,
                anchor_pos=self.init_robot_base_pos_anchor,
                anchor_ori=self.init_robot_base_ori_anchor,
                rot_dev=self.robot_spawn_deviation_rot,
                pos_dev_x=self.robot_spawn_deviation_pos_x,
                pos_dev_y=self.robot_spawn_deviation_pos_y,
            )
            self.init_robot_base_pos = robot_pos
            self.init_robot_base_ori = self.init_robot_base_ori_anchor

        # step through a few timesteps to settle objects
        action = np.zeros(self.action_spec[0].shape)  # apply empty action

        # Since the env.step frequency is slower than the mjsim timestep frequency, the internal controller will output
        # multiple torque commands in between new high level action commands. Therefore, we need to denote via
        # 'policy_step' whether the current step we're taking is simply an internal update of the controller,
        # or an actual policy update
        policy_step = True

        # Loop through the simulation at the model timestep rate until we're ready to take the next policy step
        # (as defined by the control frequency specified at the environment level)
        for i in range(10 * int(self.control_timestep / self.model_timestep)):
            self.sim.step1()
            self._pre_action(action, policy_step)
            self.sim.step2()
            policy_step = False

    def _setup_scene(self):
        pass

    def _get_obj_cfgs(self):
        """
        Returns a list of object configurations to use in the environment.
        The object configurations are usually environment-specific and should
        be implemented in the subclass.

        Returns:
            list: list of object configurations
        """

        return []

    def get_ep_meta(self):
        """
        Returns a dictionary containing episode meta data
        """

        def copy_dict_for_json(orig_dict):
            new_dict = {}
            for (k, v) in orig_dict.items():
                if isinstance(v, dict):
                    new_dict[k] = copy_dict_for_json(v)
                elif isinstance(v, Fixture):
                    new_dict[k] = v.name
                else:
                    new_dict[k] = v
            return new_dict

        ep_meta = super().get_ep_meta()
        ep_meta["layout_id"] = (
            self.layout_id if isinstance(self.layout_id, dict) else int(self.layout_id)
        )
        ep_meta["style_id"] = (
            self.style_id if isinstance(self.style_id, dict) else int(self.style_id)
        )
        ep_meta["object_cfgs"] = [copy_dict_for_json(cfg) for cfg in self.object_cfgs]

        # serialize np arrays to lists
        for cfg in ep_meta["object_cfgs"]:
            if cfg.get("reset_region", None) is not None:
                for (k, v) in cfg["reset_region"].items():
                    if isinstance(v, np.ndarray):
                        cfg["reset_region"][k] = list(v)

        ep_meta["fixtures"] = {
            k: {"cls": v.__class__.__name__} for (k, v) in self.fixtures.items()
        }
        ep_meta["gen_textures"] = self._curr_gen_fixtures or {}
        ep_meta["lang"] = ""
        ep_meta["fixture_refs"] = dict(
            {
                k: (v[0] if isinstance(v, tuple) else v).name
                for (k, v) in self.fixture_refs.items()
            }
        )
        ep_meta["cam_configs"] = deepcopy(self._cam_configs)
        ep_meta["init_robot_base_pos"] = list(self.init_robot_base_pos)
        ep_meta["init_robot_base_ori"] = list(self.init_robot_base_ori)

        return ep_meta

    def get_privileged_information(
        self, trajectory_horizon: int = 64
    ) -> Dict[str, Any]:
        """
        Return privileged scene metadata and current simulator state.

        The structure mirrors the mshab helper at a higher level:
        - ``static`` contains task / layout / object / fixture metadata that is
          mostly fixed for the episode.
        - ``dynamic`` contains the latest robot / object / fixture state.
        """

        def _to_serializable(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, (list, tuple)):
                return [_to_serializable(v) for v in value]
            if isinstance(value, dict):
                return {str(k): _to_serializable(v) for k, v in value.items()}
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            return repr(value)

        def _safe_call(fn, *args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception:
                return None

        def _safe_attr(obj, attr, default=None):
            try:
                return getattr(obj, attr, default)
            except Exception:
                return default

        def _site_local_pos(site: Any) -> Any:
            if site is None:
                return None
            try:
                raw = site.get("pos")
            except Exception:
                raw = None
            if raw is None:
                return None
            try:
                return string_to_array(raw)
            except Exception:
                return raw

        def _body_id(name: Optional[str]) -> Optional[int]:
            if not name:
                return None
            return _safe_call(self.sim.model.body_name2id, name)

        def _site_id(name: Optional[str]) -> Optional[int]:
            if not name:
                return None
            return _safe_call(self.sim.model.site_name2id, name)

        def _joint_qpos(name: str):
            addr = _safe_call(self.sim.model.get_joint_qpos_addr, name)
            if addr is None:
                return None
            try:
                return _to_serializable(self.sim.data.qpos[addr])
            except Exception:
                return None

        def _joint_qvel(name: str):
            addr = _safe_call(self.sim.model.get_joint_qvel_addr, name)
            if addr is None:
                return None
            try:
                return _to_serializable(self.sim.data.qvel[addr])
            except Exception:
                return None

        def _pose_from_body_id(body_id: Optional[int]) -> Optional[Dict[str, Any]]:
            if body_id is None:
                return None
            try:
                quat_xyzw = T.convert_quat(self.sim.data.body_xquat[body_id], to="xyzw")
                return dict(
                    position=_to_serializable(self.sim.data.body_xpos[body_id]),
                    orientation=_to_serializable(quat_xyzw),
                )
            except Exception:
                return None

        def _pose_from_site_id(site_id: Optional[int]) -> Optional[Dict[str, Any]]:
            if site_id is None:
                return None
            try:
                pos = self.sim.data.site_xpos[site_id]
                rot = self.sim.data.site_xmat[site_id].reshape(3, 3)
                quat_xyzw = Rotation.from_matrix(rot).as_quat()
                return dict(
                    position=_to_serializable(pos),
                    orientation=_to_serializable(quat_xyzw),
                )
            except Exception:
                return None

        def _velocity_from_site_id(site_id: Optional[int]) -> Optional[Dict[str, Any]]:
            if site_id is None:
                return None
            try:
                return dict(
                    linear=_to_serializable(self.sim.data.site_xvelp[site_id]),
                    angular=_to_serializable(self.sim.data.site_xvelr[site_id]),
                )
            except Exception:
                return None

        def _end_effector_velocity(
            site_id: Optional[int],
            body_id: Optional[int],
            pose: Optional[Dict[str, Any]] = None,
        ) -> Optional[Dict[str, Any]]:
            site_velocity = _velocity_from_site_id(site_id)
            if site_velocity is not None:
                return site_velocity

            body_velocity = _velocity_from_body_id(body_id)
            if body_velocity is not None:
                return body_velocity

            if pose is None:
                return None
            try:
                prev_pose = getattr(self, "_privileged_prev_eef_pose", None)
                prev_time = getattr(self, "_privileged_prev_time", None)
                curr_time = float(getattr(self.sim.data, "time", np.nan))
                curr_pos = np.asarray(pose["position"], dtype=float)
                if (
                    prev_pose is not None
                    and prev_time is not None
                    and np.isfinite(curr_time)
                    and curr_time > float(prev_time)
                ):
                    prev_pos = np.asarray(prev_pose["position"], dtype=float)
                    linear = (curr_pos - prev_pos) / float(curr_time - float(prev_time))
                    return dict(
                        linear=_to_serializable(linear),
                        angular=None,
                    )
            except Exception:
                pass
            return None

        def _velocity_from_body_id(body_id: Optional[int]) -> Optional[Dict[str, Any]]:
            if body_id is None:
                return None
            try:
                return dict(
                    linear=_to_serializable(self.sim.data.body_xvelp[body_id]),
                    angular=_to_serializable(self.sim.data.body_xvelr[body_id]),
                )
            except Exception:
                try:
                    body_name = self.sim.model.body_id2name(body_id)
                except Exception:
                    body_name = None
                if body_name:
                    linear = _safe_call(self.sim.data.get_body_xvelp, body_name)
                    angular = _safe_call(self.sim.data.get_body_xvelr, body_name)
                    if linear is not None or angular is not None:
                        return dict(
                            linear=_to_serializable(linear),
                            angular=_to_serializable(angular),
                        )
                return None

        def _pose_from_camera_name(
            camera_name: Optional[str],
        ) -> Optional[Dict[str, Any]]:
            if not camera_name:
                return None
            try:
                cam_id = int(self.sim.model.camera_name2id(camera_name))
                pos = np.asarray(self.sim.data.cam_xpos[cam_id], dtype=float)
                rot = np.asarray(self.sim.data.cam_xmat[cam_id], dtype=float).reshape(
                    3, 3
                )
                quat_xyzw = Rotation.from_matrix(rot).as_quat()
                return dict(
                    position=_to_serializable(pos),
                    orientation=_to_serializable(quat_xyzw),
                )
            except Exception:
                return None

        def _camera_dynamic() -> Dict[str, Any]:
            result = {}
            for camera_name, camera_cfg in (
                getattr(self, "_cam_configs", {}) or {}
            ).items():
                if not isinstance(camera_cfg, dict):
                    continue
                result[camera_name] = dict(
                    pose=_pose_from_camera_name(camera_name),
                    fovy=_safe_call(
                        lambda: float(
                            self.sim.model.cam_fovy[
                                int(self.sim.model.camera_name2id(camera_name))
                            ]
                        )
                    ),
                )
            return result

        def _box_corners(half_extents: np.ndarray) -> np.ndarray:
            hx, hy, hz = np.asarray(half_extents, dtype=float).reshape(3)
            return np.array(
                [
                    [-hx, -hy, -hz],
                    [hx, -hy, -hz],
                    [hx, hy, -hz],
                    [-hx, hy, -hz],
                    [-hx, -hy, hz],
                    [hx, -hy, hz],
                    [hx, hy, hz],
                    [-hx, hy, hz],
                ],
                dtype=float,
            )

        def _geom_local_corners(geom_id: int) -> Optional[np.ndarray]:
            try:
                size = np.asarray(
                    self.sim.model.geom_size[geom_id], dtype=float
                ).reshape(-1)
            except Exception:
                size = np.empty((0,), dtype=float)
            try:
                geom_type = int(self.sim.model.geom_type[geom_id])
            except Exception:
                geom_type = -1
            local_corners = None
            if (
                size.size >= 3
                and np.all(np.isfinite(size[:3]))
                and np.any(np.abs(size[:3]) > 0.0)
            ):
                local_corners = _box_corners(np.abs(size[:3]))
            elif (
                size.size == 2
                and np.all(np.isfinite(size[:2]))
                and np.any(np.abs(size[:2]) > 0.0)
            ):
                local_corners = _box_corners(
                    np.abs(np.array([size[0], size[0], size[1]], dtype=float))
                )
            elif size.size == 1 and np.isfinite(size[0]) and abs(size[0]) > 0.0:
                local_corners = _box_corners(np.full(3, abs(size[0]), dtype=float))

            # For mesh geoms, prefer the actual mesh vertices when available so the bounds
            # stay tight and don't inflate the parent/base link.
            if geom_type == 7:
                try:
                    mesh_id = int(self.sim.model.geom_dataid[geom_id])
                    vert_adr = int(self.sim.model.mesh_vertadr[mesh_id])
                    vert_num = int(self.sim.model.mesh_vertnum[mesh_id])
                    mesh_verts = np.asarray(
                        self.sim.model.mesh_vert[vert_adr : vert_adr + vert_num],
                        dtype=float,
                    )
                    if mesh_verts.size and size.size >= 3:
                        local_corners = mesh_verts * np.abs(size[:3])[None, :]
                except Exception:
                    pass

            if local_corners is None or local_corners.size == 0:
                return None

            try:
                geom_pos = np.asarray(self.sim.model.geom_pos[geom_id], dtype=float)
            except Exception:
                geom_pos = np.zeros(3, dtype=float)
            try:
                geom_quat = np.asarray(self.sim.model.geom_quat[geom_id], dtype=float)
                geom_rot = Rotation.from_quat(
                    T.convert_quat(geom_quat, to="xyzw")
                ).as_matrix()
            except Exception:
                geom_rot = np.eye(3, dtype=float)
            return local_corners @ geom_rot.T + geom_pos[None, :]

        def _body_local_bounds(body_id: Optional[int]) -> Optional[Dict[str, Any]]:
            if body_id is None:
                return None
            try:
                geom_start = int(self.sim.model.body_geomadr[body_id])
                geom_count = int(self.sim.model.body_geomnum[body_id])
            except Exception:
                return None
            if geom_count <= 0:
                return None
            mins = []
            maxs = []
            for geom_id in range(geom_start, geom_start + geom_count):
                geom_corners = _geom_local_corners(geom_id)
                if geom_corners is None:
                    continue
                mins.append(np.min(geom_corners, axis=0))
                maxs.append(np.max(geom_corners, axis=0))
            if not mins:
                return None
            bmin = np.min(np.stack(mins, axis=0), axis=0)
            bmax = np.max(np.stack(maxs, axis=0), axis=0)
            return dict(min=_to_serializable(bmin), max=_to_serializable(bmax))

        def _body_immediate_child_connector_bounds(
            body_id: Optional[int],
            *,
            min_half_extents: Iterable[float] = (0.02, 0.02, 0.02),
            pad: float = 0.01,
        ) -> Optional[Dict[str, Any]]:
            if body_id is None:
                return None
            try:
                parents = np.asarray(self.sim.model.body_parentid, dtype=int).reshape(
                    -1
                )
                body_pos = np.asarray(self.sim.data.body_xpos[body_id], dtype=float)
                body_rot = np.asarray(
                    self.sim.data.body_xmat[body_id], dtype=float
                ).reshape(3, 3)
            except Exception:
                return None

            child_ids = [
                int(idx)
                for idx, parent in enumerate(parents)
                if int(parent) == int(body_id) and int(idx) != int(body_id)
            ]
            if not child_ids:
                return None

            try:
                min_half = np.asarray(list(min_half_extents), dtype=float).reshape(3)
            except Exception:
                min_half = np.full(3, 0.02, dtype=float)

            local_points = [np.zeros(3, dtype=float)]
            body_rot_T = body_rot.T
            for child_id in child_ids:
                try:
                    child_pos = np.asarray(
                        self.sim.data.body_xpos[child_id], dtype=float
                    )
                except Exception:
                    continue
                child_local = (child_pos - body_pos) @ body_rot_T
                local_points.append(child_local)

            if len(local_points) <= 1:
                return None

            stacked = np.stack(local_points, axis=0)
            bmin = np.min(stacked, axis=0) - float(pad)
            bmax = np.max(stacked, axis=0) + float(pad)
            center = 0.5 * (bmin + bmax)
            half = np.maximum(0.5 * (bmax - bmin), min_half)
            return dict(
                min=_to_serializable(center - half),
                max=_to_serializable(center + half),
            )

        def _body_vertical_connector_bounds(
            body_id: Optional[int],
            *,
            target_child_names: Iterable[str],
            min_half_extents: Iterable[float] = (0.03, 0.03, 0.03),
            lateral_half_extents: Iterable[float] = (0.08, 0.08),
            pad_z: float = 0.02,
        ) -> Optional[Dict[str, Any]]:
            if body_id is None:
                return None
            try:
                parents = np.asarray(self.sim.model.body_parentid, dtype=int).reshape(
                    -1
                )
                body_name = self.sim.model.body_id2name(body_id)
                body_pos = np.asarray(self.sim.data.body_xpos[body_id], dtype=float)
                body_rot = np.asarray(
                    self.sim.data.body_xmat[body_id], dtype=float
                ).reshape(3, 3)
            except Exception:
                return None
            target_child_names = set(target_child_names)
            child_points = []
            body_rot_T = body_rot.T
            for child_id, parent in enumerate(parents):
                if int(parent) != int(body_id) or int(child_id) == int(body_id):
                    continue
                try:
                    child_name = self.sim.model.body_id2name(int(child_id))
                except Exception:
                    child_name = None
                if child_name not in target_child_names:
                    continue
                try:
                    child_pos = np.asarray(
                        self.sim.data.body_xpos[int(child_id)], dtype=float
                    )
                except Exception:
                    continue
                child_points.append((child_pos - body_pos) @ body_rot_T)
            if not child_points:
                return None
            try:
                min_half = np.asarray(list(min_half_extents), dtype=float).reshape(3)
                lateral_half = np.asarray(
                    list(lateral_half_extents), dtype=float
                ).reshape(2)
            except Exception:
                min_half = np.full(3, 0.03, dtype=float)
                lateral_half = np.full(2, 0.08, dtype=float)
            stacked = np.stack([np.zeros(3, dtype=float)] + child_points, axis=0)
            bmin = np.min(stacked, axis=0)
            bmax = np.max(stacked, axis=0)
            center_xy = 0.5 * (bmin[:2] + bmax[:2])
            half_xy = np.maximum(0.5 * (bmax[:2] - bmin[:2]), lateral_half)
            center_z = 0.5 * (bmin[2] + bmax[2])
            half_z = max(0.5 * (bmax[2] - bmin[2]) + float(pad_z), float(min_half[2]))
            center = np.array([center_xy[0], center_xy[1], center_z], dtype=float)
            half = np.array([half_xy[0], half_xy[1], half_z], dtype=float)
            return dict(
                min=_to_serializable(center - half),
                max=_to_serializable(center + half),
            )

        def _robot_subtree_body_ids(root_body_id: Optional[int]) -> List[int]:
            if root_body_id is None:
                return []
            try:
                parents = np.asarray(self.sim.model.body_parentid, dtype=int).reshape(
                    -1
                )
            except Exception:
                return [int(root_body_id)]
            body_ids: List[int] = []
            for body_id in range(len(parents)):
                cur = body_id
                while cur >= 0:
                    if cur == root_body_id:
                        body_ids.append(int(body_id))
                        break
                    parent = int(parents[cur])
                    if parent == cur:
                        break
                    cur = parent
            return body_ids

        def _body_subtree_local_bounds(
            body_id: Optional[int],
        ) -> Optional[Dict[str, Any]]:
            if body_id is None:
                return None
            try:
                root_pos = np.asarray(self.sim.data.body_xpos[body_id], dtype=float)
                root_rot = np.asarray(
                    self.sim.data.body_xmat[body_id], dtype=float
                ).reshape(3, 3)
            except Exception:
                return None
            mins = []
            maxs = []
            root_rot_T = root_rot.T
            for sub_body_id in _robot_subtree_body_ids(body_id):
                try:
                    geom_start = int(self.sim.model.body_geomadr[sub_body_id])
                    geom_count = int(self.sim.model.body_geomnum[sub_body_id])
                except Exception:
                    continue
                for geom_id in range(geom_start, geom_start + geom_count):
                    half_extents = _geom_half_extents(geom_id)
                    if half_extents is None:
                        continue
                    try:
                        geom_pos = np.asarray(
                            self.sim.data.geom_xpos[geom_id], dtype=float
                        )
                        geom_rot = np.asarray(
                            self.sim.data.geom_xmat[geom_id], dtype=float
                        ).reshape(3, 3)
                    except Exception:
                        continue
                    local_corners = np.array(
                        [
                            [-half_extents[0], -half_extents[1], -half_extents[2]],
                            [half_extents[0], -half_extents[1], -half_extents[2]],
                            [half_extents[0], half_extents[1], -half_extents[2]],
                            [-half_extents[0], half_extents[1], -half_extents[2]],
                            [-half_extents[0], -half_extents[1], half_extents[2]],
                            [half_extents[0], -half_extents[1], half_extents[2]],
                            [half_extents[0], half_extents[1], half_extents[2]],
                            [-half_extents[0], half_extents[1], half_extents[2]],
                        ],
                        dtype=float,
                    )
                    world_corners = local_corners @ geom_rot.T + geom_pos[None, :]
                    root_local_corners = (
                        world_corners - root_pos[None, :]
                    ) @ root_rot_T
                    mins.append(np.min(root_local_corners, axis=0))
                    maxs.append(np.max(root_local_corners, axis=0))
            if not mins:
                return None
            bmin = np.min(np.stack(mins, axis=0), axis=0)
            bmax = np.max(np.stack(maxs, axis=0), axis=0)
            return dict(min=_to_serializable(bmin), max=_to_serializable(bmax))

        def _site_local_box_bounds(
            body_id: Optional[int],
            site_id: Optional[int],
            half_extents: Iterable[float],
        ) -> Optional[Dict[str, Any]]:
            if body_id is None or site_id is None:
                return None
            try:
                body_pos = np.asarray(self.sim.data.body_xpos[body_id], dtype=float)
                body_rot = np.asarray(
                    self.sim.data.body_xmat[body_id], dtype=float
                ).reshape(3, 3)
                site_pos = np.asarray(self.sim.data.site_xpos[site_id], dtype=float)
                half = np.asarray(list(half_extents), dtype=float).reshape(3)
            except Exception:
                return None
            corners = np.array(
                [
                    [-half[0], -half[1], -half[2]],
                    [half[0], -half[1], -half[2]],
                    [half[0], half[1], -half[2]],
                    [-half[0], half[1], -half[2]],
                    [-half[0], -half[1], half[2]],
                    [half[0], -half[1], half[2]],
                    [half[0], half[1], half[2]],
                    [-half[0], half[1], half[2]],
                ],
                dtype=float,
            )
            world_corners = corners + site_pos[None, :]
            local_corners = (world_corners - body_pos[None, :]) @ body_rot.T
            return dict(
                min=_to_serializable(np.min(local_corners, axis=0)),
                max=_to_serializable(np.max(local_corners, axis=0)),
            )

        def _body_external_wrench(body_id: Optional[int]) -> Optional[Dict[str, Any]]:
            if body_id is None:
                return None
            try:
                wrench = np.asarray(
                    self.sim.data.cfrc_ext[body_id], dtype=float
                ).reshape(-1)
            except Exception:
                return None
            if wrench.size < 6 or not np.all(np.isfinite(wrench[:6])):
                return None
            force = wrench[:3]
            torque = wrench[3:6]
            return dict(
                force=_to_serializable(force),
                torque=_to_serializable(torque),
                force_norm=float(np.linalg.norm(force)),
                torque_norm=float(np.linalg.norm(torque)),
            )

        def _robot_joint_bundle() -> Dict[str, Any]:
            joint_names: List[str] = []
            seen = set()

            def _append_joint(name: Optional[str]) -> None:
                if not name or name in seen:
                    return
                seen.add(name)
                joint_names.append(name)

            for name in _safe_attr(robot, "robot_joints", []) or []:
                _append_joint(name)

            gripper_obj = _safe_attr(robot, "gripper")
            if isinstance(gripper_obj, dict):
                grippers = list(gripper_obj.values())
            elif gripper_obj is None:
                grippers = []
            else:
                grippers = [gripper_obj]
            for gripper in grippers:
                for attr_name in ("joints", "joint_names"):
                    for name in _safe_attr(gripper, attr_name, []) or []:
                        _append_joint(name)

            return dict(
                joint_names=_to_serializable(joint_names),
                joint_positions=_to_serializable(
                    [_joint_qpos(name) for name in joint_names]
                ),
                joint_velocities=_to_serializable(
                    [_joint_qvel(name) for name in joint_names]
                ),
            )

        def _physics_dynamic(
            robot_link_meta: Dict[str, Any], eef_body_name: Optional[str]
        ) -> Dict[str, Any]:
            robot_body_ids = {
                int(info.get("body_id"))
                for info in robot_link_meta.values()
                if isinstance(info, dict) and info.get("body_id") is not None
            }
            geom_bodyid = None
            try:
                geom_bodyid = np.asarray(self.sim.model.geom_bodyid, dtype=int).reshape(
                    -1
                )
            except Exception:
                geom_bodyid = None

            total_contact_count = int(_safe_attr(self.sim.data, "ncon", 0) or 0)
            robot_contact_count = 0
            for contact_idx in range(total_contact_count):
                try:
                    contact = self.sim.data.contact[contact_idx]
                    geom1 = int(contact.geom1)
                    geom2 = int(contact.geom2)
                except Exception:
                    continue
                body1 = (
                    int(geom_bodyid[geom1])
                    if geom_bodyid is not None and 0 <= geom1 < len(geom_bodyid)
                    else None
                )
                body2 = (
                    int(geom_bodyid[geom2])
                    if geom_bodyid is not None and 0 <= geom2 < len(geom_bodyid)
                    else None
                )
                if body1 in robot_body_ids or body2 in robot_body_ids:
                    robot_contact_count += 1

            per_link_force_norms = {}
            max_robot_link_force = np.nan
            robot_force_sum = 0.0
            end_effector_force_norm = np.nan
            for name, info in robot_link_meta.items():
                body_id = info.get("body_id") if isinstance(info, dict) else None
                wrench = _body_external_wrench(body_id)
                if wrench is None:
                    continue
                force_norm = float(wrench["force_norm"])
                per_link_force_norms[name] = force_norm
                robot_force_sum += force_norm
                if (
                    not np.isfinite(max_robot_link_force)
                    or force_norm > max_robot_link_force
                ):
                    max_robot_link_force = force_norm
                if eef_body_name and name == eef_body_name:
                    end_effector_force_norm = force_norm

            return dict(
                total_contact_count=total_contact_count,
                robot_contact_count=robot_contact_count,
                max_robot_link_force=max_robot_link_force,
                robot_force_sum=robot_force_sum,
                end_effector_force_norm=end_effector_force_norm,
                per_link_force_norms=_to_serializable(per_link_force_norms),
            )

        def _robot_link_static(
            body_id: int, *, fallback_bounds: Optional[Dict[str, Any]] = None
        ) -> Optional[Dict[str, Any]]:
            try:
                body_name = self.sim.model.body_id2name(body_id)
            except Exception:
                body_name = None
            if not body_name:
                return None
            direct_bounds = _body_local_bounds(body_id)
            structural_connector_bounds = None
            if direct_bounds is None and fallback_bounds is None:
                if body_name == "mobilebase0_base":
                    structural_connector_bounds = _body_vertical_connector_bounds(
                        body_id,
                        target_child_names=("mobilebase0_support",),
                        lateral_half_extents=(0.11, 0.11),
                        pad_z=0.03,
                    )
                elif body_name == "robot0_base":
                    structural_connector_bounds = _body_vertical_connector_bounds(
                        body_id,
                        target_child_names=("robot0_link0",),
                        lateral_half_extents=(0.06, 0.06),
                        pad_z=0.02,
                    )
            connector_bounds = (
                _body_immediate_child_connector_bounds(body_id)
                if direct_bounds is None
                and fallback_bounds is None
                and structural_connector_bounds is None
                else None
            )
            visual_bounds = (
                direct_bounds
                or fallback_bounds
                or structural_connector_bounds
                or connector_bounds
            )
            if direct_bounds is not None:
                visual_bounds_source = "direct_geom"
            elif fallback_bounds is not None:
                visual_bounds_source = "eef_site_fallback"
            elif structural_connector_bounds is not None:
                visual_bounds_source = "structural_connector"
            elif connector_bounds is not None:
                visual_bounds_source = "immediate_child_connector"
            else:
                visual_bounds_source = None
            return dict(
                name=body_name,
                body_id=int(body_id),
                visual_bounds=visual_bounds,
                visual_bounds_source=visual_bounds_source,
            )

        def _relative_pose_to_eef(
            target_pose: Optional[Dict[str, Any]], eef_pose: Optional[Dict[str, Any]]
        ) -> Optional[Dict[str, Any]]:
            if target_pose is None or eef_pose is None:
                return None
            try:
                target_mat = T.pose2mat(
                    (
                        np.asarray(target_pose["position"]),
                        np.asarray(target_pose["orientation"]),
                    )
                )
                eef_inv = T.pose_inv(
                    T.pose2mat(
                        (
                            np.asarray(eef_pose["position"]),
                            np.asarray(eef_pose["orientation"]),
                        )
                    )
                )
                rel = T.pose_in_A_to_pose_in_B(target_mat, eef_inv)
                rel_pos, rel_quat = T.mat2pose(rel)
                return dict(
                    position=_to_serializable(rel_pos),
                    orientation=_to_serializable(rel_quat),
                )
            except Exception:
                return None

        def _fixture_joint_state(fixture) -> Optional[Dict[str, Any]]:
            joints = _safe_attr(fixture, "_joints", []) or []
            if not joints:
                return None
            result = {}
            for joint_suffix in joints:
                joint_name = f"{fixture.name}_{joint_suffix}"
                result[joint_name] = dict(
                    qpos=_joint_qpos(joint_name),
                    qvel=_joint_qvel(joint_name),
                )
            return result or None

        def _fixture_static(fixture) -> Dict[str, Any]:
            local_bounds = {}
            for key, site in (_safe_attr(fixture, "_bounds_sites", {}) or {}).items():
                local_bounds[key] = _to_serializable(_site_local_pos(site))
            return dict(
                class_name=fixture.__class__.__name__,
                nat_lang=_safe_attr(fixture, "nat_lang"),
                size=_to_serializable(_safe_attr(fixture, "size")),
                position=_to_serializable(_safe_attr(fixture, "pos")),
                quaternion=_to_serializable(_safe_attr(fixture, "quat")),
                euler=_to_serializable(_safe_attr(fixture, "euler")),
                origin_offset=_to_serializable(_safe_attr(fixture, "origin_offset")),
                placement=_to_serializable(_safe_attr(fixture, "_placement")),
                local_bounds_sites=local_bounds or None,
            )

        def _fixture_dynamic(fixture) -> Dict[str, Any]:
            get_state_fn = _safe_attr(fixture, "get_state")
            get_door_state_fn = _safe_attr(fixture, "get_door_state")
            get_site_info_fn = _safe_attr(fixture, "get_site_info")

            state = _safe_call(get_state_fn, self) if callable(get_state_fn) else None
            if state is None:
                state = (
                    _safe_call(get_state_fn, self.sim)
                    if callable(get_state_fn)
                    else None
                )
            if state is None:
                state = _safe_call(get_state_fn) if callable(get_state_fn) else None
            door_state = (
                _safe_call(get_door_state_fn, self)
                if callable(get_door_state_fn)
                else None
            )
            site_info = (
                _safe_call(get_site_info_fn, self.sim)
                if callable(get_site_info_fn)
                else None
            )
            return dict(
                state=_to_serializable(state),
                door_state=_to_serializable(door_state),
                joints=_fixture_joint_state(fixture),
                sites=_to_serializable(site_info),
            )

        def _predicate_static_spec(static_info: Dict[str, Any]) -> Dict[str, Any]:
            return build_predicate_static_spec(self, static_info)

        def _predicate_snapshot(
            static_info: Dict[str, Any], dynamic_info: Dict[str, Any]
        ) -> Dict[str, Any]:
            return build_predicate_snapshot(self, static_info, dynamic_info)

        def _object_static(
            name: str, model, cfg: Optional[Dict[str, Any]]
        ) -> Dict[str, Any]:
            bbox_points = _safe_call(model.get_bbox_points)
            bbox_min = None
            bbox_max = None
            object_info = cfg.get("info") if isinstance(cfg, dict) else None
            if bbox_points is not None:
                try:
                    bbox_points = np.asarray(bbox_points, dtype=float)
                    bbox_min = bbox_points.min(axis=0)
                    bbox_max = bbox_points.max(axis=0)
                except Exception:
                    bbox_min = None
                    bbox_max = None
            return dict(
                root_body=_safe_attr(model, "root_body"),
                info=_to_serializable(object_info),
                category=(
                    object_info.get("cat") if isinstance(object_info, dict) else None
                ),
                config=_to_serializable(cfg),
                mjcf_path=_safe_attr(
                    model, "asset_path", _safe_attr(model, "mjcf_path")
                ),
                scale=_to_serializable(_safe_attr(model, "scale")),
                horizontal_radius=_to_serializable(
                    _safe_attr(model, "horizontal_radius")
                ),
                bbox=dict(
                    min=_to_serializable(bbox_min),
                    max=_to_serializable(bbox_max),
                )
                if bbox_min is not None
                else None,
            )

        robot = self.robots[0]
        eef_site_name = None
        try:
            eef_site_id = robot.eef_site_id["right"]
        except Exception:
            eef_site_id = None
        if eef_site_id is not None:
            try:
                eef_site_name = self.sim.model.site_id2name(eef_site_id)
            except Exception:
                eef_site_name = None

        robot_root_body = _safe_attr(robot.robot_model, "root_body")
        robot_root_body_id = _body_id(robot_root_body)

        robot_subtree_body_ids = _robot_subtree_body_ids(robot_root_body_id)

        def _choose_robot_base_body(
            default_name: Optional[str],
            default_body_id: Optional[int],
            subtree_body_ids: List[int],
        ) -> Tuple[Optional[str], Optional[int], Optional[Dict[str, Any]]]:
            default_pose = _pose_from_body_id(default_body_id)
            default_pos = (
                np.asarray(default_pose["position"], dtype=float)
                if default_pose is not None and default_pose.get("position") is not None
                else None
            )

            def _is_far_dummy(pos: Optional[np.ndarray]) -> bool:
                if pos is None or pos.shape[0] < 2:
                    return False
                return bool(np.linalg.norm(pos[:2]) > 5.0)

            if not _is_far_dummy(default_pos):
                return default_name, default_body_id, default_pose

            candidates = []
            for body_id in subtree_body_ids:
                body_name = _safe_call(self.sim.model.body_id2name, int(body_id))
                pose = _pose_from_body_id(body_id)
                pos = (
                    np.asarray(pose["position"], dtype=float)
                    if pose is not None and pose.get("position") is not None
                    else None
                )
                if pos is None or _is_far_dummy(pos):
                    continue
                lowered = str(body_name or "").lower()
                priority = 100
                if "mobilebase" in lowered and lowered.endswith("base"):
                    priority = 0
                elif "mobilebase" in lowered:
                    priority = 10
                elif "support" in lowered:
                    priority = 20
                elif "base" in lowered:
                    priority = 30
                candidates.append((priority, str(body_name), int(body_id), pose))

            if candidates:
                _, body_name, body_id, pose = sorted(candidates)[0]
                return body_name, body_id, pose

            return default_name, default_body_id, default_pose

        (
            robot_root_body,
            robot_root_body_id,
            robot_root_pose,
        ) = _choose_robot_base_body(
            robot_root_body,
            robot_root_body_id,
            robot_subtree_body_ids,
        )
        eef_body_name = None
        eef_body_id = None
        if eef_site_id is not None:
            try:
                eef_body_id = int(self.sim.model.site_bodyid[eef_site_id])
                eef_body_name = self.sim.model.body_id2name(eef_body_id)
            except Exception:
                eef_body_name = None
                eef_body_id = None
        eef_fallback_bounds = _site_local_box_bounds(
            eef_body_id,
            eef_site_id,
            half_extents=(0.035, 0.025, 0.025),
        )
        robot_link_meta = {}
        for body_id in robot_subtree_body_ids:
            link_info = _robot_link_static(
                body_id,
                fallback_bounds=(
                    eef_fallback_bounds
                    if eef_body_id is not None and int(body_id) == int(eef_body_id)
                    else None
                ),
            )
            if link_info is None:
                continue
            robot_link_meta[link_info["name"]] = link_info

        if not hasattr(self, "_privileged_static_cache"):
            ep_meta = self.get_ep_meta()
            object_cfg_by_name = {
                cfg["name"]: cfg
                for cfg in getattr(self, "object_cfgs", [])
                if "name" in cfg
            }
            self._privileged_static_cache = dict(
                task=dict(
                    env_name=self.__class__.__name__,
                    language=_to_serializable(ep_meta.get("lang", "")),
                    episode_meta=_to_serializable(ep_meta),
                ),
                robot=dict(
                    robot_class=self.robots[0].__class__.__name__,
                    robot_model_class=self.robots[0].robot_model.__class__.__name__,
                    naming_prefix=_safe_attr(
                        self.robots[0].robot_model, "naming_prefix"
                    ),
                    init_base_pos=_to_serializable(
                        _safe_attr(self.robots[0].robot_model, "base_xpos")
                        or (robot_root_pose or {}).get("position")
                    ),
                    link_metadata=_to_serializable(robot_link_meta),
                    base_link=(
                        dict(
                            name=robot_root_body,
                            info=_to_serializable(robot_link_meta.get(robot_root_body)),
                        )
                        if robot_root_body
                        else None
                    ),
                    end_effector_link=(
                        dict(
                            name=eef_body_name,
                            info=_to_serializable(robot_link_meta.get(eef_body_name)),
                        )
                        if eef_body_name
                        else None
                    ),
                ),
                scene_layout=dict(
                    layout_id=int(self.layout_id),
                    style_id=int(self.style_id),
                    fixture_refs=_to_serializable(
                        {
                            k: (v[0] if isinstance(v, tuple) else v).name
                            for k, v in getattr(self, "fixture_refs", {}).items()
                        }
                    ),
                    cameras=_to_serializable(getattr(self, "_cam_configs", {})),
                    fixtures={
                        name: _fixture_static(fixture)
                        for name, fixture in getattr(self, "fixtures", {}).items()
                    },
                    objects={
                        name: _object_static(
                            name,
                            model,
                            object_cfg_by_name.get(name),
                        )
                        for name, model in getattr(self, "objects", {}).items()
                    },
                ),
            )
            self._privileged_static_cache["predicates"] = _predicate_static_spec(
                self._privileged_static_cache
            )

        if not hasattr(self, "_privileged_history"):
            self._privileged_history = dict(
                eef_positions=[], robot_force_cumulative=0.0
            )
        eef_pose = _pose_from_site_id(eef_site_id)
        eef_velocity = _end_effector_velocity(eef_site_id, eef_body_id, eef_pose)
        if eef_pose is not None:
            self._privileged_history["eef_positions"].append(eef_pose["position"])
            if len(self._privileged_history["eef_positions"]) > trajectory_horizon:
                self._privileged_history["eef_positions"].pop(0)
        robot_joint_bundle = _robot_joint_bundle()
        physics = _physics_dynamic(robot_link_meta, eef_body_name)
        if np.isfinite(physics.get("robot_force_sum", np.nan)):
            self._privileged_history["robot_force_cumulative"] += float(
                physics["robot_force_sum"]
            )

        dynamic = dict(
            task=dict(
                timestep=int(getattr(self, "timestep", -1)),
                success=bool(_safe_call(self._check_success)),
            ),
            robot=dict(
                root_body=robot_root_body,
                root_pose=_pose_from_body_id(robot_root_body_id),
                root_velocity=_velocity_from_body_id(robot_root_body_id),
                eef_site=eef_site_name,
                end_effector_pose=eef_pose,
                end_effector_velocity=eef_velocity,
                link_poses={
                    name: _pose_from_body_id(info.get("body_id"))
                    for name, info in robot_link_meta.items()
                },
                joint_names=robot_joint_bundle["joint_names"],
                joint_positions=robot_joint_bundle["joint_positions"],
                joint_velocities=robot_joint_bundle["joint_velocities"],
                gripper_action_dim=int(len(getattr(robot, "gripper", {}))),
                trajectory=_to_serializable(self._privileged_history["eef_positions"]),
            ),
            scene=dict(
                objects={},
                fixtures={},
                cameras=_camera_dynamic(),
            ),
            physics=dict(
                **physics,
                robot_force_cumulative=float(
                    self._privileged_history["robot_force_cumulative"]
                ),
            ),
        )

        for name, model in getattr(self, "objects", {}).items():
            body_id = _body_id(_safe_attr(model, "root_body"))
            pose = _pose_from_body_id(body_id)
            dynamic["scene"]["objects"][name] = dict(
                pose=pose,
                velocity=_velocity_from_body_id(body_id),
                relative_to_eef=_relative_pose_to_eef(pose, eef_pose),
            )

        for name, fixture in getattr(self, "fixtures", {}).items():
            dynamic["scene"]["fixtures"][name] = _fixture_dynamic(fixture)

        # Evaluate the task-agnostic low-level monitor after the base privileged
        # state has been assembled so downstream symbolic logic can reuse the
        # same serialized snapshot without re-deriving basic predicates elsewhere.
        dynamic["predicates"] = _predicate_snapshot(
            self._privileged_static_cache,
            dynamic,
        )

        self._privileged_prev_eef_pose = eef_pose
        try:
            self._privileged_prev_time = float(getattr(self.sim.data, "time", np.nan))
        except Exception:
            self._privileged_prev_time = None

        return dict(
            static=self._privileged_static_cache,
            dynamic=dynamic,
        )

    def edit_model_xml(self, xml_str):
        """
        This function postprocesses the model.xml collected from a MuJoCo demonstration
        for retrospective model changes.

        Args:
            xml_str (str): Mujoco sim demonstration XML file as string

        Returns:
            str: Post-processed xml file as string
        """
        xml_str = super().edit_model_xml(xml_str)

        tree = ET.fromstring(xml_str)
        root = tree
        worldbody = root.find("worldbody")
        actuator = root.find("actuator")
        asset = root.find("asset")
        meshes = asset.findall("mesh")
        textures = asset.findall("texture")
        all_elements = meshes + textures

        robocasa_path_split = os.path.split(robocasa.__file__)[0].split("/")

        # replace robocasa-specific asset paths
        for elem in all_elements:
            old_path = elem.get("file")
            if old_path is None:
                continue

            old_path_split = old_path.split("/")
            # maybe replace all paths to robosuite assets
            if (
                ("models/assets/fixtures" in old_path)
                or ("models/assets/textures" in old_path)
                or ("models/assets/objects" in old_path)
                or ("models/assets/generative_textures" in old_path)
            ):
                if "/robosuite/" in old_path:
                    check_lst = [
                        loc
                        for loc, val in enumerate(old_path_split)
                        if val == "robosuite"
                    ]
                elif "/robocasa/" in old_path:
                    check_lst = [
                        loc
                        for loc, val in enumerate(old_path_split)
                        if val == "robocasa"
                    ]
                else:
                    raise ValueError

                ind = max(check_lst)  # last occurrence index
                new_path_split = robocasa_path_split + old_path_split[ind + 1 :]

                new_path = "/".join(new_path_split)
                elem.set("file", new_path)

        # set cameras
        for cam_name, cam_config in self._cam_configs.items():
            parent_body = cam_config.get("parent_body", None)

            cam_root = worldbody
            if parent_body is not None:
                cam_root = find_elements(
                    root=worldbody, tags="body", attribs={"name": parent_body}
                )
                if cam_root is None:
                    # camera config refers to body that doesnt exist on the robot
                    continue

            cam = find_elements(
                root=cam_root, tags="camera", attribs={"name": cam_name}
            )
            if cam is not None:
                # remove fovy if sensor size is specified to avoid
                # ValueError: XML Error: either 'fovy' or 'sensorsize' attribute can be specified, not both
                if (
                    "sensorsize" in cam_config.get("camera_attribs", {})
                    and "fovy" in cam.attrib
                ):
                    del cam.attrib["fovy"]

            if cam is None:
                cam = ET.Element("camera")
                cam.set("mode", "fixed")
                cam.set("name", cam_name)
                cam_root.append(cam)

            cam.set("pos", array_to_string(cam_config["pos"]))
            cam.set("quat", array_to_string(cam_config["quat"]))
            for (k, v) in cam_config.get("camera_attribs", {}).items():
                cam.set(k, v)

        # result = ET.tostring(root, encoding="utf8").decode("utf8")
        result = ET.tostring(root).decode("utf8")

        # replace with generative textures
        if (self.generative_textures is not None) and (
            self.generative_textures is not False
        ):
            # sample textures
            assert self.generative_textures == "100p"
            if self._curr_gen_fixtures is None or self._curr_gen_fixtures == {}:
                self._curr_gen_fixtures = get_random_textures(self.rng)

            cab_tex = self._curr_gen_fixtures["cab_tex"]
            counter_tex = self._curr_gen_fixtures["counter_tex"]
            wall_tex = self._curr_gen_fixtures["wall_tex"]
            floor_tex = self._curr_gen_fixtures["floor_tex"]

            result = replace_cab_textures(
                self.rng, result, new_cab_texture_file=cab_tex
            )
            result = replace_counter_top_texture(
                self.rng, result, new_counter_top_texture_file=counter_tex
            )
            result = replace_wall_texture(
                self.rng, result, new_wall_texture_file=wall_tex
            )
            result = replace_floor_texture(
                self.rng, result, new_floor_texture_file=floor_tex
            )

        return result

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        self.obj_body_id = {}
        for (name, model) in self.objects.items():
            self.obj_body_id[name] = self.sim.model.body_name2id(model.root_body)

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information

        # Get robot prefix and define observables modality
        pf = self.robots[0].robot_model.naming_prefix
        modality = "object"

        # for conversion to relative gripper frame
        @sensor(modality=modality)
        def world_pose_in_gripper(obs_cache):
            return (
                T.pose_inv(
                    T.pose2mat((obs_cache[f"{pf}eef_pos"], obs_cache[f"{pf}eef_quat"]))
                )
                if f"{pf}eef_pos" in obs_cache and f"{pf}eef_quat" in obs_cache
                else np.eye(4)
            )

        sensors = [world_pose_in_gripper]
        names = ["world_pose_in_gripper"]
        actives = [False]

        # add ground-truth poses (absolute and relative to eef) for all objects
        for obj_name in self.obj_body_id:
            obj_sensors, obj_sensor_names = self._create_obj_sensors(
                obj_name=obj_name, modality=modality
            )
            sensors += obj_sensors
            names += obj_sensor_names
            actives += [True] * len(obj_sensors)

        # Create observables
        for name, s, active in zip(names, sensors, actives):
            observables[name] = Observable(
                name=name,
                sensor=s,
                sampling_rate=self.control_freq,
                active=active,
            )

        return observables

    def _create_obj_sensors(self, obj_name, modality="object"):
        """
        Helper function to create sensors for a given object. This is abstracted in a separate function call so that we
        don't have local function naming collisions during the _setup_observables() call.

        Args:
            obj_name (str): Name of object to create sensors for

            modality (str): Modality to assign to all sensors

        Returns:
            2-tuple:
                sensors (list): Array of sensors for the given obj
                names (list): array of corresponding observable names
        """

        ### TODO: this was stolen from pick-place - do we want to move this into utils to share it? ###
        pf = self.robots[0].robot_model.naming_prefix

        @sensor(modality=modality)
        def obj_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.obj_body_id[obj_name]])

        @sensor(modality=modality)
        def obj_quat(obs_cache):
            return T.convert_quat(
                self.sim.data.body_xquat[self.obj_body_id[obj_name]], to="xyzw"
            )

        @sensor(modality=modality)
        def obj_to_eef_pos(obs_cache):
            # Immediately return default value if cache is empty
            if any(
                [
                    name not in obs_cache
                    for name in [
                        f"{obj_name}_pos",
                        f"{obj_name}_quat",
                        "world_pose_in_gripper",
                    ]
                ]
            ):
                return np.zeros(3)
            obj_pose = T.pose2mat(
                (obs_cache[f"{obj_name}_pos"], obs_cache[f"{obj_name}_quat"])
            )
            rel_pose = T.pose_in_A_to_pose_in_B(
                obj_pose, obs_cache["world_pose_in_gripper"]
            )
            rel_pos, rel_quat = T.mat2pose(rel_pose)
            obs_cache[f"{obj_name}_to_{pf}eef_quat"] = rel_quat
            return rel_pos

        @sensor(modality=modality)
        def obj_to_eef_quat(obs_cache):
            return (
                obs_cache[f"{obj_name}_to_{pf}eef_quat"]
                if f"{obj_name}_to_{pf}eef_quat" in obs_cache
                else np.zeros(4)
            )

        sensors = [obj_pos, obj_quat, obj_to_eef_pos, obj_to_eef_quat]
        names = [
            f"{obj_name}_pos",
            f"{obj_name}_quat",
            f"{obj_name}_to_{pf}eef_pos",
            f"{obj_name}_to_{pf}eef_quat",
        ]

        return sensors, names

    def _post_action(self, action):
        """
        Do any housekeeping after taking an action.

        Args:
            action (np.array): Action to execute within the environment

        Returns:
            3-tuple:
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) information about the current state of the environment
        """
        reward, done, info = super()._post_action(action)

        # Check if stove is turned on or not
        self.update_state()
        return reward, done, info

    def update_state(self):
        """
        Updates the state of the environment.
        This involves updating the state of all fixtures in the environment.
        """
        super().update_state()

        for fixtr in self.fixtures.values():
            fixtr.update_state(self)

    def visualize(self, vis_settings):
        """
        In addition to super call, make the robot semi-transparent

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        visual_geom_names = []

        for robot in self.robots:
            robot_model = robot.robot_model
            visual_geom_names += robot_model.visual_geoms

        for name in visual_geom_names:
            rgba = self.sim.model.geom_rgba[self.sim.model.geom_name2id(name)]
            if self.translucent_robot:
                rgba[-1] = 0.10
            else:
                rgba[-1] = 1.0

    def reward(self, action=None):
        """
        Reward function for the task. The reward function is based on the task
        and to be implemented in the subclasses. By default, returns a sparse
        reward corresponding to task success.

        Returns:
            float: Reward for the task
        """
        reward = float(self._check_success())
        return reward

    def _check_success(self):
        """
        Checks if the task has been successfully completed.
        Success condition is based on the task and to be implemented in the
        subclasses. Returns False by default.

        Returns:
            bool: True if the task is successfully completed, False otherwise
        """
        return False

    def sample_object(
        self,
        groups,
        exclude_groups=None,
        graspable=None,
        microwavable=None,
        washable=None,
        cookable=None,
        fridgable=None,
        freezable=None,
        dishwashable=None,
        split=None,
        obj_registries=None,
        max_size=(None, None, None),
        object_scale=None,
        rotate_upright=False,
    ):
        """
        Sample a kitchen object from the specified groups and within max_size bounds.

        Args:
            groups (list or str): groups to sample from or the exact xml path of the object to spawn

            exclude_groups (str or list): groups to exclude

            graspable (bool): whether the sampled object must be graspable

            washable (bool): whether the sampled object must be washable

            microwavable (bool): whether the sampled object must be microwavable

            cookable (bool): whether whether the sampled object must be cookable

            fridgable (bool): whether whether the sampled object must be fridgable

            freezable (bool): whether whether the sampled object must be freezable

            dishwashable (bool): whether whether the sampled object must be dishwashable

            split (str): split to sample from. Split "pretrain" specifies all but the last 4 object instances
                        (or the first half - whichever is larger), "target" specifies the rest, and None
                        specifies all.

            obj_registries (tuple): registries to sample from

            max_size (tuple): max size of the object. If the sampled object is not within bounds of max size,
                            function will resample

            object_scale (float): scale of the object. If set will multiply the scale of the sampled object by this value


        Returns:
            dict: kwargs to apply to the MJCF model for the sampled object

            dict: info about the sampled object - the path of the mjcf, groups which the object's category belongs to,
            the category of the object the sampling split the object came from, and the groups the object was sampled from
        """
        return sample_kitchen_object(
            groups,
            exclude_groups=exclude_groups,
            graspable=graspable,
            washable=washable,
            microwavable=microwavable,
            cookable=cookable,
            fridgable=fridgable,
            freezable=freezable,
            dishwashable=dishwashable,
            rng=self.rng,
            obj_registries=(obj_registries or self.obj_registries),
            split=(split or self.obj_instance_split),
            max_size=max_size,
            object_scale=object_scale,
            rotate_upright=rotate_upright,
        )

    def get_fixture(
        self,
        id,
        ref=None,
        size=(0.2, 0.2),
        full_name_check=False,
        return_all=False,
        full_depth_region=False,
    ):
        """
        search fixture by id (name, object, or type)

        Args:
            id (str, Fixture, FixtureType): id of fixture to search for

            ref (str, Fixture, FixtureType): if specified, will search for fixture close to ref (within 0.10m)

            size (tuple): if sampling counter, minimum size (x,y) that the counter region must be

            full_depth_region (bool): if True, will only sample island counter regions that can be accessed

        Returns:
            Fixture: fixture object
        """
        # case 1: id refers to fixture object directly
        if isinstance(id, Fixture):
            return id
        # case 2: id refers to exact name of fixture
        elif id in self.fixtures.keys():
            return self.fixtures[id]

        if ref is None:
            # find all fixtures with names containing given name
            if isinstance(id, FixtureType) or isinstance(id, int):
                matches = [
                    name
                    for (name, fxtr) in self.fixtures.items()
                    if fixture_is_type(fxtr, id)
                ]
            else:
                if full_name_check:
                    matches = [name for name in self.fixtures.keys() if name == id]
                else:
                    matches = [name for name in self.fixtures.keys() if id in name]
            if id == FixtureType.COUNTER or id == FixtureType.COUNTER_NON_CORNER:
                matches = [
                    name
                    for name in matches
                    if FixtureUtils.is_fxtr_valid(self, self.fixtures[name], size)
                ]

            if (
                len(matches) > 1
                and any("island" in name for name in matches)
                and full_depth_region
            ):
                island_matches = [name for name in matches if "island" in name]
                if len(island_matches) >= 3:
                    depths = [self.fixtures[name].size[1] for name in island_matches]
                    sorted_indices = sorted(range(len(depths)), key=lambda i: depths[i])
                    min_depth = depths[sorted_indices[0]]
                    next_min_depth = (
                        depths[sorted_indices[1]] if len(depths) > 1 else min_depth
                    )
                    if min_depth < 0.8 * next_min_depth:
                        keep = [
                            i
                            for i in range(len(island_matches))
                            if i != sorted_indices[0]
                        ]
                        filtered_islands = [island_matches[i] for i in keep]
                        matches = [
                            name for name in matches if name not in island_matches
                        ] + filtered_islands

            if len(matches) == 0:
                return None
            # sample random key
            if return_all:
                return [self.fixtures[key] for key in matches]
            else:
                key = self.rng.choice(matches)
                return self.fixtures[key]
        else:
            ref_fixture = self.get_fixture(ref)

            assert isinstance(id, FixtureType)
            cand_fixtures = []
            for fxtr in self.fixtures.values():
                if not fixture_is_type(fxtr, id):
                    continue
                if fxtr is ref_fixture:
                    continue
                if id == FixtureType.COUNTER:
                    fxtr_is_valid = FixtureUtils.is_fxtr_valid(self, fxtr, size)
                    if not fxtr_is_valid:
                        continue
                cand_fixtures.append(fxtr)

            if return_all:
                return cand_fixtures
            else:
                # first, try to find fixture "containing" the reference fixture
                for fxtr in cand_fixtures:
                    if OU.point_in_fixture(ref_fixture.pos, fxtr, only_2d=True):
                        return fxtr
                # if no fixture contains reference fixture, sample all close fixtures
                dists = [
                    OU.fixture_pairwise_dist(ref_fixture, fxtr)
                    for fxtr in cand_fixtures
                ]
                min_dist = np.min(dists)
                close_fixtures = [
                    fxtr
                    for (fxtr, d) in zip(cand_fixtures, dists)
                    if d - min_dist < 0.10
                ]
                return self.rng.choice(close_fixtures)

    def register_fixture_ref(self, ref_name, fn_kwargs):
        """
        Registers a fixture reference for later use. Initializes the fixture
        if it has not been initialized yet.

        Args:
            ref_name (str): name of the reference

            fn_kwargs (dict): keyword arguments to pass to get_fixture

        Returns:
            Fixture: fixture object
        """
        full_depth_region = fn_kwargs.pop("full_depth_region", False)
        if ref_name not in self.fixture_refs:
            fixture = self.get_fixture(**fn_kwargs, full_depth_region=full_depth_region)
            self.fixture_refs[ref_name] = (fixture, full_depth_region)

        ref_value = self.fixture_refs[ref_name]
        if isinstance(ref_value, tuple):
            return ref_value[0]
        else:
            return ref_value

    def get_obj_lang(self, obj_name="obj", get_preposition=False):
        """
        gets a formatted language string for the object (replaces underscores with spaces)

        Args:
            obj_name (str): name of object
            get_preposition (bool): if True, also returns preposition for object

        Returns:
            str: language string for object
        """
        return OU.get_obj_lang(self, obj_name=obj_name, get_preposition=get_preposition)
