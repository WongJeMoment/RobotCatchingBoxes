"""Train and evaluation configurations for Unitree G1 locomotion.

The hard parts of the MDP (robot asset, observations, actions, rewards, terrain,
terminations and randomization) come from Isaac Lab's maintained G1 task.  This
module owns the project-level choices so experiments can be changed without
editing the Isaac Lab checkout.
"""

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.rough_env_cfg import G1Rewards, G1RoughEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import EventCfg as VelocityEventCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import TerminationsCfg as VelocityTerminationsCfg
from isaaclab_assets import G1_CFG

from g1_locomotion.robots import G1_23DOF_FIXED_HAND_CFG

from . import mdp


@configclass
class BoxThrowEventsCfg(VelocityEventCfg):
    """Reset and interval events for randomized projectile boxes."""

    reset_throw_boxes = EventTerm(
        func=mdp.park_throw_boxes,
        mode="reset",
        params={"collection_name": "throw_boxes", "parking_depth": -20.0},
    )
    throw_box = EventTerm(
        func=mdp.throw_random_box,
        mode="interval",
        interval_range_s=(4.0, 7.0),
        is_global_time=False,
        params={
            "collection_name": "throw_boxes",
            "robot_name": "robot",
            "distance_range": (2.0, 3.5),
            "flight_time_range": (0.45, 0.75),
            "target_height_offset_range": (0.0, 0.45),
            "spawn_height_offset_range": (-0.15, 0.35),
            "horizontal_aim_noise": 0.15,
            "angular_velocity_range": (-6.0, 6.0),
            "gravity_z": -9.81,
            "parking_depth": -20.0,
        },
    )


_CATCH_JOINT_NAMES = [
    "torso_joint",
    ".*_shoulder_.*_joint",
    ".*_elbow_.*_joint",
    ".*_(five|three|zero|six|four|one|two)_joint",
]
_LOWER_BODY_JOINT_NAMES = [
    ".*_hip_.*_joint",
    ".*_knee_joint",
    ".*_ankle_.*_joint",
]
_WHOLE_BODY_JOINT_NAMES = [*_LOWER_BODY_JOINT_NAMES, *_CATCH_JOINT_NAMES]
_PALM_NAMES = ["left_palm_link", "right_palm_link"]
_LEFT_HAND_PATTERN = "left_(palm|five|three|zero|six|four|one|two)_link"
_RIGHT_HAND_PATTERN = "right_(palm|five|three|zero|six|four|one|two)_link"

# Exact order used by the fixed-hand training policy.  Listing every joint
# avoids asset-order changes silently invalidating a checkpoint or sim2sim map.
_FIXED_HAND_LOWER_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
]
_FIXED_HAND_UPPER_JOINT_NAMES = [
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
]
_FIXED_HAND_WHOLE_BODY_JOINT_NAMES = [
    *_FIXED_HAND_LOWER_JOINT_NAMES,
    *_FIXED_HAND_UPPER_JOINT_NAMES,
]
_FIXED_HAND_BODY_NAMES = [
    "left_wrist_roll_rubber_hand",
    "right_wrist_roll_rubber_hand",
]
_FIXED_HAND_CENTER_OFFSET = 0.108


def _catch_joint_cfg() -> SceneEntityCfg:
    return SceneEntityCfg("robot", joint_names=_CATCH_JOINT_NAMES, preserve_order=True)


def _lower_body_joint_cfg() -> SceneEntityCfg:
    return SceneEntityCfg("robot", joint_names=_LOWER_BODY_JOINT_NAMES, preserve_order=True)


def _whole_body_joint_cfg() -> SceneEntityCfg:
    return SceneEntityCfg("robot", joint_names=_WHOLE_BODY_JOINT_NAMES, preserve_order=True)


def _palm_cfg() -> SceneEntityCfg:
    return SceneEntityCfg("robot", body_names=_PALM_NAMES, preserve_order=True)


def _left_hand_sensor_cfg() -> SceneEntityCfg:
    return SceneEntityCfg("contact_forces", body_names=_LEFT_HAND_PATTERN)


def _right_hand_sensor_cfg() -> SceneEntityCfg:
    return SceneEntityCfg("contact_forces", body_names=_RIGHT_HAND_PATTERN)


def _fixed_hand_lower_joint_cfg() -> SceneEntityCfg:
    return SceneEntityCfg(
        "robot", joint_names=_FIXED_HAND_LOWER_JOINT_NAMES, preserve_order=True
    )


def _fixed_hand_upper_joint_cfg() -> SceneEntityCfg:
    return SceneEntityCfg(
        "robot", joint_names=_FIXED_HAND_UPPER_JOINT_NAMES, preserve_order=True
    )


def _fixed_hand_whole_body_joint_cfg() -> SceneEntityCfg:
    return SceneEntityCfg(
        "robot", joint_names=_FIXED_HAND_WHOLE_BODY_JOINT_NAMES, preserve_order=True
    )


def _fixed_hand_body_cfg() -> SceneEntityCfg:
    return SceneEntityCfg("robot", body_names=_FIXED_HAND_BODY_NAMES, preserve_order=True)


def _fixed_left_hand_sensor_cfg() -> SceneEntityCfg:
    return SceneEntityCfg(
        "contact_forces", body_names="left_wrist_roll_rubber_hand"
    )


def _fixed_right_hand_sensor_cfg() -> SceneEntityCfg:
    return SceneEntityCfg(
        "contact_forces", body_names="right_wrist_roll_rubber_hand"
    )


def _fixed_feet_sensor_cfg() -> SceneEntityCfg:
    return SceneEntityCfg(
        "contact_forces", body_names=".*_ankle_roll_link", preserve_order=True
    )


@configclass
class CatchBoxEventsCfg(VelocityEventCfg):
    """Launch one front-facing projectile whenever an episode resets."""

    launch_box = EventTerm(
        func=mdp.throw_random_box,
        mode="reset",
        params={
            "collection_name": "throw_boxes",
            "robot_name": "robot",
            "azimuth_range": (-0.22, 0.22),
            "distance_range": (1.0, 1.45),
            "flight_time_range": (0.55, 0.75),
            "target_height_offset_range": (0.22, 0.42),
            "spawn_height_offset_range": (-0.05, 0.20),
            "horizontal_aim_noise": 0.08,
            "angular_velocity_range": (-1.5, 1.5),
            "gravity_z": -9.81,
            "parking_depth": -20.0,
        },
    )


@configclass
class G1CatchActionsCfg:
    """Bounded position control for the torso, arms and finger joints."""

    joint_pos = base_mdp.EMAJointPositionToLimitsActionCfg(
        asset_name="robot",
        joint_names=_CATCH_JOINT_NAMES,
        scale=1.0,
        alpha=0.40,
        rescale_to_limits=True,
        preserve_order=True,
    )


@configclass
class G1WholeBodyCatchActionsCfg:
    """Two coupled action branches for balance and two-handed catching."""

    # Small residual targets around the nominal standing pose keep early PPO
    # exploration from throwing the legs across their complete joint ranges.
    lower_body = base_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=_LOWER_BODY_JOINT_NAMES,
        scale=0.25,
        use_default_offset=True,
        preserve_order=True,
    )
    # The upper body needs a larger workspace to intercept the projectile.  A
    # zero action is still the G1 default pose, which makes an untrained policy
    # substantially safer to visualize than a joint-limit-centered action.
    upper_body = base_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=_CATCH_JOINT_NAMES,
        scale={
            "torso_joint": 0.45,
            ".*_shoulder_.*_joint": 0.90,
            ".*_elbow_.*_joint": 0.90,
            ".*_(five|three|zero|six|four|one|two)_joint": 1.0,
        },
        use_default_offset=True,
        preserve_order=True,
    )


@configclass
class G1CatchObservationsCfg:
    """Proprioception plus active-box and hand-contact state."""

    @configclass
    class PolicyCfg(ObsGroup):
        box_state = ObsTerm(
            func=mdp.active_box_state_b,
            params={"collection_name": "throw_boxes", "robot_cfg": SceneEntityCfg("robot")},
            clip=(-10.0, 10.0),
        )
        palm_box_kinematics = ObsTerm(
            func=mdp.palm_box_kinematics_b,
            params={"collection_name": "throw_boxes", "robot_cfg": _palm_cfg()},
            clip=(-10.0, 10.0),
        )
        hand_contacts = ObsTerm(
            func=mdp.hand_contact_forces,
            params={
                "left_sensor_cfg": _left_hand_sensor_cfg(),
                "right_sensor_cfg": _right_hand_sensor_cfg(),
                "force_scale": 25.0,
            },
        )
        joint_pos = ObsTerm(
            func=base_mdp.joint_pos_limit_normalized,
            params={"asset_cfg": _catch_joint_cfg()},
        )
        joint_vel = ObsTerm(
            func=base_mdp.joint_vel_rel,
            params={"asset_cfg": _catch_joint_cfg()},
            scale=0.1,
            clip=(-10.0, 10.0),
        )
        actions = ObsTerm(func=base_mdp.last_action)
        # Kept as a compatibility field because the inherited flat-terrain
        # configuration explicitly disables this observation.
        height_scan: ObsTerm | None = None

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class G1WholeBodyCatchObservationsCfg:
    """Whole-body proprioception fused with projectile and hand state."""

    @configclass
    class PolicyCfg(ObsGroup):
        # Lower-body state: the policy must observe the floating base in order
        # to use its legs to reject the impulse generated by the caught box.
        base_lin_vel = ObsTerm(func=base_mdp.base_lin_vel, clip=(-10.0, 10.0))
        base_ang_vel = ObsTerm(func=base_mdp.base_ang_vel, clip=(-10.0, 10.0))
        projected_gravity = ObsTerm(func=base_mdp.projected_gravity)
        box_state = ObsTerm(
            func=mdp.active_box_state_b,
            params={"collection_name": "throw_boxes", "robot_cfg": SceneEntityCfg("robot")},
            clip=(-10.0, 10.0),
        )
        palm_box_kinematics = ObsTerm(
            func=mdp.palm_box_kinematics_b,
            params={"collection_name": "throw_boxes", "robot_cfg": _palm_cfg()},
            clip=(-10.0, 10.0),
        )
        hand_contacts = ObsTerm(
            func=mdp.hand_contact_forces,
            params={
                "left_sensor_cfg": _left_hand_sensor_cfg(),
                "right_sensor_cfg": _right_hand_sensor_cfg(),
                "force_scale": 25.0,
            },
        )
        joint_pos = ObsTerm(
            func=base_mdp.joint_pos_rel,
            params={"asset_cfg": _whole_body_joint_cfg()},
            clip=(-3.0, 3.0),
        )
        joint_vel = ObsTerm(
            func=base_mdp.joint_vel_rel,
            params={"asset_cfg": _whole_body_joint_cfg()},
            scale=0.1,
            clip=(-10.0, 10.0),
        )
        actions = ObsTerm(func=base_mdp.last_action)
        height_scan: ObsTerm | None = None

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class G1CatchRewardsCfg(G1Rewards):
    """Dense shaping rewards followed by physical-contact catch rewards."""

    hand_proximity = RewTerm(
        func=mdp.hand_target_proximity,
        weight=6.0,
        params={
            "std": 0.25,
            "collection_name": "throw_boxes",
            "robot_cfg": _palm_cfg(),
            "clearance": 0.025,
        },
    )
    centered_between_palms = RewTerm(
        func=mdp.box_centered_between_palms,
        weight=2.0,
        params={"std": 0.20, "collection_name": "throw_boxes", "robot_cfg": _palm_cfg()},
    )
    velocity_match = RewTerm(
        func=mdp.palm_box_velocity_match,
        weight=2.0,
        params={"std": 1.0, "collection_name": "throw_boxes", "robot_cfg": _palm_cfg()},
    )
    bilateral_contact = RewTerm(
        func=mdp.bilateral_hand_contact,
        weight=8.0,
        params={
            "force_threshold": 5.0,
            "left_sensor_cfg": _left_hand_sensor_cfg(),
            "right_sensor_cfg": _right_hand_sensor_cfg(),
        },
    )
    hold_height = RewTerm(
        func=mdp.box_held_above_ground,
        weight=3.0,
        params={
            "min_height": 0.45,
            "target_height": 0.90,
            "force_threshold": 5.0,
            "collection_name": "throw_boxes",
            "left_sensor_cfg": _left_hand_sensor_cfg(),
            "right_sensor_cfg": _right_hand_sensor_cfg(),
        },
    )
    catch_success = RewTerm(
        func=mdp.catch_success_reward,
        weight=25.0,
        params={
            "collection_name": "throw_boxes",
            "robot_cfg": _palm_cfg(),
            "left_sensor_cfg": _left_hand_sensor_cfg(),
            "right_sensor_cfg": _right_hand_sensor_cfg(),
            "force_threshold": 2.0,
            "max_hand_distance": 0.22,
            "max_relative_speed": 0.75,
            "min_height": 0.55,
        },
    )
    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-0.01)
    joint_acceleration = RewTerm(
        func=base_mdp.joint_acc_l2,
        weight=-1.0e-7,
        params={"asset_cfg": _catch_joint_cfg()},
    )
    joint_torque = RewTerm(
        func=base_mdp.joint_torques_l2,
        weight=-2.0e-6,
        params={"asset_cfg": _catch_joint_cfg()},
    )
    joint_limits = RewTerm(
        func=base_mdp.joint_pos_limits,
        weight=-0.25,
        params={"asset_cfg": _catch_joint_cfg()},
    )


@configclass
class G1WholeBodyCatchRewardsCfg(G1CatchRewardsCfg):
    """Catching rewards plus explicit standing and impact-recovery terms."""

    base_height = RewTerm(
        func=base_mdp.base_height_l2,
        weight=-4.0,
        params={"target_height": 0.74, "asset_cfg": SceneEntityCfg("robot")},
    )
    lower_body_posture = RewTerm(
        func=base_mdp.joint_deviation_l1,
        weight=-0.15,
        params={"asset_cfg": _lower_body_joint_cfg()},
    )


@configclass
class G1CatchTerminationsCfg(VelocityTerminationsCfg):
    """Finish on timeout, a miss, or a sustained successful catch."""

    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    box_dropped = DoneTerm(
        func=mdp.box_dropped,
        params={"min_height": 0.32, "collection_name": "throw_boxes"},
    )
    box_out_of_reach = DoneTerm(
        func=mdp.box_out_of_reach,
        params={
            "max_distance": 2.5,
            "collection_name": "throw_boxes",
            "robot_cfg": SceneEntityCfg("robot"),
        },
    )
    box_caught = DoneTerm(
        func=mdp.SustainedCatch,
        params={
            "collection_name": "throw_boxes",
            "robot_cfg": _palm_cfg(),
            "left_sensor_cfg": _left_hand_sensor_cfg(),
            "right_sensor_cfg": _right_hand_sensor_cfg(),
            "hold_time_s": 0.30,
            "force_threshold": 2.0,
            "max_hand_distance": 0.22,
            "max_relative_speed": 0.75,
            "min_height": 0.55,
        },
    )


@configclass
class G1WholeBodyCatchTerminationsCfg(G1CatchTerminationsCfg):
    """Terminate a miss, a stable catch, or loss of whole-body balance."""

    # Torso contact is allowed because it can be caused by the intended box
    # impact.  It is disabled after the upstream G1 configuration has finished
    # resolving its sensor body names; falling is detected geometrically.
    bad_orientation = DoneTerm(
        func=base_mdp.bad_orientation,
        params={"limit_angle": 0.80, "asset_cfg": SceneEntityCfg("robot")},
    )
    root_too_low = DoneTerm(
        func=base_mdp.root_height_below_minimum,
        params={"minimum_height": 0.52, "asset_cfg": SceneEntityCfg("robot")},
    )


def _make_throw_box_collection() -> RigidObjectCollectionCfg:
    """Create small, medium and large projectiles in every environment."""
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=20.0,
        max_angular_velocity=100.0,
        max_depenetration_velocity=5.0,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=1,
    )
    collision_props = sim_utils.CollisionPropertiesCfg(
        collision_enabled=True,
        contact_offset=0.01,
        rest_offset=0.0,
    )
    physics_material = sim_utils.RigidBodyMaterialCfg(
        static_friction=0.6,
        dynamic_friction=0.5,
        restitution=0.05,
    )

    specifications = {
        "small": ((0.15, 0.15, 0.15), 0.5, (0.15, 0.75, 1.0)),
        "medium": ((0.30, 0.25, 0.25), 1.5, (1.0, 0.65, 0.1)),
        "large": ((0.45, 0.35, 0.35), 3.0, (0.95, 0.2, 0.15)),
    }
    objects: dict[str, RigidObjectCfg] = {}
    for index, (name, (size, mass, color)) in enumerate(specifications.items()):
        objects[name] = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/ThrowBox_{name.title()}",
            spawn=sim_utils.CuboidCfg(
                size=size,
                rigid_props=rigid_props,
                collision_props=collision_props,
                mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                physics_material=physics_material,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=color,
                    roughness=0.7,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -20.0 - 2.0 * index)),
        )
    return RigidObjectCollectionCfg(rigid_objects=objects)


def _configure_box_throwing(env_cfg: G1FlatEnvCfg | G1RoughEnvCfg) -> None:
    """Add projectiles and allow intended impacts without instant termination."""
    env_cfg.scene.throw_boxes = _make_throw_box_collection()

    # The stock task terminates on any torso contact. Projectile contact is
    # intentional, so only terminate when the G1 has actually tipped over.
    env_cfg.terminations.base_contact.func = base_mdp.bad_orientation
    env_cfg.terminations.base_contact.params = {"limit_angle": 1.0}


def _configure_follow_camera(env_cfg: G1FlatEnvCfg | G1RoughEnvCfg) -> None:
    """Make the viewport follow the G1 in environment zero."""
    env_cfg.viewer.eye = (3.0, 3.0, 2.0)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.8)
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.asset_name = "robot"


@configclass
class G1FlatTrainEnvCfg(G1FlatEnvCfg):
    """Flat-ground curriculum used for the first training stage."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # 4096 environments is a good starting point for a 24 GB GPU.  Override
        # it at the CLI with --num_envs when memory is limited.
        self.scene.num_envs = 4096
        self.scene.env_spacing = 2.5
        self.episode_length_s = 20.0

        # Sample forward, lateral and yaw velocity commands.
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        _configure_follow_camera(self)


@configclass
class G1FlatPlayEnvCfg(G1FlatTrainEnvCfg):
    """Deterministic, lightweight configuration for policy evaluation."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 16
        self.episode_length_s = 40.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class G1RoughTrainEnvCfg(G1RoughEnvCfg):
    """Rough-terrain curriculum used to make the walking policy robust."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 4096
        self.scene.env_spacing = 2.5
        self.episode_length_s = 20.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.2, 0.2)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        _configure_follow_camera(self)


@configclass
class G1RoughPlayEnvCfg(G1RoughTrainEnvCfg):
    """Small rough-terrain scene for policy evaluation."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 16
        self.episode_length_s = 40.0
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class G1FlatBoxThrowEnvCfg(G1FlatTrainEnvCfg):
    """Flat-ground training with randomized box impacts."""

    events: BoxThrowEventsCfg = BoxThrowEventsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        _configure_box_throwing(self)


@configclass
class G1FlatBoxThrowPlayEnvCfg(G1FlatBoxThrowEnvCfg):
    """Visual evaluation of flat-ground box-impact recovery."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 4
        self.episode_length_s = 40.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class G1RoughBoxThrowEnvCfg(G1RoughTrainEnvCfg):
    """Rough-terrain training with randomized box impacts."""

    events: BoxThrowEventsCfg = BoxThrowEventsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        _configure_box_throwing(self)


@configclass
class G1RoughBoxThrowPlayEnvCfg(G1RoughBoxThrowEnvCfg):
    """Visual evaluation of rough-terrain box-impact recovery."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 4
        self.episode_length_s = 40.0
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class G1CatchBoxEnvCfg(G1FlatEnvCfg):
    """Fixed-base first-stage task for learning a two-handed projectile catch."""

    observations: G1CatchObservationsCfg = G1CatchObservationsCfg()
    actions: G1CatchActionsCfg = G1CatchActionsCfg()
    rewards: G1CatchRewardsCfg = G1CatchRewardsCfg()
    terminations: G1CatchTerminationsCfg = G1CatchTerminationsCfg()
    events: CatchBoxEventsCfg = CatchBoxEventsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        # Remove locomotion-only terms after the parent configuration has
        # finished applying its G1 defaults.
        for reward_name in (
            "termination_penalty",
            "track_lin_vel_xy_exp",
            "track_ang_vel_z_exp",
            "lin_vel_z_l2",
            "ang_vel_xy_l2",
            "dof_torques_l2",
            "dof_acc_l2",
            "action_rate_l2",
            "feet_air_time",
            "feet_slide",
            "undesired_contacts",
            "flat_orientation_l2",
            "dof_pos_limits",
            "joint_deviation_hip",
            "joint_deviation_arms",
            "joint_deviation_fingers",
            "joint_deviation_torso",
        ):
            setattr(self.rewards, reward_name, None)
        self.terminations.base_contact = None

        # Full G1 collision geometry is required for finger/box contacts.  The
        # stock locomotion task deliberately uses the minimal collision asset.
        self.scene.robot = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.articulation_props.fix_root_link = True
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = False
        self.scene.throw_boxes = _make_throw_box_collection()

        # Catching with the full hand collision model is heavier than the
        # locomotion task, so start with fewer parallel environments.
        self.scene.num_envs = 512
        self.scene.env_spacing = 3.0
        self.episode_length_s = 4.0
        self.sim.physx.gpu_max_rigid_patch_count = 20 * 2**15

        # The first stage isolates upper-body coordination.  A zero root reset
        # keeps every launch in front of the same shoulder frame.
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.reset_base.params = {
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.reset_robot_joints.params["position_range"] = (0.95, 1.05)

        # Velocity commands remain present in the inherited scene interface but
        # are intentionally inactive and are not part of the catch observation.
        self.commands.base_velocity.rel_standing_envs = 1.0
        self.commands.base_velocity.debug_vis = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        _configure_follow_camera(self)
        # The catch task has a fixed robot base, so continuous asset tracking is
        # unnecessary and would overwrite manual viewport orbit/zoom every
        # rendered frame.  A static environment origin preserves user edits.
        self.viewer.origin_type = "env"
        self.viewer.asset_name = None
        self.viewer.eye = (2.8, 2.8, 1.8)
        self.viewer.lookat = (0.35, 0.0, 0.95)


@configclass
class G1CatchBoxPlayEnvCfg(G1CatchBoxEnvCfg):
    """Single-scene visual evaluation for a trained catching policy."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 8.0
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.observations.policy.enable_corruption = False


@configclass
class G1WholeBodyCatchBoxEnvCfg(G1FlatEnvCfg):
    """Free-standing G1 task coupling leg balance and two-handed catching."""

    observations: G1WholeBodyCatchObservationsCfg = G1WholeBodyCatchObservationsCfg()
    actions: G1WholeBodyCatchActionsCfg = G1WholeBodyCatchActionsCfg()
    rewards: G1WholeBodyCatchRewardsCfg = G1WholeBodyCatchRewardsCfg()
    terminations: G1WholeBodyCatchTerminationsCfg = G1WholeBodyCatchTerminationsCfg()
    events: CatchBoxEventsCfg = CatchBoxEventsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        # The upstream G1 post-init accesses this term, so remove it only now.
        self.terminations.base_contact = None

        # Unlike the first-stage catch task, the root remains free and the full
        # collision asset supplies both foot-ground and hand-box contacts.
        self.scene.robot = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.articulation_props.fix_root_link = False
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = False
        self.scene.throw_boxes = _make_throw_box_collection()

        self.scene.num_envs = 256
        self.scene.env_spacing = 3.0
        self.episode_length_s = 5.0
        self.sim.physx.gpu_max_rigid_patch_count = 20 * 2**15

        # Start from an upright nominal stance.  The projectile itself provides
        # the perturbation, so unrelated pushes and mass randomization are off
        # for this demonstrator.
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.reset_base.params = {
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.reset_robot_joints.params["position_range"] = (0.98, 1.02)

        # A zero command turns the inherited locomotion tracking rewards into
        # standing-still rewards, while legs remain free to step after impact.
        self.commands.base_velocity.rel_standing_envs = 1.0
        self.commands.base_velocity.debug_vis = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        # Keep locomotion terms that teach balance, but remove terms that would
        # discourage the large, intentional upper-body catching motion.
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.lin_vel_z_l2.weight = -0.5
        self.rewards.ang_vel_xy_l2.weight = -0.25
        self.rewards.flat_orientation_l2.weight = -4.0
        self.rewards.feet_air_time = None
        self.rewards.feet_slide.weight = -0.2
        self.rewards.joint_deviation_arms = None
        self.rewards.joint_deviation_fingers = None
        self.rewards.joint_deviation_torso = None
        # G1CatchRewardsCfg already contains an all-action smoothness term.
        self.rewards.action_rate_l2 = None

        # A successful whole-body catch must remain physically stable; merely
        # grabbing the box while falling does not receive success credit.
        stability_params = {"max_robot_tilt": 0.55, "min_root_height": 0.60}
        self.rewards.catch_success.params.update(stability_params)
        self.terminations.box_caught.params.update(stability_params)

        # Static environment camera avoids forcing the user's viewport back on
        # every rendered frame while still showing small recovery steps.
        self.viewer.origin_type = "env"
        self.viewer.env_index = 0
        self.viewer.asset_name = None
        self.viewer.eye = (3.0, 3.0, 1.9)
        self.viewer.lookat = (0.35, 0.0, 0.90)


@configclass
class G1WholeBodyCatchBoxPlayEnvCfg(G1WholeBodyCatchBoxEnvCfg):
    """Single-environment visualization of the trained whole-body policy."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 8.0
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.observations.policy.enable_corruption = False
        # Show the stable hold for longer before the successful episode resets.
        self.terminations.box_caught.params["hold_time_s"] = 1.0


@configclass
class G1FixedHandCatchEventsCfg(VelocityEventCfg):
    """Reset events with a staged fixed-hand catching curriculum."""

    launch_box = EventTerm(
        func=mdp.throw_curriculum_box,
        mode="reset",
        params={
            "collection_name": "throw_boxes",
            "robot_name": "robot",
            "balance_warmup_steps": 20_000,
            "curriculum_steps": 100_000,
            "gravity_z": -9.81,
            "parking_depth": -20.0,
        },
    )


@configclass
class G1FixedHandWholeBodyActionsCfg:
    """Residual position targets for the true 23-DoF fixed-hand robot."""

    lower_body = base_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=_FIXED_HAND_LOWER_JOINT_NAMES,
        scale=0.25,
        use_default_offset=True,
        preserve_order=True,
    )
    upper_body = base_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=_FIXED_HAND_UPPER_JOINT_NAMES,
        scale={
            "waist_yaw_joint": 0.45,
            ".*_shoulder_pitch_joint": 0.90,
            ".*_shoulder_roll_joint": 0.90,
            ".*_shoulder_yaw_joint": 0.90,
            ".*_elbow_joint": 0.90,
            ".*_wrist_roll_joint": 0.90,
        },
        use_default_offset=True,
        preserve_order=True,
    )


@configclass
class G1FixedHandWholeBodyObservationsCfg:
    """108-D observation vector for a 23-D fixed-hand action space."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=base_mdp.base_lin_vel, clip=(-10.0, 10.0))
        base_ang_vel = ObsTerm(func=base_mdp.base_ang_vel, clip=(-10.0, 10.0))
        projected_gravity = ObsTerm(func=base_mdp.projected_gravity)
        box_state = ObsTerm(
            func=mdp.active_box_state_b,
            params={
                "collection_name": "throw_boxes",
                "robot_cfg": SceneEntityCfg("robot"),
            },
            clip=(-10.0, 10.0),
        )
        hand_box_kinematics = ObsTerm(
            func=mdp.palm_box_kinematics_b,
            params={
                "collection_name": "throw_boxes",
                "robot_cfg": _fixed_hand_body_cfg(),
                "hand_center_offset": _FIXED_HAND_CENTER_OFFSET,
            },
            clip=(-10.0, 10.0),
        )
        hand_contacts = ObsTerm(
            func=mdp.hand_contact_forces,
            params={
                "left_sensor_cfg": _fixed_left_hand_sensor_cfg(),
                "right_sensor_cfg": _fixed_right_hand_sensor_cfg(),
                "force_scale": 25.0,
            },
        )
        joint_pos = ObsTerm(
            func=base_mdp.joint_pos_rel,
            params={"asset_cfg": _fixed_hand_whole_body_joint_cfg()},
            clip=(-3.0, 3.0),
        )
        joint_vel = ObsTerm(
            func=base_mdp.joint_vel_rel,
            params={"asset_cfg": _fixed_hand_whole_body_joint_cfg()},
            scale=0.1,
            clip=(-10.0, 10.0),
        )
        actions = ObsTerm(func=base_mdp.last_action)
        height_scan: ObsTerm | None = None

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class G1FixedHandCatchRewardsCfg(G1WholeBodyCatchRewardsCfg):
    """Contact-progress rewards plus stronger stance regularization."""

    termination_penalty = RewTerm(
        func=base_mdp.is_terminated_term,
        weight=-200.0,
        params={
            "term_keys": [
                "box_dropped",
                "box_out_of_reach",
                "bad_orientation",
                "root_too_low",
            ]
        },
    )
    hand_proximity = RewTerm(
        func=mdp.hand_target_proximity,
        weight=10.0,
        params={
            "std": 0.22,
            "collection_name": "throw_boxes",
            "robot_cfg": _fixed_hand_body_cfg(),
            "clearance": 0.018,
            "hand_center_offset": _FIXED_HAND_CENTER_OFFSET,
        },
    )
    centered_between_palms = RewTerm(
        func=mdp.box_centered_between_palms,
        weight=4.0,
        params={
            "std": 0.16,
            "collection_name": "throw_boxes",
            "robot_cfg": _fixed_hand_body_cfg(),
            "hand_center_offset": _FIXED_HAND_CENTER_OFFSET,
        },
    )
    velocity_match = RewTerm(
        func=mdp.palm_box_velocity_match,
        weight=4.0,
        params={
            "std": 0.8,
            "collection_name": "throw_boxes",
            "robot_cfg": _fixed_hand_body_cfg(),
            "hand_center_offset": _FIXED_HAND_CENTER_OFFSET,
        },
    )
    contact_progress = RewTerm(
        func=mdp.hand_contact_progress,
        weight=4.0,
        params={
            "force_threshold": 3.0,
            "collection_name": "throw_boxes",
            "left_sensor_cfg": _fixed_left_hand_sensor_cfg(),
            "right_sensor_cfg": _fixed_right_hand_sensor_cfg(),
        },
    )
    bilateral_contact = RewTerm(
        func=mdp.bilateral_hand_contact,
        weight=12.0,
        params={
            "force_threshold": 3.0,
            "collection_name": "throw_boxes",
            "left_sensor_cfg": _fixed_left_hand_sensor_cfg(),
            "right_sensor_cfg": _fixed_right_hand_sensor_cfg(),
        },
    )
    hold_height = RewTerm(
        func=mdp.box_held_above_ground,
        weight=8.0,
        params={
            "min_height": 0.50,
            "target_height": 0.95,
            "force_threshold": 2.0,
            "collection_name": "throw_boxes",
            "left_sensor_cfg": _fixed_left_hand_sensor_cfg(),
            "right_sensor_cfg": _fixed_right_hand_sensor_cfg(),
        },
    )
    catch_success = RewTerm(
        func=mdp.catch_success_reward,
        weight=50.0,
        params={
            "collection_name": "throw_boxes",
            "robot_cfg": _fixed_hand_body_cfg(),
            "left_sensor_cfg": _fixed_left_hand_sensor_cfg(),
            "right_sensor_cfg": _fixed_right_hand_sensor_cfg(),
            "force_threshold": 1.5,
            "max_hand_distance": 0.18,
            "max_relative_speed": 0.80,
            "min_height": 0.52,
            "hand_center_offset": _FIXED_HAND_CENTER_OFFSET,
        },
    )
    both_feet_contact = RewTerm(
        func=mdp.bilateral_foot_contact,
        weight=1.5,
        params={"sensor_cfg": _fixed_feet_sensor_cfg(), "force_threshold": 20.0},
    )
    base_height = RewTerm(
        func=base_mdp.base_height_l2,
        weight=-8.0,
        params={"target_height": 0.74, "asset_cfg": SceneEntityCfg("robot")},
    )
    lower_body_posture = RewTerm(
        func=base_mdp.joint_deviation_l1,
        weight=-0.30,
        params={"asset_cfg": _fixed_hand_lower_joint_cfg()},
    )
    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-0.03)
    joint_acceleration = RewTerm(
        func=base_mdp.joint_acc_l2,
        weight=-1.0e-7,
        params={"asset_cfg": _fixed_hand_whole_body_joint_cfg()},
    )
    joint_torque = RewTerm(
        func=base_mdp.joint_torques_l2,
        weight=-2.0e-6,
        params={"asset_cfg": _fixed_hand_whole_body_joint_cfg()},
    )
    joint_limits = RewTerm(
        func=base_mdp.joint_pos_limits,
        weight=-0.50,
        params={"asset_cfg": _fixed_hand_whole_body_joint_cfg()},
    )


@configclass
class G1FixedHandCatchTerminationsCfg(G1WholeBodyCatchTerminationsCfg):
    """Catch, miss and balance criteria for the fixed-hand model."""

    box_caught = DoneTerm(
        func=mdp.SustainedCatch,
        params={
            "collection_name": "throw_boxes",
            "robot_cfg": _fixed_hand_body_cfg(),
            "left_sensor_cfg": _fixed_left_hand_sensor_cfg(),
            "right_sensor_cfg": _fixed_right_hand_sensor_cfg(),
            "hold_time_s": 0.25,
            "force_threshold": 1.5,
            "max_hand_distance": 0.18,
            "max_relative_speed": 0.80,
            "min_height": 0.52,
            "hand_center_offset": _FIXED_HAND_CENTER_OFFSET,
            "max_robot_tilt": 0.50,
            "min_root_height": 0.58,
        },
    )
    bad_orientation = DoneTerm(
        func=base_mdp.bad_orientation,
        params={"limit_angle": 0.65, "asset_cfg": SceneEntityCfg("robot")},
    )
    root_too_low = DoneTerm(
        func=base_mdp.root_height_below_minimum,
        params={"minimum_height": 0.54, "asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class G1FixedHandWholeBodyCatchBoxEnvCfg(G1WholeBodyCatchBoxEnvCfg):
    """Stable whole-body catch training using the real 23-DoF fixed-hand G1."""

    observations: G1FixedHandWholeBodyObservationsCfg = G1FixedHandWholeBodyObservationsCfg()
    actions: G1FixedHandWholeBodyActionsCfg = G1FixedHandWholeBodyActionsCfg()
    rewards: G1FixedHandCatchRewardsCfg = G1FixedHandCatchRewardsCfg()
    terminations: G1FixedHandCatchTerminationsCfg = G1FixedHandCatchTerminationsCfg()
    events: G1FixedHandCatchEventsCfg = G1FixedHandCatchEventsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.robot = G1_23DOF_FIXED_HAND_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.scene.num_envs = 256
        self.episode_length_s = 6.0
        self.events.reset_robot_joints.params["position_range"] = (0.995, 1.005)

        # Avoid duplicating the task-specific torque, acceleration and limit
        # terms inherited from the locomotion base configuration.
        self.rewards.dof_torques_l2 = None
        self.rewards.dof_acc_l2 = None
        self.rewards.dof_pos_limits = None
        self.rewards.joint_deviation_hip = None
        self.rewards.track_lin_vel_xy_exp.weight = 2.5
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.lin_vel_z_l2.weight = -1.0
        self.rewards.ang_vel_xy_l2.weight = -0.50
        self.rewards.flat_orientation_l2.weight = -6.0
        self.rewards.feet_slide.weight = -0.50

        stable_catch = {"max_robot_tilt": 0.50, "min_root_height": 0.58}
        self.rewards.catch_success.params.update(stable_catch)
        self.terminations.box_caught.params.update(stable_catch)


@configclass
class G1FixedHandWholeBodyCatchBoxPlayEnvCfg(G1FixedHandWholeBodyCatchBoxEnvCfg):
    """Single-scene evaluation for a fixed-hand catch checkpoint."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 8.0
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.observations.policy.enable_corruption = False
        self.terminations.box_caught.params["hold_time_s"] = 0.8
