"""Unitree G1 23-DoF configuration with fixed rubber hands."""

from __future__ import annotations

import tempfile
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_ASSET_ROOT = _PROJECT_ROOT / "assets" / "isaaclab" / "unitree_g1_23dof"


G1_23DOF_FIXED_HAND_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(_ASSET_ROOT / "g1_23dof_fixed_hand.urdf"),
        usd_dir=str(Path(tempfile.gettempdir()) / "g1_locomotion" / "g1_23dof_fixed_hand"),
        usd_file_name="g1_23dof_fixed_hand.usd",
        fix_base=False,
        merge_fixed_joints=True,
        self_collision=False,
        collider_type="convex_hull",
        activate_contact_sensors=True,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            target_type="none",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0.0,
                damping=0.0,
            ),
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            fix_root_link=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # The official 0.793 m free-root height is for straight legs.  This
        # 0.74 m height matches the bent nominal stance below and places the
        # full sole colliders on the ground without a settling drop.
        pos=(0.0, 0.0, 0.74),
        joint_pos={
            ".*_hip_pitch_joint": -0.20,
            ".*_knee_joint": 0.42,
            ".*_ankle_pitch_joint": -0.23,
            "waist_yaw_joint": 0.0,
            ".*_shoulder_pitch_joint": 0.35,
            "left_shoulder_roll_joint": 0.16,
            "right_shoulder_roll_joint": -0.16,
            ".*_shoulder_yaw_joint": 0.0,
            ".*_elbow_joint": 0.87,
            ".*_wrist_roll_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                "waist_yaw_joint",
            ],
            effort_limit_sim={
                ".*_hip_.*_joint": 88.0,
                ".*_knee_joint": 139.0,
                "waist_yaw_joint": 88.0,
            },
            velocity_limit_sim={
                ".*_hip_.*_joint": 32.0,
                ".*_knee_joint": 20.0,
                "waist_yaw_joint": 32.0,
            },
            stiffness={
                ".*_hip_yaw_joint": 150.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_joint": 200.0,
                "waist_yaw_joint": 200.0,
            },
            damping={
                ".*_hip_.*_joint": 5.0,
                ".*_knee_joint": 5.0,
                "waist_yaw_joint": 5.0,
            },
            armature=0.01,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=35.0,
            velocity_limit_sim=30.0,
            stiffness=35.0,
            damping=3.0,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
            ],
            effort_limit_sim=25.0,
            velocity_limit_sim=37.0,
            stiffness={
                ".*_shoulder_.*_joint": 40.0,
                ".*_elbow_joint": 40.0,
                ".*_wrist_roll_joint": 20.0,
            },
            damping={
                ".*_shoulder_.*_joint": 8.0,
                ".*_elbow_joint": 8.0,
                ".*_wrist_roll_joint": 4.0,
            },
            armature=0.01,
        ),
    },
)
"""Official 23-DoF G1 with project-local black fixed hands and full-sole colliders."""
