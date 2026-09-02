"""Observations, rewards and terminations for catching projectile boxes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from isaaclab.assets import Articulation, RigidObjectCollection
from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# The order matches the objects created by ``_make_throw_box_collection``.
_BOX_HALF_WIDTHS = (0.075, 0.125, 0.175)


def _active_box(
    env: ManagerBasedRLEnv,
    collection_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the active box state and its object index for every environment."""
    boxes: RigidObjectCollection = env.scene[collection_name]
    all_states = boxes.data.object_state_w
    fallback_ids = torch.argmax(all_states[..., 2], dim=1)
    active_ids = getattr(env, "_active_throw_box_ids", fallback_ids)
    active_ids = active_ids.to(device=env.device, dtype=torch.long)
    active_ids = torch.where(active_ids >= 0, active_ids, fallback_ids)
    env_ids = torch.arange(env.num_envs, device=env.device)
    return all_states[env_ids, active_ids], active_ids


def _has_active_box(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return whether each environment has left the balance-only warm-up."""
    active_ids = getattr(env, "_active_throw_box_ids", None)
    if active_ids is None:
        return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    return active_ids.to(device=env.device) >= 0


def _rotate_vectors_to_root(vectors_w: torch.Tensor, root_quat_w: torch.Tensor) -> torch.Tensor:
    """Rotate ``(N, K, 3)`` world vectors into the robot root frame."""
    num_vectors = vectors_w.shape[1]
    quaternions = root_quat_w.unsqueeze(1).expand(-1, num_vectors, -1)
    vectors_b = math_utils.quat_apply_inverse(
        quaternions.reshape(-1, 4), vectors_w.reshape(-1, 3)
    )
    return vectors_b.reshape_as(vectors_w)


def _hand_force(
    contact_sensor: ContactSensor,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return the largest recent contact force on one hand."""
    forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    return forces.norm(dim=-1).amax(dim=(1, 2))


def _hand_forces(
    env: ManagerBasedRLEnv,
    left_sensor_cfg: SceneEntityCfg,
    right_sensor_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    contact_sensor: ContactSensor = env.scene.sensors[left_sensor_cfg.name]
    return (
        _hand_force(contact_sensor, left_sensor_cfg),
        _hand_force(contact_sensor, right_sensor_cfg),
    )


def _palm_targets_w(
    env: ManagerBasedRLEnv,
    collection_name: str,
    robot_cfg: SceneEntityCfg,
    clearance: float = 0.025,
    hand_center_offset: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute lateral box-side targets and current palm state in world frame."""
    boxes, active_ids = _active_box(env, collection_name)
    robot: Articulation = env.scene[robot_cfg.name]
    palm_pos = robot.data.body_link_pos_w[:, robot_cfg.body_ids]
    palm_vel = robot.data.body_link_lin_vel_w[:, robot_cfg.body_ids]
    if hand_center_offset != 0.0:
        # A rigid rubber hand's link frame is at the wrist, while the useful
        # contact surface is near the center of its 21 cm-long mesh.  Rewards
        # and observations must track that point rather than the wrist joint.
        local_offsets = torch.zeros_like(palm_pos)
        local_offsets[..., 0] = hand_center_offset
        offset_w = math_utils.quat_apply(
            robot.data.body_link_quat_w[:, robot_cfg.body_ids].reshape(-1, 4),
            local_offsets.reshape(-1, 3),
        ).reshape_as(local_offsets)
        palm_pos = palm_pos + offset_w
        palm_vel = palm_vel + torch.linalg.cross(
            robot.data.body_link_ang_vel_w[:, robot_cfg.body_ids], offset_w
        )

    lateral_axis_b = torch.zeros((env.num_envs, 3), device=env.device)
    lateral_axis_b[:, 1] = 1.0
    lateral_axis_w = math_utils.quat_apply(robot.data.root_quat_w, lateral_axis_b)
    half_widths = torch.tensor(_BOX_HALF_WIDTHS, device=env.device)[active_ids]
    offsets = lateral_axis_w * (half_widths + clearance).unsqueeze(-1)
    targets = torch.stack((boxes[:, :3] + offsets, boxes[:, :3] - offsets), dim=1)
    return targets, palm_pos, palm_vel, boxes


def active_box_state_b(
    env: ManagerBasedRLEnv,
    collection_name: str = "throw_boxes",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Box pose/velocity in the root frame followed by a one-hot size code."""
    box, active_ids = _active_box(env, collection_name)
    robot: Articulation = env.scene[robot_cfg.name]
    box_pos_b, box_quat_b = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        box[:, :3],
        box[:, 3:7],
    )
    box_quat_b = math_utils.quat_unique(box_quat_b)
    box_lin_vel_b = math_utils.quat_apply_inverse(
        robot.data.root_quat_w, box[:, 7:10] - robot.data.root_lin_vel_w
    )
    box_ang_vel_b = math_utils.quat_apply_inverse(
        robot.data.root_quat_w, box[:, 10:13] - robot.data.root_ang_vel_w
    )
    box_size = F.one_hot(active_ids, num_classes=3).to(dtype=box.dtype)
    state = torch.cat((box_pos_b, box_quat_b, box_lin_vel_b, box_ang_vel_b, box_size), dim=-1)
    return state * _has_active_box(env).unsqueeze(-1)


def palm_box_kinematics_b(
    env: ManagerBasedRLEnv,
    collection_name: str = "throw_boxes",
    robot_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["left_palm_link", "right_palm_link"], preserve_order=True
    ),
    hand_center_offset: float = 0.0,
) -> torch.Tensor:
    """Vectors and relative velocities from both palms to the active box."""
    _, palm_pos, palm_vel, box = _palm_targets_w(
        env,
        collection_name,
        robot_cfg,
        hand_center_offset=hand_center_offset,
    )
    robot: Articulation = env.scene[robot_cfg.name]
    position_delta_w = box[:, None, :3] - palm_pos
    velocity_delta_w = box[:, None, 7:10] - palm_vel
    position_delta_b = _rotate_vectors_to_root(position_delta_w, robot.data.root_quat_w)
    velocity_delta_b = _rotate_vectors_to_root(velocity_delta_w, robot.data.root_quat_w)
    kinematics = torch.cat((position_delta_b.flatten(1), velocity_delta_b.flatten(1)), dim=-1)
    return kinematics * _has_active_box(env).unsqueeze(-1)


def hand_contact_forces(
    env: ManagerBasedRLEnv,
    left_sensor_cfg: SceneEntityCfg,
    right_sensor_cfg: SceneEntityCfg,
    force_scale: float = 25.0,
) -> torch.Tensor:
    """Normalized left/right hand contact magnitudes."""
    left_force, right_force = _hand_forces(env, left_sensor_cfg, right_sensor_cfg)
    return torch.stack((left_force, right_force), dim=-1).div(force_scale).clamp(0.0, 2.0)


def hand_target_proximity(
    env: ManagerBasedRLEnv,
    std: float,
    collection_name: str,
    robot_cfg: SceneEntityCfg,
    clearance: float = 0.025,
    hand_center_offset: float = 0.0,
) -> torch.Tensor:
    """Dense reward for moving each palm toward the corresponding box side."""
    targets, palms, _, _ = _palm_targets_w(
        env, collection_name, robot_cfg, clearance, hand_center_offset
    )
    mean_distance = torch.linalg.vector_norm(targets - palms, dim=-1).mean(dim=1)
    return torch.exp(-mean_distance / std) * _has_active_box(env)


def box_centered_between_palms(
    env: ManagerBasedRLEnv,
    std: float,
    collection_name: str,
    robot_cfg: SceneEntityCfg,
    hand_center_offset: float = 0.0,
) -> torch.Tensor:
    """Reward placing the box at the midpoint of both palms."""
    _, palms, _, box = _palm_targets_w(
        env, collection_name, robot_cfg, hand_center_offset=hand_center_offset
    )
    midpoint_error = torch.linalg.vector_norm(palms.mean(dim=1) - box[:, :3], dim=-1)
    return torch.exp(-midpoint_error / std) * _has_active_box(env)


def palm_box_velocity_match(
    env: ManagerBasedRLEnv,
    std: float,
    collection_name: str,
    robot_cfg: SceneEntityCfg,
    hand_center_offset: float = 0.0,
) -> torch.Tensor:
    """Reward matching average palm velocity to the incoming box."""
    _, _, palm_vel, box = _palm_targets_w(
        env, collection_name, robot_cfg, hand_center_offset=hand_center_offset
    )
    relative_speed = torch.linalg.vector_norm(palm_vel.mean(dim=1) - box[:, 7:10], dim=-1)
    return torch.exp(-relative_speed / std) * _has_active_box(env)


def bilateral_hand_contact(
    env: ManagerBasedRLEnv,
    force_threshold: float,
    left_sensor_cfg: SceneEntityCfg,
    right_sensor_cfg: SceneEntityCfg,
    collection_name: str | None = None,
) -> torch.Tensor:
    """Reward simultaneous contact on the left and right hands."""
    left_force, right_force = _hand_forces(env, left_sensor_cfg, right_sensor_cfg)
    left_score = (left_force / force_threshold).clamp(0.0, 1.0)
    right_score = (right_force / force_threshold).clamp(0.0, 1.0)
    contact = torch.minimum(left_score, right_score)
    if collection_name is not None:
        contact *= _has_active_box(env)
    return contact


def hand_contact_progress(
    env: ManagerBasedRLEnv,
    force_threshold: float,
    left_sensor_cfg: SceneEntityCfg,
    right_sensor_cfg: SceneEntityCfg,
    collection_name: str | None = None,
) -> torch.Tensor:
    """Give partial credit for one-hand contact on the way to a two-hand clamp."""
    left_force, right_force = _hand_forces(env, left_sensor_cfg, right_sensor_cfg)
    left_score = (left_force / force_threshold).clamp(0.0, 1.0)
    right_score = (right_force / force_threshold).clamp(0.0, 1.0)
    progress = 0.5 * (left_score + right_score)
    if collection_name is not None:
        progress *= _has_active_box(env)
    return progress


def bilateral_foot_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float = 20.0,
) -> torch.Tensor:
    """Reward maintaining support through both feet during the interception."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    foot_forces = forces.norm(dim=-1).amax(dim=1)
    scores = (foot_forces / force_threshold).clamp(0.0, 1.0)
    return scores.amin(dim=1)


def box_held_above_ground(
    env: ManagerBasedRLEnv,
    min_height: float,
    target_height: float,
    force_threshold: float,
    collection_name: str,
    left_sensor_cfg: SceneEntityCfg,
    right_sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward lifting/holding the box only while both hands are in contact."""
    box, _ = _active_box(env, collection_name)
    contacts = bilateral_hand_contact(env, force_threshold, left_sensor_cfg, right_sensor_cfg)
    height_score = ((box[:, 2] - min_height) / (target_height - min_height)).clamp(0.0, 1.0)
    return contacts * height_score * _has_active_box(env)


def _catch_condition(
    env: ManagerBasedRLEnv,
    collection_name: str,
    robot_cfg: SceneEntityCfg,
    left_sensor_cfg: SceneEntityCfg,
    right_sensor_cfg: SceneEntityCfg,
    force_threshold: float,
    max_hand_distance: float,
    max_relative_speed: float,
    min_height: float,
    hand_center_offset: float = 0.0,
    max_robot_tilt: float | None = None,
    min_root_height: float | None = None,
) -> torch.Tensor:
    targets, palms, palm_vel, box = _palm_targets_w(
        env, collection_name, robot_cfg, hand_center_offset=hand_center_offset
    )
    target_distances = torch.linalg.vector_norm(targets - palms, dim=-1)
    relative_speed = torch.linalg.vector_norm(palm_vel.mean(dim=1) - box[:, 7:10], dim=-1)
    left_force, right_force = _hand_forces(env, left_sensor_cfg, right_sensor_cfg)
    caught = (
        (left_force > force_threshold)
        & (right_force > force_threshold)
        & (target_distances.amax(dim=1) < max_hand_distance)
        & (relative_speed < max_relative_speed)
        & (box[:, 2] > min_height)
    )
    caught &= _has_active_box(env)
    # The fixed-base task does not need these checks.  They are optional so the
    # same catch definition can also require a free-standing humanoid to remain
    # upright after it absorbs the projectile impulse.
    robot: Articulation = env.scene[robot_cfg.name]
    if max_robot_tilt is not None:
        tilt = torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0))
        caught &= tilt < max_robot_tilt
    if min_root_height is not None:
        root_height = robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
        caught &= root_height > min_root_height
    return caught


def catch_success_reward(
    env: ManagerBasedRLEnv,
    collection_name: str,
    robot_cfg: SceneEntityCfg,
    left_sensor_cfg: SceneEntityCfg,
    right_sensor_cfg: SceneEntityCfg,
    force_threshold: float = 2.0,
    max_hand_distance: float = 0.22,
    max_relative_speed: float = 0.75,
    min_height: float = 0.55,
    hand_center_offset: float = 0.0,
    max_robot_tilt: float | None = None,
    min_root_height: float | None = None,
) -> torch.Tensor:
    """Sparse bonus for a physically plausible two-handed catch."""
    return _catch_condition(
        env,
        collection_name,
        robot_cfg,
        left_sensor_cfg,
        right_sensor_cfg,
        force_threshold,
        max_hand_distance,
        max_relative_speed,
        min_height,
        hand_center_offset,
        max_robot_tilt,
        min_root_height,
    ).float()


def box_dropped(
    env: ManagerBasedRLEnv,
    min_height: float,
    collection_name: str = "throw_boxes",
) -> torch.Tensor:
    """Terminate after the active projectile falls below the catch region."""
    box, _ = _active_box(env, collection_name)
    return _has_active_box(env) & (box[:, 2] < min_height)


def box_out_of_reach(
    env: ManagerBasedRLEnv,
    max_distance: float,
    collection_name: str = "throw_boxes",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate if the projectile has passed far outside the workspace."""
    box, _ = _active_box(env, collection_name)
    robot: Articulation = env.scene[robot_cfg.name]
    return _has_active_box(env) & (
        torch.linalg.vector_norm(box[:, :3] - robot.data.root_pos_w, dim=-1) > max_distance
    )


class SustainedCatch(ManagerTermBase):
    """Declare success only after the two-handed catch is held continuously."""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._consecutive_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._consecutive_steps[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        collection_name: str,
        robot_cfg: SceneEntityCfg,
        left_sensor_cfg: SceneEntityCfg,
        right_sensor_cfg: SceneEntityCfg,
        hold_time_s: float = 0.30,
        force_threshold: float = 2.0,
        max_hand_distance: float = 0.22,
        max_relative_speed: float = 0.75,
        min_height: float = 0.55,
        hand_center_offset: float = 0.0,
        max_robot_tilt: float | None = None,
        min_root_height: float | None = None,
    ) -> torch.Tensor:
        caught = _catch_condition(
            env,
            collection_name,
            robot_cfg,
            left_sensor_cfg,
            right_sensor_cfg,
            force_threshold,
            max_hand_distance,
            max_relative_speed,
            min_height,
            hand_center_offset,
            max_robot_tilt,
            min_root_height,
        )
        self._consecutive_steps[:] = torch.where(caught, self._consecutive_steps + 1, 0)
        required_steps = max(1, round(hold_time_s / env.step_dt))
        return self._consecutive_steps >= required_steps
