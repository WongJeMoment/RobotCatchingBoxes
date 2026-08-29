"""Events that throw randomized boxes at a locomotion robot."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObjectCollection
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _uniform(
    value_range: tuple[float, float],
    shape: tuple[int, ...],
    device: str,
) -> torch.Tensor:
    """Sample uniformly without creating CPU tensors."""
    low, high = value_range
    return low + (high - low) * torch.rand(shape, device=device)


def _resolve_env_ids(env: ManagerBasedEnv, env_ids: torch.Tensor | None) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    return env_ids.to(device=env.device, dtype=torch.long)


def park_throw_boxes(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    collection_name: str = "throw_boxes",
    parking_depth: float = -20.0,
) -> None:
    """Move all projectile boxes below their environments and stop them."""
    resolved_env_ids = _resolve_env_ids(env, env_ids)
    boxes: RigidObjectCollection = env.scene[collection_name]
    num_envs = len(resolved_env_ids)
    if num_envs == 0:
        return

    states = boxes.data.default_object_state[resolved_env_ids].clone()
    origins = env.scene.env_origins[resolved_env_ids]
    states[..., :3] = origins.unsqueeze(1)
    states[..., 2] += parking_depth

    # Separate parked boxes so they cannot collide with each other underground.
    object_offsets = torch.arange(boxes.num_objects, device=boxes.device, dtype=states.dtype)
    states[..., 2] -= object_offsets.unsqueeze(0) * 2.0
    states[..., 3:7] = 0.0
    states[..., 3] = 1.0
    states[..., 7:] = 0.0
    boxes.write_object_state_to_sim(states, env_ids=resolved_env_ids)

    # Catching observations use this index instead of trying to infer the
    # selected object from its pose.  Keep the buffer on the environment so it
    # naturally works with partial resets.
    if not hasattr(env, "_active_throw_box_ids"):
        env._active_throw_box_ids = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=env.device
        )
    env._active_throw_box_ids[resolved_env_ids] = -1


def throw_random_box(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    collection_name: str = "throw_boxes",
    robot_name: str = "robot",
    azimuth_range: tuple[float, float] = (0.0, 2.0 * torch.pi),
    distance_range: tuple[float, float] = (2.0, 3.5),
    flight_time_range: tuple[float, float] = (0.45, 0.75),
    target_height_offset_range: tuple[float, float] = (0.0, 0.45),
    spawn_height_offset_range: tuple[float, float] = (-0.15, 0.35),
    horizontal_aim_noise: float = 0.15,
    angular_velocity_range: tuple[float, float] = (-6.0, 6.0),
    gravity_z: float = -9.81,
    parking_depth: float = -20.0,
) -> None:
    """Throw a randomly selected box toward each selected robot.

    The size, direction, target point, flight time, orientation and spin are
    sampled independently in every environment. A ballistic velocity aims the
    box approximately at the robot's pelvis/torso under gravity.
    """
    resolved_env_ids = _resolve_env_ids(env, env_ids)
    num_envs = len(resolved_env_ids)
    if num_envs == 0:
        return

    boxes: RigidObjectCollection = env.scene[collection_name]
    robot: Articulation = env.scene[robot_name]
    device = boxes.device

    # Retire old projectiles before launching the next one.
    park_throw_boxes(env, resolved_env_ids, collection_name, parking_depth)

    target = robot.data.root_pos_w[resolved_env_ids].clone()
    target[:, :2] += _uniform(
        (-horizontal_aim_noise, horizontal_aim_noise),
        (num_envs, 2),
        device,
    )
    target[:, 2] += _uniform(target_height_offset_range, (num_envs,), device)

    azimuth = _uniform(azimuth_range, (num_envs,), device)
    distance = _uniform(distance_range, (num_envs,), device)
    spawn = target.clone()
    spawn[:, 0] += distance * torch.cos(azimuth)
    spawn[:, 1] += distance * torch.sin(azimuth)
    spawn[:, 2] += _uniform(spawn_height_offset_range, (num_envs,), device)

    flight_time = _uniform(flight_time_range, (num_envs,), device)
    linear_velocity = (target - spawn) / flight_time.unsqueeze(-1)
    linear_velocity[:, 2] -= 0.5 * gravity_z * flight_time
    angular_velocity = _uniform(angular_velocity_range, (num_envs, 3), device)
    orientations = math_utils.random_orientation(num=num_envs, device=device)

    # The collection API selects a Cartesian product of env and object IDs.
    # Grouping by size gives pairwise random choices without looping over envs.
    selected_object_ids = torch.randint(boxes.num_objects, (num_envs,), device=device)
    env._active_throw_box_ids[resolved_env_ids] = selected_object_ids
    for object_id in range(boxes.num_objects):
        selection = selected_object_ids == object_id
        selected_env_ids = resolved_env_ids[selection]
        if len(selected_env_ids) == 0:
            continue

        object_state = torch.zeros((len(selected_env_ids), 1, 13), device=device)
        object_state[:, 0, :3] = spawn[selection]
        object_state[:, 0, 3:7] = orientations[selection]
        object_state[:, 0, 7:10] = linear_velocity[selection]
        object_state[:, 0, 10:13] = angular_velocity[selection]
        object_ids = torch.tensor([object_id], dtype=torch.long, device=device)
        boxes.write_object_state_to_sim(
            object_state,
            env_ids=selected_env_ids,
            object_ids=object_ids,
        )
