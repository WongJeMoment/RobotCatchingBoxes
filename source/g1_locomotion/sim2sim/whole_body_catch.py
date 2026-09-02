"""Run the whole-body box-catching policy in MuJoCo.

This module mirrors the policy-facing contract of
``Unitree-G1-WholeBody-Catch-Box-v0``.  It intentionally does not import Isaac
Sim or IsaacLab, so deployment can run in a small CPU-only Python environment.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

try:
    import mujoco
    import numpy as np
    import torch
    import torch.nn.functional as torch_functional
except ModuleNotFoundError as exc:  # pragma: no cover - exercised by the CLI environment
    raise SystemExit(
        f"sim2sim 缺少依赖 {exc.name!r}。请先执行: pip install 'mujoco>=3.2' numpy torch"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "assets" / "mujoco" / "unitree_g1" / "scene.xml"
DEFAULT_LOG_ROOT = PROJECT_ROOT / "logs" / "rsl_rl" / "unitree_g1_whole_body_catch_box"

PHYSICS_DT = 0.005
POLICY_DECIMATION = 4
POLICY_DT = PHYSICS_DT * POLICY_DECIMATION
OBSERVATION_DIM = 150
ACTION_DIM = 37
NORMALIZER_EPS = 1.0e-2

# IsaacLab resolves each regex in the configuration in order and preserves the
# USD asset order inside that regex.  This is therefore neither the MJCF joint
# order nor Unitree's hardware motor order.  Observations, action outputs and
# last_action all use this exact sequence.
POLICY_JOINT_NAMES: tuple[str, ...] = (
    # lower_body: all hips, then knees, then ankles
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
    # upper_body: torso, shoulders, elbows, then hand joints in USD order
    "torso_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "right_elbow_pitch_joint",
    "left_elbow_roll_joint",
    "right_elbow_roll_joint",
    "left_five_joint",
    "left_three_joint",
    "left_zero_joint",
    "right_five_joint",
    "right_three_joint",
    "right_zero_joint",
    "left_six_joint",
    "left_four_joint",
    "left_one_joint",
    "right_six_joint",
    "right_four_joint",
    "right_one_joint",
    "left_two_joint",
    "right_two_joint",
)

OBSERVATION_TERMS: tuple[tuple[str, int], ...] = (
    ("base_lin_vel", 3),
    ("base_ang_vel", 3),
    ("projected_gravity", 3),
    ("box_state", 16),
    ("palm_box_kinematics", 12),
    ("hand_contacts", 2),
    ("joint_pos", ACTION_DIM),
    ("joint_vel", ACTION_DIM),
    ("actions", ACTION_DIM),
)


def _build_observation_slices() -> dict[str, slice]:
    result: dict[str, slice] = {}
    offset = 0
    for name, width in OBSERVATION_TERMS:
        result[name] = slice(offset, offset + width)
        offset += width
    if offset != OBSERVATION_DIM:
        raise RuntimeError(f"内部观测契约错误: {offset} != {OBSERVATION_DIM}")
    return result


OBSERVATION_SLICES = _build_observation_slices()


@dataclass(frozen=True)
class BoxSpec:
    """Static projectile properties shared with the IsaacLab task."""

    name: str
    half_width: float
    parking_depth: float


BOX_SPECS: tuple[BoxSpec, ...] = (
    BoxSpec("small", 0.075, -20.0),
    BoxSpec("medium", 0.125, -22.0),
    BoxSpec("large", 0.175, -24.0),
)


def _joint_defaults() -> np.ndarray:
    values = {name: 0.0 for name in POLICY_JOINT_NAMES}
    for side in ("left", "right"):
        values[f"{side}_hip_pitch_joint"] = -0.20
        values[f"{side}_knee_joint"] = 0.42
        values[f"{side}_ankle_pitch_joint"] = -0.23
        values[f"{side}_elbow_pitch_joint"] = 0.87
        values[f"{side}_shoulder_pitch_joint"] = 0.35
    values["left_shoulder_roll_joint"] = 0.16
    values["right_shoulder_roll_joint"] = -0.16
    values["left_one_joint"] = 1.0
    values["right_one_joint"] = -1.0
    values["left_two_joint"] = 0.52
    values["right_two_joint"] = -0.52
    return np.asarray([values[name] for name in POLICY_JOINT_NAMES], dtype=np.float64)


def _action_scales() -> np.ndarray:
    values: list[float] = []
    for index, name in enumerate(POLICY_JOINT_NAMES):
        if index < 12:
            values.append(0.25)
        elif name == "torso_joint":
            values.append(0.45)
        elif "shoulder" in name or "elbow" in name:
            values.append(0.90)
        else:
            values.append(1.0)
    return np.asarray(values, dtype=np.float64)


def _pd_parameters() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stiffness: list[float] = []
    damping: list[float] = []
    effort: list[float] = []
    for name in POLICY_JOINT_NAMES:
        if "hip_pitch" in name or "knee" in name:
            stiffness.append(200.0)
            damping.append(5.0)
            effort.append(300.0)
        elif "hip_roll" in name or "hip_yaw" in name:
            stiffness.append(150.0)
            damping.append(5.0)
            effort.append(300.0)
        elif "ankle" in name:
            stiffness.append(20.0)
            damping.append(2.0)
            effort.append(20.0)
        elif name == "torso_joint":
            stiffness.append(200.0)
            damping.append(5.0)
            effort.append(300.0)
        else:
            stiffness.append(40.0)
            damping.append(10.0)
            effort.append(300.0)
    return (
        np.asarray(stiffness, dtype=np.float64),
        np.asarray(damping, dtype=np.float64),
        np.asarray(effort, dtype=np.float64),
    )


DEFAULT_JOINT_POS = _joint_defaults()
ACTION_SCALE = _action_scales()
PD_STIFFNESS, PD_DAMPING, EFFORT_LIMIT = _pd_parameters()


class TorchPolicy:
    """Inference wrapper for either an RSL-RL checkpoint or exported JIT policy."""

    def __init__(self, path: Path, device: str = "cpu") -> None:
        self.path = path.expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"策略文件不存在: {self.path}")

        self.device = torch.device(device)
        self.kind = ""
        self._jit: Any | None = None
        self._layers: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._mean: torch.Tensor | None = None
        self._std: torch.Tensor | None = None

        try:
            self._jit = torch.jit.load(str(self.path), map_location=self.device)
            self._jit.eval()
            self.kind = "TorchScript"
            state = self._jit.state_dict()
        except (RuntimeError, ValueError):
            self._jit = None
            payload = self._load_checkpoint(self.path)
            if not isinstance(payload, dict) or "model_state_dict" not in payload:
                raise ValueError(
                    f"{self.path} 既不是导出的 TorchScript，也不是 RSL-RL checkpoint"
                )
            state = payload["model_state_dict"]
            self.kind = "RSL-RL checkpoint"
            self._configure_raw_actor(state)

        input_dim, output_dim = self._infer_dimensions(state)
        if (input_dim, output_dim) != (OBSERVATION_DIM, ACTION_DIM):
            raise ValueError(
                "策略维度与 WholeBody-Catch-Box 不兼容: "
                f"obs={input_dim}, action={output_dim}; 期望 "
                f"obs={OBSERVATION_DIM}, action={ACTION_DIM}"
            )

    @staticmethod
    def _load_checkpoint(path: Path) -> Any:
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # PyTorch before weights_only was added.
            return torch.load(path, map_location="cpu")

    @staticmethod
    def _infer_dimensions(state: dict[str, torch.Tensor]) -> tuple[int, int]:
        weight_pattern = re.compile(r"^actor\.(\d+)\.weight$")
        layers = sorted(
            (int(match.group(1)), value)
            for key, value in state.items()
            if (match := weight_pattern.fullmatch(key)) is not None
        )
        if not layers:
            raise ValueError("策略中找不到 actor.*.weight")
        return int(layers[0][1].shape[1]), int(layers[-1][1].shape[0])

    def _configure_raw_actor(self, state: dict[str, torch.Tensor]) -> None:
        weight_pattern = re.compile(r"^actor\.(\d+)\.weight$")
        layer_indices = sorted(
            int(match.group(1))
            for key in state
            if (match := weight_pattern.fullmatch(key)) is not None
        )
        for index in layer_indices:
            weight_key = f"actor.{index}.weight"
            bias_key = f"actor.{index}.bias"
            if bias_key not in state:
                raise ValueError(f"checkpoint 缺少 {bias_key}")
            self._layers.append(
                (
                    state[weight_key].detach().to(device=self.device, dtype=torch.float32),
                    state[bias_key].detach().to(device=self.device, dtype=torch.float32),
                )
            )

        mean_key = "actor_obs_normalizer._mean"
        std_key = "actor_obs_normalizer._std"
        if mean_key not in state or std_key not in state:
            raise ValueError("checkpoint 缺少 actor observation normalizer")
        self._mean = state[mean_key].detach().to(device=self.device, dtype=torch.float32)
        self._std = state[std_key].detach().to(device=self.device, dtype=torch.float32)

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        if observation.shape != (OBSERVATION_DIM,):
            raise ValueError(f"策略输入形状错误: {observation.shape}")
        tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.inference_mode():
            if self._jit is not None:
                output = self._jit(tensor)
            else:
                assert self._mean is not None and self._std is not None
                output = (tensor - self._mean) / (self._std + NORMALIZER_EPS)
                for index, (weight, bias) in enumerate(self._layers):
                    output = torch_functional.linear(output, weight, bias)
                    if index + 1 != len(self._layers):
                        output = torch_functional.elu(output)

        if not isinstance(output, torch.Tensor) or tuple(output.shape) != (1, ACTION_DIM):
            shape = getattr(output, "shape", None)
            raise RuntimeError(f"策略输出形状错误: {shape}")
        action = output.squeeze(0).detach().cpu().numpy().astype(np.float64, copy=False)
        if not np.all(np.isfinite(action)):
            raise FloatingPointError("策略输出出现 NaN/Inf")
        return action.copy()


@dataclass(frozen=True)
class _BoxBinding:
    body_id: int
    geom_id: int
    qpos_adr: int
    dof_adr: int


class ModelBindings:
    """Strict name-based mapping between policy tensors and the MJCF model."""

    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model
        self.root_body_id = self._id(mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.root_joint_id = self._id(mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
        if model.jnt_type[self.root_joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("floating_base_joint 必须是 free joint")
        self.root_qpos_adr = int(model.jnt_qposadr[self.root_joint_id])
        self.root_dof_adr = int(model.jnt_dofadr[self.root_joint_id])

        self.joint_ids = np.asarray(
            [self._id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in POLICY_JOINT_NAMES],
            dtype=np.int32,
        )
        self.qpos_adrs = model.jnt_qposadr[self.joint_ids].astype(np.int32, copy=True)
        self.dof_adrs = model.jnt_dofadr[self.joint_ids].astype(np.int32, copy=True)
        self.actuator_ids = np.asarray(
            [self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in POLICY_JOINT_NAMES],
            dtype=np.int32,
        )

        if len(set(self.joint_ids.tolist())) != ACTION_DIM:
            raise ValueError("MJCF 中的策略关节映射不是一一对应")
        for joint_id, actuator_id, name in zip(
            self.joint_ids, self.actuator_ids, POLICY_JOINT_NAMES, strict=True
        ):
            if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError(f"策略关节不是 hinge: {name}")
            if int(model.actuator_trnid[actuator_id, 0]) != int(joint_id):
                raise ValueError(f"actuator 与 joint 不匹配: {name}")

        self.left_palm_site_id = self._id(mujoco.mjtObj.mjOBJ_SITE, "left_palm")
        self.right_palm_site_id = self._id(mujoco.mjtObj.mjOBJ_SITE, "right_palm")

        boxes: list[_BoxBinding] = []
        for spec in BOX_SPECS:
            joint_id = self._id(mujoco.mjtObj.mjOBJ_JOINT, f"box_{spec.name}_joint")
            boxes.append(
                _BoxBinding(
                    body_id=self._id(mujoco.mjtObj.mjOBJ_BODY, f"box_{spec.name}"),
                    geom_id=self._id(mujoco.mjtObj.mjOBJ_GEOM, f"box_{spec.name}_geom"),
                    qpos_adr=int(model.jnt_qposadr[joint_id]),
                    dof_adr=int(model.jnt_dofadr[joint_id]),
                )
            )
        self.boxes = tuple(boxes)
        self.hand_geom_keys = self._find_hand_geoms()

    def _id(self, object_type: mujoco.mjtObj, name: str) -> int:
        result = int(mujoco.mj_name2id(self.model, object_type, name))
        if result < 0:
            raise ValueError(f"MJCF 缺少 {object_type.name}: {name}")
        return result

    def _find_hand_geoms(self) -> dict[int, tuple[str, str]]:
        mapping: dict[int, tuple[str, str]] = {}
        finger_link = re.compile(r"^(left|right)_(zero|one|two|three|four|five|six)_link$")
        for geom_id in range(self.model.ngeom):
            geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if geom_name in ("left_palm_collision", "right_palm_collision"):
                side = geom_name.split("_", 1)[0]
                mapping[geom_id] = (side, f"{side}_palm_link")
                continue
            if int(self.model.geom_group[geom_id]) != 3:
                continue
            body_id = int(self.model.geom_bodyid[geom_id])
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            match = finger_link.fullmatch(body_name or "")
            if match is not None:
                mapping[geom_id] = (match.group(1), body_name)
        if not any(side == "left" for side, _ in mapping.values()):
            raise ValueError("MJCF 未找到左手碰撞体")
        if not any(side == "right" for side, _ in mapping.values()):
            raise ValueError("MJCF 未找到右手碰撞体")
        return mapping


def _quat_conjugate(quaternion: np.ndarray) -> np.ndarray:
    result = quaternion.copy()
    result[1:] *= -1.0
    return result


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.asarray(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dtype=np.float64,
    )


def _random_quaternion(rng: np.random.Generator) -> np.ndarray:
    """Sample a uniform SO(3) quaternion in MuJoCo/Isaac wxyz order."""

    u1, u2, u3 = rng.random(3)
    xyzw = np.asarray(
        (
            math.sqrt(1.0 - u1) * math.sin(2.0 * math.pi * u2),
            math.sqrt(1.0 - u1) * math.cos(2.0 * math.pi * u2),
            math.sqrt(u1) * math.sin(2.0 * math.pi * u3),
            math.sqrt(u1) * math.cos(2.0 * math.pi * u3),
        ),
        dtype=np.float64,
    )
    return xyzw[[3, 0, 1, 2]]


def _quat_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    matrix = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(matrix, quaternion)
    return matrix.reshape(3, 3)


class WholeBodyCatchSim:
    """One MuJoCo environment matching the trained IsaacLab policy contract."""

    def __init__(
        self,
        model_path: Path,
        *,
        seed: int = 42,
        episode_length_s: float = 8.0,
        hold_time_s: float = 1.0,
    ) -> None:
        self.model_path = model_path.expanduser().resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"MuJoCo 模型不存在: {self.model_path}")
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.bindings = ModelBindings(self.model)
        self.rng = np.random.default_rng(seed)
        self.episode_length_s = episode_length_s
        self.hold_time_s = hold_time_s

        if not math.isclose(float(self.model.opt.timestep), PHYSICS_DT, abs_tol=1.0e-12):
            raise ValueError(
                f"MuJoCo timestep={self.model.opt.timestep}，策略要求 {PHYSICS_DT}"
            )

        # The training asset explicitly uses the following actuator effort
        # limits.  Override the lower hardware-oriented limits in the public
        # MJCF; the PD output is clipped to the same values below.
        for actuator_id, limit in zip(
            self.bindings.actuator_ids, EFFORT_LIMIT, strict=True
        ):
            self.model.actuator_ctrlrange[actuator_id] = (-limit, limit)
            self.model.actuator_ctrllimited[actuator_id] = 1

        self.last_action = np.zeros(ACTION_DIM, dtype=np.float64)
        self.position_target = DEFAULT_JOINT_POS.copy()
        self.contact_history: deque[tuple[float, float]] = deque(maxlen=3)
        self.active_box_index = 0
        self.episode_index = -1
        self.episode_time = 0.0
        self.sustained_catch_steps = 0
        self.last_launch: dict[str, Any] = {}
        self.reset()

    @property
    def policy_dt(self) -> float:
        return POLICY_DT

    def reset(self) -> dict[str, Any]:
        mujoco.mj_resetData(self.model, self.data)
        bindings = self.bindings

        root_qpos = self.data.qpos[bindings.root_qpos_adr : bindings.root_qpos_adr + 7]
        root_qpos[:] = (0.0, 0.0, 0.74, 1.0, 0.0, 0.0, 0.0)
        self.data.qvel[bindings.root_dof_adr : bindings.root_dof_adr + 6] = 0.0
        self.data.qpos[bindings.qpos_adrs] = DEFAULT_JOINT_POS
        self.data.qvel[bindings.dof_adrs] = 0.0
        self.data.ctrl[:] = 0.0

        for spec, box in zip(BOX_SPECS, bindings.boxes, strict=True):
            self.data.qpos[box.qpos_adr : box.qpos_adr + 7] = (
                0.0,
                0.0,
                spec.parking_depth,
                1.0,
                0.0,
                0.0,
                0.0,
            )
            self.data.qvel[box.dof_adr : box.dof_adr + 6] = 0.0
            self.model.geom_contype[box.geom_id] = 0
            self.model.geom_conaffinity[box.geom_id] = 0
            self.model.body_gravcomp[box.body_id] = 1.0

        self.last_action.fill(0.0)
        self.position_target[:] = DEFAULT_JOINT_POS
        self.contact_history.clear()
        self.contact_history.extend(((0.0, 0.0),) * 3)
        self.episode_time = 0.0
        self.sustained_catch_steps = 0
        self.episode_index += 1

        mujoco.mj_forward(self.model, self.data)
        self.last_launch = self._launch_box()
        mujoco.mj_forward(self.model, self.data)
        return self.last_launch.copy()

    def _launch_box(self) -> dict[str, Any]:
        root_position = self.data.xpos[self.bindings.root_body_id].copy()
        target = root_position.copy()
        target[:2] += self.rng.uniform(-0.08, 0.08, size=2)
        target[2] += self.rng.uniform(0.22, 0.42)

        azimuth = self.rng.uniform(-0.22, 0.22)
        distance = self.rng.uniform(1.0, 1.45)
        spawn = target.copy()
        spawn[0] += distance * math.cos(azimuth)
        spawn[1] += distance * math.sin(azimuth)
        spawn[2] += self.rng.uniform(-0.05, 0.20)

        flight_time = self.rng.uniform(0.55, 0.75)
        linear_velocity = (target - spawn) / flight_time
        linear_velocity[2] -= 0.5 * -9.81 * flight_time
        angular_velocity = self.rng.uniform(-1.5, 1.5, size=3)
        orientation = _random_quaternion(self.rng)

        self.active_box_index = int(self.rng.integers(len(BOX_SPECS)))
        box = self.bindings.boxes[self.active_box_index]
        self.data.qpos[box.qpos_adr : box.qpos_adr + 7] = np.concatenate(
            (spawn, orientation)
        )
        # MuJoCo stores a free-joint angular qvel in the body's local frame,
        # while Isaac's rigid-object state API takes a world-frame velocity.
        angular_velocity_local = _quat_to_matrix(orientation).T @ angular_velocity
        self.data.qvel[box.dof_adr : box.dof_adr + 6] = np.concatenate(
            (linear_velocity, angular_velocity_local)
        )
        self.model.geom_contype[box.geom_id] = 2
        self.model.geom_conaffinity[box.geom_id] = 3
        self.model.body_gravcomp[box.body_id] = 0.0
        return {
            "size": BOX_SPECS[self.active_box_index].name,
            "spawn": spawn,
            "target": target,
            "flight_time": float(flight_time),
        }

    def _object_velocity(self, object_type: mujoco.mjtObj, object_id: int) -> tuple[np.ndarray, np.ndarray]:
        velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            object_type,
            object_id,
            velocity,
            0,
        )
        return velocity[3:].copy(), velocity[:3].copy()

    def _root_kinematics(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        root_id = self.bindings.root_body_id
        position = self.data.xpos[root_id].copy()
        quaternion = self.data.xquat[root_id].copy()
        rotation = self.data.xmat[root_id].reshape(3, 3).copy()
        linear_velocity, angular_velocity = self._object_velocity(
            mujoco.mjtObj.mjOBJ_BODY, root_id
        )
        return position, quaternion, rotation, linear_velocity, angular_velocity

    def _box_kinematics(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        box = self.bindings.boxes[self.active_box_index]
        position = self.data.xpos[box.body_id].copy()
        quaternion = self.data.xquat[box.body_id].copy()
        linear_velocity, angular_velocity = self._object_velocity(
            mujoco.mjtObj.mjOBJ_BODY, box.body_id
        )
        return position, quaternion, linear_velocity, angular_velocity

    def _palm_kinematics(self) -> tuple[np.ndarray, np.ndarray]:
        site_ids = (self.bindings.left_palm_site_id, self.bindings.right_palm_site_id)
        positions = np.stack([self.data.site_xpos[site_id].copy() for site_id in site_ids])
        velocities = np.stack(
            [
                self._object_velocity(mujoco.mjtObj.mjOBJ_SITE, site_id)[0]
                for site_id in site_ids
            ]
        )
        return positions, velocities

    def _recent_hand_forces(self) -> np.ndarray:
        return np.max(np.asarray(self.contact_history, dtype=np.float64), axis=0)

    def observation_parts(self) -> dict[str, np.ndarray]:
        root_pos, root_quat, root_rotation, root_lin_w, root_ang_w = self._root_kinematics()
        box_pos, box_quat, box_lin_w, box_ang_w = self._box_kinematics()
        palm_pos, palm_lin_w = self._palm_kinematics()
        world_to_root = root_rotation.T

        base_lin_vel = np.clip(world_to_root @ root_lin_w, -10.0, 10.0)
        base_ang_vel = np.clip(world_to_root @ root_ang_w, -10.0, 10.0)
        projected_gravity = world_to_root @ np.asarray((0.0, 0.0, -1.0))

        box_pos_b = world_to_root @ (box_pos - root_pos)
        box_quat_b = _quat_multiply(_quat_conjugate(root_quat), box_quat)
        if box_quat_b[0] < 0.0:
            box_quat_b *= -1.0
        box_lin_b = world_to_root @ (box_lin_w - root_lin_w)
        box_ang_b = world_to_root @ (box_ang_w - root_ang_w)
        size_one_hot = np.zeros(3, dtype=np.float64)
        size_one_hot[self.active_box_index] = 1.0
        box_state = np.clip(
            np.concatenate((box_pos_b, box_quat_b, box_lin_b, box_ang_b, size_one_hot)),
            -10.0,
            10.0,
        )

        palm_position_delta = np.stack(
            [world_to_root @ delta for delta in box_pos[None, :] - palm_pos]
        )
        palm_velocity_delta = np.stack(
            [world_to_root @ delta for delta in box_lin_w[None, :] - palm_lin_w]
        )
        palm_box_kinematics = np.clip(
            np.concatenate((palm_position_delta.reshape(-1), palm_velocity_delta.reshape(-1))),
            -10.0,
            10.0,
        )

        hand_contacts = np.clip(self._recent_hand_forces() / 25.0, 0.0, 2.0)
        joint_pos = np.clip(
            self.data.qpos[self.bindings.qpos_adrs] - DEFAULT_JOINT_POS,
            -3.0,
            3.0,
        )
        # IsaacLab applies clip before scale for an observation term.
        joint_vel = np.clip(
            self.data.qvel[self.bindings.dof_adrs], -10.0, 10.0
        ) * 0.1
        return {
            "base_lin_vel": base_lin_vel,
            "base_ang_vel": base_ang_vel,
            "projected_gravity": projected_gravity,
            "box_state": box_state,
            "palm_box_kinematics": palm_box_kinematics,
            "hand_contacts": hand_contacts,
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "actions": self.last_action.copy(),
        }

    def observation(self) -> np.ndarray:
        parts = self.observation_parts()
        observation = np.concatenate([parts[name] for name, _ in OBSERVATION_TERMS]).astype(
            np.float32, copy=False
        )
        if observation.shape != (OBSERVATION_DIM,):
            raise RuntimeError(f"观测拼接错误: {observation.shape}")
        if not np.all(np.isfinite(observation)):
            raise FloatingPointError("MuJoCo 观测出现 NaN/Inf")
        return observation

    def _apply_pd_control(self) -> None:
        joint_pos = self.data.qpos[self.bindings.qpos_adrs]
        joint_vel = self.data.qvel[self.bindings.dof_adrs]
        torque = PD_STIFFNESS * (self.position_target - joint_pos) - PD_DAMPING * joint_vel
        torque = np.clip(torque, -EFFORT_LIMIT, EFFORT_LIMIT)
        self.data.ctrl[self.bindings.actuator_ids] = torque

    def _sample_hand_forces(self) -> tuple[float, float]:
        # IsaacLab records one net world-frame force per hand link.  Recreate
        # that by summing contact vectors for each palm/finger link, then take
        # the largest link magnitude on each side.
        link_forces: dict[tuple[str, str], np.ndarray] = {}
        contact_force = np.zeros(6, dtype=np.float64)
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            key1 = self.bindings.hand_geom_keys.get(geom1)
            key2 = self.bindings.hand_geom_keys.get(geom2)
            if key1 is None and key2 is None:
                continue
            mujoco.mj_contactForce(self.model, self.data, contact_index, contact_force)
            # contact.frame stores the normal and tangent axes as rows.
            force_world = contact.frame.reshape(3, 3).T @ contact_force[:3]
            if key1 is not None:
                link_forces.setdefault(key1, np.zeros(3, dtype=np.float64))[:] += force_world
            if key2 is not None:
                link_forces.setdefault(key2, np.zeros(3, dtype=np.float64))[:] -= force_world

        maxima = {"left": 0.0, "right": 0.0}
        for (side, _), force in link_forces.items():
            maxima[side] = max(maxima[side], float(np.linalg.norm(force)))
        return maxima["left"], maxima["right"]

    def _catch_condition(self) -> bool:
        root_pos, _, root_rotation, _, _ = self._root_kinematics()
        box_pos, _, box_lin_w, _ = self._box_kinematics()
        palm_pos, palm_lin_w = self._palm_kinematics()
        lateral_axis_w = root_rotation[:, 1]
        clearance = BOX_SPECS[self.active_box_index].half_width + 0.025
        targets = np.stack(
            (box_pos + lateral_axis_w * clearance, box_pos - lateral_axis_w * clearance)
        )
        target_distances = np.linalg.norm(targets - palm_pos, axis=1)
        relative_speed = float(np.linalg.norm(palm_lin_w.mean(axis=0) - box_lin_w))
        left_force, right_force = self._recent_hand_forces()
        projected_gravity = root_rotation.T @ np.asarray((0.0, 0.0, -1.0))
        tilt = math.acos(float(np.clip(-projected_gravity[2], -1.0, 1.0)))
        return bool(
            left_force > 2.0
            and right_force > 2.0
            and float(np.max(target_distances)) < 0.22
            and relative_speed < 0.75
            and box_pos[2] > 0.55
            and tilt < 0.55
            and root_pos[2] > 0.60
        )

    def termination_reason(self) -> str | None:
        state = np.concatenate((self.data.qpos, self.data.qvel))
        if not np.all(np.isfinite(state)):
            return "non_finite_state"

        if self._catch_condition():
            self.sustained_catch_steps += 1
        else:
            self.sustained_catch_steps = 0
        required_steps = max(1, round(self.hold_time_s / POLICY_DT))
        if self.sustained_catch_steps >= required_steps:
            return "box_caught"

        root_pos, _, root_rotation, _, _ = self._root_kinematics()
        box_pos, _, _, _ = self._box_kinematics()
        if box_pos[2] < 0.32:
            return "box_dropped"
        if float(np.linalg.norm(box_pos - root_pos)) > 2.5:
            return "box_out_of_reach"
        projected_gravity = root_rotation.T @ np.asarray((0.0, 0.0, -1.0))
        tilt = math.acos(float(np.clip(-projected_gravity[2], -1.0, 1.0)))
        if tilt > 0.80:
            return "bad_orientation"
        if root_pos[2] < 0.52:
            return "root_too_low"
        if self.episode_time >= self.episode_length_s:
            return "time_out"
        return None

    def step(self, action: np.ndarray) -> str | None:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (ACTION_DIM,):
            raise ValueError(f"action 形状错误: {action.shape}")
        if not np.all(np.isfinite(action)):
            raise FloatingPointError("action 出现 NaN/Inf")
        self.last_action[:] = action
        self.position_target[:] = DEFAULT_JOINT_POS + ACTION_SCALE * action

        for _ in range(POLICY_DECIMATION):
            self._apply_pd_control()
            mujoco.mj_step(self.model, self.data)
            self.contact_history.append(self._sample_hand_forces())
        self.episode_time += POLICY_DT
        return self.termination_reason()


def _checkpoint_iteration(path: Path) -> int:
    match = re.fullmatch(r"model_(\d+)\.pt", path.name)
    return int(match.group(1)) if match is not None else -1


def find_latest_checkpoint(log_root: Path = DEFAULT_LOG_ROOT) -> Path:
    candidates = list(log_root.glob("*/model_*.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"{log_root} 下没有 checkpoint；请通过 --checkpoint 指定 model_*.pt 或 policy.pt"
        )
    return max(candidates, key=lambda path: (path.parent.name, _checkpoint_iteration(path)))


def _format_vector(vector: np.ndarray) -> str:
    return "[" + ", ".join(f"{value:.2f}" for value in vector) + "]"


def _print_contract(sim: WholeBodyCatchSim, policy: TorchPolicy | None) -> None:
    print("[INFO] sim2sim contract validated")
    print(f"[INFO] model: {sim.model_path}")
    if policy is not None:
        print(f"[INFO] policy: {policy.path} ({policy.kind}, device={policy.device})")
    else:
        print("[INFO] policy: zero-action controller")
    print(
        f"[INFO] observation={OBSERVATION_DIM}, action={ACTION_DIM}, "
        f"physics={1.0 / PHYSICS_DT:.0f} Hz, policy={1.0 / POLICY_DT:.0f} Hz"
    )
    print("[INFO] observation layout: " + ", ".join(
        f"{name}[{OBSERVATION_SLICES[name].start}:{OBSERVATION_SLICES[name].stop}]"
        for name, _ in OBSERVATION_TERMS
    ))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="在 MuJoCo 中运行 Unitree G1 全身接箱策略（IsaacLab -> MuJoCo sim2sim）"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="RSL-RL model_*.pt 或 play.py 导出的 policy.pt；默认自动选择最新 checkpoint",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="MuJoCo scene.xml")
    parser.add_argument("--device", default="cpu", help="PyTorch 推理设备，默认 cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--headless", action="store_true", help="不打开 MuJoCo viewer")
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="总仿真秒数；0 表示 viewer 关闭前持续运行（纯 headless 默认 8 秒）",
    )
    parser.add_argument("--max-episodes", type=int, default=0, help="达到 episode 数后退出；0 不限制")
    parser.add_argument(
        "--max-policy-steps", type=int, default=0, help="达到策略步数后退出；用于冒烟测试"
    )
    parser.add_argument("--episode-length", type=float, default=8.0)
    parser.add_argument("--hold-time", type=float, default=1.0)
    parser.add_argument("--log-interval", type=float, default=1.0)
    parser.add_argument(
        "--real-time",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="按真实时间节流；viewer 默认开启，headless 默认关闭",
    )
    parser.add_argument("--zero-action", action="store_true", help="不用策略，只验证模型和 PD 站立")
    parser.add_argument("--dry-run", action="store_true", help="只加载并检查策略/模型契约")
    return parser


def run(args: argparse.Namespace) -> int:
    if args.duration < 0.0:
        raise ValueError("--duration 不能为负数")
    if args.max_episodes < 0 or args.max_policy_steps < 0:
        raise ValueError("--max-episodes/--max-policy-steps 不能为负数")
    if args.episode_length <= 0.0 or args.hold_time <= 0.0:
        raise ValueError("--episode-length/--hold-time 必须为正数")

    policy: TorchPolicy | None
    if args.zero_action:
        policy = None
    else:
        policy_path = args.checkpoint if args.checkpoint is not None else find_latest_checkpoint()
        policy = TorchPolicy(policy_path, device=args.device)

    sim = WholeBodyCatchSim(
        args.model,
        seed=args.seed,
        episode_length_s=args.episode_length,
        hold_time_s=args.hold_time,
    )
    _print_contract(sim, policy)
    if args.dry_run:
        observation = sim.observation()
        action = np.zeros(ACTION_DIM) if policy is None else policy(observation)
        print(
            f"[INFO] dry-run OK: obs_norm={np.linalg.norm(observation):.3f}, "
            f"action_norm={np.linalg.norm(action):.3f}"
        )
        return 0

    duration = float(args.duration)
    if args.headless and duration == 0.0 and args.max_episodes == 0 and args.max_policy_steps == 0:
        duration = 8.0
    real_time = (not args.headless) if args.real_time is None else bool(args.real_time)

    viewer: Any | None = None
    reset_requested = {"value": False}
    if not args.headless:
        try:
            import mujoco.viewer

            def on_key(keycode: int) -> None:
                if keycode in (ord("R"), ord("r")):
                    reset_requested["value"] = True

            viewer = mujoco.viewer.launch_passive(
                sim.model,
                sim.data,
                key_callback=on_key,
                show_left_ui=False,
                show_right_ui=False,
            )
            viewer.cam.lookat[:] = (0.2, 0.0, 0.9)
            viewer.cam.distance = 3.0
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -18.0
        except Exception as exc:
            raise RuntimeError(
                f"MuJoCo viewer 启动失败（无显示器时请加 --headless）: {exc}"
            ) from exc

    print(
        f"[INFO] episode=0 launched {sim.last_launch['size']} box, "
        f"spawn={_format_vector(sim.last_launch['spawn'])}, 按 R 可重置"
    )
    total_steps = 0
    completed_episodes = 0
    total_sim_time = 0.0
    next_log_time = 0.0
    try:
        while True:
            if viewer is not None and not viewer.is_running():
                break
            if duration > 0.0 and total_sim_time >= duration:
                break
            if args.max_policy_steps > 0 and total_steps >= args.max_policy_steps:
                break
            if args.max_episodes > 0 and completed_episodes >= args.max_episodes:
                break

            loop_start = time.perf_counter()
            observation = sim.observation()
            action = np.zeros(ACTION_DIM, dtype=np.float64) if policy is None else policy(observation)
            reason = sim.step(action)
            total_steps += 1
            total_sim_time += POLICY_DT

            if total_sim_time + 1.0e-12 >= next_log_time:
                box_pos, _, _, _ = sim._box_kinematics()
                root_pos, _, _, _, _ = sim._root_kinematics()
                hand_forces = sim._recent_hand_forces()
                print(
                    f"[INFO] t={total_sim_time:6.2f}s episode={sim.episode_index} "
                    f"root_z={root_pos[2]:.3f} box={_format_vector(box_pos)} "
                    f"hand_N={_format_vector(hand_forces)}"
                )
                next_log_time += max(float(args.log_interval), POLICY_DT)

            if reset_requested["value"]:
                reason = "manual_reset"
                reset_requested["value"] = False
            if reason is not None:
                completed_episodes += 1
                print(
                    f"[INFO] episode={sim.episode_index} finished: {reason} "
                    f"at {sim.episode_time:.2f}s"
                )
                if args.max_episodes > 0 and completed_episodes >= args.max_episodes:
                    break
                launch = sim.reset()
                print(
                    f"[INFO] episode={sim.episode_index} launched {launch['size']} box, "
                    f"spawn={_format_vector(launch['spawn'])}"
                )

            if viewer is not None:
                root_position = sim.data.xpos[sim.bindings.root_body_id]
                viewer.cam.lookat[:] = (root_position[0] + 0.2, root_position[1], 0.9)
                viewer.sync()
            if real_time:
                sleep_time = POLICY_DT - (time.perf_counter() - loop_start)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)
    finally:
        if viewer is not None:
            viewer.close()

    print(
        f"[INFO] sim2sim finished: {total_steps} policy steps, "
        f"{completed_episodes} completed episodes, {total_sim_time:.2f}s simulated"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (FileNotFoundError, ValueError, RuntimeError, FloatingPointError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
