#!/usr/bin/env python3
"""Warm-start the 23-DoF fixed-hand task from a legacy 37-DoF checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


OLD_JOINT_NAMES = (
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

NEW_JOINT_NAMES = (
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
)

RENAMED_JOINTS = {
    "waist_yaw_joint": "torso_joint",
    "left_elbow_joint": "left_elbow_pitch_joint",
    "right_elbow_joint": "right_elbow_pitch_joint",
    "left_wrist_roll_joint": "left_elbow_roll_joint",
    "right_wrist_roll_joint": "right_elbow_roll_joint",
}


def _source_joint_indices() -> list[int]:
    return [
        OLD_JOINT_NAMES.index(RENAMED_JOINTS.get(name, name))
        for name in NEW_JOINT_NAMES
    ]


def _observation_indices() -> list[int]:
    joint_indices = _source_joint_indices()
    # The first 39 entries keep the same base, box, hand-kinematics and contact
    # layout.  Each of the three 37-D joint blocks is reduced to the mapped
    # 23 physical joints.
    return [
        *range(39),
        *(39 + index for index in joint_indices),
        *(76 + index for index in joint_indices),
        *(113 + index for index in joint_indices),
    ]


def migrate_checkpoint(source: Path, destination: Path) -> None:
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    if tuple(state["actor.0.weight"].shape) != (512, 150):
        raise ValueError("源 checkpoint 不是预期的 150 维观测/37 维动作模型")
    if tuple(state["actor.6.weight"].shape) != (37, 128):
        raise ValueError("源 checkpoint 的 actor 输出维度不是 37")

    observation_indices = torch.tensor(_observation_indices(), dtype=torch.long)
    action_indices = torch.tensor(_source_joint_indices(), dtype=torch.long)
    migrated = {name: value.clone() for name, value in state.items()}

    for prefix in ("actor", "critic"):
        migrated[f"{prefix}.0.weight"] = state[f"{prefix}.0.weight"].index_select(
            1, observation_indices
        )
    for prefix in ("actor_obs_normalizer", "critic_obs_normalizer"):
        for statistic in ("_mean", "_var", "_std"):
            key = f"{prefix}.{statistic}"
            migrated[key] = state[key].index_select(1, observation_indices)

    migrated["actor.6.weight"] = state["actor.6.weight"].index_select(0, action_indices)
    migrated["actor.6.bias"] = state["actor.6.bias"].index_select(0, action_indices)
    migrated["std"] = torch.full((23,), 0.35, dtype=state["std"].dtype)

    # Parameter count and ordering do not change, but Adam moment shapes do.
    # Retain optimizer hyperparameters and restart its moment estimates.
    optimizer = checkpoint["optimizer_state_dict"]
    migrated_optimizer = {
        "state": {},
        "param_groups": optimizer["param_groups"],
    }
    output = {
        "model_state_dict": migrated,
        "optimizer_state_dict": migrated_optimizer,
        "iter": 0,
        "infos": {
            "warm_start_source": str(source.resolve()),
            "mapped_actions": list(NEW_JOINT_NAMES),
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, destination)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="把旧 37-DoF 接箱 checkpoint 转成固定手任务的 23-DoF 热启动 checkpoint"
    )
    parser.add_argument("source", type=Path, help="旧 model_*.pt 路径")
    parser.add_argument("destination", type=Path, help="输出 model_*.pt 路径")
    args = parser.parse_args()
    migrate_checkpoint(args.source.expanduser(), args.destination.expanduser())
    print(f"已生成固定手热启动 checkpoint: {args.destination.expanduser().resolve()}")


if __name__ == "__main__":
    main()
