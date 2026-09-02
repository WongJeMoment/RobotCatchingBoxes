"""Contract and smoke tests for the MuJoCo sim2sim runner."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np
import torch

from g1_locomotion.sim2sim.whole_body_catch import (
    ACTION_DIM,
    DEFAULT_JOINT_POS,
    DEFAULT_LOG_ROOT,
    DEFAULT_MODEL_PATH,
    EFFORT_LIMIT,
    OBSERVATION_DIM,
    OBSERVATION_SLICES,
    POLICY_JOINT_NAMES,
    TorchPolicy,
    WholeBodyCatchSim,
    find_latest_checkpoint,
)


class Sim2SimContractTest(unittest.TestCase):
    def test_observation_layout_and_joint_order(self) -> None:
        self.assertEqual(len(POLICY_JOINT_NAMES), ACTION_DIM)
        self.assertEqual(len(set(POLICY_JOINT_NAMES)), ACTION_DIM)
        expected_slices = {
            "base_lin_vel": slice(0, 3),
            "base_ang_vel": slice(3, 6),
            "projected_gravity": slice(6, 9),
            "box_state": slice(9, 25),
            "palm_box_kinematics": slice(25, 37),
            "hand_contacts": slice(37, 39),
            "joint_pos": slice(39, 76),
            "joint_vel": slice(76, 113),
            "actions": slice(113, 150),
        }
        self.assertEqual(OBSERVATION_SLICES, expected_slices)
        self.assertEqual(OBSERVATION_SLICES["actions"].stop, OBSERVATION_DIM)
        self.assertEqual(
            POLICY_JOINT_NAMES[:12],
            (
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
            ),
        )

    def test_model_observation_and_actuator_mapping(self) -> None:
        sim = WholeBodyCatchSim(DEFAULT_MODEL_PATH, seed=7)
        observation = sim.observation()
        self.assertEqual(observation.shape, (OBSERVATION_DIM,))
        self.assertTrue(np.all(np.isfinite(observation)))
        np.testing.assert_allclose(
            sim.data.qpos[sim.bindings.qpos_adrs], DEFAULT_JOINT_POS, atol=1.0e-12
        )
        np.testing.assert_allclose(
            observation[OBSERVATION_SLICES["projected_gravity"]],
            (0.0, 0.0, -1.0),
            atol=1.0e-7,
        )

        actuator_limits = sim.model.actuator_ctrlrange[sim.bindings.actuator_ids, 1]
        np.testing.assert_allclose(actuator_limits, EFFORT_LIMIT)
        hand_links = set(sim.bindings.hand_geom_keys.values())
        self.assertIn(("left", "left_palm_link"), hand_links)
        self.assertIn(("right", "right_palm_link"), hand_links)
        active_boxes = [
            index
            for index, box in enumerate(sim.bindings.boxes)
            if sim.model.geom_contype[box.geom_id] != 0
        ]
        self.assertEqual(active_boxes, [sim.active_box_index])

    def test_joint_velocity_is_clipped_before_scaling(self) -> None:
        sim = WholeBodyCatchSim(DEFAULT_MODEL_PATH, seed=7)
        sim.data.qvel[sim.bindings.dof_adrs[0]] = 100.0
        sim.data.qvel[sim.bindings.dof_adrs[1]] = -100.0
        joint_velocity = sim.observation()[OBSERVATION_SLICES["joint_vel"]]
        self.assertAlmostEqual(float(joint_velocity[0]), 1.0)
        self.assertAlmostEqual(float(joint_velocity[1]), -1.0)

    def test_zero_action_headless_smoke(self) -> None:
        sim = WholeBodyCatchSim(DEFAULT_MODEL_PATH, seed=11)
        for _ in range(10):
            sim.step(np.zeros(ACTION_DIM, dtype=np.float64))
            self.assertTrue(np.all(np.isfinite(sim.observation())))

    def test_palm_contact_is_included_in_hand_force(self) -> None:
        sim = WholeBodyCatchSim(DEFAULT_MODEL_PATH, seed=11)
        box = sim.bindings.boxes[sim.active_box_index]
        left_palm = sim.data.site_xpos[sim.bindings.left_palm_site_id].copy()
        sim.data.qpos[box.qpos_adr : box.qpos_adr + 7] = np.concatenate(
            (left_palm, np.asarray((1.0, 0.0, 0.0, 0.0)))
        )
        sim.data.qvel[box.dof_adr : box.dof_adr + 6] = 0.0
        mujoco.mj_forward(sim.model, sim.data)
        sim._apply_pd_control()
        mujoco.mj_step(sim.model, sim.data)
        left_force, right_force = sim._sample_hand_forces()
        self.assertGreater(left_force, 1.0)
        self.assertGreaterEqual(right_force, 0.0)

    def test_checkpoint_dimension_guard(self) -> None:
        state = {
            "actor.0.weight": torch.zeros((37, 149)),
            "actor.0.bias": torch.zeros(37),
            "actor_obs_normalizer._mean": torch.zeros((1, 149)),
            "actor_obs_normalizer._std": torch.ones((1, 149)),
        }
        with tempfile.TemporaryDirectory() as temp_directory:
            path = Path(temp_directory) / "bad_model.pt"
            torch.save({"model_state_dict": state}, path)
            with self.assertRaisesRegex(ValueError, "策略维度"):
                TorchPolicy(path)

    def test_local_checkpoint_matches_exported_jit(self) -> None:
        try:
            checkpoint = find_latest_checkpoint(DEFAULT_LOG_ROOT)
        except FileNotFoundError:
            self.skipTest("local training logs are not available")
        exported = checkpoint.parent / "exported" / "policy.pt"
        if not exported.is_file():
            self.skipTest("matching exported policy is not available")

        sim = WholeBodyCatchSim(DEFAULT_MODEL_PATH, seed=42)
        observation = sim.observation()
        checkpoint_action = TorchPolicy(checkpoint)(observation)
        exported_action = TorchPolicy(exported)(observation)
        np.testing.assert_allclose(checkpoint_action, exported_action, rtol=1.0e-6, atol=1.0e-6)


if __name__ == "__main__":
    unittest.main()
