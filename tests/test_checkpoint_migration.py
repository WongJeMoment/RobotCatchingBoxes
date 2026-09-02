"""Tests for the legacy-to-fixed-hand policy warm-start mapping."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from migrate_fixed_hand_checkpoint import (  # noqa: E402
    _observation_indices,
    _source_joint_indices,
    migrate_checkpoint,
)


class CheckpointMigrationTest(unittest.TestCase):
    def test_joint_and_observation_maps_have_unique_expected_widths(self) -> None:
        action_indices = _source_joint_indices()
        observation_indices = _observation_indices()
        self.assertEqual(len(action_indices), 23)
        self.assertEqual(len(set(action_indices)), 23)
        self.assertEqual(len(observation_indices), 108)
        self.assertEqual(len(set(observation_indices)), 108)
        self.assertEqual(observation_indices[:39], list(range(39)))

    def test_migrated_checkpoint_has_108_by_23_policy_contract(self) -> None:
        observation_columns = torch.arange(150, dtype=torch.float32).repeat(512, 1)
        action_rows = torch.arange(37, dtype=torch.float32).unsqueeze(1).repeat(1, 128)
        state = {
            "std": torch.full((37,), 0.55),
            "actor.0.weight": observation_columns,
            "actor.6.weight": action_rows,
            "actor.6.bias": torch.arange(37, dtype=torch.float32),
            "critic.0.weight": observation_columns.clone(),
        }
        for prefix in ("actor_obs_normalizer", "critic_obs_normalizer"):
            state[f"{prefix}._mean"] = torch.arange(150, dtype=torch.float32).unsqueeze(0)
            state[f"{prefix}._var"] = torch.ones((1, 150))
            state[f"{prefix}._std"] = torch.ones((1, 150))

        checkpoint = {
            "model_state_dict": state,
            "optimizer_state_dict": {
                "state": {0: {"step": torch.tensor(1.0)}},
                "param_groups": [{"params": list(range(17)), "lr": 5.0e-4}],
            },
            "iter": 2999,
            "infos": None,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "legacy.pt"
            destination = Path(temp_dir) / "fixed.pt"
            torch.save(checkpoint, source)
            migrate_checkpoint(source, destination)
            migrated = torch.load(destination, map_location="cpu", weights_only=False)

        mapped = migrated["model_state_dict"]
        self.assertEqual(tuple(mapped["actor.0.weight"].shape), (512, 108))
        self.assertEqual(tuple(mapped["critic.0.weight"].shape), (512, 108))
        self.assertEqual(tuple(mapped["actor.6.weight"].shape), (23, 128))
        self.assertEqual(tuple(mapped["std"].shape), (23,))
        torch.testing.assert_close(
            mapped["actor.0.weight"][0], torch.tensor(_observation_indices(), dtype=torch.float32)
        )
        torch.testing.assert_close(
            mapped["actor.6.weight"][:, 0],
            torch.tensor(_source_joint_indices(), dtype=torch.float32),
        )
        self.assertEqual(migrated["optimizer_state_dict"]["state"], {})
        self.assertEqual(migrated["iter"], 0)


if __name__ == "__main__":
    unittest.main()
