"""Static contract tests for the deployable 23-DoF fixed-hand task."""

from __future__ import annotations

import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = (
    PROJECT_ROOT
    / "assets"
    / "isaaclab"
    / "unitree_g1_23dof"
    / "g1_23dof_fixed_hand.urdf"
)


class FixedHandTrainingContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = ET.parse(URDF_PATH).getroot()

    def test_true_23_dof_without_finger_joints(self) -> None:
        actuated = [
            joint
            for joint in self.root.findall("joint")
            if joint.attrib["type"] in {"revolute", "continuous", "prismatic"}
        ]
        names = [joint.attrib["name"] for joint in actuated]
        self.assertEqual(len(names), 23)
        self.assertEqual(len(set(names)), 23)
        self.assertFalse(any("finger" in name for name in names))
        self.assertIn("left_wrist_roll_joint", names)
        self.assertIn("right_wrist_roll_joint", names)

    def test_rubber_hands_are_dark_rigid_links(self) -> None:
        for side in ("left", "right"):
            link_name = f"{side}_wrist_roll_rubber_hand"
            link = self.root.find(f"link[@name='{link_name}']")
            self.assertIsNotNone(link)
            self.assertEqual(link.find("visual/material").attrib["name"], "dark")
            mesh = link.find("collision/geometry/mesh")
            self.assertTrue(mesh.attrib["filename"].endswith(f"{link_name}.STL"))
            self.assertFalse(
                any(
                    joint.find("parent").attrib["link"] == link_name
                    for joint in self.root.findall("joint")
                )
            )

    def test_each_foot_has_one_full_sole_box(self) -> None:
        expected_size = "0.2031092182 0.0654692440 0.0185078794"
        for side in ("left", "right"):
            link = self.root.find(f"link[@name='{side}_ankle_roll_link']")
            collisions = link.findall("collision")
            self.assertEqual(len(collisions), 1)
            self.assertEqual(
                collisions[0].find("geometry/box").attrib["size"], expected_size
            )
            self.assertIsNone(collisions[0].find("geometry/sphere"))

    def test_task_registration_and_action_clip_are_declared(self) -> None:
        registration = (PROJECT_ROOT / "source/g1_locomotion/tasks/__init__.py").read_text()
        agent_cfg = (
            PROJECT_ROOT
            / "source/g1_locomotion/tasks/velocity/agents/rsl_rl_ppo_cfg.py"
        ).read_text()
        self.assertIn("Unitree-G1-FixedHand-WholeBody-Catch-Box-v0", registration)
        self.assertIn("G1FixedHandWholeBodyCatchBoxPPORunnerCfg", agent_cfg)
        self.assertIn("self.clip_actions = 1.0", agent_cfg)

    def test_post_catch_stability_contract_is_declared(self) -> None:
        env_cfg = (
            PROJECT_ROOT / "source/g1_locomotion/tasks/velocity/g1_env_cfg.py"
        ).read_text()
        mdp_init = (
            PROJECT_ROOT / "source/g1_locomotion/tasks/velocity/mdp/__init__.py"
        ).read_text()
        catch_mdp = (
            PROJECT_ROOT / "source/g1_locomotion/tasks/velocity/mdp/catch_box.py"
        ).read_text()
        self.assertIn("post_catch_stability = RewTerm", env_cfg)
        self.assertIn('"hold_time_s": 0.80', env_cfg)
        self.assertIn('".*_ankle_pitch_joint": 0.05', env_cfg)
        self.assertIn('".*_ankle_roll_joint": 0.05', env_cfg)
        self.assertIn("post_catch_lower_body_stability", mdp_init)
        self.assertIn("def post_catch_lower_body_stability", catch_mdp)


if __name__ == "__main__":
    unittest.main()
