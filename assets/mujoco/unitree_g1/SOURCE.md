# Asset provenance

The Unitree G1 MJCF and meshes in this directory come from the
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/cddb24f0dba18db8a330944edd0621adb91bdad0/unitree_g1)
at commit `cddb24f0dba18db8a330944edd0621adb91bdad0` (2024-07-19).

The fixed `left_wrist_roll_rubber_hand.STL` and
`right_wrist_roll_rubber_hand.STL` meshes come from the official
[Unitree Robotics `unitree_ros`](https://github.com/unitreerobotics/unitree_ros/tree/7d6075f7f58588b189b940130e3edab3c839b2df/robots/g1_description)
repository at commit `7d6075f7f58588b189b940130e3edab3c839b2df`.  Its BSD-3-Clause
terms are retained in [UNITREE_LICENSE](UNITREE_LICENSE).

That historical revision is used deliberately: its 37 actuated joints and
link names match the Isaac Sim 5.1 `G1_CFG` asset used by this repository,
including the single `torso_joint`, two elbow joints per arm, and seven hand
joints per side.  The newer 29-DOF G1 body model is not policy-compatible.

Local changes are limited to:

- collision masks that disable robot self-collisions, as in the IsaacLab task;
- cuboid foot colliders matching the G1 USD dimensions and local transforms;
- official fixed rubber-hand visuals and collision hulls, with the old finger
  bodies hidden and collision-disabled for checkpoint compatibility;
- palm sites retained at the policy-compatible fixed palm frames;
- a 5 ms simulation timestep, ground plane, and the three task projectiles.

The upstream BSD-3-Clause terms are retained in [LICENSE](LICENSE).
