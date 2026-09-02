# Asset provenance

The Unitree G1 MJCF and meshes in this directory come from the
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/cddb24f0dba18db8a330944edd0621adb91bdad0/unitree_g1)
at commit `cddb24f0dba18db8a330944edd0621adb91bdad0` (2024-07-19).

That historical revision is used deliberately: its 37 actuated joints and
link names match the Isaac Sim 5.1 `G1_CFG` asset used by this repository,
including the single `torso_joint`, two elbow joints per arm, and seven hand
joints per side.  The newer 29-DOF G1 body model is not policy-compatible.

Local changes are limited to:

- collision masks that disable robot self-collisions, as in the IsaacLab task;
- palm sites placed at the fixed `left_palm_link` and `right_palm_link` frames;
- a 5 ms simulation timestep, ground plane, and the three task projectiles.

The upstream BSD-3-Clause terms are retained in [LICENSE](LICENSE).
