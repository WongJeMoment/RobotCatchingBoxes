# Unitree G1 23-DoF fixed-hand asset

The URDF and meshes in this directory come from Unitree Robotics' official
`unitree_ros` repository, revision
`7d6075f7f58588b189b940130e3edab3c839b2df`:

- `robots/g1_description/g1_23dof.urdf`
- the mesh files referenced by that URDF

Upstream: <https://github.com/unitreerobotics/unitree_ros>

Local changes:

- renamed the local URDF to `g1_23dof_fixed_hand.urdf`;
- changed both fixed rubber-hand visual materials to the dark material;
- replaced each four-point 5 mm foot collision set with one measured full-sole
  box collider to make PhysX stance and gait contacts stable.

The upstream BSD-3-Clause license is retained in `LICENSE`.
