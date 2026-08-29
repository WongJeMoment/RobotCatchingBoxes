"""RSL-RL PPO configurations for the custom G1 tasks."""

from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import (
    G1FlatPPORunnerCfg as IsaacLabG1FlatPPORunnerCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import (
    G1RoughPPORunnerCfg as IsaacLabG1RoughPPORunnerCfg,
)


@configclass
class G1FlatPPORunnerCfg(IsaacLabG1FlatPPORunnerCfg):
    """PPO defaults for flat-ground G1 locomotion."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "unitree_g1_flat"
        self.run_name = "ppo"
        self.max_iterations = 1500
        self.save_interval = 50


@configclass
class G1RoughPPORunnerCfg(IsaacLabG1RoughPPORunnerCfg):
    """PPO defaults for rough-terrain G1 locomotion."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "unitree_g1_rough"
        self.run_name = "ppo"
        self.max_iterations = 3000
        self.save_interval = 50


@configclass
class G1FlatBoxThrowPPORunnerCfg(G1FlatPPORunnerCfg):
    """PPO defaults for flat-ground impact recovery."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "unitree_g1_flat_box_throw"
        self.max_iterations = 2000


@configclass
class G1RoughBoxThrowPPORunnerCfg(G1RoughPPORunnerCfg):
    """PPO defaults for rough-terrain impact recovery."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "unitree_g1_rough_box_throw"
        self.max_iterations = 3500


@configclass
class G1CatchBoxPPORunnerCfg(G1FlatPPORunnerCfg):
    """PPO defaults for fixed-base two-handed projectile catching."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "unitree_g1_catch_box"
        self.run_name = "ppo"
        self.num_steps_per_env = 48
        self.max_iterations = 3000
        self.save_interval = 50
        self.policy.init_noise_std = 0.8
        self.policy.actor_obs_normalization = True
        self.policy.critic_obs_normalization = True
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]
        self.algorithm.entropy_coef = 0.006


@configclass
class G1WholeBodyCatchBoxPPORunnerCfg(G1CatchBoxPPORunnerCfg):
    """PPO defaults for the coupled standing and catching demonstrator."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "unitree_g1_whole_body_catch_box"
        self.num_steps_per_env = 64
        self.max_iterations = 5000
        self.policy.init_noise_std = 0.55
        self.algorithm.entropy_coef = 0.004
