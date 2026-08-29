"""Register the project-specific G1 Gymnasium environments.

Only Gymnasium is imported here. Isaac Lab configuration modules are deliberately
loaded lazily by their string entry points, after Isaac Sim has been launched.
"""
#
#
# 导入 Gymnasium
import gymnasium as gym

# 注册任务
def _register(task_id: str, env_cfg: str, agent_cfg: str) -> None:
    """Register a task once, making repeated imports harmless."""
    if task_id in gym.registry:
        return
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "rsl_rl_cfg_entry_point": agent_cfg,
        },
    )


_ENV_MODULE = "g1_locomotion.tasks.velocity.g1_env_cfg"
_AGENT_MODULE = "g1_locomotion.tasks.velocity.agents.rsl_rl_ppo_cfg"

_register(
    "Unitree-G1-Velocity-Flat-v0",
    f"{_ENV_MODULE}:G1FlatTrainEnvCfg",
    f"{_AGENT_MODULE}:G1FlatPPORunnerCfg",
)
_register(
    "Unitree-G1-Velocity-Flat-Play-v0",
    f"{_ENV_MODULE}:G1FlatPlayEnvCfg",
    f"{_AGENT_MODULE}:G1FlatPPORunnerCfg",
)
_register(
    "Unitree-G1-Velocity-Rough-v0",
    f"{_ENV_MODULE}:G1RoughTrainEnvCfg",
    f"{_AGENT_MODULE}:G1RoughPPORunnerCfg",
)
_register(
    "Unitree-G1-Velocity-Rough-Play-v0",
    f"{_ENV_MODULE}:G1RoughPlayEnvCfg",
    f"{_AGENT_MODULE}:G1RoughPPORunnerCfg",
)
_register(
    "Unitree-G1-Velocity-Flat-BoxThrow-v0",
    f"{_ENV_MODULE}:G1FlatBoxThrowEnvCfg",
    f"{_AGENT_MODULE}:G1FlatBoxThrowPPORunnerCfg",
)
_register(
    "Unitree-G1-Velocity-Flat-BoxThrow-Play-v0",
    f"{_ENV_MODULE}:G1FlatBoxThrowPlayEnvCfg",
    f"{_AGENT_MODULE}:G1FlatBoxThrowPPORunnerCfg",
)
_register(
    "Unitree-G1-Velocity-Rough-BoxThrow-v0",
    f"{_ENV_MODULE}:G1RoughBoxThrowEnvCfg",
    f"{_AGENT_MODULE}:G1RoughBoxThrowPPORunnerCfg",
)
_register(
    "Unitree-G1-Velocity-Rough-BoxThrow-Play-v0",
    f"{_ENV_MODULE}:G1RoughBoxThrowPlayEnvCfg",
    f"{_AGENT_MODULE}:G1RoughBoxThrowPPORunnerCfg",
)
_register(
    "Unitree-G1-Catch-Box-v0",
    f"{_ENV_MODULE}:G1CatchBoxEnvCfg",
    f"{_AGENT_MODULE}:G1CatchBoxPPORunnerCfg",
)
_register(
    "Unitree-G1-Catch-Box-Play-v0",
    f"{_ENV_MODULE}:G1CatchBoxPlayEnvCfg",
    f"{_AGENT_MODULE}:G1CatchBoxPPORunnerCfg",
)
_register(
    "Unitree-G1-WholeBody-Catch-Box-v0",
    f"{_ENV_MODULE}:G1WholeBodyCatchBoxEnvCfg",
    f"{_AGENT_MODULE}:G1WholeBodyCatchBoxPPORunnerCfg",
)
_register(
    "Unitree-G1-WholeBody-Catch-Box-Play-v0",
    f"{_ENV_MODULE}:G1WholeBodyCatchBoxPlayEnvCfg",
    f"{_AGENT_MODULE}:G1WholeBodyCatchBoxPPORunnerCfg",
)
