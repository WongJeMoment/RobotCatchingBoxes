"""Run an upstream Isaac Lab workflow after registering this project.

Keeping the RSL-RL runner in Isaac Lab avoids maintaining a fork of a script that
changes between Isaac Lab releases.  This bootstrap only adds the local tasks to
Gymnasium and then executes the matching upstream script unchanged.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "source"


def _pop_option(name: str) -> str | None:
    """Remove one wrapper-only option from sys.argv and return its value."""
    prefix = f"{name}="
    for index, argument in enumerate(sys.argv[1:], start=1):
        if argument.startswith(prefix):
            sys.argv.pop(index)
            return argument[len(prefix) :]
        if argument == name:
            if index + 1 >= len(sys.argv):
                raise SystemExit(f"{name} 需要一个目录参数")
            sys.argv.pop(index)
            return sys.argv.pop(index)
    return None


def _find_isaaclab_root(explicit_path: str | None) -> Path:
    configured = explicit_path or os.environ.get("ISAACLAB_PATH")
    if not configured:
        raise SystemExit(
            "找不到 Isaac Lab。请设置 ISAACLAB_PATH，或添加参数 "
            "--isaaclab-path /path/to/IsaacLab"
        )

    root = Path(configured).expanduser().resolve()
    if not (root / "isaaclab.sh").is_file() and not (root / "isaaclab.bat").is_file():
        raise SystemExit(f"不是有效的 Isaac Lab 根目录: {root}")
    return root


def _has_option(name: str) -> bool:
    return any(argument == name or argument.startswith(f"{name}=") for argument in sys.argv[1:])


def run_upstream(script_name: str, default_task: str) -> None:
    """Register local tasks and execute Isaac Lab's RSL-RL workflow."""
    # Find the Isaac Lab root and the upstream script to run.
    isaaclab_root = _find_isaaclab_root(_pop_option("--isaaclab-path"))
    # Add the Isaac Lab RSL-RL runner to sys.path so that it can be imported.
    runner_dir = isaaclab_root / "scripts" / "reinforcement_learning" / "rsl_rl"
    # Add the upstream script to sys.path so that it can be imported.
    runner = runner_dir / script_name
    if not runner.is_file():
        raise SystemExit(
            f"当前 Isaac Lab 中不存在兼容脚本: {runner}\n"
            "本项目面向 Isaac Lab 2.3.x；请检查版本或 README 的安装说明。"
        )

    sys.path.insert(0, str(SOURCE_ROOT))
    sys.path.insert(0, str(runner_dir))  # upstream script imports its cli_args.py

    # Registration is intentionally performed before the upstream Hydra decorator
    # resolves the task ID.
    import g1_locomotion.tasks  # noqa: F401, PLC0415

    if not _has_option("--task"):
        sys.argv.extend(["--task", default_task])

    runpy.run_path(str(runner), run_name="__main__")

