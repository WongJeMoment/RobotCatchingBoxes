#!/usr/bin/env python3
"""Train Unitree G1 with PPO in Isaac Sim/Isaac Lab."""

from _bootstrap import run_upstream


if __name__ == "__main__":
    run_upstream("train.py", "Unitree-G1-Velocity-Flat-v0")

