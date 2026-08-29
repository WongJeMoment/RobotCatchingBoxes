#!/usr/bin/env python3
"""Evaluate and export a trained Unitree G1 policy."""

from _bootstrap import run_upstream


if __name__ == "__main__":
    run_upstream("play.py", "Unitree-G1-Velocity-Flat-Play-v0")

