#!/usr/bin/env python3
"""Standalone entry point for IsaacLab-to-MuJoCo policy transfer."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "source"))

from g1_locomotion.sim2sim.whole_body_catch import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
