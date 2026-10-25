#!/usr/bin/env python3
"""
idle.py — "alive" breathing mode: subtle, organic micro-motions when not commanded.

The arm drifts slowly around a neutral pose — a gentle "breath" (shoulder rise/fall),
a faint sway, and an occasional soft glance to the side. Layered incommensurate
frequencies make the motion non-repeating, like a living thing, not a metronome.

Run (from the LeRobot repo folder):
  uv run python idle.py --port /dev/tty.usbmodemXXXX --id my_arm
Stop with Ctrl-C (it returns to neutral and releases torque).
"""

from __future__ import annotations

import argparse
import math
import random
import time

from emotions import ALL, JOINTS, clamp, denorm, make_robot, move_to, read_ranges

# Breath layer: (normalized amplitude, frequency Hz). Slow + subtle.
BREATH = {
    "shoulder_lift": (0.030, 0.20),   # main "breath" — ~5 s cycle
    "shoulder_pan": (0.020, 0.07),    # faint sway
    "elbow_flex": (0.020, 0.12),
    "wrist_flex": (0.015, 0.10),
    "wrist_roll": (0.020, 0.05),
    "gripper": (0.010, 0.08),
}
# Drift layer: even slower, incommensurate frequencies -> never exactly repeats.
DRIFT = {
    "shoulder_lift": (0.012, 0.043),
    "shoulder_pan": (0.010, 0.031),
    "elbow_flex": (0.010, 0.057),
    "wrist_roll": (0.008, 0.023),
}

GLANCE_EVERY_S = 18.0  # how often it softly looks around
GLANCE_AMP = 0.06      # glance size (shoulder_pan)
GLANCE_DUR_S = 2.0     # glance duration

# Fixed per-joint phases so the waves don't all crest together.
PHASES = {
    "shoulder_pan": 0.0, "shoulder_lift": 1.3, "elbow_flex": 2.7,
    "wrist_flex": 4.1, "wrist_roll": 5.2, "gripper": 0.9,
}

CENTER = 0.5  # normalized idle center


def idle_pose(t):
    """Normalized pose at time `t`. Pure + deterministic (unit-tested)."""
    pose = {j: CENTER for j in ALL}
    for j, (amp, freq) in BREATH.items():
        pose[j] += amp * math.sin(2 * math.pi * freq * t + PHASES[j])
    for j, (amp, freq) in DRIFT.items():
        pose[j] += amp * math.sin(2 * math.pi * freq * t + PHASES[j] * 2.1)
    cyc = t % GLANCE_EVERY_S
    if cyc < GLANCE_DUR_S:
        pose["shoulder_pan"] += GLANCE_AMP * math.sin(math.pi * cyc / GLANCE_DUR_S)
    return {j: clamp(pose[j], 0.0, 1.0) for j in ALL}


def main():
    ap = argparse.ArgumentParser(description="Alive idle breathing mode.")
    ap.add_argument("--port", required=True, help="e.g. /dev/tty.usbmodemXXXX")
    ap.add_argument("--id", dest="robot_id", default="my_arm")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    robot = make_robot(args.port, args.robot_id)
    ranges = read_ranges(robot)
    neutral = {j: CENTER for j in ALL}
    t_offset = random.uniform(0.0, 1000.0)  # each run starts at a different point

    try:
        move_to(robot, ranges, neutral, 1.5)
        print("Alive mode — subtle breathing. Ctrl-C to stop.")
        period = 1.0 / args.fps
        t0 = time.perf_counter()
        while True:
            t = time.perf_counter() - t0 + t_offset
            target = denorm(idle_pose(t), ranges)
            robot.send_action({f"{j}.pos": target[j] for j in ALL})
            time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        move_to(robot, ranges, neutral, 1.0)
        robot.disconnect()
        print("Stopped.")


if __name__ == "__main__":
    main()
