#!/usr/bin/env python3
"""
emotions.py — expressive body-language gestures for the SO-100 / SO-101 arm.

Each emotion is a parametric motion: joints oscillate around a rest pose, and the
emotion comes from *timing + amplitude* (fast & sharp = angry, slow & tiny = sad,
etc.) — the same principle animators use. No camera, no ML.

All motions live inside the arm's calibrated range, so nothing can over-drive.

Usage (from the LeRobot repo folder):
  uv run python emotions.py wave --port /dev/tty.usbmodemXXXX
  uv run python emotions.py --list
  uv run python emotions.py demo   --port /dev/tty.usbmodemXXXX   # cycle all
  uv run python emotions.py random --port /dev/tty.usbmodemXXXX   # pick one
"""

from __future__ import annotations

import argparse
import math
import random
import time

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
ALL = JOINTS + ["gripper"]
RES = 4096  # STS3215 encoder steps per turn
SAFE_MARGIN_DEG = 4.0

# Normalized rest pose (0.5 = middle of each joint's calibrated range).
REST = {j: 0.5 for j in ALL}

# Each gesture: dur (s), base (static offsets from rest), osc (joint -> amp, freq Hz, phase).
GESTURES = {
    "wave": dict(
        dur=3.0,
        base={"shoulder_lift": -0.10, "elbow_flex": -0.08, "gripper": 0.25},
        osc={"shoulder_pan": (0.35, 1.6, 0.0)},
    ),
    "nod": dict(dur=2.5, base={}, osc={"shoulder_lift": (0.16, 1.0, 0.0)}),
    "excited": dict(
        dur=3.0,
        base={},
        osc={
            "shoulder_pan": (0.12, 4.5, 0.0),
            "shoulder_lift": (0.10, 5.5, 1.0),
            "gripper": (0.25, 4.0, 0.0),
        },
    ),
    "happy": dict(
        dur=3.5,
        base={"shoulder_lift": -0.08},
        osc={"shoulder_pan": (0.40, 0.8, 0.0), "shoulder_lift": (0.12, 1.4, 0.6)},
    ),
    "sad": dict(
        dur=4.0,
        base={},
        osc={"shoulder_pan": (0.05, 0.3, 0.0), "shoulder_lift": (0.03, 0.45, 2.0)},
    ),
    "angry": dict(
        dur=2.5,
        base={"shoulder_lift": -0.05},
        osc={"shoulder_pan": (0.45, 3.0, 0.0), "elbow_flex": (0.30, 3.2, 1.5)},
    ),
    "thinking": dict(
        dur=4.5,
        base={"shoulder_pan": 0.12},
        osc={"shoulder_pan": (0.06, 0.35, 0.0), "wrist_roll": (0.10, 0.5, 1.0)},
    ),
    "shrug": dict(
        dur=1.8,
        base={},
        osc={"shoulder_lift": (0.22, 0.9, 0.0), "wrist_roll": (0.15, 0.9, 0.0)},
    ),
    "no": dict(dur=2.5, base={}, osc={"shoulder_pan": (0.25, 2.0, 0.0)}),
}


# ------------------------------ pure math ------------------------------

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def lerp(a, b, t):
    return a + (b - a) * t


def envelope(t, dur, fade=0.35):
    """0 at the start and end, 1 in the middle — avoids sudden jumps."""
    return min(clamp(t / fade, 0.0, 1.0), clamp((dur - t) / fade, 0.0, 1.0))


def gesture_pose(name, t):
    """Normalized joint pose for `name` at time `t` (s). Every value in [0, 1]."""
    g = GESTURES[name]
    e = envelope(t, g["dur"])
    pose = dict(REST)
    for j, b in g.get("base", {}).items():
        pose[j] += b
    for j, (amp, freq, phase) in g.get("osc", {}).items():
        pose[j] += e * amp * math.sin(2 * math.pi * freq * t + phase)
    return {j: clamp(pose[j], 0.0, 1.0) for j in ALL}


# ------------------------------ robot helpers ------------------------------

def make_robot(port, robot_id):
    cfg = SO101FollowerConfig(port=port, id=robot_id, use_degrees=True)
    robot = SO101Follower(cfg)
    robot.connect()
    return robot


def read_ranges(robot):
    ranges = {}
    for j in JOINTS:
        cal = robot.bus.calibration[j]
        mid = (cal.range_min + cal.range_max) / 2.0
        k = 360.0 / (RES - 1)
        ranges[j] = ((cal.range_min - mid) * k + SAFE_MARGIN_DEG, (cal.range_max - mid) * k - SAFE_MARGIN_DEG)
    ranges["gripper"] = (5.0, 95.0)
    return ranges


def denorm(norm, ranges):
    return {j: lerp(ranges[j][0], ranges[j][1], norm[j]) for j in ALL}


def move_to(robot, ranges, target_norm, seconds, fps=40):
    """Smoothly interpolate from the current pose to `target_norm` over `seconds`."""
    target = denorm(target_norm, ranges)
    obs = robot.get_observation()
    start = {j: float(obs[f"{j}.pos"]) for j in ALL}
    steps = max(1, round(seconds * fps))
    for i in range(steps + 1):
        t = i / steps
        cur = {j: lerp(start[j], target[j], t) for j in ALL}
        robot.send_action({f"{j}.pos": cur[j] for j in ALL})
        time.sleep(1.0 / fps)


def play_gesture(robot, ranges, name, fps=30):
    g = GESTURES[name]
    print(f"Playing '{name}' ({g['dur']}s)")
    move_to(robot, ranges, REST, 0.8)
    period = 1.0 / fps
    for i in range(int(g["dur"] * fps)):
        target = denorm(gesture_pose(name, i * period), ranges)
        robot.send_action({f"{j}.pos": target[j] for j in ALL})
        time.sleep(period)
    move_to(robot, ranges, REST, 0.8)


def main():
    ap = argparse.ArgumentParser(description="Play expressive gestures on the SO arm.")
    ap.add_argument("emotion", nargs="?", help="emotion name, or 'demo' / 'random'")
    ap.add_argument("--port", required=True, help="e.g. /dev/tty.usbmodemXXXX")
    ap.add_argument("--id", dest="robot_id", default="my_arm")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--list", action="store_true", help="list available emotions")
    args = ap.parse_args()

    if args.list:
        print("Emotions:", ", ".join(GESTURES))
        return

    robot = make_robot(args.port, args.robot_id)
    ranges = read_ranges(robot)
    try:
        if args.emotion == "demo":
            for name in GESTURES:
                play_gesture(robot, ranges, name, args.fps)
        elif args.emotion == "random":
            play_gesture(robot, ranges, random.choice(list(GESTURES)), args.fps)
        elif args.emotion in GESTURES:
            play_gesture(robot, ranges, args.emotion, args.fps)
        else:
            print(f"Unknown emotion '{args.emotion}'. Available: {', '.join(GESTURES)}")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
