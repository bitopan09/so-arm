#!/usr/bin/env python3
"""
teach_replay.py — kinesthetic teach-and-replay for ONE SO-100 / SO-101 arm.

No camera. No leader arm. No neural network.

The same arm plays both roles:
  TEACH  (master): torque OFF -> you move the arm by hand -> it records joint positions.
  REPLAY (slave):  torque ON  -> it re-plays the recorded motion on itself.

Run from the LeRobot repo folder so `uv run` picks up its environment:

  uv run python teach_replay.py teach   --port /dev/tty.usbmodemXXXX --id my_arm
  uv run python teach_replay.py replay  --port /dev/tty.usbmodemXXXX --id my_arm --delay 3
  uv run python teach_replay.py replay  --port /dev/tty.usbmodemXXXX --id my_arm --loop
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

FPS = 30
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def make_robot(port: str, robot_id: str) -> SO101Follower:
    """Build + connect the arm. connect() loads/creates calibration and enables torque."""
    cfg = SO101FollowerConfig(port=port, id=robot_id, use_degrees=True)
    robot = SO101Follower(cfg)
    robot.connect()
    return robot


def read_joints(robot: SO101Follower) -> dict[str, float]:
    """Read the current normalized joint positions (6 motors)."""
    obs = robot.get_observation()
    return {m: float(obs[f"{m}.pos"]) for m in JOINTS}


def teach(robot: SO101Follower, out_path: Path, fps: int) -> None:
    period = 1.0 / fps

    # Torque off => the arm is free to be moved by hand (backdriven).
    robot.bus.disable_torque()
    input(
        "Torque is OFF — the arm is free to move by hand.\n"
        "1) Move the arm to its START pose.\n"
        "2) Press ENTER to begin recording, then move the arm through the motion.\n"
        "3) Press ENTER again to STOP.\n"
    )

    print("Recording... move the arm now. Press ENTER to STOP.")

    stop = threading.Event()
    threading.Thread(target=lambda: (input(), stop.set()), daemon=True).start()

    frames: list[dict[str, float]] = []
    next_t = time.perf_counter()
    while not stop.is_set():
        frames.append(read_joints(robot))
        next_t += period
        time.sleep(max(0.0, next_t - time.perf_counter()))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"fps": fps, "joints": JOINTS, "frames": frames}, indent=2))
    print(f"Saved {len(frames)} frames ({len(frames) / fps:.1f}s) -> {out_path}")


def replay(robot: SO101Follower, in_path: Path, delay: float, loop: bool) -> None:
    data = json.loads(in_path.read_text())
    frames = data["frames"]
    fps = int(data.get("fps", FPS))
    period = 1.0 / fps

    # Put the arm back at the recorded start pose by hand, then engage torque.
    robot.bus.disable_torque()
    input("Torque is OFF. Move the arm to the SAME start pose you used when recording, then press ENTER.")
    robot.bus.enable_torque()

    if delay > 0:
        print(f"Waiting {delay}s before replay...")
        time.sleep(delay)

    input("The arm is about to move on its own. Press ENTER to START (Ctrl-C to abort).")

    def run_once() -> None:
        for frame in frames:
            t0 = time.perf_counter()
            action = {f"{m}.pos": frame[m] for m in JOINTS}
            robot.send_action(action)
            dt = time.perf_counter() - t0
            if dt < period:
                time.sleep(period - dt)

    while True:
        run_once()
        if not loop:
            print("Replay finished.")
            break
        print("Loop done — repeating...")

    robot.bus.disable_torque()
    print("Torque released.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Teach-and-replay for a single SO-100/SO-101 arm.")
    ap.add_argument("mode", choices=["teach", "replay"])
    ap.add_argument("--port", required=True, help="e.g. /dev/tty.usbmodemXXXX (from lerobot-find-port)")
    ap.add_argument("--id", dest="robot_id", default="my_arm", help="calibration id (same one used at calibrate time)")
    ap.add_argument("--file", default="teach_replay.json", help="trajectory file to save/load")
    ap.add_argument("--fps", type=int, default=FPS)
    ap.add_argument("--delay", type=float, default=0.0, help="seconds to wait before replay starts")
    ap.add_argument("--loop", action="store_true", help="repeat the replay forever")
    args = ap.parse_args()

    robot = make_robot(args.port, args.robot_id)
    try:
        if args.mode == "teach":
            teach(robot, Path(args.file), args.fps)
        else:
            replay(robot, Path(args.file), args.delay, args.loop)
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
