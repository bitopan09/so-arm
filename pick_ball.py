#!/usr/bin/env python3
"""
pick_ball.py — find an orange ping-pong ball with a top-down camera and pick it up.

How it works (no depth, no IK):
  1. `--teach`  : record a "home" pose + N grab poses (ball placed at N table spots,
                  arm moved by hand to grab each). Saved to pick_refs.json.
  2. (default)   : detect the orange ball -> map its pixel position to an arm pose via
                  a linear fit over the taught points -> approach, grab, lift.

Run from the LeRobot repo / so-arm folder:
  uv run python pick_ball.py teach --port /dev/ttyACM0 --id my_arm --camera 0
  uv run python pick_ball.py --port /dev/ttyACM0 --id my_arm --camera 0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from brain import move_to, read_pose, set_gripper
from emotions import ALL, JOINTS, make_robot
from vision import detect_orange_ball

SCRIPT_DIR = Path(__file__).resolve().parent
REFS_FILE = SCRIPT_DIR / "pick_refs.json"

GRIP_OPEN = 5.0
GRIP_CLOSED = 95.0


def fit_mapping(refs):
    """Fit a per-joint affine map (px, py, 1) -> joint value by least squares."""
    X = np.array([[r["px"], r["py"], 1.0] for r in refs])
    joints = list(refs[0]["pose"].keys())
    coefs = {}
    for j in joints:
        y = np.array([r["pose"][j] for r in refs])
        coefs[j], *_ = np.linalg.lstsq(X, y, rcond=None)
    return coefs


def map_to_pose(coefs, px, py):
    v = np.array([px, py, 1.0])
    return {j: float(np.dot(coefs[j], v)) for j in coefs}


def wait_for_ball(cap, timeout=10.0, fps=10):
    """Read frames until an orange ball is seen; return (px, py, radius) or (None,...)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        ok, frame = cap.read()
        if not ok:
            continue
        det = detect_orange_ball(frame)
        if det is not None:
            return det
        time.sleep(1.0 / fps)
    return None


def teach(robot, cap):
    refs = []
    robot.bus.disable_torque()
    input("Move the arm to its HOME (raised, out of the way) pose, then press Enter.")
    home = {j: read_pose(robot)[j] for j in JOINTS}
    robot.bus.enable_torque()

    while True:
        ans = input("Place the ball somewhere on the table. Type 'r' to record a grab pose, or 'done' to finish: ").strip()
        if ans.lower() == "done":
            break
        if ans.lower() != "r":
            continue
        det = wait_for_ball(cap)
        if det is None:
            print("  ! ball not seen by the camera")
            continue
        px, py, _ = det
        robot.bus.disable_torque()
        input(f"  ball at pixel ({px:.0f},{py:.0f}) — move the arm to grab it, then press Enter.")
        pose = {j: read_pose(robot)[j] for j in JOINTS}
        robot.bus.enable_torque()
        refs.append({"px": px, "py": py, "pose": pose})
        print(f"  recorded ref #{len(refs)}")

    REFS_FILE.write_text(json.dumps({"home": home, "refs": refs}, indent=2))
    print(f"Saved {len(refs)} refs + home to {REFS_FILE}")


def pick(robot, cap, refs, home):
    coefs = fit_mapping(refs)
    print("Looking for the orange ball...")
    det = wait_for_ball(cap)
    if det is None:
        print("No ball found.")
        return
    px, py, _ = det
    pose = map_to_pose(coefs, px, py)
    pose["gripper"] = GRIP_OPEN  # add gripper so move_to doesn't KeyError
    home["gripper"] = GRIP_OPEN
    print(f"Ball at ({px:.0f},{py:.0f}) -> arm pose { {j: round(pose[j],1) for j in pose} }")

    set_gripper(robot, GRIP_OPEN)
    move_to(robot, pose, 2.0)          # approach (gripper open)
    set_gripper(robot, GRIP_CLOSED)    # grab
    time.sleep(0.3)
    move_to(robot, home, 2.0)          # lift
    print("Grabbed the ball.")


def main():
    ap = argparse.ArgumentParser(description="Detect an orange ball and pick it up.")
    ap.add_argument("mode", nargs="?", choices=["teach"], default=None)
    ap.add_argument("--port", required=True, help="e.g. /dev/ttyACM0")
    ap.add_argument("--id", dest="robot_id", default="my_arm")
    ap.add_argument("--camera", type=int, default=0, help="webcam index for the RGB feed")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}.")

    robot = make_robot(args.port, args.robot_id)
    try:
        if args.mode == "teach":
            teach(robot, cap)
            return
        if not REFS_FILE.exists():
            print(f"No {REFS_FILE.name} — run 'teach' first.")
            return
        data = json.loads(REFS_FILE.read_text())
        if len(data["refs"]) < 3:
            print("Need at least 3 taught refs for a linear fit.")
            return
        pick(robot, cap, data["refs"], data["home"])
    finally:
        cap.release()
        robot.disconnect()


if __name__ == "__main__":
    main()
