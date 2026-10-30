#!/usr/bin/env python3
"""
watch_grab.py — live camera feed with orange-ball detection; press 'g' to grab.

Overlays:
  - green circle + red dot  = the detected ball
  - blue crosses            = your taught calibration points (from pick_refs.json)

This lets you see whether the ball is INSIDE the taught region (mapping is reliable)
or OUTSIDE it (the linear map is extrapolating = inaccurate, teach more points).

Controls:  'g' = grab the ball    'q' = quit
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

from brain import move_to, set_gripper
from emotions import make_robot
from pick_ball import GRIP_CLOSED, GRIP_OPEN, REFS_FILE, fit_mapping, map_to_pose
from vision import detect_orange_ball


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--id", dest="robot_id", default="my_arm")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--refs", default=str(REFS_FILE))
    args = ap.parse_args()

    data = json.loads(Path(args.refs).read_text())
    refs = data["refs"]
    home = data["home"]
    if len(refs) < 3:
        print("Need >= 3 refs. Run: python pick_ball.py teach --port ... --camera ...")
        return
    coefs = fit_mapping(refs)
    taught = [(r["px"], r["py"]) for r in refs]
    xs = [p[0] for p in taught]
    ys = [p[1] for p in taught]

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"camera {args.camera} not open")

    robot = make_robot(args.port, args.robot_id)
    try:
        print("Live feed: press 'g' to GRAB, 'q' to quit.")
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            for tx, ty in taught:  # blue crosses = taught points
                cv2.drawMarker(frame, (int(tx), int(ty)), (255, 0, 0), cv2.MARKER_CROSS, 20, 1)

            det = detect_orange_ball(frame)
            pose = None
            if det:
                x, y, r = det
                cv2.circle(frame, (int(x), int(y)), int(r), (0, 255, 0), 2)
                cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)
                pose = map_to_pose(coefs, x, y)
                inside = min(xs) <= x <= max(xs) and min(ys) <= y <= max(ys)
                status = f"ball ({x:.0f},{y:.0f}) pan={pose['shoulder_pan']:.0f} lift={pose['shoulder_lift']:.0f}"
                if not inside:
                    status += "  [OUTSIDE taught area!]"
            else:
                status = "no ball"

            cv2.putText(frame, status, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "g = grab    q = quit", (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.imshow("ball tracker", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("g") and det:
                pose["gripper"] = GRIP_OPEN
                home["gripper"] = GRIP_OPEN
                print("GRABBING ->", {j: round(pose[j], 1) for j in pose})
                set_gripper(robot, GRIP_OPEN)
                move_to(robot, pose, 2.0)
                set_gripper(robot, GRIP_CLOSED)
                time.sleep(0.3)
                move_to(robot, home, 2.0)
                print("grabbed")
            elif key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        robot.disconnect()


if __name__ == "__main__":
    main()
