#!/usr/bin/env python3
"""
pick_ball.py — detect a colored ping-pong ball with a top-down camera and pick it up.

Modes:
  teach        record a home pose + N grab poses (ball at N table spots)
  (default)    LIVE: stream an annotated feed to your browser, type 'grab' to pick
  --once       detect + grab once, then exit (no stream)

Live feed — open in any browser on the same network:
  http://<jetson-ip>:8080/

The overlay shows the detected ball (green circle + crosshair) and every taught
reference point (red dots). If the ball sits OUTSIDE the taught dots, the mapping is
extrapolating and will miss — teach more points that cover the whole reachable area.
"""

from __future__ import annotations

import argparse
import json
import select
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

from brain import move_to, read_pose, set_gripper
from emotions import JOINTS, clamp, make_robot, read_ranges
from vision import BALL_COLORS, detect_ball

SCRIPT_DIR = Path(__file__).resolve().parent
REFS_FILE = SCRIPT_DIR / "pick_refs.json"

GRIP_OPEN = 5.0
GRIP_CLOSED = 95.0
COLOR = "blue"          # ball color: blue | orange | green | red | yellow | pink
LIFT = 18.0             # degrees to raise shoulder_lift after closing (lifts ball off table)

_latest_frame = np.zeros((480, 640, 3), dtype=np.uint8)


def _set_latest(frame):
    global _latest_frame
    _latest_frame = frame


class _StreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send_index()
        elif self.path == "/stream.mjpg":
            self._send_stream()
        else:
            self.send_error(404)

    def _send_index(self):
        html = b'<html><body style="margin:0;background:#000"><img src="/stream.mjpg" style="width:100%"></body></html>'
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html)

    def _send_stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            while True:
                ok, buf = cv2.imencode(".jpg", _latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if not ok:
                    time.sleep(0.05)
                    continue
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                self.wfile.write(buf.tobytes())
                self.wfile.write(b"\r\n")
                time.sleep(0.03)
        except (BrokenPipeError, ConnectionResetError):
            pass


def start_stream(port=8080):
    srv = ThreadingHTTPServer(("0.0.0.0", port), _StreamHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def fit_mapping(refs):
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


def annotate(frame, det, coefs, refs):
    out = frame.copy()
    if det is not None:
        x, y, r = det
        cx, cy = int(x), int(y)
        cv2.circle(out, (cx, cy), max(2, int(r)), (0, 255, 0), 2)
        cv2.line(out, (cx - 20, cy), (cx + 20, cy), (0, 255, 0), 1)
        cv2.line(out, (cx, cy - 20), (cx, cy + 20), (0, 255, 0), 1)
        cv2.putText(out, f"({cx},{cy})", (cx + 12, cy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if coefs is not None:
            pose = map_to_pose(coefs, x, y)
            cv2.putText(out, f"pan={pose.get('shoulder_pan', 0):.0f} dist~elbow={pose.get('elbow_flex', 0):.0f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    for r in refs:
        cv2.circle(out, (int(r["px"]), int(r["py"])), 4, (0, 0, 255), -1)
    return out


def wait_for_ball(cap, timeout=10.0, fps=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        ok, frame = cap.read()
        if not ok:
            continue
        det = detect_ball(frame, COLOR)
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
        ans = input("Type 'r' to record a grab pose (ball placed somewhere), or 'done': ").strip().lower()
        if ans == "done":
            break
        if ans != "r":
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


def do_pick(robot, coefs, home, px, py):
    """Approach -> close -> lift -> carry home."""
    ranges = read_ranges(robot)
    raw = map_to_pose(coefs, px, py)
    pose = {j: clamp(raw[j], ranges[j][0], ranges[j][1]) for j in JOINTS}
    if any(abs(pose[j] - raw[j]) > 0.5 for j in JOINTS):
        print("  ! mapped pose out of range -> clamped. Teach more spread-out points.")
    pose["gripper"] = GRIP_OPEN

    print(f"grab at ({px:.0f},{py:.0f}) -> { {j: round(pose[j], 1) for j in JOINTS} }")
    set_gripper(robot, GRIP_OPEN)
    move_to(robot, pose, 2.0)          # descend onto the ball
    time.sleep(0.2)
    set_gripper(robot, GRIP_CLOSED)    # close
    time.sleep(0.4)

    lift = dict(pose)
    lift["shoulder_lift"] = clamp(pose["shoulder_lift"] + LIFT, *ranges["shoulder_lift"])
    move_to(robot, lift, 0.8)          # lift straight up, ball off the table
    time.sleep(0.3)
    move_to(robot, {**home, "gripper": GRIP_CLOSED}, 1.8)
    print("grabbed — ball is in the claw")


def interactive(robot, cap, refs, home, stream_port):
    coefs = fit_mapping(refs) if len(refs) >= 3 else None
    if coefs is None:
        print("Need >= 3 taught refs for a mapping — run 'teach' first.")
    start_stream(stream_port)
    print(f"Live feed: open http://<jetson-ip>:{stream_port}/ in a browser")
    print("Type 'grab' (or 'g') to pick up the ball. 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if ok:
            det = detect_ball(frame, COLOR)
            _set_latest(annotate(frame, det, coefs, refs))
        else:
            time.sleep(0.05)

        cmd = None
        if select.select([sys.stdin], [], [], 0)[0]:
            cmd = sys.stdin.readline().strip().lower()
        if cmd in ("q", "quit", "exit"):
            break
        if cmd in ("g", "grab", "pick"):
            if coefs is None:
                print("  ! need >= 3 taught refs")
            else:
                ok2, f2 = cap.read()
                det2 = detect_ball(f2, COLOR) if ok2 else None
                if det2 is None:
                    print("  ! no ball in view")
                else:
                    do_pick(robot, coefs, home, *det2[:2])
        time.sleep(0.03)


def main():
    global COLOR
    ap = argparse.ArgumentParser(description="Detect a colored ball and pick it up.")
    ap.add_argument("mode", nargs="?", choices=["teach"], default=None)
    ap.add_argument("--port", required=True, help="arm serial port, e.g. /dev/ttyACM0")
    ap.add_argument("--id", dest="robot_id", default="my_arm")
    ap.add_argument("--camera", type=int, default=0, help="camera index for the RGB feed")
    ap.add_argument("--color", default=COLOR, choices=list(BALL_COLORS), help=f"ball color (default {COLOR})")
    ap.add_argument("--stream-port", type=int, default=8080)
    ap.add_argument("--once", action="store_true", help="grab once and exit (no stream)")
    args = ap.parse_args()

    COLOR = args.color

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
        refs, home = data["refs"], data["home"]
        if args.once:
            det = wait_for_ball(cap)
            if det is None:
                print("No ball found.")
            else:
                do_pick(robot, fit_mapping(refs), home, *det[:2])
        else:
            interactive(robot, cap, refs, home, args.stream_port)
    finally:
        cap.release()
        robot.disconnect()


if __name__ == "__main__":
    main()
