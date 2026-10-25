#!/usr/bin/env python3
"""
hand_teleop.py — full-arm teleop: the SO-100/SO-101 mirrors YOUR arm.

Tracks your arm with MediaPipe (Pose -> shoulder/elbow/wrist, Hand -> fingers) and
maps your actual joint angles to the robot's joints:

  upper-arm azimuth  ->  shoulder_pan   (swing the arm left / right)
  upper-arm elevation-> shoulder_lift   (raise / lower the arm)
  elbow bend         ->  elbow_flex     (bend the elbow)
  wrist bend         ->  wrist_flex     (bend the hand up / down at the wrist)
  pinch (thumb-index)->  gripper        (pinch to close)
  wrist_roll         ->  held at its calibrated middle

Every target is clamped to the arm's calibrated range, so it cannot over-drive.

Models needed (auto-downloaded on first run if missing):
  pose_landmarker_full.task   (arm angles)
  hand_landmarker.task        (gripper pinch)

Run from the LeRobot repo folder:

  uv run python hand_teleop.py --port /dev/tty.usbmodemXXXX --id my_arm

Controls:
  ENTER -> enable control (arm starts moving)
  q / ESC -> quit and release torque
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

# ---------------------------------------------------------------------------
# Tuning — raw-input ranges (degrees / image units) -> normalized [0,1].
# If a joint feels reversed, flip the matching INVERT flag.
# ---------------------------------------------------------------------------
ELBOW_MIN, ELBOW_MAX = 15.0, 160.0   # 15deg = near-straight, 160deg = fully folded
LIFT_MIN, LIFT_MAX = -80.0, 80.0     # upper-arm elevation, + = up
PAN_MIN, PAN_MAX = -50.0, 50.0       # upper-arm azimuth (narrower = less base travel)
PAN_DEADZONE = 10.0                  # ignore small base movements near center
PAN_GAIN = 0.6                       # scale base motion (lower = less twitchy)
WRIST_MIN, WRIST_MAX = 15.0, 100.0   # wrist bend (forearm vs hand), + = more bent
GRIP_MIN, GRIP_MAX = 0.55, 1.9       # fingertip-to-palm ratio: fist .. open
ROLL_MIN, ROLL_MAX = -60.0, 60.0     # palm roll -> wrist_roll
INVERT_ELBOW = False
INVERT_LIFT = False
INVERT_PAN = False
INVERT_WRIST = False
INVERT_ROLL = False
INVERT_GRIP = False

GRAB_RATIO = 0.8        # below this (fist) -> latch gripper shut
RELEASE_RATIO = 1.3     # above this (open hand) -> release

SMOOTH = 0.40           # EMA smoothing (lower = smoother/slower)
GRIP_SMOOTH = 0.30      # gripper responsiveness (higher = faster open/close)
SAFE_MARGIN_DEG = 4.0   # stay this far inside the calibrated range

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
RES = 4096  # STS3215 encoder steps per turn

# Pose landmark indices (MediaPipe 33-point body).
ARM_INDICES = {
    "right": {"shoulder": 12, "elbow": 14, "wrist": 16},
    "left": {"shoulder": 11, "elbow": 13, "wrist": 15},
}
# Hand landmark indices (MediaPipe 21-point hand).
HAND = {"wrist": 0, "index_mcp": 5, "index_tip": 8, "middle_mcp": 9, "thumb_tip": 4, "pinky_mcp": 17}

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

MODEL_URLS = {
    "pose_landmarker_full.task": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_full/float16/1/pose_landmarker_full.task"
    ),
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    ),
}


# --------------------------- pure math (testable) ---------------------------

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def lerp(a, b, t):
    return a + (b - a) * t


def norm_map(raw, lo, hi, invert=False):
    """Map raw -> [0,1] linearly, clamped. invert=True flips the direction."""
    t = clamp((raw - lo) / (hi - lo), 0.0, 1.0)
    return 1.0 - t if invert else t


def pan_target(pan):
    """Deadzone + gain on the base so small hand moves don't spin it."""
    if pan is None:
        return None
    if abs(pan) <= PAN_DEADZONE:
        p = 0.0
    else:
        p = pan - math.copysign(PAN_DEADZONE, pan)
    t = norm_map(p, PAN_MIN, PAN_MAX, INVERT_PAN)
    return 0.5 + PAN_GAIN * (t - 0.5)


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _norm(v):
    m = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    return (v[0] / m, v[1] / m, v[2] / m) if m > 1e-9 else (0.0, 0.0, 1.0)


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def angle_between(u, v):
    """Angle between two vectors in degrees (0 = same direction, 180 = opposite)."""
    return math.degrees(math.acos(clamp(_dot(_norm(u), _norm(v)), -1.0, 1.0)))


def arm_angles(shoulder, elbow, wrist):
    """Upper-arm angles in degrees from 3D points (x, y, z; y down, z toward camera).

    Returns {'shoulder_pan', 'shoulder_lift', 'elbow_flex'}. `shoulder_pan` is None when the
    arm is nearly vertical (azimuth is ill-defined) so the caller can hold the last value.
    """
    U = _norm(_sub(elbow, shoulder))  # shoulder -> elbow (upper arm)
    F = _norm(_sub(wrist, elbow))     # elbow -> wrist (forearm)

    elbow_flex = angle_between(U, F)                      # 0 straight .. 180 folded
    shoulder_lift = math.degrees(math.asin(clamp(-U[1], -1.0, 1.0)))  # + up

    horiz = math.hypot(U[0], U[2])
    if horiz < 0.2:  # arm nearly vertical -> azimuth undefined
        shoulder_pan = None
    else:
        shoulder_pan = math.degrees(math.atan2(U[0], -U[2]))  # 0 = pointing forward

    return {"shoulder_pan": shoulder_pan, "shoulder_lift": shoulder_lift, "elbow_flex": elbow_flex}


def hand_metrics(hand_lm):
    """Fingertip-to-palm ratio (fist ~ small, open ~ large) and palm roll (degrees)."""
    wrist = hand_lm[HAND["wrist"]]
    middle_mcp = hand_lm[HAND["middle_mcp"]]
    index_mcp = hand_lm[HAND["index_mcp"]]
    pinky_mcp = hand_lm[HAND["pinky_mcp"]]

    hand_size = math.hypot(middle_mcp.x - wrist.x, middle_mcp.y - wrist.y)
    tips = (8, 12, 16, 20)  # index, middle, ring, pinky fingertips
    tip_dist = sum(math.hypot(hand_lm[i].x - wrist.x, hand_lm[i].y - wrist.y) for i in tips) / 4.0
    grip_ratio = tip_dist / max(hand_size, 1e-6)
    roll = math.degrees(math.atan2(pinky_mcp.y - index_mcp.y, pinky_mcp.x - index_mcp.x))
    return grip_ratio, roll


def pick_hand(hand_landmarks, wrist):
    """Return the hand whose wrist landmark is closest to the pose wrist (x, y), else None."""
    if not hand_landmarks:
        return None

    def _dist(hl):
        wl = hl[HAND["wrist"]]
        return (wl.x - wrist[0]) ** 2 + (wl.y - wrist[1]) ** 2

    return min(hand_landmarks, key=_dist)


def deg_range(min_raw, max_raw):
    """Calibrated [range_min, range_max] in raw ticks -> (deg_min, deg_max)."""
    mid = (min_raw + max_raw) / 2.0
    k = 360.0 / (RES - 1)
    return (min_raw - mid) * k, (max_raw - mid) * k


def desired_targets(
    angles: dict[str, float | None],
    roll: float | None,
    grip_ratio: float | None,
    ranges: dict[str, tuple[float, float]],
):
    """Map raw measurements to robot targets. A None measurement -> None target (hold)."""
    t_pan = pan_target(angles["shoulder_pan"])
    t_lift = norm_map(angles["shoulder_lift"], LIFT_MIN, LIFT_MAX, INVERT_LIFT)
    t_elbow = norm_map(angles["elbow_flex"], ELBOW_MIN, ELBOW_MAX, INVERT_ELBOW)
    t_wflex = None if angles["wrist_flex"] is None else norm_map(angles["wrist_flex"], WRIST_MIN, WRIST_MAX, INVERT_WRIST)
    t_wroll = None if roll is None else norm_map(roll, ROLL_MIN, ROLL_MAX, INVERT_ROLL)
    t_grip = None if grip_ratio is None else norm_map(grip_ratio, GRIP_MIN, GRIP_MAX, invert=INVERT_GRIP)

    def _m(key: str, t: float | None) -> float | None:
        return None if t is None else lerp(ranges[key][0], ranges[key][1], t)

    return {
        "shoulder_pan": _m("shoulder_pan", t_pan),
        "shoulder_lift": _m("shoulder_lift", t_lift),
        "elbow_flex": _m("elbow_flex", t_elbow),
        "wrist_flex": _m("wrist_flex", t_wflex),
        "wrist_roll": _m("wrist_roll", t_wroll),
        "gripper": _m("gripper", t_grip),
    }


# ------------------------------ robot helpers ------------------------------

def make_robot(port: str, robot_id: str) -> SO101Follower:
    cfg = SO101FollowerConfig(port=port, id=robot_id, use_degrees=True)
    robot = SO101Follower(cfg)
    robot.connect()
    return robot


def read_ranges(robot: SO101Follower):
    ranges = {}
    for name in JOINTS:
        cal = robot.bus.calibration[name]
        dmin, dmax = deg_range(cal.range_min, cal.range_max)
        ranges[name] = (dmin + SAFE_MARGIN_DEG, dmax - SAFE_MARGIN_DEG)
    ranges["gripper"] = (5.0, 95.0)
    return ranges


# -------------------------------- main loop --------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Mirror your arm with the SO arm (webcam).")
    ap.add_argument("--port", required=True, help="e.g. /dev/tty.usbmodemXXXX")
    ap.add_argument("--id", dest="robot_id", default="my_arm")
    ap.add_argument("--camera", type=int, default=0, help="webcam index")
    ap.add_argument("--arm", default="auto", choices=["left", "right", "auto"])
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    for name, url in MODEL_URLS.items():
        p = script_dir / name
        if not p.exists():
            print(f"Downloading {name} ...")
            import urllib.request

            urllib.request.urlretrieve(url, p)
            print(f"  saved to {p}")

    pose_model = script_dir / "pose_landmarker_full.task"
    hand_model = script_dir / "hand_landmarker.task"

    robot = make_robot(args.port, args.robot_id)
    ranges = read_ranges(robot)

    # Start targets at the arm's current pose so there's no jump on enable.
    obs = robot.get_observation()
    current = {j: float(obs[f"{j}.pos"]) for j in JOINTS}
    current["gripper"] = float(obs["gripper.pos"])

    pose_detector = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(pose_model)),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )
    hand_detector = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(hand_model)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}.")

    input(
        "\nTorque is ON. Your arm will control the robot.\n"
        "Keep hands/objects clear of the arm.\n"
        "Press ENTER to ENABLE control ('q' or ESC quits)."
    )
    print("Control enabled. Face the camera and move your arm slowly.")

    frame_no = 0
    grip_latched = False
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)  # mirror so left/right feel natural
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))

            pose_result = pose_detector.detect(mp_img)
            hand_result = hand_detector.detect(mp_img)

            status = "NO PERSON (holding)"
            if pose_result.pose_landmarks:
                lm = pose_result.pose_landmarks[0]
                side = args.arm
                if side == "auto":
                    vis = {}
                    for s in ("left", "right"):
                        idx = ARM_INDICES[s]
                        vis[s] = sum(lm[idx[k]].visibility for k in idx) / 3.0
                    side = "right" if vis["right"] >= vis["left"] else "left"

                idx = ARM_INDICES[side]
                S = (lm[idx["shoulder"]].x, lm[idx["shoulder"]].y, lm[idx["shoulder"]].z)
                E = (lm[idx["elbow"]].x, lm[idx["elbow"]].y, lm[idx["elbow"]].z)
                W = (lm[idx["wrist"]].x, lm[idx["wrist"]].y, lm[idx["wrist"]].z)

                angles = arm_angles(S, E, W)
                angles["wrist_flex"] = None  # set below if a hand is visible

                # Draw arm skeleton
                h, w = frame.shape[:2]
                pts = [(int(p[0] * w), int(p[1] * h)) for p in (S, E, W)]
                cv2.line(frame, pts[0], pts[1], (255, 0, 0), 2)
                cv2.line(frame, pts[1], pts[2], (255, 0, 0), 2)
                for x, y in pts:
                    cv2.circle(frame, (x, y), 4, (0, 255, 255), -1)

                # Hand for pinch + wrist_flex (the one closest to the tracked wrist)
                hand_lm = pick_hand(hand_result.hand_landmarks, W)

                grip_ratio = None
                roll = None
                if hand_lm is not None:
                    grip_ratio, roll = hand_metrics(hand_lm)
                    # wrist bend = angle between forearm (elbow->wrist) and hand (wrist->middle mcp)
                    mcp = hand_lm[HAND["middle_mcp"]]
                    hand_dir = (mcp.x - W[0], mcp.y - W[1], mcp.z - W[2])
                    angles["wrist_flex"] = angle_between(_sub(E, W), hand_dir)
                    # overlay hand
                    hpts = [(int(p.x * w), int(p.y * h)) for p in hand_lm]
                    for a, b in HAND_CONNECTIONS:
                        cv2.line(frame, hpts[a], hpts[b], (0, 255, 0), 1)

                # gripper latch: fist to grab, open hand to release (hysteresis)
                if grip_ratio is not None:
                    if not grip_latched and grip_ratio < GRAB_RATIO:
                        grip_latched = True
                    elif grip_latched and grip_ratio > RELEASE_RATIO:
                        grip_latched = False

                desired = desired_targets(angles, roll, grip_ratio, ranges)

                for j in JOINTS:
                    d = desired[j]
                    if d is None:  # e.g. vertical arm -> hold pan
                        continue
                    current[j] += SMOOTH * (d - current[j])
                for j in JOINTS:
                    lo, hi = ranges[j]
                    current[j] = clamp(current[j], lo, hi)

                # gripper: hold while latched, else smooth toward target
                if not grip_latched and desired["gripper"] is not None:
                    current["gripper"] += GRIP_SMOOTH * (desired["gripper"] - current["gripper"])
                current["gripper"] = clamp(current["gripper"], *ranges["gripper"])

                robot.send_action({f"{j}.pos": current[j] for j in JOINTS + ["gripper"]})
                status = f"ARM {side}" + (" [HAND]" if hand_lm is not None else " [NO HAND]")
                if grip_ratio is not None:
                    status += f" grip={grip_ratio:.2f}"
                status += " [GRIP]" if grip_latched else ""

            cv2.putText(frame, status, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow("arm teleop", frame)

            frame_no += 1
            if frame_no % 30 == 0:
                print(
                    f"[{status}] pan={current['shoulder_pan']:.1f} lift={current['shoulder_lift']:.1f} "
                    f"elbow={current['elbow_flex']:.1f} wrist={current['wrist_flex']:.1f} "
                    f"gripper={current['gripper']:.0f}"
                )

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        pose_detector.close()
        hand_detector.close()
        cv2.destroyAllWindows()
        robot.disconnect()
        print("Released torque and disconnected.")


if __name__ == "__main__":
    main()
