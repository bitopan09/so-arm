#!/usr/bin/env python3
"""
brain.py — the arm's "brain": talk to it, it reacts + does pick-and-place.

Stage 1 (fixed locations, no vision yet):
  - Type a command; keyword matching understands grab / put / wave / emotions / chat.
  - Grab & place move between three taught workspace poses (home / pick / place).
  - Speaks back via macOS `say`.
  - Optional LLM (any OpenAI-compatible endpoint) for richer chat replies.

One-time setup — teach the three poses by hand (torque off, move arm, Enter):
  uv run python brain.py teach --port /dev/tty.usbmodemXXXX --id my_arm

Run:
  uv run python brain.py --port /dev/tty.usbmodemXXXX --id my_arm

To enable the LLM personality, set env vars:
  OPENAI_API_KEY=... OPENAI_BASE_URL=https://api.openai.com/v1 OPENAI_MODEL=gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from emotions import GESTURES, JOINTS, make_robot, play_gesture, read_ranges

ALL = JOINTS + ["gripper"]
SCRIPT_DIR = Path(__file__).resolve().parent
POSES_FILE = SCRIPT_DIR / "poses.json"

# Gripper actual values (0..100). Flip these if the claw moves the wrong way.
GRIP_OPEN = 5.0
GRIP_CLOSED = 95.0

REPLIES = {
    "grab": "Grabbing it now.",
    "place": "Placing it.",
    "wave": "Hi there!",
    "howareyou": "Doing great, thanks for asking.",
    "chat": "I'm just a little arm, but I'm listening.",
    "quit": "Bye!",
}
for _g in GESTURES:
    REPLIES.setdefault(_g, f"Feeling {_g}.")


def say(text):
    """Speak via the platform's TTS; always print so there's a visible reply."""
    print(f"[bot] {text}")
    for cmd in (["say", text], ["espeak", text], ["spd-say", text]):
        try:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except FileNotFoundError:
            continue


def parse_command(text):
    """Keyword -> intent. Pure function (unit-tested)."""
    t = text.lower()
    if any(w in t for w in ("grab", "pick", "take", "hold", "grasp")):
        return "grab"
    if any(w in t for w in ("put", "place", "drop", "release", "give", "throw")):
        return "place"
    if any(w in t for w in ("quit", "exit", "stop", "bye", "shutdown")):
        return "quit"
    if any(w in t for w in ("how are you", "how's it going", "how you doing")):
        return "howareyou"
    for g in GESTURES:
        if g in t:
            return g
    if any(w in t for w in ("hi", "hello", "hey", "wave")):
        return "wave"
    return "chat"


def llm_available():
    return bool(os.environ.get("OPENAI_API_KEY"))


def llm_reply(text):
    """Optional OpenAI-compatible reply. Returns None on any failure."""
    if not llm_available():
        return None
    try:
        import requests
    except ImportError:
        return None
    key = os.environ["OPENAI_API_KEY"]
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    try:
        r = requests.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a friendly little robot arm. Reply in ONE short sentence (max 12 words), warm and playful.",
                    },
                    {"role": "user", "content": text},
                ],
                "max_tokens": 40,
                "temperature": 0.8,
            },
            timeout=10,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:  # noqa: BLE001
        print(f"(LLM error: {e})")
        return None


def move_to(robot, target, seconds, fps=40):
    """Interpolate from the current pose to `target` (actual joint values) over `seconds`."""
    obs = robot.get_observation()
    start = {j: float(obs[f"{j}.pos"]) for j in ALL}
    steps = max(1, round(seconds * fps))
    for i in range(steps + 1):
        t = i / steps
        cur = {j: start[j] + (target[j] - start[j]) * t for j in ALL}
        robot.send_action({f"{j}.pos": cur[j] for j in ALL})
        time.sleep(1.0 / fps)


def set_gripper(robot, value, seconds=0.7, fps=30):
    """Move just the gripper to `value` (0..100)."""
    obs = robot.get_observation()
    start = float(obs["gripper.pos"])
    steps = max(1, round(seconds * fps))
    for i in range(steps + 1):
        cur = start + (value - start) * (i / steps)
        robot.send_action({"gripper.pos": cur})
        time.sleep(1.0 / fps)


def do_grab(robot, poses):
    say("Grabbing")
    move_to(robot, poses["pick"], 2.0)
    set_gripper(robot, GRIP_CLOSED)
    time.sleep(0.3)
    move_to(robot, poses["home"], 2.0)
    say("Got it")


def do_place(robot, poses):
    say("Placing")
    move_to(robot, poses["place"], 2.0)
    set_gripper(robot, GRIP_OPEN)
    time.sleep(0.3)
    move_to(robot, poses["home"], 2.0)
    say("Done")


def read_pose(robot):
    obs = robot.get_observation()
    return {j: float(obs[f"{j}.pos"]) for j in ALL}


def teach(robot):
    poses = {}
    for name in ("home", "pick", "place"):
        robot.bus.disable_torque()
        input(f"\nMove the arm to the '{name}' position (by hand), then press Enter.")
        poses[name] = read_pose(robot)
        robot.bus.enable_torque()
        print(f"  recorded '{name}': gripper={poses[name]['gripper']:.0f}")
    POSES_FILE.write_text(json.dumps(poses, indent=2))
    print(f"Saved poses to {POSES_FILE}")


def main():
    ap = argparse.ArgumentParser(description="The arm's brain.")
    ap.add_argument("mode", nargs="?", choices=["teach"], default=None)
    ap.add_argument("--port", required=True, help="e.g. /dev/tty.usbmodemXXXX")
    ap.add_argument("--id", dest="robot_id", default="my_arm")
    args = ap.parse_args()

    robot = make_robot(args.port, args.robot_id)
    ranges = read_ranges(robot)
    try:
        if args.mode == "teach":
            teach(robot)
            return

        if POSES_FILE.exists():
            poses = json.loads(POSES_FILE.read_text())
            print(f"Loaded poses from {POSES_FILE.name}")
        else:
            print(f"No {POSES_FILE.name} yet — using current pose as home/pick/place. Run 'teach' to set real ones.")
            rest = read_pose(robot)
            poses = {"home": rest, "pick": rest, "place": rest}

        say("I'm awake. Tell me what to do.")
        print("Commands: grab | put | wave | <emotion> | how are you | quit")
        while True:
            try:
                text = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not text:
                continue

            cmd = parse_command(text)
            reply = llm_reply(text)

            if cmd == "quit":
                say(reply or REPLIES["quit"])
                break
            if cmd == "grab":
                do_grab(robot, poses)
                say(reply or REPLIES["grab"])
            elif cmd == "place":
                do_place(robot, poses)
                say(reply or REPLIES["place"])
            elif cmd in GESTURES:
                play_gesture(robot, ranges, cmd)
                say(reply or REPLIES[cmd])
            elif cmd == "howareyou":
                say(reply or REPLIES["howareyou"])
            else:
                say(reply or REPLIES["chat"])
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
