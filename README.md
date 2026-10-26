# SO-Arm — a living, voice-controlled robot arm

Scripts to run a Hugging Face **LeRobot SO-100 / SO-101 arm** (6× Feetech STS3215
servos) on a **NVIDIA Jetson Orin Nano (JetPack 6)** — no laptop required.

The arm has no brain of its own (just servos + a serial controller board), so these
scripts run on the Jetson, which plugs into the controller board over USB.

## What each file does

| File | What it does | Camera? |
|------|--------------|:-------:|
| `teach_replay.py` | Record a motion by moving the arm by hand, then replay it on the same arm | ❌ |
| `emotions.py`     | Play expressive body-language gestures (wave, sad, excited, thinking, …) | ❌ |
| `idle.py`         | "Alive" breathing micro-motions when idle — subtle, organic, non-repeating | ❌ |
| `brain.py`        | Command brain: *grab*, *put*, *wave*, emotions, chat (optional LLM) | ❌ |
| `hand_teleop.py`  | Mirror your arm via the camera (hand/arm tracking) + grab-and-place | ✅ |
| `pick_ball.py`    | Detect an orange ball (top-down camera) and pick it up | ✅ |
| `vision.py`       | Orange-ball color detection (imported by `pick_ball.py`) | — |

All motion is clamped to the arm's calibrated range, so nothing can over-drive.

`tests/` holds unit tests for the pure (math) logic — they run with no arm attached.

---

## 1. Hardware

- SO-100 / SO-101 arm (assembled) + its controller board + power supply
- Jetson Orin Nano (JetPack 6) + USB cable to the controller board
- *(optional)* USB webcam or CSI camera — only needed for `hand_teleop.py`

> ⚠️ The arm ships in a **5V/7.4V** build and a **12V** build — they are **not
> interchangeable**. Use the supply that matches your kit. Blinking motor LEDs =
> wrong voltage or a joint forced past its limit.

## 2. One-time Jetson setup

```bash
# Python 3.12 + package manager (works on aarch64 Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Clone this repo
cd ~
git clone git@github.com:bitopan09/so-arm.git
cd so-arm

# Create a Python 3.12 virtualenv
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate

# PyTorch for JetPack 6 (Jetson-optimised wheels — install BEFORE lerobot)
uv pip install torch torchvision --index-url https://pypi.jetson-ai-lab.dev/jp6/cu126

# LeRobot (Feetech motor stack) + the camera/vision deps
uv pip install 'lerobot[feetech]' mediapipe opencv-python

# (optional) a Linux text-to-speech voice for brain.py
sudo apt install -y espeak

# sanity checks
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import lerobot; print('lerobot', lerobot.__version__)"
```

> If `pip install mediapipe` fails on your board, that only affects `hand_teleop.py`
> — the four motion scripts still work.

## 3. One-time arm setup

Run once per arm (power the arm on first; check the motor LEDs are steady red):

```bash
lerobot-find-port          # note the port, e.g. /dev/ttyACM0

lerobot-setup-motors --robot.type=so101_follower --robot.port=/dev/ttyACM0
#   connect ONE motor at a time as prompted (gripper -> ... -> shoulder_pan)

lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_arm
#   mid-range -> Enter -> sweep each joint -> Enter
```

Use the same `--port` and `--id` in every command below.

## 4. Run it

```bash
# expressive gestures
python emotions.py wave --port /dev/ttyACM0 --id my_arm
python emotions.py demo --port /dev/ttyACM0 --id my_arm      # cycle through all

# alive idle breathing (Ctrl-C to stop)
python idle.py --port /dev/ttyACM0 --id my_arm

# record a motion by hand, then replay it
python teach_replay.py teach  --port /dev/ttyACM0 --id my_arm
python teach_replay.py replay --port /dev/ttyACM0 --id my_arm --delay 3

# the brain: teach 3 workspace poses once, then talk to it
python brain.py teach --port /dev/ttyACM0 --id my_arm        # home / pick / place
python brain.py --port /dev/ttyACM0 --id my_arm              # then type: grab | put | wave | sad

# hand/arm teleop (needs a camera)
python hand_teleop.py --port /dev/ttyACM0 --id my_arm --camera 0
```

### brain.py — the LLM personality (optional)

`brain.py` works offline with keyword matching. To give it an LLM personality, set:

```bash
export OPENAI_API_KEY=...          # any OpenAI-compatible key
export OPENAI_BASE_URL=https://api.openai.com/v1   # optional
export OPENAI_MODEL=gpt-4o-mini    # optional
```

It needs internet for the LLM; without it, it still understands *grab / put / wave / sad*.

## 5. Troubleshooting

- **"Motor not found" / board seen but no motors** → check power supply, the 3-pin
  daisy-chain, and (Waveshare board) the jumpers set to **B (USB)**.
- **`hand_teleop.py` shows `[NO HAND]`** → move your hand fully into frame, more light.
- **Gripper moves the wrong way** → flip `INVERT_GRIP` in `hand_teleop.py`, or swap
  `GRIP_OPEN` / `GRIP_CLOSED` in `brain.py`.
- **Port not `/dev/ttyACM0`** → `ls /dev/ttyACM*` and use what appears.

## 6. Model files (hand tracking)

`hand_teleop.py` auto-downloads `pose_landmarker_full.task` and `hand_landmarker.task`
on first run (they're ~18 MB, not committed). If the Jetson has no internet at that
moment, download them manually and drop them next to `hand_teleop.py`.

## License / credit

Built on [Hugging Face LeRobot](https://github.com/huggingface/lerobot). Educational
use — keep hands and objects clear of the arm; it moves on its own.
