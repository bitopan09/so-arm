"""Unit tests for the pure logic in hand_teleop.py (no arm, no camera needed)."""

import types

import pytest

import hand_teleop as ht


def _lm3(x, y, z=0.0):
    return (x, y, z)


def _hand_lm(finger_dist=0.1):
    """21 landmarks: wrist (0.5,0.6), middle_mcp (0.5,0.5) -> hand_size 0.1;
    the 4 fingertips (8,12,16,20) sit `finger_dist` to the right of the wrist."""
    lm = [types.SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(21)]
    lm[0] = types.SimpleNamespace(x=0.5, y=0.6, z=0.0)
    lm[9] = types.SimpleNamespace(x=0.5, y=0.5, z=0.0)
    for i in (8, 12, 16, 20):
        lm[i] = types.SimpleNamespace(x=0.5 + finger_dist, y=0.6, z=0.0)
    return lm


def test_clamp():
    assert ht.clamp(5, 0, 10) == 5
    assert ht.clamp(-1, 0, 10) == 0
    assert ht.clamp(11, 0, 10) == 10


def test_lerp():
    assert ht.lerp(0, 10, 0.5) == 5.0
    assert ht.lerp(-90, 90, 0.0) == -90.0
    assert ht.lerp(-90, 90, 1.0) == 90.0


def test_norm_map():
    assert ht.norm_map(50, 0, 100) == pytest.approx(0.5)
    assert ht.norm_map(-5, 0, 100) == 0.0
    assert ht.norm_map(500, 0, 100) == 1.0
    assert ht.norm_map(0, 0, 100, invert=True) == 1.0


def test_pan_target_deadzone_and_gain():
    assert ht.pan_target(None) is None
    assert ht.pan_target(0.0) == pytest.approx(0.5)  # inside deadzone
    assert ht.pan_target(5.0) == pytest.approx(0.5)  # inside deadzone
    # pan=50 -> p=40 -> t=(40+50)/100=0.9 -> 0.5 + 0.6*0.4 = 0.74
    assert ht.pan_target(50.0) == pytest.approx(0.74)
    assert ht.pan_target(-50.0) == pytest.approx(0.26)


def test_angle_between():
    assert ht.angle_between((1, 0, 0), (1, 0, 0)) == pytest.approx(0.0)
    assert ht.angle_between((1, 0, 0), (0, 1, 0)) == pytest.approx(90.0)
    assert ht.angle_between((1, 0, 0), (-1, 0, 0)) == pytest.approx(180.0)


def test_deg_range_full_turn():
    lo, hi = ht.deg_range(0, 4095)
    assert lo == pytest.approx(-180.0, abs=0.1)
    assert hi == pytest.approx(180.0, abs=0.1)


# --- arm_angles: straight / bent / raised / pan ---

def test_arm_angles_straight_forward():
    # shoulder -> elbow -> wrist all pointing "forward/up" collinear
    a = ht.arm_angles((0.5, 0.6, 0.0), (0.5, 0.5, -0.2), (0.5, 0.4, -0.4))
    assert a["elbow_flex"] == pytest.approx(0.0, abs=1.0)
    assert a["shoulder_pan"] == pytest.approx(0.0, abs=1.0)
    assert a["shoulder_lift"] > 0  # arm above horizontal


def test_arm_angles_bent_elbow():
    # upper arm straight down, forearm forward -> 90 degree elbow
    a = ht.arm_angles((0.5, 0.5, 0.0), (0.5, 0.6, 0.0), (0.5, 0.6, -0.2))
    assert a["elbow_flex"] == pytest.approx(90.0, abs=1.0)
    assert a["shoulder_lift"] == pytest.approx(-90.0, abs=1.0)  # arm down


def test_arm_angles_raised():
    a = ht.arm_angles((0.5, 0.5, 0.0), (0.5, 0.3, 0.0), (0.5, 0.2, -0.1))
    assert a["shoulder_lift"] == pytest.approx(90.0, abs=1.0)


def test_arm_angles_pan_right():
    a = ht.arm_angles((0.5, 0.5, 0.0), (0.7, 0.5, -0.1), (0.8, 0.5, -0.1))
    assert a["shoulder_pan"] > 0


def test_arm_angles_vertical_pan_undefined():
    # arm pointing straight down -> azimuth is ill-defined -> hold
    a = ht.arm_angles((0.5, 0.5, 0.0), (0.5, 0.6, 0.0), (0.5, 0.6, -0.2))
    assert a["shoulder_pan"] is None
    assert a["elbow_flex"] == pytest.approx(90.0, abs=1.0)


# --- hand_metrics ---

def test_hand_metrics_grip_ratio():
    # hand_size = |middle(0.5,0.5) - wrist(0.5,0.6)| = 0.1; each tip is `finger_dist` from wrist
    ratio, _ = ht.hand_metrics(_hand_lm(finger_dist=0.1))
    assert ratio == pytest.approx(1.0)
    fist, _ = ht.hand_metrics(_hand_lm(finger_dist=0.03))
    assert fist < ratio


def test_pick_hand_closest_wrist():
    def hand_at(wx, wy):
        lm = [types.SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(21)]
        lm[0] = types.SimpleNamespace(x=wx, y=wy, z=0.0)  # wrist
        return lm

    hands = [hand_at(0.3, 0.3), hand_at(0.7, 0.6), hand_at(0.6, 0.55)]
    assert ht.pick_hand(hands, (0.65, 0.55)) is hands[2]
    assert ht.pick_hand([], (0.5, 0.5)) is None


# --- desired_targets stays in calibrated range ---

def test_desired_targets_always_in_range():
    ranges = {j: (-90.0, 90.0) for j in ht.JOINTS}
    ranges["gripper"] = (5.0, 95.0)
    for pan in (-150, -30, 0, 30, 150):
        for lift in (-150, -30, 0, 30, 150):
            for elbow in (-10, 0, 45, 90, 180):
                for wrist in (0, 30, 90, 180):
                    for roll in (-150, -30, 0, 30, 150):
                        for pinch in (-0.5, 0.1, 0.5, 1.0, 2.5):
                            angles = {
                                "shoulder_pan": pan,
                                "shoulder_lift": lift,
                                "elbow_flex": elbow,
                                "wrist_flex": wrist,
                            }
                            t = ht.desired_targets(angles, roll, pinch, ranges)
                            for j in ht.JOINTS:
                                lo, hi = ranges[j]
                                assert lo <= t[j] <= hi, f"{j}={t[j]} out of [{lo},{hi}]"
                            assert 5.0 <= t["gripper"] <= 95.0


def test_desired_targets_none_holds():
    ranges = {j: (-90.0, 90.0) for j in ht.JOINTS}
    ranges["gripper"] = (5.0, 95.0)
    angles = {"shoulder_pan": None, "shoulder_lift": 0.0, "elbow_flex": 45.0, "wrist_flex": None}
    t = ht.desired_targets(angles, None, None, ranges)
    assert t["shoulder_pan"] is None
    assert t["wrist_flex"] is None
    assert t["wrist_roll"] is None
    assert t["gripper"] is None
    assert t["elbow_flex"] is not None


def test_read_ranges_clamped_and_safe():
    class Cal:
        def __init__(self, mn, mx):
            self.range_min, self.range_max = mn, mx

    class FakeBus:
        def __init__(self):
            self.calibration = {j: Cal(0, 4095) for j in ht.JOINTS}

    class FakeRobot:
        def __init__(self):
            self.bus = FakeBus()

    r = ht.read_ranges(FakeRobot())  # type: ignore[arg-type]
    for j in ht.JOINTS:
        lo, hi = r[j]
        assert -180.0 <= lo < hi <= 180.0
    assert r["gripper"] == (5.0, 95.0)
