"""Unit tests for the pure gesture logic in emotions.py (no arm needed)."""

import math

import pytest

import emotions as em


def test_clamp():
    assert em.clamp(5, 0, 10) == 5
    assert em.clamp(-1, 0, 10) == 0


def test_lerp():
    assert em.lerp(0, 10, 0.5) == 5.0


def test_envelope_zero_at_edges():
    assert em.envelope(0.0, 3.0) == pytest.approx(0.0)
    assert em.envelope(3.0, 3.0) == pytest.approx(0.0)
    assert em.envelope(1.5, 3.0) == pytest.approx(1.0)


def test_gesture_pose_returns_all_joints_in_range():
    for name in em.GESTURES:
        for i in range(30):
            t = i * 0.1
            pose = em.gesture_pose(name, t)
            assert set(pose) == set(em.ALL), f"{name} missing joints"
            for j, v in pose.items():
                assert 0.0 <= v <= 1.0, f"{name} {j}={v} out of [0,1]"


def _peak_deviation(name):
    """Max |value - 0.5| across any joint over the gesture's full duration."""
    g = em.GESTURES[name]
    peak = 0.0
    for i in range(int(g["dur"] * 30)):
        pose = em.gesture_pose(name, i / 30.0)
        peak = max(peak, max(abs(v - 0.5) for v in pose.values()))
    return peak


def test_sad_is_smaller_than_excited():
    assert _peak_deviation("sad") < _peak_deviation("excited")


def test_angry_is_sharper_than_thinking():
    # angry has higher frequency content (higher freq in its osc table) than thinking
    angry_freq = max(f for _, (_, f, _) in em.GESTURES["angry"]["osc"].items())
    think_freq = max(f for _, (_, f, _) in em.GESTURES["thinking"]["osc"].items())
    assert angry_freq > think_freq
