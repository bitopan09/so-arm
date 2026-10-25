"""Unit tests for idle.py pure logic (no arm needed)."""

import idle


def test_idle_pose_returns_all_joints_in_range():
    for i in range(200):
        pose = idle.idle_pose(i * 0.05)
        assert set(pose) == set(idle.ALL)
        for j, v in pose.items():
            assert 0.0 <= v <= 1.0, f"{j}={v} out of [0,1]"


def test_idle_pose_is_subtle():
    # breathing should stay close to center — well under 15% of the range
    peak = 0.0
    for i in range(600):
        pose = idle.idle_pose(i * 0.05)
        peak = max(peak, max(abs(v - idle.CENTER) for v in pose.values()))
    assert peak < 0.15, f"idle motion too large: {peak}"


def test_idle_pose_is_non_repeating():
    # incommensurate frequencies -> pose differs at t=0 vs much later
    a = idle.idle_pose(0.0)
    b = idle.idle_pose(123.45)
    assert any(abs(a[j] - b[j]) > 1e-4 for j in idle.ALL)
