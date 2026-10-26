"""Unit tests for pick_ball.py mapping logic (no arm/camera needed)."""

import pick_ball


def test_fit_mapping_linear():
    refs = [
        {"px": 0.0, "py": 0.0, "pose": {"a": 0.0, "b": 0.0}},
        {"px": 100.0, "py": 0.0, "pose": {"a": 10.0, "b": 0.0}},
        {"px": 0.0, "py": 100.0, "pose": {"a": 0.0, "b": 10.0}},
    ]
    coefs = pick_ball.fit_mapping(refs)
    pose = pick_ball.map_to_pose(coefs, 50.0, 50.0)
    assert abs(pose["a"] - 5.0) < 0.2
    assert abs(pose["b"] - 5.0) < 0.2


def test_fit_mapping_interpolates_corners():
    refs = [
        {"px": 0.0, "py": 0.0, "pose": {"j": 0.0}},
        {"px": 200.0, "py": 0.0, "pose": {"j": 20.0}},
        {"px": 0.0, "py": 200.0, "pose": {"j": 10.0}},
    ]
    coefs = pick_ball.fit_mapping(refs)
    assert abs(pick_ball.map_to_pose(coefs, 200, 0)["j"] - 20.0) < 0.2
    assert abs(pick_ball.map_to_pose(coefs, 0, 200)["j"] - 10.0) < 0.2
