"""Unit tests for vision.py (colored ball detection)."""

import cv2
import numpy as np

import vision

BLUE_BGR = (255, 0, 0)
ORANGE_BGR = (0, 140, 255)


def _ball(bgr, xy=(120, 80), r=20, size=(200, 200)):
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    cv2.circle(img, xy, r, bgr, -1)
    return img


def test_detect_blue_ball():
    det = vision.detect_ball(_ball(BLUE_BGR), "blue")
    assert det is not None
    x, y, _ = det
    assert abs(x - 120) < 5 and abs(y - 80) < 5


def test_blue_is_the_default():
    assert vision.detect_ball(_ball(BLUE_BGR)) is not None


def test_blue_detector_ignores_orange():
    assert vision.detect_ball(_ball(ORANGE_BGR), "blue") is None


def test_orange_still_works():
    det = vision.detect_ball(_ball(ORANGE_BGR), "orange")
    assert det is not None
    assert vision.detect_orange_ball(_ball(ORANGE_BGR)) is not None


def test_no_ball():
    assert vision.detect_ball(np.zeros((100, 100, 3), np.uint8), "blue") is None


def test_min_radius_filter():
    assert vision.detect_ball(_ball(BLUE_BGR, r=3), "blue", min_radius=10) is None


def test_unknown_color_raises():
    try:
        vision.detect_ball(_ball(BLUE_BGR), "chartreuse")
    except KeyError:
        return
    raise AssertionError("expected KeyError for unknown color")
