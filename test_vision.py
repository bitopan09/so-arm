"""Unit tests for vision.py (orange ball detection)."""

import cv2
import numpy as np

import vision


def test_detect_orange_ball():
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(img, (120, 80), 20, (0, 140, 255), -1)  # orange in BGR
    det = vision.detect_orange_ball(img)
    assert det is not None
    x, y, r = det
    assert abs(x - 120) < 5 and abs(y - 80) < 5


def test_no_ball():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    assert vision.detect_orange_ball(img) is None


def test_ignores_non_orange():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(img, (50, 50), 20, (255, 0, 0), -1)  # blue
    assert vision.detect_orange_ball(img) is None
