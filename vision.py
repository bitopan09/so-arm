#!/usr/bin/env python3
"""vision.py — orange ping-pong ball detection (pure OpenCV, unit-testable)."""

import cv2
import numpy as np

# OpenCV HSV: hue 0..179, sat/val 0..255. Orange hue ~ 5..25.
_ORANGE_LOWER = np.array([5, 120, 120])
_ORANGE_UPPER = np.array([25, 255, 255])


def orange_mask(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, _ORANGE_LOWER, _ORANGE_UPPER)
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=3)
    return mask


def detect_orange_ball(frame_bgr, min_radius=5):
    """Return (x, y, radius) of the largest orange blob, or None."""
    mask = orange_mask(frame_bgr)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    (x, y), r = cv2.minEnclosingCircle(c)
    if r < min_radius:
        return None
    return (x, y, r)
