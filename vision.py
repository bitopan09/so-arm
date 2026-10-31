#!/usr/bin/env python3
"""vision.py — colored ping-pong ball detection (pure OpenCV, unit-testable)."""

import cv2
import numpy as np

# OpenCV HSV: hue 0..179, sat/val 0..255.
BALL_COLORS = {
    "orange": ((5, 120, 120), (25, 255, 255)),
    "blue": ((95, 120, 90), (135, 255, 255)),
    "green": ((40, 90, 70), (85, 255, 255)),
    "red": ((0, 140, 90), (8, 255, 255)),  # red wraps; see _red_mask
    "yellow": ((22, 120, 120), (38, 255, 255)),
    "pink": ((140, 90, 120), (170, 255, 255)),
}

DEFAULT_COLOR = "blue"


def color_mask(frame_bgr, color=DEFAULT_COLOR):
    """Binary HSV mask for a named ball color."""
    lower, upper = BALL_COLORS[color]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    if color == "red":  # hue wraps through 0
        mask |= cv2.inRange(hsv, np.array([168, 140, 90]), np.array([179, 255, 255]))
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=3)
    return mask


def detect_ball(frame_bgr, color=DEFAULT_COLOR, min_radius=5):
    """Return (x, y, radius) of the largest ball of `color`, or None."""
    contours, _ = cv2.findContours(color_mask(frame_bgr, color), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    (x, y), r = cv2.minEnclosingCircle(c)
    if r < min_radius:
        return None
    return (float(x), float(y), float(r))


# --- backwards-compatible orange helpers -------------------------------------
def orange_mask(frame_bgr):
    return color_mask(frame_bgr, "orange")


def detect_orange_ball(frame_bgr, min_radius=5):
    return detect_ball(frame_bgr, "orange", min_radius)
