"""
SmartSign - shared helpers.
Training and live prediction MUST use the same preprocessing,
so it lives in one place.
"""

import numpy as np


def normalize_landmarks(row, eps=1e-6):
    """
    Min-max normalize one sample of 63 values (21 landmarks x [x, y, z]).

        X_norm = (X - X_min) / (X_max - X_min)

    Applied separately to the x, y and z axes of each sample, so the result
    does not depend on hand size or how far the hand is from the camera.
    """
    pts = np.asarray(row, dtype=np.float32).reshape(21, 3)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    norm = (pts - mins) / (maxs - mins + eps)
    return norm.flatten()


def extract_landmarks(hand_landmarks):
    """Flatten MediaPipe's 21 (x, y, z) landmarks into 63 values."""
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    return coords