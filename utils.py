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


class WordBuilder:
    """
    Turns a stream of per-frame predictions into typed text.

    A letter is added only after it has been predicted steadily for
    `stable_frames` frames (about half a second). It cannot repeat until the
    hand leaves the camera or the sign changes, so holding "A" adds one A.
    Labels longer than one character (word signs such as "hello") are added
    as whole words.
    """

    def __init__(self, stable_frames=15):
        self.stable_frames = stable_frames
        self.text = ""
        self.candidate = None
        self.count = 0
        self.locked = None

    def update(self, label):
        """label = predicted label, or None if the model is not confident.
        Returns True when a letter/word was just added."""
        if label is None:
            self.candidate, self.count = None, 0
            return False
        if label != self.candidate:
            self.candidate, self.count = label, 1
            if self.locked is not None and label != self.locked:
                self.locked = None
            return False
        self.count += 1
        if self.count >= self.stable_frames and self.locked != label:
            self._add(label)
            self.locked = label
            return True
        return False

    def release(self):
        """Call when no hand is visible: allows the same letter again."""
        self.candidate, self.count, self.locked = None, 0, None

    def _add(self, label):
        if len(label) > 1:
            self.text += label.lower() + " "
        else:
            self.text += label

    @property
    def progress(self):
        if self.candidate is None:
            return 0.0
        if self.candidate == self.locked:
            return 1.0
        return min(self.count / self.stable_frames, 1.0)

    def space(self):
        if self.text and not self.text.endswith(" "):
            self.text += " "

    def backspace(self):
        self.text = self.text[:-1]

    def clear(self):
        self.text = ""

    def pop_text(self):
        out = self.text.strip()
        self.text = ""
        return out
