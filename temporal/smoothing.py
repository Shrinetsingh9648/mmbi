# mmbi/temporal/smoothing.py
"""
Lightweight exponential-moving-average smoother for the per-frame
micro-expression probability stream.

Kept deliberately tiny and dependency-free (pure Python) so it can be
unit tested without torch/cv2/mediapipe and reused identically by both
the real-time engine and offline evaluation scripts.

Micro-expressions are short (40-500 ms), so do NOT use a long window —
a 1-second moving average would smear/erase the signal entirely. The
default alpha=0.3 keeps roughly the last 3-4 frames' worth of influence
dominant while cutting single-frame sensor noise.
"""

from __future__ import annotations


class EMASmoother:
    """
    smoothed[t] = alpha * raw[t] + (1 - alpha) * smoothed[t-1]

    alpha closer to 1.0  -> less smoothing, follows raw signal closely.
    alpha closer to 0.0  -> more smoothing, slower to react (risk of
                             smearing/erasing genuine short ME spikes).
    """

    def __init__(self, alpha: float = 0.3, initial_value: float = 0.0):
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = float(alpha)
        self._value = float(initial_value)
        self._initialized = False

    def update(self, raw_value: float) -> float:
        raw_value = float(raw_value)
        if not self._initialized:
            # First sample: no history yet, so start exactly at the raw
            # value instead of blending with an arbitrary initial_value.
            self._value = raw_value
            self._initialized = True
        else:
            self._value = self.alpha * raw_value + (1 - self.alpha) * self._value
        return self._value

    @property
    def value(self) -> float:
        return self._value

    def reset(self, initial_value: float = 0.0) -> None:
        self._value = float(initial_value)
        self._initialized = False
