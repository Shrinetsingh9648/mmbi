import numpy as np
from collections import deque
import time

class BlinkDetector:
    """
    Detects blinks using Eye Aspect Ratio (EAR).
    EAR < threshold for 2+ consecutive frames = blink.
    Tracks blink rate per minute and stress signal.

    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    Normal blink rate: 12–20 blinks/minute.
    Stress signal: > 25 blinks/minute.
    """

    # MediaPipe indices for EAR calculation (6 points per eye)
    LEFT_EYE_EAR  = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE_EAR = [33,  160, 158, 133, 153, 144]

    EAR_THRESHOLD     = 0.20   # below this = eye closing
    CONSEC_FRAMES     = 2      # frames eye must be closed to count as blink
    BLINK_WINDOW_SECS = 60     # rolling window for blink rate

    def __init__(self, fps=120):
        self.fps          = fps
        self.counter      = 0          # consecutive below-threshold frames
        self.total_blinks = 0
        self.blink_times  = deque()    # timestamps of blinks
        self.in_blink     = False

    def _ear(self, lm, indices):
        p = lm[indices, :2]
        vertical_1 = np.linalg.norm(p[1] - p[5])
        vertical_2 = np.linalg.norm(p[2] - p[4])
        horizontal = np.linalg.norm(p[0] - p[3])
        return (vertical_1 + vertical_2) / (2.0 * horizontal + 1e-6)

    def update(self, lm):
        """
        Call every frame with landmarks array.
        Returns dict: ear, blink_detected, total_blinks,
                      blink_rate_per_min, stress_signal
        """
        left_ear  = self._ear(lm, self.LEFT_EYE_EAR)
        right_ear = self._ear(lm, self.RIGHT_EYE_EAR)
        ear        = (left_ear + right_ear) / 2.0

        blink_detected = False

        if ear < self.EAR_THRESHOLD:
            self.counter += 1
        else:
            if self.counter >= self.CONSEC_FRAMES:
                # Eye was closed long enough — count as blink
                self.total_blinks += 1
                now = time.time()
                self.blink_times.append(now)
                blink_detected = True
            self.counter = 0

        # Clean up old blink timestamps outside rolling window
        now = time.time()
        while self.blink_times and (now - self.blink_times[0]) > self.BLINK_WINDOW_SECS:
            self.blink_times.popleft()

        blink_rate = len(self.blink_times)   # blinks per last 60 seconds

        # Stress signal from blink rate
        if blink_rate < 5:
            stress_signal = "STARING"        # too few blinks = intense focus or discomfort
        elif blink_rate <= 20:
            stress_signal = "NORMAL"
        elif blink_rate <= 30:
            stress_signal = "ELEVATED"
        else:
            stress_signal = "HIGH_STRESS"

        return {
            'ear':               round(float(ear), 3),
            'blink_detected':    blink_detected,
            'total_blinks':      self.total_blinks,
            'blink_rate_per_min': blink_rate,
            'stress_signal':     stress_signal
        }