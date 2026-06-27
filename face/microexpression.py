# mmbi/face/microexpression.py
import cv2, numpy as np
from collections import deque

FPS = 120
MIN_FRAMES = 5    # 40 ms minimum
MAX_FRAMES = 60   # 500 ms maximum

class MicroExpressionDetector:
    def __init__(self):
        self.au_buffer = deque(maxlen=MAX_FRAMES)
        self.prev_gray = None

    # def update(self, frame_gray, au_values):
    #     self.au_buffer.append(au_values)
    #     flow = None
    #     if self.prev_gray is not None:
    #         flow = cv2.calcOpticalFlowFarneback(
    #             self.prev_gray, frame_gray,
    #             None, 0.5, 3, 15, 3, 5, 1.2, 0)
    #     self.prev_gray = frame_gray
    #     return self._detect_spike(flow)

    def update(self, frame_gray, au_values):
        self.au_buffer.append(au_values)
        # Only compute optical flow every 4 frames to reduce CPU load
        flow = None
        if self.prev_gray is not None and len(self.au_buffer) % 4 == 0:
            # Resize to half resolution before optical flow — 4x faster
            small_curr = cv2.resize(frame_gray, (0, 0), fx=0.5, fy=0.5)
            small_prev = cv2.resize(self.prev_gray, (0, 0), fx=0.5, fy=0.5)
            flow = cv2.calcOpticalFlowFarneback(
                small_prev, small_curr,
                None, 0.5, 2, 10, 2, 5, 1.1, 0)  # fewer pyramid levels
        self.prev_gray = frame_gray
        return self._detect_spike(flow)

    def _detect_spike(self, flow):
        if len(self.au_buffer) < MIN_FRAMES:
            return None
        recent = list(self.au_buffer)[-MIN_FRAMES:]
        deltas = []
        for i in range(1, len(recent)):
            for k in recent[i]:
                deltas.append(abs(recent[i][k] - recent[i-1].get(k, 0)))
        magnitude = np.mean(deltas) if deltas else 0
        if magnitude > 0.015:           # tunable threshold
            return {'spike': True, 'magnitude': round(magnitude, 4)}
        return None