# mmbi/face/microexpression.py
"""
Cheap, always-on heuristic micro-expression CUE detector.

IMPORTANT — what this class is and is NOT:
  - It is NOT a trained micro-expression classifier.
  - It does NOT produce onset/apex/offset phase labels.
  - Its "magnitude"/"combined_score" outputs are hand-tuned heuristic
    thresholds, not calibrated probabilities.
  - The actual instance-wise ME probability that should drive the
    real-time graph and event detector comes from the trained
    MicroLSTM's spotting_head (models/lstm_temporal.py), once that head
    has been trained on real labeled data.

What this class IS good for:
  - A fast, dependency-light signal that feeds INTO the fused feature
    vector (temporal/buffer.py FEATURE_KEYS: flow_brow/flow_eye/
    flow_nose/flow_mouth/flow_cheek) that the trained model consumes.
  - A fallback/sanity-check display value ("heuristic AU+flow spike")
    for when no trained model weights are available at all.

Previous bug (fixed here): optical flow used to be computed every 4th
call via cv2.calcOpticalFlowFarneback(...) and then silently discarded
-- the `flow` variable was passed into `_detect_spike()` but never read.
This version actually consumes the flow field: it computes overall
mean/max flow magnitude AND per-region magnitude (brow/eye/nose/mouth/
cheek), using the same MediaPipe landmark indices AUCalculator already
uses, so no new landmark scheme is introduced.
"""
import cv2
import numpy as np
from collections import deque

# NOTE: this FPS figure describes what the ORIGINAL author assumed, not a
# measured value. See capture/camera.py.measure_actual_fps() /
# temporal/buffer.py for the runtime-measured figure that should be used
# for real duration math. MIN_FRAMES/MAX_FRAMES are expressed in frames
# because that's what the deque needs; their real-world duration depends
# on the actual camera FPS, not the constant below.
ASSUMED_FPS = 120  # historical assumption only -- do not trust for timing
MIN_FRAMES = 5      # AU-delta window length, in frames
MAX_FRAMES = 60      # AU-delta buffer capacity, in frames

FLOW_EVERY_N_CALLS = 4          # keep the CPU-saving stride: Farneback on
                                 # every frame is unnecessarily expensive
                                 # for a heuristic secondary signal.
FLOW_DOWNSCALE = 0.5            # resize before Farneback -- 4x faster

# Regional ROI landmark index groups (reuses AUCalculator's MediaPipe
# indices, plus a couple of extras, so both classes agree on "where the
# eyebrow/eye/mouth/etc. is").
REGION_LANDMARKS = {
    'brow':  [107, 336, 70, 300, 65, 295],
    'eye':   [159, 145, 386, 374, 33, 263],
    'nose':  [1, 4, 5, 197],
    'mouth': [61, 291, 13, 14, 0, 17],
    'cheek': [116, 345, 187, 411],
}

AU_SPIKE_THRESHOLD = 0.015   # tunable — AU-delta heuristic
FLOW_SPIKE_THRESHOLD = 0.35  # tunable — mean regional flow magnitude
                              # (units: pixels/frame in the downsampled,
                              # half-resolution flow field)


class MicroExpressionDetector:
    def __init__(self):
        self.au_buffer = deque(maxlen=MAX_FRAMES)
        self.prev_gray = None
        self._call_count = 0
        self._last_flow_regions = {f'flow_{r}': 0.0 for r in REGION_LANDMARKS}
        self._last_flow_mean = 0.0
        self._last_flow_max = 0.0

    def update(self, frame_gray, au_values, landmarks=None):
        """
        Call every frame.

        frame_gray : (H, W) grayscale frame.
        au_values  : dict from AUCalculator.compute() (or {}).
        landmarks  : optional (468, 3) pixel-space array from
                     LandmarkExtractor.extract(); if provided, enables
                     the regional flow breakdown. If omitted, regional
                     flow features default to 0.0 (whole-frame flow mean/
                     max are still computed since they don't need
                     landmarks).

        Returns a dict (never None-only-on-spike anymore, since callers
        now want the continuous flow features every call, not just when
        a heuristic spike fires):
            {
              'spike':          bool,   # AU-delta heuristic fired
              'magnitude':      float,  # AU-delta heuristic magnitude
              'flow_mean':      float,
              'flow_max':       float,
              'flow_regions':   {'flow_brow':..., 'flow_eye':..., ...},
              'combined_score': float,  # heuristic AU+flow blend, NOT a
                                         # trained probability
            }
        or None only while the AU buffer is still warming up (fewer than
        MIN_FRAMES samples collected).
        """
        self.au_buffer.append(au_values or {})
        self._call_count += 1

        flow = None
        if (self.prev_gray is not None
                and self._call_count % FLOW_EVERY_N_CALLS == 0):
            small_curr = cv2.resize(frame_gray, (0, 0),
                                     fx=FLOW_DOWNSCALE, fy=FLOW_DOWNSCALE)
            small_prev = cv2.resize(self.prev_gray, (0, 0),
                                     fx=FLOW_DOWNSCALE, fy=FLOW_DOWNSCALE)
            flow = cv2.calcOpticalFlowFarneback(
                small_prev, small_curr,
                None, 0.5, 2, 10, 2, 5, 1.1, 0)
            self._consume_flow(flow, frame_gray.shape, landmarks)

        self.prev_gray = frame_gray

        return self._build_result()

    # ── Optical flow: ACTUALLY used now ─────────────────────────────────

    def _consume_flow(self, flow, full_frame_shape, landmarks):
        """Turns the raw (h', w', 2) Farneback flow field into scalar
        magnitude features and updates the cached region values."""
        mag = np.linalg.norm(flow, axis=2)  # (h', w') flow magnitude
        self._last_flow_mean = float(mag.mean())
        self._last_flow_max = float(mag.max())

        if landmarks is None:
            # No landmarks this call -> can't localize regions; keep the
            # previous regional values rather than zeroing them out on
            # every frame where the face happens not to be detected.
            return

        full_h, full_w = full_frame_shape[:2]
        flow_h, flow_w = mag.shape
        sx = flow_w / float(full_w)
        sy = flow_h / float(full_h)

        for region, idxs in REGION_LANDMARKS.items():
            pts = landmarks[idxs, :2]  # pixel coords in full-frame space
            x1 = max(0, int(pts[:, 0].min() * sx) - 2)
            x2 = min(flow_w, int(pts[:, 0].max() * sx) + 2)
            y1 = max(0, int(pts[:, 1].min() * sy) - 2)
            y2 = min(flow_h, int(pts[:, 1].max() * sy) + 2)

            if x2 <= x1 or y2 <= y1:
                self._last_flow_regions[f'flow_{region}'] = 0.0
                continue

            roi = mag[y1:y2, x1:x2]
            self._last_flow_regions[f'flow_{region}'] = float(roi.mean()) if roi.size else 0.0

    # ── AU-delta heuristic (unchanged logic, kept as a secondary signal) ─

    def _au_delta_magnitude(self):
        if len(self.au_buffer) < MIN_FRAMES:
            return None
        recent = list(self.au_buffer)[-MIN_FRAMES:]
        deltas = []
        for i in range(1, len(recent)):
            for k in recent[i]:
                deltas.append(abs(recent[i][k] - recent[i - 1].get(k, 0)))
        return float(np.mean(deltas)) if deltas else 0.0

    def _build_result(self):
        magnitude = self._au_delta_magnitude()
        if magnitude is None:
            return None

        au_spike = magnitude > AU_SPIKE_THRESHOLD
        flow_spike = self._last_flow_mean > FLOW_SPIKE_THRESHOLD

        # Simple heuristic blend -- NOT a calibrated probability, just a
        # human-inspectable combination of the two cheap signals. The
        # model's trained spotting_head output is the number that should
        # actually drive the event detector / graph once trained.
        combined_score = 0.5 * min(magnitude / (AU_SPIKE_THRESHOLD * 4), 1.0) \
                        + 0.5 * min(self._last_flow_mean / (FLOW_SPIKE_THRESHOLD * 4), 1.0)

        return {
            'spike':          bool(au_spike or flow_spike),
            'magnitude':      round(magnitude, 4),
            'flow_mean':      round(self._last_flow_mean, 4),
            'flow_max':       round(self._last_flow_max, 4),
            'flow_regions':   {k: round(v, 4) for k, v in self._last_flow_regions.items()},
            'combined_score': round(float(np.clip(combined_score, 0.0, 1.0)), 4),
        }
