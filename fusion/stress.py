"""
Stress Index Module — Day 14
=============================
Tracks multi-signal stress over time.
Outputs:
  - stress_index         0.0 – 1.0 (from FusionLayer)
  - stress_label         CALM / MILD / MODERATE / HIGH
  - stress_trend         RISING / STABLE / FALLING
  - stress_drivers       list of strings explaining WHAT is driving stress
  - sustained_stress_secs how long stress has been MODERATE+ continuously

Why stress ≠ (1 - engagement):
  A person can be highly engaged AND stressed simultaneously
  (e.g. solving a hard problem intensely).
  Stress signals come from blink rate, micro-expression suppression,
  brow furrow, and head tension — not from attention direction.
"""

from __future__ import annotations
import numpy as np
from collections import deque
import time


TREND_WINDOW_SECS     = 15
SUSTAINED_THRESHOLD   = 0.45     # stress above this = "sustained stress"
SESSION_HISTORY_MAX   = 7200


class StressScorer:

    def __init__(self, fps: int = 120):
        self.fps              = fps
        self._window_frames   = fps * TREND_WINDOW_SECS
        self._history         = deque(maxlen=SESSION_HISTORY_MAX)
        self._session_start   = time.time()
        self._sustained_start = None
        self._sustained_secs  = 0.0

    def update(self,
               stress_score:  float,
               stress_label:  str,
               raw:           dict,
               timestamp:     float) -> dict:
        """
        raw dict must contain: blink, head, au sub-dicts and micro dict.
        Returns enriched stress dict with drivers and trend.
        """
        self._history.append((timestamp, stress_score))

        # Sustained stress tracking
        if stress_score >= SUSTAINED_THRESHOLD:
            if self._sustained_start is None:
                self._sustained_start = timestamp
            self._sustained_secs = timestamp - self._sustained_start
        else:
            self._sustained_start = None
            self._sustained_secs  = 0.0

        # Identify stress drivers
        drivers = self._identify_drivers(stress_score, raw)

        # Trend
        trend = self._compute_trend()

        # Session peak
        scores    = [s for _, s in self._history]
        peak      = float(np.max(scores)) if scores else 0.0
        session_avg = float(np.mean(scores)) if scores else 0.0

        return {
            'stress_index':         round(stress_score,     3),
            'stress_label':         stress_label,
            'trend':                trend,
            'drivers':              drivers,
            'sustained_stress_secs':round(self._sustained_secs, 1),
            'session_peak':         round(peak,         3),
            'session_avg':          round(session_avg,  3),
        }

    def _identify_drivers(self, score: float, raw: dict) -> list[str]:
        """
        Returns a plain-language list of what is causing the stress signal.
        Each driver has a threshold check to avoid false positives.
        """
        drivers = []
        blink   = raw.get('blink', {})
        head    = raw.get('head',  {})
        au      = raw.get('au',    {})
        micro   = raw.get('micro', {})

        # Blink signals
        blink_stress = blink.get('stress_signal', 'NORMAL')
        if blink_stress == 'HIGH_STRESS':
            drivers.append("High blink rate (>30/min)")
        elif blink_stress == 'ELEVATED':
            drivers.append("Elevated blink rate (>25/min)")
        elif blink_stress == 'STARING':
            drivers.append("Fixed stare (very low blink rate)")

        # EAR
        ear = float(blink.get('ear', 0.25))
        if ear < 0.18:
            drivers.append("Eyes partially closed (fatigue/stress)")

        # Brow furrow AU4
        au4 = abs(float(au.get('AU4', 0.0))) * 20
        if au4 > 0.4:
            drivers.append("Brow furrowed (AU4 active)")

        # Lip corner depressor AU15
        au15 = abs(float(au.get('AU15', 0.0))) * 20
        if au15 > 0.3:
            drivers.append("Lip corners down (AU15 — distress signal)")

        # Head tension (large roll)
        roll = abs(float(head.get('roll', 0.0)))
        if roll > 15:
            drivers.append(f"Head tilt {roll:.0f}° (tension/discomfort)")

        # Micro-expression suppression
        if micro and micro.get('magnitude', 0) > 0.015:
            drivers.append(f"Micro-expression spike (mag={micro['magnitude']:.3f})")

        if not drivers and score > 0.3:
            drivers.append("Combined signal elevation (no single dominant driver)")

        return drivers

    def _compute_trend(self) -> str:
        n = min(len(self._history), self._window_frames)
        if n < 10:
            return 'STABLE'
        scores = np.array([s for _, s in list(self._history)[-n:]])
        x      = np.arange(len(scores), dtype=np.float32)
        slope  = np.polyfit(x, scores, 1)[0] * self.fps
        if slope > 0.004:   return 'RISING'
        if slope < -0.004:  return 'FALLING'
        return 'STABLE'

    def session_summary(self) -> dict:
        if not self._history:
            return {}
        scores   = [s for _, s in self._history]
        duration = time.time() - self._session_start
        high_stress_frames = sum(1 for s in scores if s >= SUSTAINED_THRESHOLD)
        return {
            'duration_secs':       round(duration, 1),
            'avg_stress':          round(float(np.mean(scores)), 3),
            'peak_stress':         round(float(np.max(scores)),  3),
            'high_stress_pct':     round(high_stress_frames / max(len(scores), 1), 3),
            'max_sustained_secs':  round(self._sustained_secs, 1)
        }

    def reset(self):
        self._history.clear()
        self._session_start   = time.time()
        self._sustained_start = None
        self._sustained_secs  = 0.0