"""
Engagement Scorer — Day 13
==========================
Tracks engagement trajectory over a rolling window.
Produces:
  - current engagement score   (from FusionLayer)
  - engagement trend           RISING / STABLE / FALLING
  - peak_engagement_time       timestamp of highest engagement
  - session_engagement_avg     mean engagement since session start
  - engagement_events          list of notable transitions logged over time

Why a separate module:
  FusionLayer gives per-frame engagement.
  EngagementScorer adds temporal context — knowing that engagement
  has been falling for 30 seconds is more actionable than knowing
  the current frame value.
"""

from __future__ import annotations
import numpy as np
from collections import deque
import time


TREND_WINDOW_SECS   = 10     # seconds of history to compute trend
SESSION_HISTORY_MAX = 7200   # max frames stored (60 sec × 120 FPS)
TRANSITION_THRESHOLD= 0.15   # engagement delta to log an event


class EngagementScorer:

    def __init__(self, fps: int = 120):
        self.fps                = fps
        self._window_frames     = max(1, int(round(self.fps * TREND_WINDOW_SECS)))
        self._history           = deque(maxlen=SESSION_HISTORY_MAX)
        self._events: list[dict]= []
        self._session_start     = time.time()
        self._peak_score        = 0.0
        self._peak_time         = self._session_start
        self._last_event_score  = 0.5

    def update(self, score: float, label: str, timestamp: float) -> dict:
        """
        Call every frame with the engagement score from FusionLayer.
        Returns a dict with trend, events, and session stats.
        """
        self._history.append((timestamp, score))

        # Update peak
        if score > self._peak_score:
            self._peak_score = score
            self._peak_time  = timestamp

        # Trend calculation using linear regression on recent window
        trend = self._compute_trend()

        # Event logging — log when engagement crosses a notable threshold
        delta = score - self._last_event_score
        if abs(delta) >= TRANSITION_THRESHOLD:
            event = {
                'time':    round(timestamp - self._session_start, 1),
                'score':   round(score, 3),
                'delta':   round(delta, 3),
                'label':   label,
                'type':    'RISE' if delta > 0 else 'DROP'
            }
            self._events.append(event)
            self._last_event_score = score

        # Session average
        scores = [s for _, s in self._history]
        session_avg = float(np.mean(scores)) if scores else 0.5

        return {
            'trend':           trend,
            'session_avg':     round(session_avg, 3),
            'peak_score':      round(self._peak_score, 3),
            'peak_time_ago':   round(timestamp - self._peak_time, 1),
            'session_secs':    round(timestamp - self._session_start, 1),
            'recent_events':   self._events[-5:],   # last 5 transitions
            'total_events':    len(self._events)
        }

    def _compute_trend(self) -> str:
        """
        Fits a linear slope to the engagement history over TREND_WINDOW_SECS.
        Returns 'RISING', 'STABLE', or 'FALLING'.
        """
        n = min(len(self._history), self._window_frames)
        if n < 10:
            return 'STABLE'

        scores = np.array([s for _, s in list(self._history)[-n:]])
        x      = np.arange(len(scores), dtype=np.float32)

        # Fit y = mx + b, extract slope m
        coeffs = np.polyfit(x, scores, 1)
        slope  = coeffs[0]

        # Scale: at 120 FPS, slope per frame × 120 = change per second
        slope_per_sec = slope * self.fps

        if slope_per_sec > 0.005:
            return 'RISING'
        elif slope_per_sec < -0.005:
            return 'FALLING'
        return 'STABLE'

    def session_summary(self) -> dict:
        """Call at end of session for a full summary."""
        if not self._history:
            return {}
        scores    = [s for _, s in self._history]
        duration  = time.time() - self._session_start
        return {
            'duration_secs':     round(duration, 1),
            'avg_engagement':    round(float(np.mean(scores)), 3),
            'max_engagement':    round(float(np.max(scores)),  3),
            'min_engagement':    round(float(np.min(scores)),  3),
            'std_engagement':    round(float(np.std(scores)),  3),
            'peak_time_secs':    round(self._peak_time - self._session_start, 1),
            'total_events':      len(self._events),
            'events':            self._events
        }

    def reset(self):
        self._history.clear()
        self._events.clear()
        self._session_start    = time.time()
        self._peak_score       = 0.0
        self._peak_time        = self._session_start
        self._last_event_score = 0.5