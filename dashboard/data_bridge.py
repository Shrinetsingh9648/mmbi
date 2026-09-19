"""
Data Bridge — Day 19
=====================
Connects the live engine to the Streamlit dashboard.
Uses a shared JSON file as the bridge between the two processes.
Engine writes → dashboard reads → browser displays.

Save to: mmbi/dashboard/data_bridge.py
"""

import json
import os
import time
import threading
from collections import deque
from mmbi.fusion.fusion_layer import BehaviourState

BRIDGE_FILE   = os.path.join(os.path.dirname(__file__), 'live_data.json')
HISTORY_FILE  = os.path.join(os.path.dirname(__file__), 'session_history.json')
MAX_HISTORY   = 500   # max data points kept in history


class DataBridge:
    """
    Attach to engine. Call write(state) every frame.
    Throttled to write at most 10 times per second to avoid disk thrash.
    """

    def __init__(self):
        self._last_write   = 0.0
        self._write_interval = 0.1   # write every 100ms = 10 fps to dashboard
        self._history      = {
            'timestamps':        deque(maxlen=MAX_HISTORY),
            'engagement':        deque(maxlen=MAX_HISTORY),
            'stress':            deque(maxlen=MAX_HISTORY),
            'emotions':          deque(maxlen=MAX_HISTORY),
            'eye_contact':       deque(maxlen=MAX_HISTORY),
            'blink_rate':        deque(maxlen=MAX_HISTORY),
            # NEW: micro-expression probability stream (Phase 12 of the
            # implementation spec). Previously there was NO plumbing at
            # all for this — the dashboard had nothing to plot the
            # requested "ME probability over time" graph from.
            'micro_prob':        deque(maxlen=MAX_HISTORY),
            'micro_smoothed':    deque(maxlen=MAX_HISTORY),
        }
        os.makedirs(os.path.dirname(BRIDGE_FILE), exist_ok=True)

    def write(self, state: BehaviourState,
              eng_info: dict, stress_info: dict,
              micro_info: dict = None):
        """Call every frame from engine. Throttled internally.

        `micro_info` (new, optional for backward compatibility) is the
        dict engine.py builds each frame with keys:
          prob, smoothed, threshold_on, threshold_off, in_event,
          latest_event, events, untrained_heads, warnings
        """
        now = time.time()
        if now - self._last_write < self._write_interval:
            return
        self._last_write = now

        if state is None:
            return

        micro_info = micro_info or {}

        # Update history
        self._history['timestamps'].append(round(now, 2))
        self._history['engagement'].append(round(state.engagement_score, 3))
        self._history['stress'].append(round(state.stress_index, 3))
        self._history['emotions'].append(state.dominant_emotion)
        self._history['eye_contact'].append(round(state.eye_contact_score, 3))
        self._history['blink_rate'].append(state.blink_rate)
        self._history['micro_prob'].append(round(float(micro_info.get('prob', 0.0)), 4))
        self._history['micro_smoothed'].append(round(float(micro_info.get('smoothed', 0.0)), 4))

        # Current state snapshot
        exp = state.explanation or {}
        payload = {
            'timestamp':          round(now, 2),
            'engagement_score':   round(state.engagement_score, 3),
            'engagement_label':   state.engagement_label,
            'stress_index':       round(state.stress_index, 3),
            'stress_label':       state.stress_label,
            'dominant_emotion':   state.dominant_emotion,
            'emotion_confidence': round(state.emotion_confidence, 3),
            'emotion_source':     state.emotion_source,
            'attention_zone':     state.attention_zone,
            'eye_contact_score':  round(state.eye_contact_score, 3),
            'active_aus':         state.active_aus,
            'lean_signal':        state.lean_signal,
            'blink_rate':         state.blink_rate,
            'blink_stress':       state.blink_stress,
            'natural_language':   exp.get('natural_language', ''),
            'conflict_flags':     exp.get('conflict_flags', []),
            'stress_drivers':     stress_info.get('drivers', []),
            'eng_trend':          eng_info.get('trend', 'STABLE'),
            'str_trend':          stress_info.get('trend', 'STABLE'),
            'session_eng_avg':    round(eng_info.get('session_avg', 0), 3),
            'session_str_avg':    round(stress_info.get('session_avg', 0), 3),
            'sustained_stress':   round(stress_info.get('sustained_stress_secs', 0), 1),
            # NEW: micro-expression section. `untrained_heads`/`warnings`
            # are surfaced explicitly so the dashboard can (and must, per
            # audit §14) grey out / label any head that has no trained
            # weights rather than presenting it as a real detection.
            'micro': {
                'prob':            round(float(micro_info.get('prob', 0.0)), 4),
                'smoothed':        round(float(micro_info.get('smoothed', 0.0)), 4),
                'threshold_on':    micro_info.get('threshold_on'),
                'threshold_off':   micro_info.get('threshold_off'),
                'in_event':        bool(micro_info.get('in_event', False)),
                'latest_event':    micro_info.get('latest_event'),
                'events':          micro_info.get('events', []),
                'untrained_heads': micro_info.get('untrained_heads', []),
                'warnings':        micro_info.get('warnings', []),
            },
            'history': {
                'timestamps':     list(self._history['timestamps']),
                'engagement':     list(self._history['engagement']),
                'stress':         list(self._history['stress']),
                'emotions':       list(self._history['emotions']),
                'eye_contact':    list(self._history['eye_contact']),
                'blink_rate':     list(self._history['blink_rate']),
                'micro_prob':     list(self._history['micro_prob']),
                'micro_smoothed': list(self._history['micro_smoothed']),
            }
        }

        try:
            with open(BRIDGE_FILE, 'w') as f:
                json.dump(payload, f)
        except Exception:
            pass

    def clear(self):
        try:
            if os.path.exists(BRIDGE_FILE):
                os.remove(BRIDGE_FILE)
        except Exception:
            pass


def read_bridge() -> dict | None:
    """Called by dashboard to read latest engine data."""
    try:
        if not os.path.exists(BRIDGE_FILE):
            return None
        with open(BRIDGE_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return None