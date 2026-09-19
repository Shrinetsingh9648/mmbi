# mmbi/temporal/event_detector.py
"""
Converts a per-frame (smoothed) micro-expression probability stream into
discrete micro-expression events using hysteresis thresholding, so that
N consecutive positive frames become ONE event, not N events.

This module is intentionally pure Python + stdlib only (no torch/cv2/
mediapipe) so it can be exercised by fast unit tests independent of the
rest of the pipeline (see tests/test_event_detector.py) and reused
identically by the live engine and any offline evaluation script.

State machine:

    IDLE --(prob > THRESHOLD_ON)--> IN_EVENT
    IN_EVENT --(prob keeps > THRESHOLD_OFF)--> IN_EVENT (keep accumulating)
    IN_EVENT --(prob < THRESHOLD_OFF)--> IDLE, event emitted

THRESHOLD_ON > THRESHOLD_OFF on purpose (hysteresis): this prevents a
probability wobbling right around a single threshold from re-opening a
new event a moment after the previous one closed.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


DEFAULT_THRESHOLD_ON = 0.50
DEFAULT_THRESHOLD_OFF = 0.30
DEFAULT_MIN_DURATION_FRAMES = 3      # drop shorter spikes as noise
DEFAULT_MAX_DURATION_FRAMES = None   # None = no cap; set e.g. to
                                      # int(0.5 * fps) to force-split
                                      # anything longer than ~500ms


@dataclass
class MicroExpressionEvent:
    event_id: int
    start_frame: int
    end_frame: int
    apex_frame: int
    apex_prob: float
    frame_probs: List[float] = field(default_factory=list)

    start_time: Optional[float] = None
    apex_time: Optional[float] = None
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None

    # Filled in by the caller after the event closes, from the
    # classification/phase/intensity heads run on the apex window.
    emotion_class: Optional[str] = None
    confidence: Optional[float] = None
    intensity: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id":    self.event_id,
            "start_frame": self.start_frame,
            "apex_frame":  self.apex_frame,
            "end_frame":   self.end_frame,
            "start_time":  self.start_time,
            "apex_time":   self.apex_time,
            "end_time":    self.end_time,
            "duration_ms": self.duration_ms,
            "class":       self.emotion_class,
            "confidence":  self.confidence,
            "intensity":   self.intensity,
        }


class MicroExpressionEventDetector:
    """
    Feed it one (frame_index, probability, timestamp) triple per
    processed frame via update(). It returns a closed
    MicroExpressionEvent the instant one is finalized, or None otherwise.

    Call flush() at end-of-stream to force-close any still-open event
    (otherwise a trailing in-progress event is silently lost).
    """

    def __init__(
        self,
        threshold_on: float = DEFAULT_THRESHOLD_ON,
        threshold_off: float = DEFAULT_THRESHOLD_OFF,
        min_duration_frames: int = DEFAULT_MIN_DURATION_FRAMES,
        max_duration_frames: Optional[int] = DEFAULT_MAX_DURATION_FRAMES,
    ):
        if threshold_off >= threshold_on:
            raise ValueError(
                "threshold_off must be strictly less than threshold_on "
                "(hysteresis requires a gap between ON and OFF)"
            )
        self.threshold_on = float(threshold_on)
        self.threshold_off = float(threshold_off)
        self.min_duration_frames = int(min_duration_frames)
        self.max_duration_frames = max_duration_frames

        self._state = "IDLE"
        self._event: Optional[MicroExpressionEvent] = None
        self._next_event_id = 1

        # For debugging / the dashboard's "is anything happening right now" flag
        self.in_event: bool = False

    def reset(self) -> None:
        self._state = "IDLE"
        self._event = None
        self.in_event = False

    def update(
        self,
        frame_index: int,
        probability: float,
        timestamp: Optional[float] = None,
    ) -> Optional[MicroExpressionEvent]:
        """
        Call once per processed frame with the (smoothed) ME probability.
        Returns a finalized MicroExpressionEvent if one just closed on
        this call, else None.

        Note on event boundaries: the frame that first drops BELOW
        threshold_off is treated as the frame that CLOSES the event and
        is not itself counted as part of the event's [start_frame,
        end_frame] range — end_frame is the last frame that was still
        >= threshold_off.
        """
        p = float(probability)

        if self._state == "IDLE":
            if p > self.threshold_on:
                self._state = "IN_EVENT"
                self.in_event = True
                self._event = MicroExpressionEvent(
                    event_id=self._next_event_id,
                    start_frame=frame_index,
                    end_frame=frame_index,
                    apex_frame=frame_index,
                    apex_prob=p,
                    frame_probs=[p],
                    start_time=timestamp,
                    apex_time=timestamp,
                    end_time=timestamp,
                )
            return None

        # self._state == "IN_EVENT"
        assert self._event is not None

        if p < self.threshold_off:
            # This frame is NOT part of the event — it's what closes it.
            return self._close_event()

        # Still part of the event: extend it with this frame.
        self._event.frame_probs.append(p)
        self._event.end_frame = frame_index
        self._event.end_time = timestamp

        if p > self._event.apex_prob:
            self._event.apex_prob = p
            self._event.apex_frame = frame_index
            self._event.apex_time = timestamp

        force_split = (
            self.max_duration_frames is not None
            and (self._event.end_frame - self._event.start_frame + 1)
            >= self.max_duration_frames
        )
        if force_split:
            return self._close_event()

        return None

    def _close_event(self) -> Optional[MicroExpressionEvent]:
        event = self._event
        self._state = "IDLE"
        self.in_event = False
        self._event = None

        if event is None:
            return None

        duration_frames = event.end_frame - event.start_frame + 1

        if duration_frames < self.min_duration_frames:
            # Too short to be a real micro-expression event -> discard,
            # do NOT emit it and do NOT increment the event id counter.
            return None

        if event.start_time is not None and event.end_time is not None:
            event.duration_ms = round((event.end_time - event.start_time) * 1000.0, 1)

        self._next_event_id += 1
        return event

    def flush(self, end_timestamp: Optional[float] = None) -> Optional[MicroExpressionEvent]:
        """Force-close a still-open event at end of stream (e.g. video ended
        while probability was still above threshold_off)."""
        if self._state != "IN_EVENT":
            return None
        if end_timestamp is not None and self._event is not None:
            self._event.end_time = end_timestamp
        return self._close_event()
