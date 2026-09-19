# mmbi/tests/test_event_detector.py
"""
Tests for temporal/event_detector.py — pure Python/stdlib, no torch/cv2
required, so these can run in any environment including CI.

Run with:
    python -m unittest mmbi.tests.test_event_detector -v
or, from inside the mmbi/ package directory:
    python -m unittest tests.test_event_detector -v
"""

import unittest

from mmbi.temporal.event_detector import MicroExpressionEventDetector


def feed(detector, probs, start_frame=0, dt=1.0 / 30):
    """Helper: feed a list of probabilities in, one per simulated frame.
    Returns the list of events emitted (None entries filtered out)."""
    events = []
    t = 0.0
    for i, p in enumerate(probs):
        ev = detector.update(frame_index=start_frame + i, probability=p, timestamp=t)
        if ev is not None:
            events.append(ev)
        t += dt
    return events


class TestNoDetection(unittest.TestCase):
    """Test 1: a stream with no positive frames -> no events."""

    def test_all_below_threshold(self):
        det = MicroExpressionEventDetector(threshold_on=0.5, threshold_off=0.3)
        probs = [0.0, 0.1, 0.2, 0.15, 0.25, 0.1, 0.0, 0.2]
        events = feed(det, probs)
        flushed = det.flush()
        if flushed:
            events.append(flushed)
        self.assertEqual(len(events), 0)


class TestConsecutivePositivesAreOneEvent(unittest.TestCase):
    """Test 2: 10 consecutive positive frames -> exactly 1 event, not 10."""

    def test_ten_positive_frames_one_event(self):
        det = MicroExpressionEventDetector(
            threshold_on=0.5, threshold_off=0.3, min_duration_frames=3
        )
        probs = [0.9] * 10 + [0.1]  # trailing frame closes the event
        events = feed(det, probs)
        self.assertEqual(len(events), 1)
        ev = events[0]
        self.assertEqual(ev.start_frame, 0)
        self.assertEqual(ev.end_frame, 9)
        self.assertEqual(len(ev.frame_probs), 10)


class TestTwoSeparateSpikes(unittest.TestCase):
    """Test 3: two separate ME spikes -> 2 events."""

    def test_two_spikes_two_events(self):
        det = MicroExpressionEventDetector(
            threshold_on=0.5, threshold_off=0.3, min_duration_frames=3
        )
        probs = (
            [0.1, 0.1]
            + [0.8, 0.9, 0.85, 0.7]      # spike 1
            + [0.1, 0.1, 0.1, 0.1]       # gap
            + [0.75, 0.95, 0.6]          # spike 2
            + [0.05]                     # close
        )
        events = feed(det, probs)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].event_id, 1)
        self.assertEqual(events[1].event_id, 2)
        # events must not overlap
        self.assertLess(events[0].end_frame, events[1].start_frame)


class TestShortNoisySpikeFiltered(unittest.TestCase):
    """Test 4: a single-frame noise spike shorter than min_duration is dropped."""

    def test_single_frame_spike_filtered(self):
        det = MicroExpressionEventDetector(
            threshold_on=0.5, threshold_off=0.3, min_duration_frames=3
        )
        probs = [0.1, 0.1, 0.9, 0.1, 0.1, 0.1]  # one frame above ON, then drop
        events = feed(det, probs)
        self.assertEqual(len(events), 0)


class TestApexDetection(unittest.TestCase):
    """Test 5: apex should be the frame with maximum probability inside the event."""

    def test_apex_is_max_probability_frame(self):
        det = MicroExpressionEventDetector(
            threshold_on=0.5, threshold_off=0.3, min_duration_frames=3
        )
        probs = [0.2, 0.3, 0.6, 0.8, 0.7, 0.4, 0.2]
        events = feed(det, probs)
        self.assertEqual(len(events), 1)
        ev = events[0]
        self.assertAlmostEqual(ev.apex_prob, 0.8)
        self.assertEqual(ev.apex_frame, 3)  # index of 0.8 in the list


class TestHysteresisPreventsFlicker(unittest.TestCase):
    """A probability that dips between ON and OFF (but not below OFF)
    must NOT split into multiple events — that's the point of hysteresis."""

    def test_dip_above_off_does_not_split_event(self):
        det = MicroExpressionEventDetector(threshold_on=0.5, threshold_off=0.3,
                                            min_duration_frames=1)
        probs = [0.8, 0.6, 0.35, 0.7, 0.9, 0.2]  # dips to 0.35 (> OFF), never split
        events = feed(det, probs)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].start_frame, 0)
        self.assertEqual(events[0].end_frame, 4)


class TestFlushClosesTrailingEvent(unittest.TestCase):
    def test_flush_emits_open_event(self):
        det = MicroExpressionEventDetector(threshold_on=0.5, threshold_off=0.3,
                                            min_duration_frames=1)
        probs = [0.9, 0.85, 0.7]  # never drops below OFF -> still open at end
        events = feed(det, probs)
        self.assertEqual(len(events), 0)
        flushed = det.flush(end_timestamp=1.0)
        self.assertIsNotNone(flushed)
        self.assertEqual(flushed.end_frame, 2)


class TestInvalidThresholds(unittest.TestCase):
    def test_off_must_be_less_than_on(self):
        with self.assertRaises(ValueError):
            MicroExpressionEventDetector(threshold_on=0.3, threshold_off=0.5)


if __name__ == "__main__":
    unittest.main()
