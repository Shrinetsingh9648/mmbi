# mmbi/tests/test_sliding_window.py
"""
Test 6 from the spec: verify that querying TemporalBuffer every frame
produces a true overlapping sliding window (window 1, window 2, window 3, ...)
rather than the old buggy pattern (window 1, wait N frames, window 2, ...).

Only depends on numpy (via temporal/buffer.py) — no torch/cv2 required.
"""

import unittest
import numpy as np

from mmbi.temporal.buffer import TemporalBuffer, FEATURE_KEYS, FEATURE_DIM


def push_dummy_frame(buf: TemporalBuffer, frame_idx: int):
    # Fill every feature with the frame index so we can identify which
    # frames ended up in a returned window just by reading the values.
    au = {k: float(frame_idx) for k in FEATURE_KEYS}
    buf.push(au_dict=au, gaze_dict={}, head_dict={}, blink_dict={})


class TestTrueSlidingWindow(unittest.TestCase):
    def test_overlapping_windows_every_frame(self):
        window_len = 5
        buf = TemporalBuffer()

        # Push enough frames to fill the window once.
        for i in range(window_len):
            push_dummy_frame(buf, i)

        self.assertTrue(buf.is_ready(window_len))

        # Simulate "one prediction per processed frame": for each new
        # frame pushed, immediately query a fresh window (this is the
        # fixed engine.py behaviour — no frame_count % 60 gate).
        windows = []
        for i in range(window_len, window_len + 10):
            push_dummy_frame(buf, i)
            w = buf.get_window(window_len)
            self.assertIsNotNone(w)
            windows.append(w[:, 0].copy())  # first feature column identifies frame idx

        # Consecutive windows must differ (they are NOT the same window
        # repeated, and they are NOT separated by a long gap) — each
        # window should be the previous one shifted by exactly one frame.
        for a, b in zip(windows[:-1], windows[1:]):
            # b's first (window_len - 1) entries == a's last (window_len - 1) entries
            np.testing.assert_array_equal(b[:-1], a[1:])
            # exactly one new frame introduced per step
            self.assertEqual(b[-1], a[-1] + 1)

        # And we got exactly one window per pushed frame (10 pushes -> 10
        # windows) — i.e. ~1 prediction per processed frame, not 1 every
        # `window_len` or `60` frames.
        self.assertEqual(len(windows), 10)

    def test_old_buggy_pattern_would_have_gaps(self):
        """Demonstrates what the OLD engine.py bug looked like, as a
        negative example: querying only every `interval` frames produces
        non-overlapping (or widely-gapped) windows -- this is the pattern
        the fix removes."""
        window_len = 5
        interval = 60
        buf = TemporalBuffer()

        predictions = 0
        for i in range(200):
            push_dummy_frame(buf, i)
            if buf.is_ready(window_len) and i % interval == 0:
                buf.get_window(window_len)
                predictions += 1

        # Over 200 frames with interval=60, the buggy pattern predicts
        # only ~3-4 times instead of ~196 times (200 - window_len + 1).
        self.assertLess(predictions, 10)


if __name__ == "__main__":
    unittest.main()
