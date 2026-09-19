# mmbi/tests/test_smoothing.py
import unittest
from mmbi.temporal.smoothing import EMASmoother


class TestEMASmoother(unittest.TestCase):
    def test_first_value_passes_through_unchanged(self):
        s = EMASmoother(alpha=0.3)
        self.assertAlmostEqual(s.update(0.8), 0.8)

    def test_smooths_towards_new_values(self):
        s = EMASmoother(alpha=0.5)
        s.update(0.0)
        v = s.update(1.0)
        self.assertAlmostEqual(v, 0.5)

    def test_does_not_over_smooth_short_spike(self):
        # A single-frame spike should still visibly raise the smoothed
        # value with alpha=0.3 (not be erased).
        s = EMASmoother(alpha=0.3)
        for _ in range(5):
            s.update(0.0)
        v = s.update(1.0)
        self.assertGreater(v, 0.25)  # visible bump, not erased to ~0

    def test_invalid_alpha_rejected(self):
        with self.assertRaises(ValueError):
            EMASmoother(alpha=0.0)
        with self.assertRaises(ValueError):
            EMASmoother(alpha=1.5)

    def test_reset(self):
        s = EMASmoother(alpha=0.3)
        s.update(0.9)
        s.reset()
        self.assertAlmostEqual(s.update(0.2), 0.2)


if __name__ == "__main__":
    unittest.main()
