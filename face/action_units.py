# mmbi/face/action_units.py
import numpy as np

class AUCalculator:
    """Computes 12 core AUs from 468 MediaPipe landmarks.
    All distances are normalized by face width so face size
    does not affect the output."""

    # MediaPipe landmark indices
    IDX = {
        'left_brow_inner': 107,  'right_brow_inner': 336,
        'left_brow_outer': 70,   'right_brow_outer': 300,
        'left_eye_top':    159,  'left_eye_bot':     145,
        'right_eye_top':   386,  'right_eye_bot':    374,
        'nose_tip':        1,
        'lip_left':        61,   'lip_right':        291,
        'lip_top':         13,   'lip_bot':          14,
        'left_cheek':      116,  'right_cheek':      345,
        'jaw_left':        234,  'jaw_right':        454,
    }

    def __init__(self):
        self.baseline = None

    def _dist(self, lm, a, b):
        return np.linalg.norm(lm[a, :2] - lm[b, :2])

    def calibrate(self, lm):
        """Call once on neutral face to set baseline."""
        self.baseline = self._compute_raw(lm)
        self.face_w = self._dist(lm,
                                  self.IDX['jaw_left'],
                                  self.IDX['jaw_right'])

    def compute(self, lm):
        if self.baseline is None:
            self.calibrate(lm)
        raw = self._compute_raw(lm)
        norm = {}
        for k in raw:
            base = self.baseline[k]
            norm[k] = (raw[k] - base) / (self.face_w + 1e-6)
        return norm

    def _compute_raw(self, lm):
        i = self.IDX
        return {
            'AU1':  self._dist(lm, i['left_brow_inner'],  i['left_eye_top']),
            'AU2':  self._dist(lm, i['left_brow_outer'],  i['left_eye_top']),
            'AU4':  self._dist(lm, i['left_brow_inner'],  i['right_brow_inner']),
            'AU5':  self._dist(lm, i['left_eye_top'],     i['left_eye_bot']),
            'AU6':  self._dist(lm, i['left_cheek'],       i['left_eye_bot']),
            'AU12': self._dist(lm, i['lip_left'],         i['lip_right']),
            'AU15': self._dist(lm, i['lip_left'],         i['lip_bot']),
            'AU17': self._dist(lm, i['lip_top'],          i['lip_bot']),
            'AU25': self._dist(lm, i['lip_top'],          i['lip_bot']),
            'AU26': self._dist(lm, i['nose_tip'],         i['lip_top']),
        }