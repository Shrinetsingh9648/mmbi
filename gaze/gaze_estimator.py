import numpy as np
import cv2

class GazeEstimator:
    """
    Estimates gaze direction from MediaPipe iris landmarks.
    Refined landmarks 468-472 are the left iris,
    473-477 are the right iris.
    Outputs a normalized gaze vector and an eye-contact score.
    """

    # Eye outline landmark indices (MediaPipe Face Mesh)
    LEFT_EYE   = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    RIGHT_EYE  = [33,  7,   163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    LEFT_IRIS  = [474, 475, 476, 477]
    RIGHT_IRIS = [469, 470, 471, 472]

    def __init__(self):
        self.baseline_left_ratio  = None
        self.baseline_right_ratio = None
        self.calibrated = False

    def _iris_ratio(self, lm, iris_idx, eye_idx):
        """
        Compute where the iris center sits within the eye bounding box.
        Returns (horizontal_ratio, vertical_ratio).
        0.5, 0.5 = looking straight ahead.
        """
        iris_pts = lm[iris_idx, :2]
        eye_pts  = lm[eye_idx,  :2]
        iris_cx  = iris_pts[:, 0].mean()
        iris_cy  = iris_pts[:, 1].mean()
        eye_xmin = eye_pts[:, 0].min()
        eye_xmax = eye_pts[:, 0].max()
        eye_ymin = eye_pts[:, 1].min()
        eye_ymax = eye_pts[:, 1].max()
        h_ratio  = (iris_cx - eye_xmin) / (eye_xmax - eye_xmin + 1e-6)
        v_ratio  = (iris_cy - eye_ymin) / (eye_ymax - eye_ymin + 1e-6)
        return h_ratio, v_ratio

    def calibrate(self, lm):
        """Call once when face is looking straight at camera."""
        lh, lv = self._iris_ratio(lm, self.LEFT_IRIS,  self.LEFT_EYE)
        rh, rv = self._iris_ratio(lm, self.RIGHT_IRIS, self.RIGHT_EYE)
        self.baseline_left_ratio  = (lh, lv)
        self.baseline_right_ratio = (rh, rv)
        self.calibrated = True

    def compute(self, lm):
        """
        Returns dict with:
          gaze_h     : -1 (left) to +1 (right), 0 = center
          gaze_v     : -1 (up)   to +1 (down),  0 = center
          eye_contact: 0.0 to 1.0 (1.0 = direct gaze at camera)
          direction  : string label
        """
        if not self.calibrated:
            self.calibrate(lm)

        lh, lv = self._iris_ratio(lm, self.LEFT_IRIS,  self.LEFT_EYE)
        rh, rv = self._iris_ratio(lm, self.RIGHT_IRIS, self.RIGHT_EYE)

        # Deviation from calibrated baseline
        bl_h, bl_v = self.baseline_left_ratio
        br_h, br_v = self.baseline_right_ratio

        gaze_h = ((lh - bl_h) + (rh - br_h)) / 2.0
        gaze_v = ((lv - bl_v) + (rv - br_v)) / 2.0

        # Scale to -1..+1 range (typical max deviation is ~0.3)
        gaze_h = float(np.clip(gaze_h / 0.3, -1.0, 1.0))
        gaze_v = float(np.clip(gaze_v / 0.3, -1.0, 1.0))

        # Eye contact score: how close to center
        eye_contact = float(1.0 - min(1.0, (abs(gaze_h) + abs(gaze_v)) / 1.5))

        # Direction label
        if abs(gaze_h) < 0.2 and abs(gaze_v) < 0.2:
            direction = "CENTER"
        elif gaze_h < -0.2:
            direction = "LEFT"
        elif gaze_h > 0.2:
            direction = "RIGHT"
        elif gaze_v < -0.2:
            direction = "UP"
        else:
            direction = "DOWN"

        return {
            'gaze_h':      round(gaze_h, 3),
            'gaze_v':      round(gaze_v, 3),
            'eye_contact': round(eye_contact, 3),
            'direction':   direction
        }