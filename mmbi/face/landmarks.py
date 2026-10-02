# mmbi/face/landmarks.py
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import urllib.request
from typing import Optional

class LandmarkExtractor:
    """
    Extracts 468/478 facial landmarks using MediaPipe FaceLandmarker Tasks API.
    Implements a deterministic primary-face selection strategy to ensure
    consistent tracking when multiple faces appear in frame.
    """
    def __init__(self, max_faces: int = 4):
        # Path to the modern Tasks face landmarker model
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'face_landmarker.task')
        if not os.path.exists(model_path):
            print(f"Downloading face landmarker model tasks file to {model_path}...")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            try:
                urllib.request.urlretrieve(url, model_path)
                print("Model downloaded successfully.")
            except Exception as e:
                print(f"ERROR: Could not download model file: {e}")
                raise e
            
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=max_faces,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        self.last_primary_center: Optional[np.ndarray] = None

    def _select_primary_face(self, face_landmarks_list):
        """
        Deterministic primary face selection strategy:
        1. Evaluates all detected faces.
        2. Calculates normalized bounding area and distance to frame center.
        3. Prioritizes largest prominent foreground face closest to center.
        4. Applies temporal hysteresis so tracking stays locked to the same individual.
        """
        if len(face_landmarks_list) == 1:
            lm = face_landmarks_list[0]
            pts = np.array([[p.x, p.y] for p in lm])
            self.last_primary_center = np.mean(pts, axis=0)
            return lm

        best_score = -1.0
        best_lm = None
        best_center = None

        for lm in face_landmarks_list:
            pts = np.array([[p.x, p.y] for p in lm])
            min_xy = np.min(pts, axis=0)
            max_xy = np.max(pts, axis=0)
            center = (min_xy + max_xy) / 2.0
            area = (max_xy[0] - min_xy[0]) * (max_xy[1] - min_xy[1])
            
            # Distance to frame center (0.5, 0.5)
            center_dist = np.linalg.norm(center - np.array([0.5, 0.5]))
            score = area * max(0.2, 1.0 - 0.6 * center_dist)

            # Spatial consistency tracking bonus
            if self.last_primary_center is not None:
                track_dist = np.linalg.norm(center - self.last_primary_center)
                if track_dist < 0.15:
                    score += 0.25 * (1.0 - track_dist / 0.15)

            if score > best_score:
                best_score = score
                best_lm = lm
                best_center = center

        if best_center is not None:
            self.last_primary_center = best_center

        return best_lm if best_lm is not None else face_landmarks_list[0]

    def extract(self, frame_rgb, normalized=False):
        if frame_rgb is None:
            return None
        h, w = frame_rgb.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = self.detector.detect(mp_image)
        if not results.face_landmarks:
            return None

        # Select deterministic primary face
        lm = self._select_primary_face(results.face_landmarks)

        if normalized:
            return np.array([[p.x, p.y, p.z] for p in lm])
        else:
            return np.array([[p.x * w, p.y * h, p.z] for p in lm])