# mmbi/face/landmarks.py
import mediapipe as mp
import numpy as np

class LandmarkExtractor:
    def __init__(self):
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,   # gives iris landmarks
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract(self, frame_rgb):
        results = self.face_mesh.process(frame_rgb)
        if not results.multi_face_landmarks:
            return None
        lm = results.multi_face_landmarks[0].landmark
        h, w = frame_rgb.shape[:2]
        return np.array([[p.x * w, p.y * h, p.z] for p in lm])