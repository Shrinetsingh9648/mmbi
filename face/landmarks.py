# mmbi/face/landmarks.py
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import urllib.request

class LandmarkExtractor:
    def __init__(self):
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
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def extract(self, frame_rgb, normalized=False):
        h, w = frame_rgb.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = self.detector.detect(mp_image)
        if not results.face_landmarks:
            return None
        lm = results.face_landmarks[0]
        if normalized:
            return np.array([[p.x, p.y, p.z] for p in lm])
        else:
            return np.array([[p.x * w, p.y * h, p.z] for p in lm])