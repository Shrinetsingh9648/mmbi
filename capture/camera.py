# mmbi/capture/camera.py
import cv2

class Camera:
    def __init__(self, index=1, fps=120, width=1920, height=1080):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FOURCC,
                     cv2.VideoWriter_fourcc('M','J','P','G'))
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"Camera initialized: {self.actual_fps} FPS")

    def read(self):
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        self.cap.release()