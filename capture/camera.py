# mmbi/capture/camera.py
import cv2
import time

# NOTE: `cv2.CAP_PROP_FPS` reflects the DRIVER-REPORTED FPS, which is
# frequently just an echo of what was requested and NOT the frame rate
# the hardware can actually sustain (most consumer USB webcams top out
# around 30-60 FPS regardless of what is requested here). Do not trust
# this value for anything timing-sensitive (like micro-expression
# window duration) without also measuring real throughput — see
# `measure_actual_fps()` below, which the engine calls once at startup.


class Camera:
    def __init__(self, index=1, fps=120, width=1920, height=1080):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FOURCC,
                     cv2.VideoWriter_fourcc('M','J','P','G'))
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.requested_fps = fps
        # Driver-reported value -- may or may not reflect real throughput.
        self.reported_fps = self.cap.get(cv2.CAP_PROP_FPS)
        # Filled in by measure_actual_fps(); None until measured.
        self.actual_fps = None
        print(f"Camera initialized. Requested FPS: {self.requested_fps}  "
              f"Driver-reported FPS: {self.reported_fps}")

    def measure_actual_fps(self, n_samples: int = 30) -> float:
        """
        Reads `n_samples` real frames and times the wall-clock throughput.
        This is the number that should actually be used to convert frame
        counts into durations (e.g. for temporal window sizing), NOT
        `requested_fps` and NOT necessarily `reported_fps`.

        Called once by the engine at startup. Safe to call again later
        (e.g. if you suspect the camera has throttled) but costs
        n_samples frame-read latencies each time.
        """
        # Warm up (first few reads after opening a camera are often slow
        # / unrepresentative while auto-exposure settles).
        for _ in range(5):
            self.cap.read()

        t0 = time.perf_counter()
        got = 0
        for _ in range(n_samples):
            ret, _ = self.cap.read()
            if ret:
                got += 1
        elapsed = time.perf_counter() - t0

        if got == 0 or elapsed <= 0:
            # Could not measure (e.g. no camera / mocked in tests) --
            # fall back to the driver-reported value rather than
            # silently assuming the originally requested FPS.
            self.actual_fps = self.reported_fps if self.reported_fps > 0 else 30.0
            print(f"WARNING: could not measure real FPS (got {got}/{n_samples} "
                  f"frames). Falling back to reported FPS: {self.actual_fps}")
            return self.actual_fps

        self.actual_fps = got / elapsed
        print(f"Requested FPS: {self.requested_fps}")
        print(f"Driver-reported FPS: {self.reported_fps}")
        print(f"Actual measured FPS: {self.actual_fps:.1f}")
        print(f"Using temporal FPS: {self.actual_fps:.1f}")
        return self.actual_fps

    def read(self):
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        self.cap.release()