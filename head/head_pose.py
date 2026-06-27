import cv2
import numpy as np

class HeadPoseEstimator:
    """
    Estimates head yaw, pitch, roll using cv2.solvePnP.
    Uses 6 stable facial landmarks as the 3D reference model.
    All angles are in degrees.

    Yaw   : negative = turned left,   positive = turned right
    Pitch : negative = looking up,    positive = looking down
    Roll  : negative = tilted left,   positive = tilted right
    """

    # Standard 3D face model points (in mm, origin at nose tip)
    MODEL_POINTS_3D = np.array([
        (0.0,    0.0,    0.0),     # Nose tip          → landmark 1
        (0.0,   -330.0, -65.0),    # Chin              → landmark 152
        (-225.0, 170.0, -135.0),   # Left eye corner   → landmark 33
        (225.0,  170.0, -135.0),   # Right eye corner  → landmark 263
        (-150.0,-150.0, -125.0),   # Left mouth corner → landmark 61
        (150.0, -150.0, -125.0),   # Right mouth corner→ landmark 291
    ], dtype=np.float64)

    # Corresponding MediaPipe landmark indices
    LANDMARK_IDS = [1, 152, 33, 263, 61, 291]

    def __init__(self, frame_w=1920, frame_h=1080):
        self.frame_w = frame_w
        self.frame_h = frame_h
        # Approximate camera intrinsic matrix
        focal = frame_w
        cx, cy = frame_w / 2.0, frame_h / 2.0
        self.camera_matrix = np.array([
            [focal, 0,     cx],
            [0,     focal, cy],
            [0,     0,     1 ]
        ], dtype=np.float64)
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    def compute(self, lm):
        """
        lm : (468, 3) numpy array from LandmarkExtractor
        Returns dict: yaw, pitch, roll, lean_signal
        """
        image_points = np.array(
            [lm[i, :2] for i in self.LANDMARK_IDS],
            dtype=np.float64
        )

        success, rvec, tvec = cv2.solvePnP(
            self.MODEL_POINTS_3D,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0, 'lean_signal': 'NEUTRAL'}

        # Convert rotation vector to rotation matrix
        rmat, _ = cv2.Rodrigues(rvec)

        # Decompose into Euler angles
        sy = np.sqrt(rmat[0, 0]**2 + rmat[1, 0]**2)
        singular = sy < 1e-6

        if not singular:
            roll  = np.degrees(np.arctan2( rmat[2, 1], rmat[2, 2]))
            pitch = np.degrees(np.arctan2(-rmat[2, 0], sy))
            yaw   = np.degrees(np.arctan2( rmat[1, 0], rmat[0, 0]))
        else:
            roll  = np.degrees(np.arctan2(-rmat[1, 2], rmat[1, 1]))
            pitch = np.degrees(np.arctan2(-rmat[2, 0], sy))
            yaw   = 0.0

        # Lean signal for engagement detection
        if pitch < -10:
            lean_signal = "FORWARD"      # leaning toward camera = interest
        elif pitch > 15:
            lean_signal = "BACKWARD"     # leaning back = disengaged
        elif abs(yaw) > 20:
            lean_signal = "AWAY"         # turned away
        else:
            lean_signal = "NEUTRAL"

        return {
            'yaw':         round(float(yaw),   2),
            'pitch':       round(float(pitch), 2),
            'roll':        round(float(roll),  2),
            'lean_signal': lean_signal,
            'rvec':        rvec,    # kept for 3D axis drawing
            'tvec':        tvec
        }

    def draw_axes(self, frame, pose_result):
        """Draws 3D coordinate axes on the face for visual verification."""
        if 'rvec' not in pose_result:
            return frame
        axis_pts = np.float32([
            [0, 0, 0], [100, 0, 0], [0, 100, 0], [0, 0, 100]
        ])
        img_pts, _ = cv2.projectPoints(
            axis_pts,
            pose_result['rvec'], pose_result['tvec'],
            self.camera_matrix, self.dist_coeffs
        )
        img_pts = img_pts.astype(int)
        origin = tuple(img_pts[0].ravel())
        cv2.arrowedLine(frame, origin, tuple(img_pts[1].ravel()), (0,0,255),   2)  # X = Red
        cv2.arrowedLine(frame, origin, tuple(img_pts[2].ravel()), (0,255,0),   2)  # Y = Green
        cv2.arrowedLine(frame, origin, tuple(img_pts[3].ravel()), (255,0,0),   2)  # Z = Blue
        return frame