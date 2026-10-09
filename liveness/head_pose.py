"""
head_pose.py

Second liveness signal: head-turn / head-pose check via solvePnP.

WHY THIS AS A SECOND SIGNAL: a static printed photo or a single frozen
frame cannot produce a consistent, continuous head rotation matching a
requested motion ("turn slightly left, then right"). Combining "did the
eyes blink" with "did the head rotate on command" is a basic
challenge-response pair -- still spoofable by a pre-recorded video of a
real person, but harder to spoof than either signal alone.

METHOD: We use OpenCV's solvePnP with a generic (non-person-specific) 3D
face landmark model and a small set of corresponding 2D MediaPipe Face
Mesh points, then decompose the resulting rotation matrix into yaw /
pitch / roll via Euler angles.

HONEST LIMITATION: the 3D reference model is a generic average face, not
calibrated per-person, and we don't have real camera intrinsics (we
approximate focal length from frame width). So the yaw/pitch/roll values
are directionally reliable (good enough to detect "turned left" vs
"facing forward") but not metrically precise -- do not quote them as
exact degrees in an interview without caveats.
"""

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

# A small set of MediaPipe Face Mesh indices with well-known correspondences
# to a generic 3D face model (nose tip, chin, eye corners, mouth corners).
LANDMARK_IDX = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_left_corner": 33,
    "right_eye_right_corner": 263,
    "mouth_left": 61,
    "mouth_right": 291,
}

# Generic 3D model points (in an arbitrary unit, roughly millimeters),
# taken from a canonical average-face reference used widely for solvePnP
# head-pose demos. Origin approx at the nose tip.
MODEL_POINTS_3D = np.array(
    [
        (0.0, 0.0, 0.0),  # nose tip
        (0.0, -330.0, -65.0),  # chin
        (-225.0, 170.0, -135.0),  # left eye left corner
        (225.0, 170.0, -135.0),  # right eye right corner
        (-150.0, -150.0, -125.0),  # mouth left
        (150.0, -150.0, -125.0),  # mouth right
    ],
    dtype=np.float64,
)


@dataclass
class HeadPoseResult:
    yaw: float    # left/right turn, degrees
    pitch: float  # up/down nod, degrees
    roll: float   # tilt, degrees


def _rotation_matrix_to_euler(rmat: np.ndarray) -> Tuple[float, float, float]:
    """Convert a 3x3 rotation matrix to (pitch, yaw, roll) in degrees."""
    sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = np.arctan2(rmat[2, 1], rmat[2, 2])
        yaw = np.arctan2(-rmat[2, 0], sy)
        roll = np.arctan2(rmat[1, 0], rmat[0, 0])
    else:
        pitch = np.arctan2(-rmat[1, 2], rmat[1, 1])
        yaw = np.arctan2(-rmat[2, 0], sy)
        roll = 0.0

    return np.degrees(pitch), np.degrees(yaw), np.degrees(roll)


class HeadPoseEstimator:
    def __init__(self, frame_width: int, frame_height: int):
        # Approximate camera intrinsics: no calibration performed, focal
        # length is estimated from frame width, principal point assumed
        # to be the frame center. This is a standard approximation for
        # webcam demos, not something you'd ship without calibration.
        focal_length = frame_width
        center = (frame_width / 2.0, frame_height / 2.0)
        self.camera_matrix = np.array(
            [
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        self.dist_coeffs = np.zeros((4, 1))  # assume no lens distortion

    def estimate(self, landmarks: np.ndarray) -> HeadPoseResult:
        """
        landmarks: (468, 2) array of pixel-space (x, y) face mesh points.
        """
        image_points = np.array(
            [landmarks[LANDMARK_IDX[name]] for name in [
                "nose_tip", "chin", "left_eye_left_corner",
                "right_eye_right_corner", "mouth_left", "mouth_right",
            ]],
            dtype=np.float64,
        )

        success, rotation_vec, _translation_vec = cv2.solvePnP(
            MODEL_POINTS_3D,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return HeadPoseResult(yaw=0.0, pitch=0.0, roll=0.0)

        rmat, _ = cv2.Rodrigues(rotation_vec)
        pitch, yaw, roll = _rotation_matrix_to_euler(rmat)
        return HeadPoseResult(yaw=yaw, pitch=pitch, roll=roll)


class HeadTurnChallenge:
    """
    Tracks whether the user's yaw has crossed a left AND a right threshold
    within a session -- a crude "turn your head" challenge-response check.
    """

    def __init__(self, yaw_threshold_deg: float = 15.0):
        self.yaw_threshold_deg = yaw_threshold_deg
        self.turned_left = False
        self.turned_right = False

    def update(self, yaw: float) -> None:
        if yaw > self.yaw_threshold_deg:
            self.turned_right = True
        elif yaw < -self.yaw_threshold_deg:
            self.turned_left = True

    @property
    def passed(self) -> bool:
        return self.turned_left and self.turned_right

    def reset(self) -> None:
        self.turned_left = False
        self.turned_right = False
