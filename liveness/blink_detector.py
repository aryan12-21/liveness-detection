"""
blink_detector.py

Blink detection via Eye Aspect Ratio (EAR) on MediaPipe Face Mesh landmarks.

WHY EAR: the eye is roughly an ellipse when open. EAR compares the vertical
eyelid distance to the horizontal eye width. When the eye closes, the
vertical distances collapse toward zero while the horizontal width stays
roughly constant, so EAR drops sharply. This is the same idea as the
classic Soukupova & Cech (2016) dlib-based approach, just re-pointed at
MediaPipe's 468-point face mesh instead of dlib's 68-point model.

HONESTY NOTE (say this in the interview if asked): EAR thresholding is a
simple geometric heuristic, not a learned liveness model. It is fooled by
a printed photo with a physically cut-out eye region, or a video replay of
someone blinking. It is a reasonable *first signal* to combine with others
(e.g. head-pose challenge-response), not a production-grade anti-spoof
system on its own.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

# MediaPipe Face Mesh landmark indices for the 6-point EAR eye contour.
# These are a widely-used subset of the 468 mesh points that approximate
# the classic 6-landmark eye outline (p1..p6) used in the original EAR paper.
# Order per eye: [left_corner, top_1, top_2, right_corner, bottom_2, bottom_1]
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]


def _euclidean(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.linalg.norm(p1 - p2))


def eye_aspect_ratio(eye_points: List[np.ndarray]) -> float:
    """
    Compute EAR for a single eye given its 6 (x, y) landmark points,
    ordered [p1, p2, p3, p4, p5, p6] as in Soukupova & Cech:

        p1 --- p4   (horizontal: eye corners)
         \\   /
          p2-p6      (vertical pairs)
          p3-p5

    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    """
    p1, p2, p3, p4, p5, p6 = eye_points
    vertical_1 = _euclidean(p2, p6)
    vertical_2 = _euclidean(p3, p5)
    horizontal = _euclidean(p1, p4)
    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


@dataclass
class BlinkResult:
    left_ear: float
    right_ear: float
    avg_ear: float
    eyes_closed: bool
    blink_completed: bool  # True on the frame a blink is confirmed (close -> open)


class BlinkDetector:
    """
    Stateful blink detector. Feed it landmarks frame-by-frame; it tracks
    EAR over time and flags a completed blink once the eye has gone
    below threshold for a minimum number of consecutive frames and then
    reopened. Requiring consecutive low-EAR frames avoids false positives
    from single-frame landmark jitter.
    """

    def __init__(
        self,
        ear_threshold: float = 0.21,
        min_consecutive_frames: int = 2,
    ):
        # NOTE ON THRESHOLD: 0.21 is a commonly-cited starting point in EAR
        # literature, but it is camera/resolution/face-angle dependent.
        # In practice we calibrated this against our own test recordings
        # (see calibrate_threshold.py) rather than trusting the default
        # blindly -- be ready to say what value you measured and why.
        self.ear_threshold = ear_threshold
        self.min_consecutive_frames = min_consecutive_frames
        self._closed_frame_count = 0
        self.total_blinks = 0

    def update(self, landmarks: np.ndarray) -> BlinkResult:
        """
        landmarks: (468, 2) array of pixel-space (x, y) face mesh points
        for the current frame.
        """
        left_eye = [landmarks[i] for i in LEFT_EYE_IDX]
        right_eye = [landmarks[i] for i in RIGHT_EYE_IDX]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        eyes_closed = avg_ear < self.ear_threshold
        blink_completed = False

        if eyes_closed:
            self._closed_frame_count += 1
        else:
            if self._closed_frame_count >= self.min_consecutive_frames:
                blink_completed = True
                self.total_blinks += 1
            self._closed_frame_count = 0

        return BlinkResult(
            left_ear=left_ear,
            right_ear=right_ear,
            avg_ear=avg_ear,
            eyes_closed=eyes_closed,
            blink_completed=blink_completed,
        )
