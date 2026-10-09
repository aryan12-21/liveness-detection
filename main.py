"""
main.py

Entry point. Runs the liveness pipeline on either a live webcam feed or
a saved video file, using MediaPipe FaceLandmarker (Tasks API, mediapipe >= 1.0)
for landmarks, then feeds those landmarks into BlinkDetector and HeadPoseEstimator.

USAGE:
    python main.py --source webcam
    python main.py --source test_recordings/real_sample_1.mp4 --label real
    python main.py --source test_recordings/photo_spoof_1.mp4 --label spoof

Press 'q' to quit early. On exit, logs one row to results/session_log.csv.

HONEST SCOPE NOTE: this is a prototype for demonstrating understanding
of the building blocks (landmark-based EAR blink detection, solvePnP
head-pose estimation, simple challenge-response logging) -- not a
production anti-spoof system. It has no anti-photo-attack texture
analysis, no depth sensing, and was tested on a small, self-recorded set.
"""

import argparse
import os
import sys
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np

from liveness.blink_detector import BlinkDetector
from liveness.head_pose import HeadPoseEstimator, HeadTurnChallenge
from liveness.logger import SessionLogger, compute_accuracy

# ---------------------------------------------------------------------------
# MediaPipe Tasks API setup (replaces mp.solutions.face_mesh in mediapipe 1.0+)
# ---------------------------------------------------------------------------
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)


def ensure_model():
    """Download the face landmarker model file if not already present."""
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading face landmarker model to '{MODEL_PATH}' ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")


def landmarks_to_pixel_array(face_landmarks, frame_width: int, frame_height: int) -> np.ndarray:
    """Convert MediaPipe's normalized (0-1) landmarks to pixel-space (x, y)."""
    pts = np.array(
        [(lm.x * frame_width, lm.y * frame_height) for lm in face_landmarks],
        dtype=np.float64,
    )
    return pts


def run(source: str, label: str, max_seconds: int = 20) -> None:
    ensure_model()

    is_webcam = source == "webcam"
    cap = cv2.VideoCapture(0 if is_webcam else source)

    if not cap.isOpened():
        print(f"ERROR: could not open source '{source}'", file=sys.stderr)
        sys.exit(1)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    blink_detector = BlinkDetector(ear_threshold=0.21, min_consecutive_frames=2)
    head_pose_estimator = HeadPoseEstimator(frame_width, frame_height)
    head_turn_challenge = HeadTurnChallenge(yaw_threshold_deg=15.0)
    logger = SessionLogger()

    # Build FaceLandmarker using the new Tasks API
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=mp_vision.RunningMode.IMAGE,  # frame-by-frame (VIDEO or IMAGE mode)
    )

    start_time = time.time()

    with mp_vision.FaceLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            if time.time() - start_time > max_seconds:
                break

            frame = cv2.flip(frame, 1) if is_webcam else frame

            # Convert to mediapipe Image format
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            detection_result = landmarker.detect(mp_image)

            if detection_result.face_landmarks:
                face_landmarks = detection_result.face_landmarks[0]  # first face
                landmarks_px = landmarks_to_pixel_array(face_landmarks, frame_width, frame_height)

                blink_result = blink_detector.update(landmarks_px)
                pose_result = head_pose_estimator.estimate(landmarks_px)
                head_turn_challenge.update(pose_result.yaw)

                # --- On-screen debug overlay ---
                cv2.putText(frame, f"EAR: {blink_result.avg_ear:.3f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Blinks: {blink_detector.total_blinks}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Yaw: {pose_result.yaw:.1f} deg", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                turn_status = "PASS" if head_turn_challenge.passed else "waiting..."
                cv2.putText(frame, f"Head-turn check: {turn_status}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No face detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Liveness Prototype (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

    blink_detected = blink_detector.total_blinks > 0
    result = logger.log(
        recording_name=source,
        ground_truth_is_real=(label == "real"),
        blink_detected=blink_detected,
        head_turn_passed=head_turn_challenge.passed,
        total_blinks=blink_detector.total_blinks,
    )

    print("\n--- Session result ---")
    print(f"Blinks detected:     {result.total_blinks}")
    print(f"Head-turn check:     {'PASS' if result.head_turn_passed else 'FAIL'}")
    print(f"Final verdict:       {'LIVE' if result.final_verdict_live else 'NOT LIVE'}")
    print(f"Correct vs label:    {result.correct}")
    print(f"Running accuracy over all logged sessions: {compute_accuracy():.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Liveness detection prototype")
    parser.add_argument("--source", default="webcam",
                         help="'webcam' or a path to a video file")
    parser.add_argument("--label", default="real", choices=["real", "spoof"],
                         help="Ground-truth label for this recording, for accuracy logging")
    parser.add_argument("--max-seconds", type=int, default=20,
                         help="Stop automatically after this many seconds")
    args = parser.parse_args()

    run(args.source, args.label, args.max_seconds)
