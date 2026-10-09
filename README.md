# Face Liveness Detection Prototype

A Python prototype that combines facial landmark signals from MediaPipe with OpenCV head-pose estimation. It can process a webcam stream or a saved video file.

## What it does

- Detects blinks using an Eye Aspect Ratio (EAR) threshold over face landmarks.
- Estimates head pose with OpenCV `solvePnP` and tracks whether the user turns left and right.
- Combines the blink and head-turn checks into a session verdict.
- Logs labeled session results to `results/session_log.csv`; `liveness/logger.py` includes a function to calculate accuracy from the sessions you have logged.

This is a heuristic prototype, not a production anti-spoofing system. It has not been validated on a large test set and this repository makes no accuracy claim. In particular, a replay video of a real person may pass these checks. Head-pose values use approximate camera parameters.

## Requirements and setup

Python and the packages in `requirements.txt` are required. The MediaPipe face-landmarker model is downloaded automatically on the first run and saved as `face_landmarker.task`.

```sh
python -m venv .venv
```

Activate the environment, then install dependencies:

```sh
python -m pip install -r requirements.txt
```

## Run

For a webcam session labeled as a real attempt:

```sh
python main.py --source webcam --label real
```

For a video file, provide its path and the ground-truth label you want recorded:

```sh
python main.py --source path/to/your_clip.mp4 --label spoof
```

Press `q` to stop. Session logs and your test clips are local data; they are ignored by Git and should not be committed, especially if they contain identifiable faces.

## Project files

- `main.py` — webcam/video input and pipeline orchestration.
- `liveness/blink_detector.py` — blink detection.
- `liveness/head_pose.py` — pose estimation and left/right turn challenge.
- `liveness/logger.py` — CSV session logging and accuracy calculation.

