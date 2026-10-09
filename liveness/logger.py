"""
logger.py

Logs each liveness-check session's result to a CSV so you can go back
and compute an honest accuracy number from your own test recordings,
rather than claiming a number you haven't measured.
"""

import csv
import os
from dataclasses import asdict, dataclass
from datetime import datetime


@dataclass
class SessionResult:
    timestamp: str
    recording_name: str
    ground_truth_is_real: bool  # you label this yourself per test recording
    blink_detected: bool
    head_turn_passed: bool
    total_blinks: int
    final_verdict_live: bool  # blink_detected AND head_turn_passed
    correct: bool  # final_verdict_live == ground_truth_is_real


class SessionLogger:
    def __init__(self, log_path: str = "results/session_log.csv"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._ensure_header()

    def _ensure_header(self) -> None:
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(SessionResult.__annotations__))
                writer.writeheader()

    def log(
        self,
        recording_name: str,
        ground_truth_is_real: bool,
        blink_detected: bool,
        head_turn_passed: bool,
        total_blinks: int,
    ) -> SessionResult:
        final_verdict_live = blink_detected and head_turn_passed
        result = SessionResult(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            recording_name=recording_name,
            ground_truth_is_real=ground_truth_is_real,
            blink_detected=blink_detected,
            head_turn_passed=head_turn_passed,
            total_blinks=total_blinks,
            final_verdict_live=final_verdict_live,
            correct=(final_verdict_live == ground_truth_is_real),
        )
        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(SessionResult.__annotations__))
            writer.writerow(asdict(result))
        return result


def compute_accuracy(log_path: str = "results/session_log.csv") -> float:
    """
    Read back the session log and compute accuracy honestly from
    whatever test recordings you've actually run -- do not hardcode
    a number, generate it from this function and quote whatever it
    outputs, caveats included (small sample size, your own test set only).
    """
    if not os.path.exists(log_path):
        return 0.0

    with open(log_path, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return 0.0

    correct = sum(1 for r in rows if r["correct"] == "True")
    return correct / len(rows)
