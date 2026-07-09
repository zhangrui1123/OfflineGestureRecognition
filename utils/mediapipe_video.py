"""Shared MediaPipe VIDEO-mode landmark extraction from MP4 files."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from utils.features import KEEP_INDICES, normalize_landmarks
from utils.hand_tracking import detect_video, make_landmarker, resolve_model


@dataclass(frozen=True)
class LandmarkSequence:
    """Normalized palm landmarks plus image-space points for one video."""

    landmarks: np.ndarray
    image_landmarks: np.ndarray
    valid: np.ndarray
    fps: float
    width: int
    height: int


ProgressCallback = Callable[[int, int, float], None]


def make_video_landmarker(
    model_cache_dir: str,
    *,
    delegate: str = "CPU",
) -> object:
    """Create a VIDEO-mode landmarker, falling back to CPU if GPU fails."""
    model_path = resolve_model(model_cache_dir)
    try:
        return make_landmarker(model_path, running_mode="VIDEO", delegate=delegate)
    except Exception as exc:
        if delegate.upper() == "GPU":
            print(f"[warn] MediaPipe GPU delegate unavailable, using CPU: {exc}")
            return make_landmarker(model_path, running_mode="VIDEO", delegate="CPU")
        raise


def extract_landmark_sequence(
    video_path: Path,
    landmarker,
    *,
    hand_side: Optional[str] = None,
    on_progress: Optional[ProgressCallback] = None,
) -> LandmarkSequence:
    """Read *video_path* and return per-frame normalized landmarks."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    landmarks: list[np.ndarray] = []
    image_landmarks: list[np.ndarray] = []
    valid: list[bool] = []
    last = np.zeros((21, 3), dtype=np.float32)
    last_img = np.zeros((21, 2), dtype=np.float32)
    t0 = time.time()
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        timestamp_ms = int(round(frame_idx * 1000 / fps))
        lms = detect_video(frame, landmarker, timestamp_ms, hand_side=hand_side)
        if lms is None:
            landmarks.append(last.copy())
            image_landmarks.append(last_img.copy())
            valid.append(False)
        else:
            raw = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
            last_img = raw[KEEP_INDICES, :2].copy()
            last = normalize_landmarks(raw[KEEP_INDICES])
            landmarks.append(last.copy())
            image_landmarks.append(last_img.copy())
            valid.append(True)
        frame_idx += 1
        if on_progress is not None and frame_idx % 200 == 0:
            speed = frame_idx / max(time.time() - t0, 1e-6)
            on_progress(frame_idx, total, speed)

    cap.release()

    if not landmarks:
        return LandmarkSequence(
            landmarks=np.zeros((0, 21, 3), dtype=np.float32),
            image_landmarks=np.zeros((0, 21, 2), dtype=np.float32),
            valid=np.zeros(0, dtype=bool),
            fps=fps,
            width=width,
            height=height,
        )

    return LandmarkSequence(
        landmarks=np.stack(landmarks, axis=0).astype(np.float32),
        image_landmarks=np.stack(image_landmarks, axis=0).astype(np.float32),
        valid=np.array(valid, dtype=bool),
        fps=fps,
        width=width,
        height=height,
    )


def detect_progress_line(done: int, total: int, speed: float) -> None:
    """Default progress printer used by finger-detect."""
    pct = done / max(total, 1) * 100
    print(f"\r  MediaPipe {done}/{total} ({pct:.0f}%) {speed:.1f} fps", end="", flush=True)
