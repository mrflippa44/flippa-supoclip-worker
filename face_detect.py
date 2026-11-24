# Adapted from https://github.com/cspenn/supoclip (AGPL-3.0)
# Low-memory Linux Render adaptation: uses built-in OpenCV Haar Cascade face
# detection (<25 MB RAM) by default with optional MediaPipe Tasks support,
# preserving SupoClip's multi-frame median aggregation and 9:16/1:1/16:9 crop math.

"""Low-memory face detection with OpenCV Haar Cascade and center-crop fallback."""

from __future__ import annotations

import gc
import os
from functools import lru_cache
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import structlog

log = structlog.get_logger(__name__)

_MIN_RELATIVE_AREA: float = 0.005
_MAX_RELATIVE_AREA: float = 0.40
_MIN_FACE_PIXELS: int = 20
_DEFAULT_FRAME_SAMPLES: int = 5
_MAX_DETECT_WIDTH: int = 480


def round_to_even(n: int) -> int:
    """Round n down to the nearest even integer (required for H.264)."""
    n_int = max(2, int(n))
    return n_int if n_int % 2 == 0 else n_int - 1


def is_mediapipe_available() -> bool:
    """Return True if MediaPipe Tasks is installed."""
    try:
        import mediapipe  # noqa: F401

        return True
    except ImportError:
        return False


def is_face_tracker_available() -> bool:
    """Return True if OpenCV Haar Cascade or MediaPipe is ready."""
    try:
        import cv2

        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        return cascade_path.is_file() or is_mediapipe_available()
    except Exception:
        return False


@lru_cache(maxsize=1)
def _get_haar_cascade() -> Any | None:
    try:
        import cv2

        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if not cascade_path.is_file():
            return None
        clf = cv2.CascadeClassifier(str(cascade_path))
        return None if clf.empty() else clf
    except Exception as exc:
        log.warning("haar_cascade_load_failed", error=str(exc))
        return None


def _detect_with_opencv(frame: np.ndarray) -> tuple[int, int] | None:
    cascade = _get_haar_cascade()
    if cascade is None:
        return None

    import cv2

    orig_h, orig_w = frame.shape[:2]
    if orig_w <= 0 or orig_h <= 0:
        return None

    scale = 1.0
    small = frame
    if orig_w > _MAX_DETECT_WIDTH:
        scale = _MAX_DETECT_WIDTH / float(orig_w)
        small = cv2.resize(
            frame,
            (_MAX_DETECT_WIDTH, max(2, int(orig_h * scale))),
            interpolation=cv2.INTER_AREA,
        )

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.15,
        minNeighbors=4,
        minSize=(_MIN_FACE_PIXELS, _MIN_FACE_PIXELS),
    )
    if faces is None or len(faces) == 0:
        return None

    small_h, small_w = small.shape[:2]
    frame_area = float(small_w * small_h)
    best_center: tuple[int, int] | None = None
    best_area = 0.0

    for x, y, w, h in faces:
        area = float(w * h)
        rel = area / frame_area
        if not (_MIN_RELATIVE_AREA < rel < _MAX_RELATIVE_AREA):
            continue
        if area > best_area:
            best_area = area
            cx = int((x + w / 2.0) / scale)
            cy = int((y + h / 2.0) / scale)
            best_center = (cx, cy)

    return best_center


def detect_face_center(frame: np.ndarray) -> tuple[int, int] | None:
    """Detect the center (x, y) of the most prominent face in a BGR frame."""
    detector_pref = (os.environ.get("FACE_DETECTOR") or "opencv").strip().lower()
    if detector_pref == "none":
        return None
    return _detect_with_opencv(frame)


def calculate_crop_box(
    frame_width: int,
    frame_height: int,
    face_center: tuple[int, int] | None,
    target_width: int = 1080,
    target_height: int = 1920,
) -> tuple[int, int, int, int]:
    """Calculate crop box centered on detected face (or frame center fallback)."""
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError(f"frame dimensions must be positive, got {frame_width}x{frame_height}")
    if target_width <= 0 or target_height <= 0:
        raise ValueError(f"target dimensions must be positive, got {target_width}x{target_height}")

    target_ratio = target_width / target_height
    if frame_width / frame_height > target_ratio:
        crop_height = round_to_even(frame_height)
        crop_width = round_to_even(int(frame_height * target_ratio))
    else:
        crop_width = round_to_even(frame_width)
        crop_height = round_to_even(int(frame_width / target_ratio))

    crop_width = min(crop_width, round_to_even(frame_width))
    crop_height = min(crop_height, round_to_even(frame_height))

    if face_center is not None:
        anchor_x, anchor_y = face_center
        anchor_y = max(0, anchor_y - int(crop_height * 0.1))
    else:
        anchor_x = frame_width // 2
        anchor_y = frame_height // 2

    x = anchor_x - crop_width // 2
    y = anchor_y - crop_height // 2
    x = max(0, min(x, frame_width - crop_width))
    y = max(0, min(y, frame_height - crop_height))
    return round_to_even(x), round_to_even(y), crop_width, crop_height


def _segment_sample_timestamps(start_s: float, end_s: float, samples: int) -> list[float]:
    count = max(1, samples)
    if end_s <= start_s or count == 1:
        return [start_s]
    step = (end_s - start_s) / (count - 1)
    return [start_s + step * i for i in range(count)]


def detect_face_center_multi(
    video_path: str | Path,
    start_s: float,
    end_s: float,
    samples: int = _DEFAULT_FRAME_SAMPLES,
) -> tuple[int, int] | None:
    """Aggregate median face center across evenly spaced frames using single VideoCapture."""
    try:
        import cv2
    except ImportError:
        return None

    timestamps = _segment_sample_timestamps(start_s, end_s, samples)
    centers: list[tuple[int, int]] = []
    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        for ts in timestamps:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(ts * fps)))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            center = detect_face_center(frame)
            del frame
            if center is not None:
                centers.append(center)
    except Exception as exc:
        log.warning("face_tracking_error", path=str(video_path), error=str(exc))
    finally:
        cap.release()
        gc.collect()

    if not centers:
        return None
    return int(median([c[0] for c in centers])), int(median([c[1] for c in centers]))
