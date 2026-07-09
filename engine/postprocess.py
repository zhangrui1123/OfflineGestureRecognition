"""Convert frame-wise class probabilities into gesture events."""

from __future__ import annotations

import numpy as np

from configs.defaults import CONF_THRESHOLD, MAX_GAP_MS, MIN_EVENT_MS, SMOOTH_WIDTH
from utils.labels import BACKGROUND_LABEL, GESTURE_ORDER


def probabilities_to_events(
    probs: np.ndarray,
    fps: float,
    *,
    boundary_probs: np.ndarray | None = None,
    conf_threshold: float = CONF_THRESHOLD,
    min_event_ms: int = MIN_EVENT_MS,
    max_gap_ms: int = MAX_GAP_MS,
    smooth: int = SMOOTH_WIDTH,
) -> list[dict]:
    """Convert frame probabilities ``[T,C]`` to gesture event dictionaries."""
    if probs.size == 0:
        return []
    labels = probs.argmax(axis=1).astype(np.int64)
    confs = probs[np.arange(len(labels)), labels]
    labels = np.where(confs >= conf_threshold, labels, BACKGROUND_LABEL)
    if smooth > 1:
        labels = _median_like_smooth(labels, probs, smooth)
    min_frames = max(1, int(round(min_event_ms * fps / 1000.0)))
    max_gap = max(0, int(round(max_gap_ms * fps / 1000.0)))

    raw: list[dict] = []
    i = 0
    while i < len(labels):
        label = int(labels[i])
        if label == BACKGROUND_LABEL:
            i += 1
            continue
        j = i + 1
        while j < len(labels) and int(labels[j]) == label:
            j += 1
        if j - i >= min_frames:
            start, end = _refine_boundaries(i, j - 1, boundary_probs)
            mean_conf = float(probs[i:j, label].mean())
            raw.append(_event(label, start, end, fps, mean_conf))
        i = j
    return _merge_events(raw, max_gap, fps)


def _median_like_smooth(labels: np.ndarray, probs: np.ndarray, width: int) -> np.ndarray:
    half = width // 2
    out = labels.copy()
    for i in range(len(labels)):
        lo = max(0, i - half)
        hi = min(len(labels), i + half + 1)
        votes: dict[int, float] = {}
        for k in range(lo, hi):
            label = int(labels[k])
            if label == BACKGROUND_LABEL:
                continue
            votes[label] = votes.get(label, 0.0) + float(probs[k, label])
        out[i] = max(votes, key=votes.get) if votes else BACKGROUND_LABEL
    return out


def _refine_boundaries(start: int, end: int, boundary_probs: np.ndarray | None) -> tuple[int, int]:
    if boundary_probs is None or boundary_probs.size == 0:
        return start, end
    n = boundary_probs.shape[1]
    pad = max(3, min(12, (end - start + 1) // 2))
    s0, s1 = max(0, start - pad), min(n, start + pad + 1)
    e0, e1 = max(0, end - pad), min(n, end + pad + 1)
    if s1 > s0:
        start = s0 + int(np.argmax(boundary_probs[0, s0:s1]))
    if e1 > e0:
        end = e0 + int(np.argmax(boundary_probs[1, e0:e1]))
    if end < start:
        end = start
    return start, end


def _event(label: int, start: int, end: int, fps: float, confidence: float) -> dict:
    name = GESTURE_ORDER[label] if 0 <= label < len(GESTURE_ORDER) else str(label)
    return {
        "gesture": name,
        "label": int(label),
        "start_frame": int(start),
        "end_frame": int(end),
        "start_ms": int(round(start * 1000 / fps)),
        "end_ms": int(round(end * 1000 / fps)),
        "duration_ms": int(round((end - start + 1) * 1000 / fps)),
        "mean_conf": round(float(confidence), 4),
    }


def _merge_events(events: list[dict], max_gap: int, fps: float) -> list[dict]:
    if not events:
        return []
    merged = [events[0].copy()]
    for ev in events[1:]:
        prev = merged[-1]
        gap = ev["start_frame"] - prev["end_frame"] - 1
        if ev["label"] == prev["label"] and gap <= max_gap:
            prev_len = prev["end_frame"] - prev["start_frame"] + 1
            cur_len = ev["end_frame"] - ev["start_frame"] + 1
            prev["end_frame"] = ev["end_frame"]
            prev["end_ms"] = ev["end_ms"]
            prev["duration_ms"] = int(round((prev["end_frame"] - prev["start_frame"] + 1) * 1000 / fps))
            prev["mean_conf"] = round(
                (prev["mean_conf"] * prev_len + ev["mean_conf"] * cur_len) / (prev_len + cur_len),
                4,
            )
        else:
            merged.append(ev.copy())
    return merged
