"""Offline video gesture event detection.

The command extracts MediaPipe hand landmarks in VIDEO mode, runs the dense
segmentation model over the whole sequence, and writes gesture events with
start/end times.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from utils.cli import (
    add_delegate_arg,
    add_hand_side_arg,
    add_model_cache_arg,
    default_out_json,
    default_results_dir,
    ensure_utf8_console,
    make_parser,
    print_event_table,
    print_next_steps_detect,
    resolve_checkpoint,
    save_json,
    select_device,
)
from configs.defaults import (
    CONF_THRESHOLD,
    DETECT_CHUNK_LEN,
    DETECT_OVERLAP,
    MAX_GAP_MS,
    MIN_EVENT_MS,
    SMOOTH_WIDTH,
)
from utils.features import build_motion_features
from utils.hand_tracking import (
    HAND_CONNECTIONS,
    HIGHLIGHT_COLORS,
)
from utils.labels import GESTURE_ZH, LABEL_NAMES, NUM_CLASSES
from utils.mediapipe_video import (
    detect_progress_line,
    extract_landmark_sequence,
    make_video_landmarker,
)
from engine.model import GestureSegmenter
from engine.postprocess import probabilities_to_events
from utils.video_io import make_writer


COLORS = [
    (50, 230, 80),
    (30, 140, 255),
    (255, 210, 50),
    (200, 80, 255),
    (50, 200, 255),
    (255, 80, 80),
    (100, 100, 100),
]


def extract_video_features(
    video_path: Path,
    *,
    model_cache_dir: str,
    hand_side: str | None,
    delegate: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int, int]:
    lmkr = make_video_landmarker(model_cache_dir, delegate=delegate)
    try:
        seq = extract_landmark_sequence(
            video_path,
            lmkr,
            hand_side=hand_side,
            on_progress=detect_progress_line,
        )
    finally:
        if lmkr is not None:
            lmkr.close()
    print(f"\r  MediaPipe done: {len(seq.landmarks)} frames.                     ")
    features = build_motion_features(seq.landmarks, seq.valid)
    return features, seq.image_landmarks, seq.valid, seq.fps, seq.width, seq.height


@torch.no_grad()
def predict_probabilities(
    model: GestureSegmenter,
    features: np.ndarray,
    device: torch.device,
    *,
    chunk_len: int,
    overlap: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(features)
    if n == 0:
        return np.zeros((0, NUM_CLASSES), dtype=np.float32), np.zeros((2, 0), dtype=np.float32)
    c_expected = model.input_channels
    if features.shape[2] < c_expected:
        pad = np.zeros((*features.shape[:2], c_expected - features.shape[2]), dtype=np.float32)
        features = np.concatenate([features, pad], axis=2)
    elif features.shape[2] > c_expected:
        features = features[:, :, :c_expected]

    stride = max(1, chunk_len - overlap)
    logits_sum = np.zeros((n, NUM_CLASSES), dtype=np.float32)
    boundary_sum = np.zeros((2, n), dtype=np.float32)
    counts = np.zeros(n, dtype=np.float32)
    for start in range(0, n, stride):
        end = min(n, start + chunk_len)
        length = end - start
        chunk = np.zeros((chunk_len, 21, c_expected), dtype=np.float32)
        chunk[:length] = features[start:end]
        x = torch.from_numpy(chunk).permute(2, 0, 1).unsqueeze(0).to(device)
        logits, boundary_logits, _ = model(x)
        probs = torch.softmax(logits[:, :, :length], dim=1).squeeze(0).T.cpu().numpy()
        b_probs = torch.sigmoid(boundary_logits[:, :, :length]).squeeze(0).cpu().numpy()
        logits_sum[start:end] += probs
        boundary_sum[:, start:end] += b_probs
        counts[start:end] += 1.0
        if end >= n:
            break
    counts = np.maximum(counts, 1.0)
    return logits_sum / counts[:, None], boundary_sum / counts[None, :]


def load_model(checkpoint: Path, device: torch.device) -> GestureSegmenter:
    ckpt = torch.load(checkpoint, map_location=device)
    args = ckpt.get("args", {})
    model = GestureSegmenter(
        input_channels=int(args.get("input_channels", 12)),
        hidden_dim=int(args.get("hidden_dim", 128)),
        temporal_channels=int(args.get("temporal_channels", 128)),
        temporal_layers=int(args.get("temporal_layers", 6)),
        temporal_stages=int(args.get("temporal_stages", 2)),
        dropout=float(args.get("dropout", 0.25)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def _draw_panel(frame: np.ndarray, alpha: float = 0.72) -> None:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 118), (18, 18, 18), -1)
    cv2.rectangle(overlay, (0, h - 178), (w, h), (18, 18, 18), -1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


def _draw_skeleton(frame: np.ndarray, pts_norm: np.ndarray, valid: bool) -> None:
    if pts_norm.size == 0:
        return
    h, w = frame.shape[:2]
    pts = np.round(pts_norm * np.array([w, h], dtype=np.float32)).astype(np.int32)
    line_color = (230, 230, 230) if valid else (80, 80, 220)
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, tuple(pts[a]), tuple(pts[b]), line_color, 2, cv2.LINE_AA)
    for i, pt in enumerate(pts):
        color = HIGHLIGHT_COLORS.get(i, (180, 180, 180))
        if not valid:
            color = (80, 80, 220)
        radius = 7 if i in HIGHLIGHT_COLORS else 4
        cv2.circle(frame, tuple(pt), radius, color, -1, cv2.LINE_AA)
        if i in HIGHLIGHT_COLORS:
            cv2.circle(frame, tuple(pt), radius + 2, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_prob_bars(frame: np.ndarray, probs: np.ndarray, x: int, y: int, width: int) -> None:
    for i, (name, prob) in enumerate(zip(LABEL_NAMES, probs)):
        yy = y + i * 22
        color = COLORS[i]
        cv2.rectangle(frame, (x, yy), (x + width, yy + 14), (48, 48, 48), -1)
        cv2.rectangle(frame, (x, yy), (x + int(width * float(prob)), yy + 14), color, -1)
        cv2.putText(
            frame,
            f"{name:<18} {float(prob) * 100:5.1f}%",
            (x + width + 10, yy + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )


def _draw_boundary_strip(
    frame: np.ndarray,
    idx: int,
    boundary_probs: np.ndarray,
    events: list[dict],
    fps: float,
) -> None:
    h, w = frame.shape[:2]
    left, right = 20, w - 20
    y = h - 40
    cv2.rectangle(frame, (left, y), (right, y + 16), (45, 45, 45), -1)
    window = 150
    lo = max(0, idx - window)
    hi = min(boundary_probs.shape[1], idx + window + 1)
    if hi > lo:
        values = np.maximum(boundary_probs[0, lo:hi], boundary_probs[1, lo:hi])
        for k, value in enumerate(values):
            x = left + int((right - left) * k / max(1, len(values) - 1))
            color = (60, 210, 255) if boundary_probs[0, lo + k] >= boundary_probs[1, lo + k] else (255, 130, 80)
            cv2.line(frame, (x, y + 16), (x, y + 16 - int(16 * float(value))), color, 1)
    center_x = left + (right - left) // 2
    cv2.line(frame, (center_x, y - 4), (center_x, y + 20), (255, 255, 255), 2, cv2.LINE_AA)
    for ev in events:
        if lo <= ev["start_frame"] <= hi:
            x = left + int((right - left) * (ev["start_frame"] - lo) / max(1, hi - lo))
            cv2.circle(frame, (x, y - 5), 4, COLORS[ev["label"]], -1, cv2.LINE_AA)
        if lo <= ev["end_frame"] <= hi:
            x = left + int((right - left) * (ev["end_frame"] - lo) / max(1, hi - lo))
            cv2.circle(frame, (x, y + 24), 4, COLORS[ev["label"]], -1, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"boundary strip +/- {window / max(fps, 1e-6):.1f}s   cyan=start orange=end",
        (left, y - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )


def _topk_text(row: np.ndarray, k: int = 3) -> str:
    ids = np.argsort(row)[::-1][:k]
    return "  ".join(f"{LABEL_NAMES[i]}={row[i] * 100:.1f}%" for i in ids)


def write_overlay_video(
    video_path: Path,
    out_path: Path,
    events: list[dict],
    probs: np.ndarray,
    boundary_probs: np.ndarray,
    image_landmarks: np.ndarray,
    valid: np.ndarray,
    fps: float,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = make_writer(out_path, fps, width, height)
    event_by_frame: list[dict | None] = [None] * len(probs)
    for ev in events:
        for i in range(max(0, ev["start_frame"]), min(len(probs), ev["end_frame"] + 1)):
            event_by_frame[i] = ev
    idx = 0
    while idx < len(probs):
        ok, frame = cap.read()
        if not ok:
            break
        label = int(probs[idx].argmax())
        conf = float(probs[idx, label])
        color = COLORS[label]
        active = event_by_frame[idx]
        valid_now = bool(valid[idx]) if idx < len(valid) else False
        _draw_panel(frame)
        if idx < len(image_landmarks):
            _draw_skeleton(frame, image_landmarks[idx], valid_now)

        title = LABEL_NAMES[label] if active is None else active["gesture"]
        zh = GESTURE_ZH.get(title, title)
        status = "MP_OK" if valid_now else "MP_MISS"
        cv2.putText(frame, f"#{idx:05d}  t={idx / max(fps, 1e-6):7.2f}s  {status}",
                    (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (235, 235, 235), 2, cv2.LINE_AA)
        cv2.putText(frame, f"PRED {title} / {zh}  {conf * 100:.1f}%",
                    (18, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.86, color, 2, cv2.LINE_AA)
        if active is not None:
            cv2.putText(
                frame,
                f"EVENT {active['start_ms']}ms - {active['end_ms']}ms  duration={active['duration_ms']}ms  mean={active['mean_conf'] * 100:.1f}%",
                (18, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(frame, "EVENT none", (18, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (170, 170, 170), 1, cv2.LINE_AA)

        start_p = float(boundary_probs[0, idx]) if boundary_probs.size else 0.0
        end_p = float(boundary_probs[1, idx]) if boundary_probs.size else 0.0
        cv2.putText(frame, f"top3: {_topk_text(probs[idx])}",
                    (width - 760, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(frame, f"boundary_start={start_p:.3f}  boundary_end={end_p:.3f}",
                    (width - 760, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (60, 210, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"model: class_probs + boundary_head + postprocess",
                    (width - 760, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (210, 210, 210), 1, cv2.LINE_AA)

        _draw_prob_bars(frame, probs[idx], 22, height - 164, 190)
        _draw_boundary_strip(frame, idx, boundary_probs, events, fps)
        writer.write(frame)
        idx += 1
    cap.release()
    writer.release()


def detect_events_in_video(args: argparse.Namespace) -> None:
    video = Path(args.video)
    if not video.exists():
        raise FileNotFoundError(
            f"Video not found: {video}\n"
            "Place MP4 files under data/video/ or pass an absolute path."
        )
    checkpoint = resolve_checkpoint(args.checkpoint)
    results_dir = default_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)
    out_json = Path(args.out_json) if args.out_json else default_out_json(video, results_dir)
    out_video = Path(args.out_video) if args.out_video else None

    device = select_device()
    print(f"[info] torch device={device}")
    features, image_landmarks, valid, fps, width, height = extract_video_features(
        video,
        model_cache_dir=args.model_cache_dir,
        hand_side=None if args.hand_side == "Any" else args.hand_side,
        delegate=args.delegate,
    )
    print(f"[info] video={video.name} {width}x{height} @ {fps:.3f}fps frames={len(features)} valid={valid.mean() if len(valid) else 0:.1%}")
    model = load_model(checkpoint, device)
    probs, boundary_probs = predict_probabilities(
        model,
        features,
        device,
        chunk_len=args.chunk_len,
        overlap=args.overlap,
    )
    events = probabilities_to_events(
        probs,
        fps,
        boundary_probs=boundary_probs,
        conf_threshold=args.conf_threshold,
        min_event_ms=args.min_event_ms,
        max_gap_ms=args.max_gap_ms,
        smooth=args.smooth,
    )

    print(f"\n[result] {len(events)} gesture event(s)")
    print_event_table(events)

    payload = {
        "task": "offline_gesture_event_detection",
        "video": str(video),
        "checkpoint": str(checkpoint),
        "fps": fps,
        "total_frames": len(probs),
        "mediapipe": {
            "running_mode": "VIDEO",
            "delegate": args.delegate,
            "valid_rate": float(valid.mean()) if len(valid) else 0.0,
        },
        "postprocess": {
            "conf_threshold": args.conf_threshold,
            "min_event_ms": args.min_event_ms,
            "max_gap_ms": args.max_gap_ms,
            "smooth": args.smooth,
        },
        "events": events,
    }
    if args.include_frames:
        payload["frames"] = [
            {
                "frame": i,
                "time_ms": int(round(i * 1000 / fps)),
                "label": int(row.argmax()),
                "label_name": LABEL_NAMES[int(row.argmax())],
                "confidence": round(float(row.max()), 4),
                "boundary_start": round(float(boundary_probs[0, i]), 4) if boundary_probs.size else 0.0,
                "boundary_end": round(float(boundary_probs[1, i]), 4) if boundary_probs.size else 0.0,
                "top3": [
                    {
                        "label": int(j),
                        "label_name": LABEL_NAMES[int(j)],
                        "prob": round(float(row[int(j)]), 4),
                    }
                    for j in np.argsort(row)[::-1][:3]
                ],
                "mediapipe_valid": bool(valid[i]) if i < len(valid) else False,
            }
            for i, row in enumerate(probs)
        ]
    out_json.parent.mkdir(parents=True, exist_ok=True)
    save_json(out_json, payload)
    if out_video is not None:
        write_overlay_video(
            video,
            out_video,
            events,
            probs,
            boundary_probs,
            image_landmarks,
            valid,
            fps,
        )
        print(f"[save] {out_video}")
    print_next_steps_detect(out_json, out_video)


def main() -> None:
    ensure_utf8_console()
    ap = make_parser(
        "Detect gesture events and start/end times in a video",
        epilog=(
            "Examples:\n"
            "  py -3.11 inference.py --video data/video/session_c10.mp4\n"
            "  py -3.11 inference.py --video clip.mp4 --out-video results/clip_pred.mp4"
        ),
    )
    ap.add_argument("--video", required=True, help="Input MP4 path")
    ap.add_argument("--checkpoint", default=None, help="Model checkpoint; auto-discovers best.pt")
    ap.add_argument("--out-json", default=None, help="Output JSON (default: results/<stem>.events.json)")
    ap.add_argument("--out-video", default=None, help="Optional overlay MP4 path")
    ap.add_argument("--include-frames", action="store_true", help="Include per-frame predictions (large JSON)")
    add_hand_side_arg(ap, default="Right")
    add_model_cache_arg(ap)
    add_delegate_arg(ap)
    ap.add_argument("--chunk-len", type=int, default=DETECT_CHUNK_LEN)
    ap.add_argument("--overlap", type=int, default=DETECT_OVERLAP)
    ap.add_argument("--conf-threshold", type=float, default=CONF_THRESHOLD)
    ap.add_argument("--min-event-ms", type=int, default=MIN_EVENT_MS)
    ap.add_argument("--max-gap-ms", type=int, default=MAX_GAP_MS)
    ap.add_argument("--smooth", type=int, default=SMOOTH_WIDTH)
    detect_events_in_video(ap.parse_args())


if __name__ == "__main__":
    main()
