"""Shared CLI helpers: path resolution, console output, and device selection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

from configs.defaults import DATA_DIR, MODEL_CACHE_DIR, RESULTS_DIR
from utils.labels import GESTURE_ZH

CHECKPOINT_CANDIDATES: tuple[str, ...] = (
    "models/best.pt",
    "checkpoints/best.pt",
)


def ensure_utf8_console() -> None:
    """Best-effort UTF-8 console on Windows so Chinese labels render correctly."""
    if sys.platform != "win32":
        return
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8")
            except Exception:
                pass


def make_parser(description: str, *, epilog: str | None = None) -> argparse.ArgumentParser:
    """Create an argparse parser with consistent defaults display."""
    return argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=epilog,
    )


def add_data_dir_arg(parser: argparse.ArgumentParser, *, default: str = DATA_DIR) -> None:
    parser.add_argument("--data-dir", default=default, help="Data root directory")


def add_model_cache_arg(parser: argparse.ArgumentParser, *, default: str = MODEL_CACHE_DIR) -> None:
    parser.add_argument("--model-cache-dir", default=default, help="MediaPipe model cache")


def add_hand_side_arg(
    parser: argparse.ArgumentParser,
    *,
    default: str | None = "Right",
    choices: tuple[str, ...] = ("Left", "Right", "Any"),
) -> None:
    parser.add_argument("--hand-side", default=default, choices=choices)


def add_delegate_arg(parser: argparse.ArgumentParser, *, default: str = "CPU") -> None:
    parser.add_argument("--delegate", default=default, choices=["CPU", "GPU"], help="MediaPipe delegate")


def add_fail_on_miss_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--fail-on-miss",
        action="store_true",
        help="Exit with code 1 when quality/readiness gates fail",
    )


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save] {path}")


def exit_on_fail(passed: bool, *, enabled: bool) -> None:
    if enabled and not passed:
        sys.exit(1)


def subject_from_stem(stem: str) -> str:
    """Extract subject id from session stem (e.g. 20260429_130158_c10 -> c10)."""
    return stem.rsplit("_", 1)[-1].lower() if "_" in stem else stem.lower()


def select_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = True
        name = torch.cuda.get_device_name(0)
        print(f"[info] device: cuda ({name})")
        return device
    if torch.backends.mps.is_available():
        print("[info] device: mps")
        return torch.device("mps")
    print("[info] device: cpu")
    return torch.device("cpu")


def resolve_checkpoint(path: str | Path | None) -> Path:
    """Find a trained checkpoint when the user omits --checkpoint."""
    if path is not None:
        candidate = Path(path)
        if candidate.exists():
            return candidate.resolve()
        raise FileNotFoundError(
            f"Checkpoint not found: {candidate}\n"
            "Train first: py -3.11 train.py --data-dir data"
        )

    for name in CHECKPOINT_CANDIDATES:
        candidate = Path(name)
        if candidate.exists():
            resolved = candidate.resolve()
            print(f"[info] checkpoint: {resolved}")
            return resolved

    searched = ", ".join(CHECKPOINT_CANDIDATES)
    raise FileNotFoundError(
        f"No checkpoint found. Looked for: {searched}\n"
        "Train first or pass --checkpoint path/to/best.pt"
    )


def default_results_dir() -> Path:
    return Path(RESULTS_DIR)


def default_out_json(video: Path, results_dir: Path | None = None) -> Path:
    root = results_dir or default_results_dir()
    return root / f"{video.stem}.events.json"


def default_out_video(video: Path, results_dir: Path | None = None) -> Path:
    root = results_dir or default_results_dir()
    return root / f"{video.stem}_pred.mp4"


def discover_videos(video_dir: Path) -> dict[str, Path]:
    """Map session stem -> video path (supports nested folders under video/)."""
    if not video_dir.exists():
        return {}
    return {path.stem: path for path in sorted(video_dir.rglob("*.mp4"))}


def discover_labels(label_dir: Path) -> dict[str, Path]:
    if not label_dir.exists():
        return {}
    return {path.stem: path for path in sorted(label_dir.rglob("*.json"))}


def discover_features(feature_dir: Path) -> dict[str, Path]:
    if not feature_dir.exists():
        return {}
    return {path.stem: path for path in sorted(feature_dir.rglob("*.npz"))}


def match_session_pairs(data_dir: Path) -> list[tuple[Path, Path]]:
    """Pair videos and labels that share the same session stem."""
    video_map = discover_videos(data_dir / "video")
    label_map = discover_labels(data_dir / "labels")
    common = sorted(set(video_map) & set(label_map))
    if not common:
        raise FileNotFoundError(
            f"No matching video+label pairs in {data_dir}.\n"
            f"  videos: {len(video_map)} under {data_dir / 'video'}\n"
            f"  labels: {len(label_map)} under {data_dir / 'labels'}\n"
            "Run finger-collect first, or check that stems match."
        )
    return [(video_map[k], label_map[k]) for k in common]


def resolve_session_pair(
    data_dir: Path,
    *,
    session: str | None = None,
    video: str | Path | None = None,
    label: str | Path | None = None,
) -> tuple[Path, Path]:
    """Resolve a video/label pair from explicit paths or a session stem."""
    video_map = discover_videos(data_dir / "video")
    label_map = discover_labels(data_dir / "labels")

    if video is not None:
        video_path = Path(video)
        if label is not None:
            label_path = Path(label)
        else:
            label_path = label_map.get(video_path.stem, data_dir / "labels" / f"{video_path.stem}.json")
    elif label is not None:
        label_path = Path(label)
        video_path = video_map.get(label_path.stem, data_dir / "video" / f"{label_path.stem}.mp4")
    else:
        if session:
            stem = session
        else:
            if not label_map:
                raise FileNotFoundError(f"No labels found under {data_dir / 'labels'}")
            stem = max(label_map.values(), key=lambda p: p.stat().st_mtime).stem
        label_path = label_map.get(stem, data_dir / "labels" / f"{stem}.json")
        video_path = video_map.get(stem, data_dir / "video" / f"{stem}.mp4")

    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    return video_path, label_path


def print_event_table(events: list[dict]) -> None:
    if not events:
        print("  (no gesture events detected)")
        return
    header = f"{'start':>8}  {'end':>8}  {'dur':>6}  {'gesture':<20}  {'zh':<8}  conf"
    print(header)
    print("-" * len(header))
    for ev in events:
        zh = GESTURE_ZH.get(ev["gesture"], ev["gesture"])
        print(
            f"{ev['start_ms']:>7}ms  {ev['end_ms']:>7}ms  "
            f"{ev.get('duration_ms', ev['end_ms'] - ev['start_ms']):>5}ms  "
            f"{ev['gesture']:<20}  {zh:<8}  {ev['mean_conf']:.3f}"
        )


def print_next_steps_detect(out_json: Path, out_video: Path | None) -> None:
    print("\n[next]")
    print(f"  results JSON : {out_json}")
    if out_video is not None:
        print(f"  preview video: {out_video}")
    print("  evaluate       : py -3.11 -m finger_ml.eval_events --label data/labels/<session>.json --pred-json", out_json)


def print_next_steps_preprocess(data_dir: Path) -> None:
    print("\n[next]")
    print(f"  audit data : py -3.11 -m finger_ml.audit --data-dir {data_dir}")
    print(f"  train model: py -3.11 train.py --data-dir {data_dir}")


def print_next_steps_train(checkpoint: Path) -> None:
    print("\n[next]")
    print(f"  detect video: py -3.11 inference.py --video data/video/<session>.mp4 --checkpoint {checkpoint}")


def print_next_steps_audit(data_dir: Path, passed: bool) -> None:
    if passed:
        print("\n[next]")
        print(f"  train: py -3.11 train.py --data-dir {data_dir}")
    else:
        print("\n[next] fix readiness issues, then re-run audit with --fail-on-miss before training")
