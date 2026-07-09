"""Run offline gesture event detection on a video.

Usage:
    py -3.11 inference.py --video data/video/session.mp4
    py -3.11 inference.py --video clip.mp4 --out-video results/clip_pred.mp4
"""

from engine.predictor import main

if __name__ == "__main__":
    main()
