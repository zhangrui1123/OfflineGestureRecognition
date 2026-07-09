"""Central defaults for paths, training, detection, and quality gates.

Import from here instead of hard-coding magic numbers in CLI modules.
"""

from __future__ import annotations

# ── Paths ────────────────────────────────────────────────────────────────────

DATA_DIR = "data"
MODEL_CACHE_DIR = ".models"
CHECKPOINT_DIR = "models"
RESULTS_DIR = "results"

# ── Preprocess / label masking ─────────────────────────────────────────────────

BOUNDARY_MARGIN = 2
PRE_IGNORE_FRAMES = 4
POST_IGNORE_SECONDS = 1.0

# ── Training (shorter chunks for stochastic mini-batch sampling) ───────────────

TRAIN_CHUNK_LEN = 256
TRAIN_HOP = 64
VAL_HOP = 128
TRAIN_BATCH_SIZE = 8
TRAIN_EPOCHS = 80
TRAIN_LR = 3e-4
TRAIN_WEIGHT_DECAY = 1e-4
TRAIN_DROPOUT = 0.25
HIDDEN_DIM = 128
TEMPORAL_CHANNELS = 128
TEMPORAL_LAYERS = 6
TEMPORAL_STAGES = 2
BOUNDARY_RADIUS = 2
LAMBDA_BOUNDARY = 0.2
LAMBDA_SMOOTH = 0.15
BCE_POS_WEIGHT = 8.0
TMSE_TAU = 4.0
GRAD_CLIP_NORM = 5.0

# ── Detection (longer chunks for full-video inference) ─────────────────────────

DETECT_CHUNK_LEN = 512
DETECT_OVERLAP = 128
CONF_THRESHOLD = 0.55
MIN_EVENT_MS = 120
MAX_GAP_MS = 120
SMOOTH_WIDTH = 7

# ── Audit readiness gates ──────────────────────────────────────────────────────

AUDIT_TARGET_EVENTS_PER_CLASS = 500
AUDIT_MIN_SUBJECTS = 5
AUDIT_MIN_SESSIONS_PER_SUBJECT = 3
AUDIT_MIN_QUALITY = 90.0
AUDIT_MIN_VALID_RATE = 0.98
AUDIT_MAX_BG_JITTER_P95 = 0.06
AUDIT_MAX_DETECT_GAP_FRAMES = 5

# ── Event evaluation gates ─────────────────────────────────────────────────────

EVAL_IOU_THRESHOLD = 0.30
EVAL_TARGET_F1 = 0.98
EVAL_TARGET_CLASS_PRECISION = 0.97
EVAL_TARGET_CLASS_RECALL = 0.97
