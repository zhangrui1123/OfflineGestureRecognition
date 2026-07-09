"""finger-ml — offline hand gesture event detection."""

from configs.defaults import DATA_DIR, MODEL_CACHE_DIR, RESULTS_DIR
from utils.labels import GESTURE_ORDER, LABEL_NAMES, NUM_CLASSES

__version__ = "0.1.0"
__all__ = [
    "__version__",
    "DATA_DIR",
    "MODEL_CACHE_DIR",
    "RESULTS_DIR",
    "GESTURE_ORDER",
    "LABEL_NAMES",
    "NUM_CLASSES",
]
