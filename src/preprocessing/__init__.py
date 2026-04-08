"""Data loading and preprocessing for LSC gesture recognition."""

from src.preprocessing.preprocessing import (
    extract_features,
    normalize_landmarks,
    FEATURE_MODES,
    DEFAULT_FEATURE_MODE,
)

__all__ = [
    "extract_features",
    "normalize_landmarks",
    "FEATURE_MODES",
    "DEFAULT_FEATURE_MODE",
]

# Dataset utilities (optional — not needed for API inference)
try:
    from src.preprocessing.dataset import LandmarksDataset, create_dataloaders

    __all__ += ["LandmarksDataset", "create_dataloaders"]
except ImportError:
    pass
