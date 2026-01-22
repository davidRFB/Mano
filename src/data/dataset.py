"""
PyTorch Dataset classes for LSC gesture recognition.

Supports both:
- Static letters (single frame or averaged across sequence)
- Dynamic letters (full sequence for RNN models)
"""

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

from src.data.preprocessing import (
    normalize_landmarks,
    extract_features,
    FEATURE_MODES,
    DEFAULT_FEATURE_MODE,
)

# Configuration
DATA_DIR = Path("./data/raw_landmarks")
SEQUENCE_LENGTH = 20
BATCH_SIZE = 32
NUM_WORKERS = 4
RANDOM_SEED = 42

# Dynamic letters that require movement (LSC)
DYNAMIC_LETTERS = {"j", "h", "z", "nn", "s"}  # nn = ñ


class LandmarksDataset(Dataset):
    """
    Dataset for landmark sequences.

    Can output:
    - Full sequences for RNN models (mode='sequence')
    - Single averaged frame for MLP models (mode='static')
    """

    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        feature_mode: str = DEFAULT_FEATURE_MODE,
        output_mode: Literal["sequence", "static"] = "sequence",
        augment: bool = False,
        letters: list[str] | None = None,
    ):
        """
        Args:
            data_dir: Path to landmark data
            feature_mode: Feature extraction mode (xy, xy_angles, etc.)
            output_mode: 'sequence' for RNN, 'static' for MLP (averages frames)
            augment: Apply data augmentation
            letters: Filter to specific letters (None = all)
        """
        self.data_dir = Path(data_dir)
        self.feature_mode = feature_mode
        self.output_mode = output_mode
        self.augment = augment
        self.feature_dim = FEATURE_MODES[feature_mode]

        self.samples: list[tuple[Path, int]] = []
        self.classes: list[str] = []
        self.class_to_idx: dict[str, int] = {}

        self._load_samples(letters)

    def _load_samples(self, letters: list[str] | None) -> None:
        """Load .npy file paths and labels."""
        letter_dirs = sorted([
            d for d in self.data_dir.iterdir()
            if d.is_dir() and list(d.glob("*.npy"))
        ])

        # Filter letters if specified
        if letters:
            letter_dirs = [d for d in letter_dirs if d.name in letters]

        self.classes = [d.name for d in letter_dirs]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        for letter_dir in letter_dirs:
            label = self.class_to_idx[letter_dir.name]
            for npy_path in letter_dir.glob("*.npy"):
                self.samples.append((npy_path, label))

        print(f"Loaded {len(self.samples)} samples from {len(self.classes)} classes")

    def _augment(self, landmarks: np.ndarray) -> np.ndarray:
        """Apply data augmentation to landmarks (seq_len, 21, 3)."""
        # Random rotation in X-Y plane
        if np.random.random() < 0.7:
            angle = np.random.uniform(-20, 20) * np.pi / 180
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            x, y = landmarks[:, :, 0], landmarks[:, :, 1]
            landmarks[:, :, 0] = x * cos_a - y * sin_a
            landmarks[:, :, 1] = x * sin_a + y * cos_a

        # Random scale
        if np.random.random() < 0.5:
            scale = np.random.uniform(0.9, 1.1)
            landmarks[:, :, :2] *= scale

        # Random translation
        if np.random.random() < 0.5:
            shift = np.random.uniform(-0.05, 0.05, size=(1, 1, 2))
            landmarks[:, :, :2] += shift

        # Random noise
        if np.random.random() < 0.5:
            noise = np.random.normal(0, 0.015, landmarks[:, :, :2].shape)
            landmarks[:, :, :2] += noise

        # Horizontal flip (mirror)
        if np.random.random() < 0.3:
            landmarks[:, :, 0] = -landmarks[:, :, 0]

        return landmarks

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        npy_path, label = self.samples[idx]

        # Load and normalize
        landmarks = np.load(npy_path)
        landmarks = normalize_landmarks(landmarks)

        if self.augment:
            landmarks = self._augment(landmarks.copy())

        # Extract features
        features = extract_features(landmarks, mode=self.feature_mode)

        if self.output_mode == "static":
            # Average across sequence for static classification
            features = features.mean(axis=0)

        return torch.from_numpy(features).float(), label

    @property
    def num_classes(self) -> int:
        return len(self.classes)


def split_leave_one_out(
    dataset: LandmarksDataset,
    seed: int = RANDOM_SEED,
) -> tuple[list[int], list[int]]:
    """
    Leave-one-out split: 1 sample per class for validation.

    For small datasets where stratified split fails.
    """
    np.random.seed(seed)

    # Group by class
    class_indices: dict[int, list[int]] = {}
    for idx, (_, label) in enumerate(dataset.samples):
        class_indices.setdefault(label, []).append(idx)

    train_idx, val_idx = [], []
    for indices in class_indices.values():
        shuffled = indices.copy()
        np.random.shuffle(shuffled)
        val_idx.append(shuffled[0])
        train_idx.extend(shuffled[1:])

    return train_idx, val_idx


def create_dataloaders(
    data_dir: Path = DATA_DIR,
    feature_mode: str = DEFAULT_FEATURE_MODE,
    output_mode: Literal["sequence", "static"] = "sequence",
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    seed: int = RANDOM_SEED,
    letters: list[str] | None = None,
) -> tuple[DataLoader, DataLoader, int, list[str], int]:
    """
    Create train and validation DataLoaders.

    Args:
        data_dir: Path to landmark data
        feature_mode: Feature extraction mode
        output_mode: 'sequence' for RNN, 'static' for MLP
        batch_size: Batch size
        num_workers: Data loading workers
        seed: Random seed
        letters: Filter to specific letters

    Returns:
        (train_loader, val_loader, num_classes, class_names, feature_dim)
    """
    # Base dataset for splitting
    base = LandmarksDataset(
        data_dir=data_dir,
        feature_mode=feature_mode,
        output_mode=output_mode,
        augment=False,
        letters=letters,
    )

    if len(base) == 0:
        raise ValueError(f"No data found in {data_dir}")

    train_idx, val_idx = split_leave_one_out(base, seed)
    print(f"Split: {len(train_idx)} train, {len(val_idx)} val")

    # Create datasets with appropriate augmentation
    train_ds = LandmarksDataset(
        data_dir=data_dir,
        feature_mode=feature_mode,
        output_mode=output_mode,
        augment=True,
        letters=letters,
    )
    val_ds = LandmarksDataset(
        data_dir=data_dir,
        feature_mode=feature_mode,
        output_mode=output_mode,
        augment=False,
        letters=letters,
    )

    train_loader = DataLoader(
        Subset(train_ds, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(val_ds, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, base.num_classes, base.classes, base.feature_dim


def get_static_letters() -> list[str]:
    """Get list of static letter classes (excluding dynamic ones)."""
    all_letters = [chr(i) for i in range(ord('a'), ord('z') + 1)] + ['nn']
    return [l for l in all_letters if l not in DYNAMIC_LETTERS]


def get_dynamic_letters() -> list[str]:
    """Get list of dynamic letter classes."""
    return list(DYNAMIC_LETTERS)


if __name__ == "__main__":
    # Test dataset
    print("Testing dataset...")
    train_loader, val_loader, n_classes, classes, feat_dim = create_dataloaders()
    print(f"Classes: {classes}")
    print(f"Feature dim: {feat_dim}")

    x, y = next(iter(train_loader))
    print(f"Batch shape: {x.shape}")
    print(f"Labels: {y}")
