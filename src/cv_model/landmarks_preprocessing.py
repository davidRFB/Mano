"""
Data preprocessing for landmark-based LSC gesture recognition.

Loads .npy landmark sequences (20 frames x 21 landmarks x 3 coords).

Feature modes:
- "xy": Just X, Y coordinates (42 features) - recommended
- "xyz": X, Y, Z coordinates (63 features) - original, Z is noisy
- "xy_angles": X, Y + finger angles (42 + 14 = 56 features)
- "xy_angles_distances": X, Y + angles + distances (42 + 14 + 10 = 66 features)
- "full": All features including velocities (66 + 42 = 108 features)

Usage:
    from src.cv_model.landmarks_preprocessing import create_dataloaders
    train_loader, val_loader, test_loader, num_classes, classes = create_dataloaders()
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
import numpy as np

# Configuration
DATA_DIR = Path("./data/raw_landmarks")
SEQUENCE_LENGTH = 20  # frames per sequence
NUM_LANDMARKS = 21

# Feature modes and their dimensions
FEATURE_MODES = {
    "xy": 42,                    # 21 landmarks × 2 coords
    "xyz": 63,                   # 21 landmarks × 3 coords (original)
    "xy_angles": 56,             # 42 + 14 finger angles
    "xy_angles_distances": 66,   # 42 + 14 angles + 10 distances
    "full": 108,                 # 66 + 42 velocities
}
DEFAULT_FEATURE_MODE = "xy_angles"

BATCH_SIZE = 32
NUM_WORKERS = 4
RANDOM_SEED = 42

# MediaPipe hand landmark indices
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

# Finger joint triplets for angle calculation (joint is middle point)
FINGER_ANGLES = [
    # Thumb (2 angles - no DIP)
    (THUMB_CMC, THUMB_MCP, THUMB_IP),
    (THUMB_MCP, THUMB_IP, THUMB_TIP),
    # Index (3 angles)
    (WRIST, INDEX_MCP, INDEX_PIP),
    (INDEX_MCP, INDEX_PIP, INDEX_DIP),
    (INDEX_PIP, INDEX_DIP, INDEX_TIP),
    # Middle (3 angles)
    (WRIST, MIDDLE_MCP, MIDDLE_PIP),
    (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP),
    (MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    # Ring (3 angles)
    (WRIST, RING_MCP, RING_PIP),
    (RING_MCP, RING_PIP, RING_DIP),
    (RING_PIP, RING_DIP, RING_TIP),
    # Pinky (3 angles)
    (WRIST, PINKY_MCP, PINKY_PIP),
    (PINKY_MCP, PINKY_PIP, PINKY_DIP),
    (PINKY_PIP, PINKY_DIP, PINKY_TIP),
]

# Key distances to measure
KEY_DISTANCES = [
    # Thumb to fingertips (pinch gestures)
    (THUMB_TIP, INDEX_TIP),
    (THUMB_TIP, MIDDLE_TIP),
    (THUMB_TIP, RING_TIP),
    (THUMB_TIP, PINKY_TIP),
    # Fingertip spread
    (INDEX_TIP, MIDDLE_TIP),
    (MIDDLE_TIP, RING_TIP),
    (RING_TIP, PINKY_TIP),
    # Palm width
    (INDEX_MCP, PINKY_MCP),
    # Wrist to key points
    (WRIST, MIDDLE_TIP),
    (WRIST, THUMB_TIP),
]


def compute_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Compute angle at p2 formed by vectors p1->p2 and p2->p3.

    Returns angle in radians, normalized to [0, 1] range (0=straight, 1=folded).
    """
    v1 = p1 - p2
    v2 = p3 - p2

    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cos_angle = np.clip(cos_angle, -1, 1)
    angle = np.arccos(cos_angle)

    # Normalize: 0 = straight (180°), 1 = fully bent (0°)
    return 1.0 - (angle / np.pi)


def extract_angles(landmarks: np.ndarray) -> np.ndarray:
    """
    Extract finger joint angles from landmarks.

    Args:
        landmarks: (21, 2) or (21, 3) landmark coordinates

    Returns:
        (15,) array of normalized angles
    """
    angles = []
    for p1_idx, p2_idx, p3_idx in FINGER_ANGLES:
        angle = compute_angle(
            landmarks[p1_idx, :2],  # Only use X, Y
            landmarks[p2_idx, :2],
            landmarks[p3_idx, :2]
        )
        angles.append(angle)
    return np.array(angles, dtype=np.float32)


def extract_distances(landmarks: np.ndarray) -> np.ndarray:
    """
    Extract key distances between landmarks.

    Args:
        landmarks: (21, 2) or (21, 3) landmark coordinates

    Returns:
        (10,) array of normalized distances
    """
    distances = []
    for p1_idx, p2_idx in KEY_DISTANCES:
        dist = np.linalg.norm(landmarks[p1_idx, :2] - landmarks[p2_idx, :2])
        distances.append(dist)
    return np.array(distances, dtype=np.float32)


def extract_features(
    sequence: np.ndarray,
    mode: str = DEFAULT_FEATURE_MODE,
) -> np.ndarray:
    """
    Extract features from landmark sequence based on mode.

    Args:
        sequence: (seq_len, 21, 3) raw landmarks
        mode: Feature extraction mode

    Returns:
        (seq_len, feature_dim) feature array
    """
    seq_len = sequence.shape[0]

    # Flip Y axis (MediaPipe Y goes down, we want Y up)
    sequence = sequence.copy()
    sequence[:, :, 1] = 1.0 - sequence[:, :, 1]

    if mode == "xyz":
        # Original: use all 3 coordinates
        return sequence.reshape(seq_len, -1)

    # For all other modes, start with X, Y only
    xy = sequence[:, :, :2]  # (seq_len, 21, 2)
    features_list = [xy.reshape(seq_len, -1)]  # 42 features

    if mode == "xy":
        return features_list[0]

    # Add angles
    if mode in ["xy_angles", "xy_angles_distances", "full"]:
        angles = np.array([extract_angles(xy[i]) for i in range(seq_len)])  # (seq_len, 15)
        features_list.append(angles)

    # Add distances
    if mode in ["xy_angles_distances", "full"]:
        distances = np.array([extract_distances(xy[i]) for i in range(seq_len)])  # (seq_len, 10)
        features_list.append(distances)

    # Add velocities (frame-to-frame differences)
    if mode == "full":
        xy_flat = xy.reshape(seq_len, -1)
        velocities = np.zeros_like(xy_flat)
        velocities[1:] = xy_flat[1:] - xy_flat[:-1]
        features_list.append(velocities)  # 42 features

    return np.concatenate(features_list, axis=1)


class LandmarksDataset(Dataset):
    """Dataset for landmark sequences with configurable feature extraction."""

    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        normalize: bool = True,
        augment: bool = False,
        feature_mode: str = DEFAULT_FEATURE_MODE,
    ):
        self.data_dir = Path(data_dir)
        self.normalize = normalize
        self.augment = augment
        self.feature_mode = feature_mode
        self.feature_dim = FEATURE_MODES[feature_mode]
        self.samples: list[tuple[Path, int]] = []
        self.classes: list[str] = []
        self.class_to_idx: dict[str, int] = {}

        self._load_samples()

    def _load_samples(self) -> None:
        """Load all .npy file paths and their labels."""
        # Get all letter directories that have .npy files
        letter_dirs = sorted(
            [d for d in self.data_dir.iterdir() if d.is_dir() and list(d.glob("*.npy"))]
        )

        self.classes = [d.name for d in letter_dirs]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        for letter_dir in letter_dirs:
            label = self.class_to_idx[letter_dir.name]
            for npy_path in letter_dir.glob("*.npy"):
                self.samples.append((npy_path, label))

        print(f"Loaded {len(self.samples)} sequences from {len(self.classes)} classes")

    def _normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Normalize landmarks relative to wrist and scale.

        Input: (seq_len, 21, 3)
        Output: (seq_len, 21, 3) normalized

        Uses only X, Y for distance calculation (Z is unreliable).
        """
        # Wrist is landmark 0 - use as reference point
        wrist = landmarks[:, 0:1, :]  # (seq_len, 1, 3)

        # Center relative to wrist
        centered = landmarks - wrist

        # Scale by max distance from wrist using X, Y only (per frame)
        distances_xy = np.linalg.norm(centered[:, :, :2], axis=2)  # (seq_len, 21)
        max_dist = distances_xy.max(axis=1, keepdims=True)  # (seq_len, 1)
        max_dist = np.maximum(max_dist, 1e-6)  # avoid division by zero

        # Scale all coordinates (including Z for compatibility)
        normalized = centered / max_dist[:, :, np.newaxis]

        return normalized

    def _augment_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """Apply data augmentation to landmark sequences (2D focused)."""
        # Random rotation in X-Y plane
        if np.random.random() < 0.7:
            angle = np.random.uniform(-20, 20) * np.pi / 180
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            # Rotate X, Y only
            x = landmarks[:, :, 0]
            y = landmarks[:, :, 1]
            landmarks[:, :, 0] = x * cos_a - y * sin_a
            landmarks[:, :, 1] = x * sin_a + y * cos_a

        # Random scale
        if np.random.random() < 0.5:
            scale = np.random.uniform(0.9, 1.1)
            landmarks[:, :, :2] = landmarks[:, :, :2] * scale

        # Random translation (X, Y only)
        if np.random.random() < 0.5:
            shift = np.random.uniform(-0.05, 0.05, size=(1, 1, 2))
            landmarks[:, :, :2] = landmarks[:, :, :2] + shift

        # Random noise (X, Y only)
        if np.random.random() < 0.5:
            noise = np.random.normal(0, 0.015, (landmarks.shape[0], landmarks.shape[1], 2))
            landmarks[:, :, :2] = landmarks[:, :, :2] + noise

        # Temporal augmentation: random speed variation
        if np.random.random() < 0.3:
            landmarks = self._temporal_warp(landmarks)

        # Mirror horizontally (flip x-axis) - creates "other hand" version
        if np.random.random() < 0.3:
            landmarks[:, :, 0] = -landmarks[:, :, 0]

        return landmarks

    def _temporal_warp(self, landmarks: np.ndarray) -> np.ndarray:
        """Apply random temporal warping (speed up/slow down)."""
        seq_len = landmarks.shape[0]
        # Random speed factor
        speed = np.random.uniform(0.8, 1.2)
        new_len = int(seq_len * speed)
        new_len = max(new_len, seq_len // 2)  # Don't shrink too much

        # Interpolate to new length, then resample to original length
        old_indices = np.linspace(0, seq_len - 1, new_len)
        new_indices = np.linspace(0, new_len - 1, seq_len)

        warped = np.zeros_like(landmarks)
        for i in range(landmarks.shape[1]):  # For each landmark
            for j in range(landmarks.shape[2]):  # For each coord
                # Interpolate to new length
                interp_vals = np.interp(old_indices, np.arange(seq_len), landmarks[:, i, j])
                # Resample back to original length
                warped[:, i, j] = np.interp(new_indices, np.arange(new_len), interp_vals)

        return warped

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        npy_path, label = self.samples[idx]

        # Load landmarks: (seq_len, 21, 3)
        landmarks = np.load(npy_path)

        # Normalize (before augmentation)
        if self.normalize:
            landmarks = self._normalize_landmarks(landmarks)

        # Augment (only if enabled)
        if self.augment:
            landmarks = self._augment_landmarks(landmarks)

        # Extract features based on mode (handles Y flip, drops Z if needed)
        features = extract_features(landmarks, mode=self.feature_mode)

        # Convert to tensor
        features_tensor = torch.from_numpy(features).float()

        return features_tensor, label

    @property
    def num_classes(self) -> int:
        return len(self.classes)


def split_dataset_leave_one_out(
    dataset: LandmarksDataset,
    seed: int = RANDOM_SEED,
) -> tuple[list[int], list[int]]:
    """
    Split dataset: leave 1 sample per class for validation, rest for training.

    For small datasets where stratified splitting fails, this ensures
    every class is represented in both train and val sets.
    """
    np.random.seed(seed)

    # Group indices by class
    class_indices: dict[int, list[int]] = {}
    for idx, (_, label) in enumerate(dataset.samples):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)

    train_idx = []
    val_idx = []

    for label, indices in class_indices.items():
        shuffled = indices.copy()
        np.random.shuffle(shuffled)

        # Leave 1 for validation, rest for training
        val_idx.append(shuffled[0])
        train_idx.extend(shuffled[1:])

    print(f"Split sizes - Train: {len(train_idx)}, Val: {len(val_idx)} (1 per class)")

    return train_idx, val_idx


def create_dataloaders(
    data_dir: Path = DATA_DIR,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    seed: int = RANDOM_SEED,
    feature_mode: str = DEFAULT_FEATURE_MODE,
) -> tuple[DataLoader, DataLoader, DataLoader, int, list[str], int]:
    """
    Create train, validation, and test DataLoaders.

    Uses leave-one-out split: 1 sample per class for validation, rest for training.
    Test loader uses same data as val (for compatibility with training script).

    Args:
        data_dir: Path to landmark data
        batch_size: Batch size
        num_workers: Number of data loading workers
        seed: Random seed for reproducibility
        feature_mode: Feature extraction mode ('xy', 'xyz', 'xy_angles', etc.)

    Returns:
        Tuple of (train_loader, val_loader, test_loader, num_classes, class_names, feature_dim)
    """
    # Create base dataset for splitting
    base_dataset = LandmarksDataset(
        data_dir=data_dir, normalize=True, augment=False, feature_mode=feature_mode
    )

    if len(base_dataset) == 0:
        raise ValueError(f"No .npy files found in {data_dir}. Run capture_landmarks.py first.")

    # Get split indices (leave-one-out per class)
    train_idx, val_idx = split_dataset_leave_one_out(base_dataset, seed=seed)

    # Create datasets with appropriate settings
    train_dataset = LandmarksDataset(
        data_dir=data_dir, normalize=True, augment=True, feature_mode=feature_mode
    )
    val_dataset = LandmarksDataset(
        data_dir=data_dir, normalize=True, augment=False, feature_mode=feature_mode
    )

    # Create subsets
    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(val_dataset, val_idx)

    # Create DataLoaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Test loader = val loader (not enough data for separate test set)
    test_loader = val_loader

    return (
        train_loader,
        val_loader,
        test_loader,
        base_dataset.num_classes,
        base_dataset.classes,
        base_dataset.feature_dim,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test landmarks preprocessing")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/raw_landmarks",
        help="Path to landmarks data directory",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    print(f"Data directory: {data_dir.absolute()}")
    print("=" * 50)

    try:
        train_loader, val_loader, test_loader, num_classes, classes = create_dataloaders(
            data_dir=data_dir
        )

        print(f"\nNum classes: {num_classes}")
        print(f"Classes: {classes}")

        # Test a batch
        sequences, labels = next(iter(train_loader))
        print(f"\nBatch shape: {sequences.shape}")  # (batch, seq_len, 63)
        print(f"Labels shape: {labels.shape}")
        print(f"Sequence range: [{sequences.min():.3f}, {sequences.max():.3f}]")

        print("\nPreprocessing pipeline working correctly!")

    except ValueError as e:
        print(f"\nError: {e}")
