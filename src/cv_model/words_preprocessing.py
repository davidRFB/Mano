"""
Data preprocessing for word-level LSC gesture recognition.

Loads holistic landmark sequences (variable frames x 51 landmarks x 3 coords).
51 landmarks = 9 upper body pose + 21 left hand + 21 right hand.

Feature modes:
- "xy": Just X, Y coordinates (102 features)
- "xyz": X, Y, Z coordinates (153 features)
- "xy_angles": X, Y + finger angles for both hands (102 + 28 = 130 features)
- "full": X, Y + angles + velocities (130 + 102 = 232 features)

Usage:
    from src.cv_model.words_preprocessing import create_dataloaders
    train_loader, val_loader, test_loader, num_classes, classes, feature_dim = create_dataloaders()
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
import numpy as np

# Configuration
DATA_DIR = Path("./data/raw_words")
MAX_SEQUENCE_LENGTH = 90  # ~3 seconds at 30fps
MIN_SEQUENCE_LENGTH = 15  # minimum frames to keep

# Holistic landmark structure
NUM_POSE_LANDMARKS = 9
NUM_HAND_LANDMARKS = 21
NUM_TOTAL_LANDMARKS = NUM_POSE_LANDMARKS + 2 * NUM_HAND_LANDMARKS  # 51

# Landmark indices
POSE_START = 0
POSE_END = NUM_POSE_LANDMARKS  # 0-8
LEFT_HAND_START = POSE_END
LEFT_HAND_END = LEFT_HAND_START + NUM_HAND_LANDMARKS  # 9-29
RIGHT_HAND_START = LEFT_HAND_END
RIGHT_HAND_END = RIGHT_HAND_START + NUM_HAND_LANDMARKS  # 30-50

# Feature modes and their dimensions
FEATURE_MODES = {
    "xy": NUM_TOTAL_LANDMARKS * 2,  # 102
    "xyz": NUM_TOTAL_LANDMARKS * 3,  # 153
    "xy_angles": NUM_TOTAL_LANDMARKS * 2 + 28,  # 102 + 14*2 hands = 130
    "full": NUM_TOTAL_LANDMARKS * 2 + 28 + NUM_TOTAL_LANDMARKS * 2,  # 130 + 102 = 232
}
DEFAULT_FEATURE_MODE = "xy_angles"

BATCH_SIZE = 32
NUM_WORKERS = 4
RANDOM_SEED = 42

# Hand landmark indices (relative to hand start)
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

# Finger joint triplets for angle calculation
FINGER_ANGLES = [
    (THUMB_CMC, THUMB_MCP, THUMB_IP),
    (THUMB_MCP, THUMB_IP, THUMB_TIP),
    (WRIST, INDEX_MCP, INDEX_PIP),
    (INDEX_MCP, INDEX_PIP, INDEX_DIP),
    (INDEX_PIP, INDEX_DIP, INDEX_TIP),
    (WRIST, MIDDLE_MCP, MIDDLE_PIP),
    (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP),
    (MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    (WRIST, RING_MCP, RING_PIP),
    (RING_MCP, RING_PIP, RING_DIP),
    (RING_PIP, RING_DIP, RING_TIP),
    (WRIST, PINKY_MCP, PINKY_PIP),
    (PINKY_MCP, PINKY_PIP, PINKY_DIP),
    (PINKY_PIP, PINKY_DIP, PINKY_TIP),
]


def compute_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Compute angle at p2 formed by vectors p1->p2 and p2->p3."""
    v1 = p1 - p2
    v2 = p3 - p2
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cos_angle = np.clip(cos_angle, -1, 1)
    angle = np.arccos(cos_angle)
    return 1.0 - (angle / np.pi)  # Normalize: 0=straight, 1=folded


def extract_hand_angles(hand_landmarks: np.ndarray) -> np.ndarray:
    """Extract 14 finger joint angles from hand landmarks (21, 2)."""
    angles = []
    for p1_idx, p2_idx, p3_idx in FINGER_ANGLES:
        angle = compute_angle(
            hand_landmarks[p1_idx, :2],
            hand_landmarks[p2_idx, :2],
            hand_landmarks[p3_idx, :2],
        )
        angles.append(angle)
    return np.array(angles, dtype=np.float32)


def extract_features(sequence: np.ndarray, mode: str = DEFAULT_FEATURE_MODE) -> np.ndarray:
    """
    Extract features from holistic landmark sequence.

    Args:
        sequence: (seq_len, 51, 3) raw landmarks
        mode: Feature extraction mode

    Returns:
        (seq_len, feature_dim) feature array
    """
    seq_len = sequence.shape[0]
    sequence = sequence.copy()

    # Flip Y axis (MediaPipe Y goes down)
    sequence[:, :, 1] = 1.0 - sequence[:, :, 1]

    if mode == "xyz":
        return sequence.reshape(seq_len, -1)

    # X, Y only
    xy = sequence[:, :, :2]  # (seq_len, 51, 2)
    features_list = [xy.reshape(seq_len, -1)]  # 102 features

    if mode == "xy":
        return features_list[0]

    # Add angles for both hands
    if mode in ["xy_angles", "full"]:
        left_hand = xy[:, LEFT_HAND_START:LEFT_HAND_END, :]  # (seq_len, 21, 2)
        right_hand = xy[:, RIGHT_HAND_START:RIGHT_HAND_END, :]  # (seq_len, 21, 2)

        left_angles = np.array([extract_hand_angles(left_hand[i]) for i in range(seq_len)])
        right_angles = np.array([extract_hand_angles(right_hand[i]) for i in range(seq_len)])

        features_list.append(left_angles)  # 14 features
        features_list.append(right_angles)  # 14 features

    # Add velocities
    if mode == "full":
        xy_flat = xy.reshape(seq_len, -1)
        velocities = np.zeros_like(xy_flat)
        velocities[1:] = xy_flat[1:] - xy_flat[:-1]
        features_list.append(velocities)  # 102 features

    return np.concatenate(features_list, axis=1)


def pad_or_truncate(sequence: np.ndarray, target_len: int) -> np.ndarray:
    """Pad or truncate sequence to target length."""
    seq_len = sequence.shape[0]

    if seq_len == target_len:
        return sequence
    elif seq_len > target_len:
        # Truncate from center (keep middle portion)
        start = (seq_len - target_len) // 2
        return sequence[start : start + target_len]
    else:
        # Pad with zeros at the end
        pad_len = target_len - seq_len
        padding = np.zeros((pad_len,) + sequence.shape[1:], dtype=sequence.dtype)
        return np.concatenate([sequence, padding], axis=0)


class WordsDataset(Dataset):
    """Dataset for word-level holistic landmark sequences."""

    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        normalize: bool = True,
        augment: bool = False,
        feature_mode: str = DEFAULT_FEATURE_MODE,
        max_seq_len: int = MAX_SEQUENCE_LENGTH,
        min_samples_per_class: int = 1,
    ):
        self.data_dir = Path(data_dir)
        self.normalize = normalize
        self.augment = augment
        self.feature_mode = feature_mode
        self.feature_dim = FEATURE_MODES[feature_mode]
        self.max_seq_len = max_seq_len
        self.min_samples_per_class = min_samples_per_class
        self.samples: list[tuple[Path, int]] = []
        self.classes: list[str] = []
        self.class_to_idx: dict[str, int] = {}

        self._load_samples()

    def _load_samples(self) -> None:
        """Load all .npy file paths and their labels."""
        # Get word directories with enough samples
        word_dirs = []
        for d in sorted(self.data_dir.iterdir()):
            if d.is_dir():
                npy_files = list(d.glob("*.npy"))
                if len(npy_files) >= self.min_samples_per_class:
                    word_dirs.append(d)

        self.classes = [d.name for d in word_dirs]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        for word_dir in word_dirs:
            label = self.class_to_idx[word_dir.name]
            for npy_path in word_dir.glob("*.npy"):
                self.samples.append((npy_path, label))

        print(f"Loaded {len(self.samples)} sequences from {len(self.classes)} words")

    def _normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Normalize holistic landmarks.

        Centers on torso (pose landmarks) and scales by body size.
        """
        # Use shoulder midpoint as reference (pose indices 1, 2 = shoulders)
        left_shoulder = landmarks[:, 1:2, :]
        right_shoulder = landmarks[:, 2:2, :]
        center = (left_shoulder + right_shoulder) / 2  # (seq_len, 1, 3)

        # Center relative to torso
        centered = landmarks - center

        # Scale by shoulder width (consistent body scale)
        shoulder_dist = np.linalg.norm(
            landmarks[:, 1, :2] - landmarks[:, 2, :2], axis=1, keepdims=True
        )
        shoulder_dist = np.maximum(shoulder_dist, 1e-6)

        # Scale X, Y by shoulder distance
        normalized = centered.copy()
        normalized[:, :, :2] = centered[:, :, :2] / shoulder_dist[:, :, np.newaxis]

        return normalized

    def _augment_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """Apply data augmentation."""
        # Random rotation in X-Y plane
        if np.random.random() < 0.5:
            angle = np.random.uniform(-15, 15) * np.pi / 180
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            x = landmarks[:, :, 0]
            y = landmarks[:, :, 1]
            landmarks[:, :, 0] = x * cos_a - y * sin_a
            landmarks[:, :, 1] = x * sin_a + y * cos_a

        # Random scale
        if np.random.random() < 0.5:
            scale = np.random.uniform(0.9, 1.1)
            landmarks[:, :, :2] = landmarks[:, :, :2] * scale

        # Random noise
        if np.random.random() < 0.5:
            noise = np.random.normal(0, 0.01, landmarks[:, :, :2].shape)
            landmarks[:, :, :2] = landmarks[:, :, :2] + noise

        # Temporal jitter (drop random frames)
        if np.random.random() < 0.3 and len(landmarks) > MIN_SEQUENCE_LENGTH + 5:
            drop_count = np.random.randint(1, 5)
            keep_indices = np.random.choice(
                len(landmarks), size=len(landmarks) - drop_count, replace=False
            )
            keep_indices = np.sort(keep_indices)
            landmarks = landmarks[keep_indices]

        return landmarks

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        npy_path, label = self.samples[idx]

        # Load landmarks: (seq_len, 51, 3)
        landmarks = np.load(npy_path)

        # Filter very short sequences
        if len(landmarks) < MIN_SEQUENCE_LENGTH:
            # Repeat to minimum length
            repeats = (MIN_SEQUENCE_LENGTH // len(landmarks)) + 1
            landmarks = np.tile(landmarks, (repeats, 1, 1))[:MIN_SEQUENCE_LENGTH]

        # Normalize
        if self.normalize:
            landmarks = self._normalize_landmarks(landmarks)

        # Augment
        if self.augment:
            landmarks = self._augment_landmarks(landmarks)

        # Pad or truncate to fixed length
        landmarks = pad_or_truncate(landmarks, self.max_seq_len)

        # Extract features
        features = extract_features(landmarks, mode=self.feature_mode)

        return torch.from_numpy(features).float(), label

    @property
    def num_classes(self) -> int:
        return len(self.classes)


def split_dataset(
    dataset: WordsDataset,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = RANDOM_SEED,
) -> tuple[list[int], list[int], list[int]]:
    """
    Split dataset into train/val/test, stratified by class.

    Ensures at least 1 sample per class in train.
    """
    np.random.seed(seed)

    class_indices: dict[int, list[int]] = {}
    for idx, (_, label) in enumerate(dataset.samples):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)

    train_idx, val_idx, test_idx = [], [], []

    for label, indices in class_indices.items():
        np.random.shuffle(indices)
        n = len(indices)

        if n == 1:
            # Single sample: use for training only
            train_idx.extend(indices)
        elif n == 2:
            # Two samples: train + val
            train_idx.append(indices[0])
            val_idx.append(indices[1])
        else:
            # Normal split
            n_test = max(1, int(n * test_ratio))
            n_val = max(1, int(n * val_ratio))
            n_train = n - n_val - n_test

            train_idx.extend(indices[:n_train])
            val_idx.extend(indices[n_train : n_train + n_val])
            test_idx.extend(indices[n_train + n_val :])

    print(f"Split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    return train_idx, val_idx, test_idx


def create_dataloaders(
    data_dir: Path = DATA_DIR,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    seed: int = RANDOM_SEED,
    feature_mode: str = DEFAULT_FEATURE_MODE,
    max_seq_len: int = MAX_SEQUENCE_LENGTH,
    min_samples_per_class: int = 1,
) -> tuple[DataLoader, DataLoader, DataLoader, int, list[str], int]:
    """
    Create train, validation, and test DataLoaders.

    Returns:
        (train_loader, val_loader, test_loader, num_classes, class_names, feature_dim)
    """
    base_dataset = WordsDataset(
        data_dir=data_dir,
        normalize=True,
        augment=False,
        feature_mode=feature_mode,
        max_seq_len=max_seq_len,
        min_samples_per_class=min_samples_per_class,
    )

    if len(base_dataset) == 0:
        raise ValueError(f"No .npy files found in {data_dir}")

    train_idx, val_idx, test_idx = split_dataset(base_dataset, seed=seed)

    # Create datasets with appropriate settings
    train_dataset = WordsDataset(
        data_dir=data_dir,
        normalize=True,
        augment=True,
        feature_mode=feature_mode,
        max_seq_len=max_seq_len,
        min_samples_per_class=min_samples_per_class,
    )
    val_dataset = WordsDataset(
        data_dir=data_dir,
        normalize=True,
        augment=False,
        feature_mode=feature_mode,
        max_seq_len=max_seq_len,
        min_samples_per_class=min_samples_per_class,
    )

    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(val_dataset, val_idx)
    test_subset = Subset(val_dataset, test_idx) if test_idx else val_subset

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

    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        base_dataset.num_classes,
        base_dataset.classes,
        base_dataset.feature_dim,
    )


if __name__ == "__main__":
    print("Testing words preprocessing...")
    print(f"Feature modes: {FEATURE_MODES}")

    try:
        train_loader, val_loader, test_loader, num_classes, classes, feat_dim = (
            create_dataloaders()
        )
        print(f"\nClasses ({num_classes}): {classes[:10]}...")
        print(f"Feature dim: {feat_dim}")

        seq, label = next(iter(train_loader))
        print(f"Batch shape: {seq.shape}")  # (batch, seq_len, features)

    except ValueError as e:
        print(f"Error: {e}")
