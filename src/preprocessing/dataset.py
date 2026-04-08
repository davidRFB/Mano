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
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split

from src.preprocessing.preprocessing import (
    normalize_landmarks,
    extract_features,
    FEATURE_MODES,
    DEFAULT_FEATURE_MODE,
)

# Configuration
DATA_DIR = Path("./data/raw_landmarks")
IMAGE_DATA_DIR = Path("./data/raw")
SEQUENCE_LENGTH = 20
BATCH_SIZE = 32
NUM_WORKERS = 4
RANDOM_SEED = 42
IMAGE_SIZE = 224  # MobileNet default

# Dynamic letters that require movement (LSC)
DYNAMIC_LETTERS = {"j", "h", "z", "nn", "s"}  # nn = ñ


class ImageDataset(Dataset):
    """
    Dataset for raw images (Static gestures).
    """

    def __init__(
        self,
        data_dir: Path = IMAGE_DATA_DIR,
        augment: bool = False,
        letters: list[str] | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.augment = augment
        
        self.samples: list[tuple[Path, int]] = []
        self.classes: list[str] = []
        self.class_to_idx: dict[str, int] = {}

        self._load_samples(letters)

        # Transformations (ImageNet normalization)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Direct resize, no crop
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Direct resize, no crop
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

    def _load_samples(self, letters: list[str] | None) -> None:
        """Load image file paths and labels."""
        class_dirs = sorted([
            d for d in self.data_dir.iterdir()
            if d.is_dir()
        ])

        # Filter letters if specified
        if letters:
            class_dirs = [d for d in class_dirs if d.name in letters]

        self.classes = [d.name for d in class_dirs]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        valid_exts = {'.jpg', '.jpeg', '.png'}
        for class_dir in class_dirs:
            label = self.class_to_idx[class_dir.name]
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in valid_exts:
                    self.samples.append((img_path, label))

        print(f"Loaded {len(self.samples)} images from {len(self.classes)} classes")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

    @property
    def num_classes(self) -> int:
        return len(self.classes)


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
        static_frame_override: dict[str, int] | None = None,
    ):
        """
        Args:
            data_dir: Path to landmark data
            feature_mode: Feature extraction mode (xy, xy_angles, etc.)
            output_mode: 'sequence' for RNN, 'static' for MLP (averages frames)
            augment: Apply data augmentation
            letters: Filter to specific letters (None = all)
            static_frame_override: Map class name to frame index for static mode
                (e.g. {"j": 18} to use frame 18 for J instead of frame 0)
        """
        self.data_dir = Path(data_dir)
        self.feature_mode = feature_mode
        self.output_mode = output_mode
        self.augment = augment
        self.feature_dim = FEATURE_MODES[feature_mode]
        self.static_frame_override = static_frame_override or {}

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
            # Use specific frame per class (default: frame 0)
            class_name = self.classes[label]
            frame_idx = self.static_frame_override.get(class_name, 0)
            features = features[frame_idx]

        return torch.from_numpy(features).float(), label

    @property
    def num_classes(self) -> int:
        return len(self.classes)


def stratified_split(
    samples: list[tuple],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = RANDOM_SEED,
) -> tuple[list[int], list[int], list[int]]:
    """
    Split into train/val/test sets. Uses stratified split if possible,
    falls back to simple random split if not enough samples.

    Args:
        samples: List of (path, label) tuples
        train_ratio: Fraction for training (default: 0.70)
        val_ratio: Fraction for validation (default: 0.15)
        test_ratio: Fraction for testing (default: 0.15, use 0 for no test set)
        seed: Random seed for reproducibility

    Returns:
        (train_indices, val_indices, test_indices)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

    indices = list(range(len(samples)))
    labels = [samples[i][1] for i in indices]
    num_classes = len(set(labels))

    # Check if stratified split is possible
    min_ratio = val_ratio if test_ratio == 0 else min(val_ratio, test_ratio)
    min_samples_needed = int(num_classes / min_ratio) + 1
    use_stratify = len(samples) >= min_samples_needed

    if not use_stratify:
        print(f"Warning: Too few samples ({len(samples)}) for stratified split. Using random split.")

    # No test set case
    if test_ratio == 0:
        try:
            train_idx, val_idx = train_test_split(
                indices,
                labels,
                train_size=train_ratio,
                stratify=labels if use_stratify else None,
                random_state=seed,
            )
        except ValueError:
            print("Warning: Stratified split failed. Using random split.")
            train_idx, val_idx = train_test_split(
                indices, train_size=train_ratio, random_state=seed
            )
        return train_idx, val_idx, []

    try:
        # First split: train vs (val + test)
        train_idx, temp_idx, _, temp_labels = train_test_split(
            indices,
            labels,
            train_size=train_ratio,
            stratify=labels if use_stratify else None,
            random_state=seed,
        )

        # Second split: val vs test
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        val_idx, test_idx = train_test_split(
            temp_idx,
            train_size=val_ratio_adjusted,
            stratify=temp_labels if use_stratify else None,
            random_state=seed,
        )
    except ValueError:
        # Fallback to simple random split
        print("Warning: Stratified split failed. Using random split.")
        train_idx, temp_idx = train_test_split(
            indices, train_size=train_ratio, random_state=seed
        )
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        val_idx, test_idx = train_test_split(
            temp_idx, train_size=val_ratio_adjusted, random_state=seed
        )

    return train_idx, val_idx, test_idx


def create_dataloaders(
    data_dir: Path = DATA_DIR,
    feature_mode: str = DEFAULT_FEATURE_MODE,
    output_mode: Literal["sequence", "static"] = "sequence",
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    seed: int = RANDOM_SEED,
    letters: list[str] | None = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    static_frame_override: dict[str, int] | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, int, list[str], int]:
    """
    Create train, validation, and test DataLoaders with stratified split.

    Args:
        data_dir: Path to landmark data
        feature_mode: Feature extraction mode
        output_mode: 'sequence' for RNN, 'static' for MLP
        batch_size: Batch size
        num_workers: Data loading workers
        seed: Random seed
        letters: Filter to specific letters
        train_ratio: Fraction for training (default: 0.70)
        val_ratio: Fraction for validation (default: 0.15)
        test_ratio: Fraction for testing (default: 0.15)
        static_frame_override: Map class name to frame index for static mode

    Returns:
        (train_loader, val_loader, test_loader, num_classes, class_names, feature_dim)
    """
    # Base dataset for splitting
    base = LandmarksDataset(
        data_dir=data_dir,
        feature_mode=feature_mode,
        output_mode=output_mode,
        augment=False,
        letters=letters,
        static_frame_override=static_frame_override,
    )

    if len(base) == 0:
        raise ValueError(f"No data found in {data_dir}")

    train_idx, val_idx, test_idx = stratified_split(
        base.samples, train_ratio, val_ratio, test_ratio, seed
    )
    print(f"Split: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")

    # Create datasets with appropriate augmentation
    train_ds = LandmarksDataset(
        data_dir=data_dir,
        feature_mode=feature_mode,
        output_mode=output_mode,
        augment=True,
        letters=letters,
        static_frame_override=static_frame_override,
    )
    val_ds = LandmarksDataset(
        data_dir=data_dir,
        feature_mode=feature_mode,
        output_mode=output_mode,
        augment=False,
        letters=letters,
        static_frame_override=static_frame_override,
    )
    test_ds = LandmarksDataset(
        data_dir=data_dir,
        feature_mode=feature_mode,
        output_mode=output_mode,
        augment=False,
        letters=letters,
        static_frame_override=static_frame_override,
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
    test_loader = DataLoader(
        Subset(test_ds, test_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, base.num_classes, base.classes, base.feature_dim


def create_image_dataloaders(
    data_dir: Path = IMAGE_DATA_DIR,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    seed: int = RANDOM_SEED,
    letters: list[str] | None = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[DataLoader, DataLoader, DataLoader, int, list[str], int]:
    """
    Create train, validation, and test DataLoaders for images with stratified split.

    Args:
        data_dir: Path to image data
        batch_size: Batch size
        num_workers: Data loading workers
        seed: Random seed
        letters: Filter to specific letters
        train_ratio: Fraction for training (default: 0.70)
        val_ratio: Fraction for validation (default: 0.15)
        test_ratio: Fraction for testing (default: 0.15)

    Returns:
        (train_loader, val_loader, test_loader, num_classes, class_names, feature_dim)
    """
    # Base dataset for splitting
    base = ImageDataset(
        data_dir=data_dir,
        augment=False,
        letters=letters,
    )

    if len(base) == 0:
        raise ValueError(f"No images found in {data_dir}")

    # Stratified split
    train_idx, val_idx, test_idx = stratified_split(
        base.samples, train_ratio, val_ratio, test_ratio, seed
    )
    print(f"Split: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")

    # Create datasets
    train_ds = ImageDataset(
        data_dir=data_dir,
        augment=True,
        letters=letters,
    )
    val_ds = ImageDataset(
        data_dir=data_dir,
        augment=False,
        letters=letters,
    )
    test_ds = ImageDataset(
        data_dir=data_dir,
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
    test_loader = DataLoader(
        Subset(test_ds, test_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # For MobileNet, input is (3, 224, 224), so dim isn't just an int
    # MobileNet doesn't use input_dim param in our factory wrapper.
    return train_loader, val_loader, test_loader, base.num_classes, base.classes, 0


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
    train_loader, val_loader, test_loader, n_classes, classes, feat_dim = create_dataloaders()
    print(f"Classes: {classes}")
    print(f"Feature dim: {feat_dim}")

    x, y = next(iter(train_loader))
    print(f"Train batch shape: {x.shape}")
    print(f"Train labels: {y}")

    x, y = next(iter(test_loader))
    print(f"Test batch shape: {x.shape}")
    print(f"Test labels: {y}")
