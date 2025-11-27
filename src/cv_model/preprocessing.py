"""
Data preprocessing and DataLoader creation for LSC gesture recognition.

Usage:
    from src.cv_model.preprocessing import create_dataloaders
    train_loader, val_loader, test_loader = create_dataloaders()
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from pathlib import Path
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Optional


# Configuration
DATA_DIR = Path("./data/raw")
IMAGE_SIZE = 224  # Standard for most pretrained models
BATCH_SIZE = 32
NUM_WORKERS = 4
RANDOM_SEED = 42

# ImageNet normalization (for pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class LSCDataset(Dataset):
    """Dataset for Colombian Sign Language hand gestures."""

    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        transform: Optional[transforms.Compose] = None,
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        self.classes: list[str] = []
        self.class_to_idx: dict[str, int] = {}

        self._load_samples()

    def _load_samples(self) -> None:
        """Load all image paths and their labels."""
        # Get all letter directories that have images
        letter_dirs = sorted(
            [d for d in self.data_dir.iterdir() if d.is_dir() and list(d.glob("*.jpg"))]
        )

        self.classes = [d.name for d in letter_dirs]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        for letter_dir in letter_dirs:
            label = self.class_to_idx[letter_dir.name]
            for img_path in letter_dir.glob("*.jpg"):
                self.samples.append((img_path, label))

        print(f"Loaded {len(self.samples)} images from {len(self.classes)} classes")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label

    @property
    def num_classes(self) -> int:
        return len(self.classes)


def get_train_transforms(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Get training transforms with augmentation."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_val_transforms(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Get validation/test transforms (no augmentation)."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def split_dataset(
    dataset: LSCDataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = RANDOM_SEED,
) -> tuple[list[int], list[int], list[int]]:
    """
    Split dataset indices into train/val/test with stratification.

    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    indices = list(range(len(dataset)))
    labels = [dataset.samples[i][1] for i in indices]

    # First split: train vs (val + test)
    train_idx, temp_idx, train_labels, temp_labels = train_test_split(
        indices,
        labels,
        train_size=train_ratio,
        stratify=labels,
        random_state=seed,
    )

    # Second split: val vs test
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=val_ratio_adjusted,
        stratify=temp_labels,
        random_state=seed,
    )

    print(f"Split sizes - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    return train_idx, val_idx, test_idx


def create_dataloaders(
    data_dir: Path = DATA_DIR,
    image_size: int = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = RANDOM_SEED,
) -> tuple[DataLoader, DataLoader, DataLoader, int, list[str]]:
    """
    Create train, validation, and test DataLoaders.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, num_classes, class_names)
    """
    # Create base dataset (without transforms for splitting)
    base_dataset = LSCDataset(data_dir=data_dir, transform=None)

    if len(base_dataset) == 0:
        raise ValueError(f"No images found in {data_dir}. Run capture_data.py first.")

    # Get split indices
    train_idx, val_idx, test_idx = split_dataset(
        base_dataset,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    # Create datasets with appropriate transforms
    train_transform = get_train_transforms(image_size)
    val_transform = get_val_transforms(image_size)

    train_dataset = LSCDataset(data_dir=data_dir, transform=train_transform)
    val_dataset = LSCDataset(data_dir=data_dir, transform=val_transform)
    test_dataset = LSCDataset(data_dir=data_dir, transform=val_transform)

    # Create subsets
    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(val_dataset, val_idx)
    test_subset = Subset(test_dataset, test_idx)

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
    )


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Denormalize a tensor for visualization."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return tensor * std + mean


if __name__ == "__main__":
    # Test the preprocessing pipeline
    print("Testing preprocessing pipeline...")
    print("=" * 50)

    try:
        train_loader, val_loader, test_loader, num_classes, classes = create_dataloaders()

        print(f"\nNum classes: {num_classes}")
        print(f"Classes: {classes}")

        # Test a batch
        images, labels = next(iter(train_loader))
        print(f"\nBatch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")

        print("\n✓ Preprocessing pipeline working correctly!")
        #Saving the dataloaders in "./data/processed/dataloaders.pth"
        torch.save((train_loader, val_loader, test_loader), "./data/processed/dataloaders.pth")
        print(f"Dataloaders saved to './data/processed/dataloaders.pth'")
    except ValueError as e:
        print(f"\n✗ Error: {e}")

