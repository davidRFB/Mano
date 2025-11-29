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


def save_preprocessed_tensors(
    data_dir: Path = DATA_DIR,
    output_path: str = "./data/processed/tensors.pth",
    image_size: int = IMAGE_SIZE,
    seed: int = RANDOM_SEED,
) -> None:
    """
    Preprocess all images and save as tensors for fast loading in Colab.
    
    Saves: {
        'train_images': Tensor, 'train_labels': Tensor,
        'val_images': Tensor, 'val_labels': Tensor,
        'test_images': Tensor, 'test_labels': Tensor,
        'classes': list, 'num_classes': int
    }
    """
    print("Preprocessing dataset and saving tensors...")
    
    # Create dataset with val transforms (no augmentation for saved tensors)
    transform = get_val_transforms(image_size)
    dataset = LSCDataset(data_dir=data_dir, transform=transform)
    
    # Get splits
    train_idx, val_idx, test_idx = split_dataset(dataset, seed=seed)
    
    def extract_tensors(indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        images = []
        labels = []
        for idx in indices:
            img, label = dataset[idx]
            images.append(img)
            labels.append(label)
        return torch.stack(images), torch.tensor(labels)
    
    print("Extracting train tensors...")
    train_images, train_labels = extract_tensors(train_idx)
    print("Extracting val tensors...")
    val_images, val_labels = extract_tensors(val_idx)
    print("Extracting test tensors...")
    test_images, test_labels = extract_tensors(test_idx)
    
    data = {
        'train_images': train_images,
        'train_labels': train_labels,
        'val_images': val_images,
        'val_labels': val_labels,
        'test_images': test_images,
        'test_labels': test_labels,
        'classes': dataset.classes,
        'num_classes': dataset.num_classes,
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, output_path)
    
    print(f"\n✓ Saved preprocessed tensors to {output_path}")
    print(f"  Train: {train_images.shape}, Val: {val_images.shape}, Test: {test_images.shape}")
    print(f"  File size: {Path(output_path).stat().st_size / 1024 / 1024:.1f} MB")


def load_preprocessed_tensors(
    tensor_path: str,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> tuple[DataLoader, DataLoader, DataLoader, int, list[str]]:
    """
    Load preprocessed tensors and create DataLoaders.
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, num_classes, class_names)
    """
    from torch.utils.data import TensorDataset
    
    print(f"Loading preprocessed tensors from {tensor_path}...")
    data = torch.load(tensor_path, weights_only=False)
    
    train_dataset = TensorDataset(data['train_images'], data['train_labels'])
    val_dataset = TensorDataset(data['val_images'], data['val_labels'])
    test_dataset = TensorDataset(data['test_images'], data['test_labels'])
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    print(f"✓ Loaded {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test samples")
    
    return (
        train_loader,
        val_loader,
        test_loader,
        data['num_classes'],
        data['classes'],
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess LSC dataset and save tensors for fast loading"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/raw",
        help="Path to input data directory (default: ./data/raw)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/processed/tensors.pth",
        help="Path to output tensor file (default: ./data/processed/tensors.pth)",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only test the pipeline, don't save tensors",
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
        images, labels = next(iter(train_loader))
        print(f"\nBatch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")

        print("\n✓ Preprocessing pipeline working correctly!")

        if not args.test_only:
            # Save preprocessed tensors for fast Colab loading
            save_preprocessed_tensors(
                data_dir=data_dir,
                output_path=args.output,
            )

    except ValueError as e:
        print(f"\n✗ Error: {e}")

