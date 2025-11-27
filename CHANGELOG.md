# Changelog

All notable changes to LSC-Connect will be documented in this file.

Format: `## [Version] - YYYY-MM-DD`

---

## [Unreleased]

### Added
- `scripts/capture_data.py` - Data collection tool for capturing hand gesture images
  - Press letter keys (a-z) to capture photos
  - Auto-saves to `data/raw/{letter}/` directories
  - Visual feedback on capture
  - **Hand detection** with MediaPipe (21 landmark points)
  - **Auto-crop** to hand region with configurable padding
  - Only captures when hand is detected
- `src/cv_model/preprocessing.py` - Data preprocessing pipeline
  - `LSCDataset` PyTorch Dataset class
  - Train/val/test stratified splitting (70/15/15)
  - Augmentation: rotation, flip, color jitter, affine transforms
  - ImageNet normalization for pretrained models
- `src/cv_model/train.py` - Training script
  - Supports: MobileNetV2, MobileNetV3, EfficientNet-B0, ResNet18
  - Pretrained weights from torchvision
  - AdamW optimizer with cosine annealing LR
  - Early stopping with patience
  - Checkpoint saving with metadata JSON
  - **MLflow integration**: logs params, metrics per epoch, artifacts
  - MLflow tracking stored in `models/mlruns/`
- `requirements.txt` with opencv, numpy, mediapipe, torch, torchvision, sklearn

### Added
- Initial project structure

### Changed

### Fixed

### Removed

---

## [0.1.0] - YYYY-MM-DD

### Added
- Project initialization
- Basic directory structure
- Development environment setup
- CLAUDE.md, CHANGELOG.md, STRUCTURE.md created

---

## Template for New Entries

```markdown
## [Version] - YYYY-MM-DD

### Added
- New features, files, or capabilities

### Changed
- Modifications to existing functionality

### Fixed
- Bug fixes

### Removed
- Deprecated or removed features
```

---

## Version Guidelines

- **Major (1.0.0)**: Breaking changes, major refactors
- **Minor (0.1.0)**: New features, non-breaking changes
- **Patch (0.0.1)**: Bug fixes, small improvements