# Changelog

All notable changes to MANO (Colombian Sign Language Translator) will be documented in this file.

Format: `## [Version] - YYYY-MM-DD`

---

## [Unreleased]

### Added
- **LLM word correction** for sign language letter sequences
  - `src/llm/corrector.py` - Fast Spanish word correction
  - Groq backend (default, ~100-200ms latency, free API)
  - Ollama backend (local fallback, ~5-10s latency)
  - Auto-loads `.env` for API keys
- **Letter capture in inference** with dual modes
  - Manual capture (SPACE key)
  - Auto-capture when letter stable for 0.5s
  - Buffer management (BACKSPACE, C to clear)
  - ENTER triggers LLM correction
- **DVC data versioning** for reproducible experiments
  - Configured Google Drive as remote storage (`G:\My Drive\dvc-storage\mano`)
  - `data/raw/` tracked as v1 (1,871 images, 26 letters)
  - Workflow: `dvc add data/raw` → `dvc push` → `git commit`
- **Quarto blog** for documenting project journey
  - `blog/` directory with posts and figures
  - Notebooks converted to polished blog posts
  - Figures stored in `blog/figures/` with ordered naming

---

## [0.2.0] - 2025-11-27

### Added
- `src/cv_model/inference.py` - Real-time gesture recognition
  - Webcam capture with MediaPipe hand detection
  - Frame-by-frame prediction with confidence display
  - Letter buffer accumulation
  - Preprocessing matching training pipeline
- `notebooks/02_model_analysis.ipynb` - Model diagnosis notebook
  - Training curves visualization
  - Data leakage analysis (temporal correlation)
  - Class imbalance detection
  - Per-class performance metrics
  - Inference condition simulation
- MLflow experiment tracking integration
  - Tracks params, metrics per epoch, model artifacts
  - Stored in `models/mlruns/`

### Analysis Findings
- **Data leakage detected**: 100% test accuracy due to near-duplicate images
- **High capture rate**: 12+ images/second creates nearly identical frames
- **Class imbalance**: 2.8x ratio (T: 129, B: 46)
- **Recommendations**: More diverse data, temporal splits, stronger augmentation

---

## [0.1.0] - 2025-11-27

### Added
- `scripts/capture_data.py` - Data collection tool
  - Press letter keys (a-z) to capture photos
  - Auto-saves to `data/raw/{letter}/` directories
  - Hand detection with MediaPipe (21 landmark points)
  - Auto-crop to hand region with configurable padding
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
- `notebooks/01_data_analysis.ipynb` - Dataset exploration
  - Image count statistics
  - Input range analysis
  - Sample visualization
- `requirements.txt` - Core dependencies
  - opencv-python, numpy, mediapipe, matplotlib
  - torch, torchvision, scikit-learn, mlflow

---

## [0.0.1] - 2025-11-27

### Added
- Initial project structure
- `CLAUDE.md` - AI assistant guidelines
- `CHANGELOG.md` - This file
- `STRUCTURE.md` - Project organization reference
- `.gitignore` configured for Python, data, models

---

## Version Guidelines

- **Major (1.0.0)**: Breaking changes, production-ready release
- **Minor (0.x.0)**: New features, non-breaking changes
- **Patch (0.0.x)**: Bug fixes, small improvements
