# Changelog

All notable changes to MANO (Colombian Sign Language Translator) will be documented in this file.

Format: `## [Version] - YYYY-MM-DD`

---

## [Unreleased]

### Changed
- **Major scripts reorganization** - Unified capture scripts with consistent interface
  - All capture scripts now support `--mode` and `--output` parameters
  - Numbered workflow: 01 (static) → 02 (sequences) → 03 (words) → 04 (train) → 05 (eval) → 06 (demo)

### Added
- **`scripts/01_capture_static.py`** - Unified single-frame capture
  - Modes: `photo` (raw), `photo_landmarks` (with overlay), `landmarks` (.npy)
  - Replaces old `capture_data.py`

- **`scripts/02_capture_dynamic.py`** - Sequence capture for ALL letters
  - Modes: `landmarks` (.npy), `video` (.avi), `both`
  - Configurable frame count (`--frames`, default 20)
  - Works for both static and dynamic letters (uniform training approach)

- **`scripts/03_capture_words.py`** - Word gesture capture with Holistic
  - Modes: `landmarks`, `video`, `both`
  - Variable length recording (SPACE to stop)
  - Uses MediaPipe Holistic: 51 landmarks (9 pose + 21 L-hand + 21 R-hand)

- **`scripts/utils/`** - Utility scripts for data processing
  - `video_to_landmarks.py` - Convert videos to .npy
  - `add_landmarks_to_dataset.py` - Add landmarks overlay to photos
  - `combine_word_datasets.py` - Merge INSOR + YouTube datasets
  - `process_youtube_dataset.py` - Process scraped YouTube videos

### Removed
- `scripts/capture_data.py` - Replaced by `01_capture_static.py`
- `scripts/capture_landmarks.py` - Replaced by `01_capture_static.py --mode landmarks`
- `scripts/capture_video.py` - Replaced by `02_capture_dynamic.py --mode video`
- `scripts/cap_video.py` - Test script, no longer needed
- `scripts/capture_words.py` - Replaced by `03_capture_words.py`

---

## [0.4.0] - 2025-12-31

### Added
- **Word-level sign language recognition pipeline**
  - New holistic landmark approach: 51 landmarks (9 pose + 21 left hand + 21 right hand)
  - Variable sequence length support (up to 90 frames, ~3 seconds at 30fps)
  - Combined dataset: 1482 videos, 1251 unique words (INSOR + YouTube)

- **Words preprocessing** (`src/cv_model/words_preprocessing.py`)
  - Feature modes: `xy` (102), `xyz` (153), `xy_angles` (130, default), `full` (232)
  - Normalization centered on shoulders for body-scale invariance
  - Angle extraction for both hands (28 features total)
  - Stratified train/val/test split with min_samples filtering

- **Words models** (`src/cv_model/words_model.py`)
  - GRU, BiGRU, LSTM, GRU+Attention architectures
  - Transformer encoder for longer sequences
  - Larger default hidden dimension (256 vs 128)

- **Words training** (`src/cv_model/words_train.py`)
  - MLflow tracking with top-5 accuracy metrics
  - Label smoothing (0.1) for large vocabulary
  - CosineAnnealingWarmRestarts scheduler
  - Gradient clipping for stability
  - `--min-samples N` flag to filter rare words

- **Words inference** (`src/cv_model/words_inference.py`)
  - Video file or webcam input
  - Top-K predictions display

- **Data processing scripts**
  - `scripts/combine_word_datasets.py` - Combine INSOR + YouTube via symlinks
  - `scripts/process_youtube_dataset.py` - Cut first 2.5s, clean filenames, parallel processing
  - `scripts/video_to_landmarks.py` - Added `--skip-existing`, `--workers`, `.m4v` support

### Dataset Statistics
- INSOR: 431 single-word videos (filtered from phrases)
- YouTube: 1051 processed videos
- Words with 1 sample: 1054 (84%)
- Words with 2+ samples: 197 (16%)

---

## [0.3.0] - 2025-12-30

### Changed
- **Landmarks model improvements**
  - Dropped Z axis (depth unreliable from single camera)
  - Fixed Y axis orientation (MediaPipe Y=0 is top, now flipped)
  - Added derived features: finger angles, key distances

### Added
- **New feature modes for letters** (21 landmarks)
  - `xy`: 42 features (X, Y only)
  - `xyz`: 63 features (original, for compatibility)
  - `xy_angles`: 56 features (X, Y + 14 finger angles) - DEFAULT
  - `xy_angles_distances`: 66 features (above + 10 distances)
  - `full`: 108 features (above + 42 velocity features)

- **Derived features**
  - 14 finger joint angles (2 thumb + 12 for other fingers)
  - 10 key distances (thumb to fingertips, adjacent tips, palm width, wrist distances)

### Technical
- `--features MODE` flag in `landmarks_train.py`
- Feature mode saved in checkpoint, auto-loaded in inference
- Y-flip applied in preprocessing for correct visualization

---

## [0.2.1] - 2025-12-15

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
