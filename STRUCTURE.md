# STRUCTURE.md - Project Organization

Quick reference for file locations and purposes. **Update this when adding new files/directories.**

---

## Directory Tree

```
Mano/
├── .dvc/                   # DVC configuration
│   └── config              # Remote storage settings (Google Drive)
├── .github/
│   └── workflows/          # CI/CD pipelines
│       ├── ci.yml          # Testing, linting, docker build
│       └── cd.yml          # Deployment to cloud
├── blog/                   # Quarto blog (project journey)
│   ├── _quarto.yml         # Quarto configuration
│   ├── index.qmd           # Blog home page
│   ├── posts/              # Blog posts (polished notebooks)
│   │   ├── 01-project-kickoff/
│   │   ├── 02-model-analysis/
│   │   ├── 03-hyperparameter-optimization/
│   │   ├── 04-landmark-experiment/
│   │   ├── 05-word-correction/
│   │   └── ...
│   └── figures/            # All blog figures (ordered naming)
│       ├── 01-class-distribution.png
│       ├── 02-training-curves.png
│       └── ...
├── data/
│   ├── raw/                # Original gesture images (DVC tracked)
│   ├── raw.dvc             # DVC pointer file (Git tracked)
│   ├── raw_landmarks/      # Letter landmark sequences (.npy, 21 landmarks)
│   ├── raw_words/          # Word landmark sequences (.npy, 51 landmarks)
│   ├── webscrapping/       # Video datasets
│   │   ├── insor_dataset/  # INSOR sign language videos
│   │   ├── youtube_dataset_proc/  # Processed YouTube videos
│   │   └── combined_words/ # Combined dataset (symlinks)
│   ├── processed/          # Preprocessed data
│   └── splits/             # Train/val/test indices
├── models/                 # Trained model checkpoints
│   ├── mlruns/             # MLflow experiment tracking
│   └── *.pth               # Model weights (gitignored)
├── notebooks/              # Working Jupyter notebooks (exploratory)
│   ├── 01_data_analysis.ipynb
│   ├── 02_model_analysis.ipynb
│   ├── 05_landmark_comparison.ipynb
│   └── 06_llm_correction.ipynb
├── scripts/                # Utility scripts
│   ├── capture_data.py     # Data collection tool
│   ├── video_to_landmarks.py     # Extract landmarks from videos
│   ├── combine_word_datasets.py  # Combine INSOR + YouTube datasets
│   └── process_youtube_dataset.py # Process YouTube videos (cut, rename)
├── src/
│   ├── cv_model/           # Computer vision model code
│   │   ├── __init__.py
│   │   ├── preprocessing.py      # Image data transforms (letters)
│   │   ├── landmarks_preprocessing.py  # Landmark features (letters)
│   │   ├── words_preprocessing.py      # Holistic landmarks (words)
│   │   ├── train.py              # Image model training
│   │   ├── landmarks_train.py    # Landmark model training (letters)
│   │   ├── words_train.py        # Word model training
│   │   ├── landmarks_model.py    # Sequence models (letters)
│   │   ├── words_model.py        # Sequence models (words)
│   │   ├── inference.py          # Real-time letter prediction
│   │   ├── landmarks_inference.py # Landmark-based inference (letters)
│   │   └── words_inference.py    # Word prediction from video
│   ├── api/                # FastAPI application (planned)
│   ├── llm/                # LLM word correction
│   │   ├── __init__.py
│   │   └── corrector.py    # Groq/Ollama Spanish word corrector
│   └── frontend/           # Streamlit interface (planned)
├── tests/                  # Test suite
├── docker/                 # Dockerfiles
├── .dvcignore              # Files DVC should ignore
├── .env.example            # Environment variable template
├── .gitignore
├── requirements.txt        # Python dependencies
├── CLAUDE.md               # AI assistant guidelines
├── CHANGELOG.md            # Project changes log
├── STRUCTURE.md            # This file
└── README.md               # User documentation
```

---

## Data Versioning (DVC)

### Setup
- **Remote**: Google Drive (`G:\My Drive\dvc-storage\mano`)
- **Tracked**: `data/raw/` (gesture images)

### Workflow
```powershell
# After capturing new images
dvc add data/raw
dvc push
git add data/raw.dvc
git commit -m "Add more training data (v1.1)"
```

### Restore Data
```powershell
# On new machine or after checkout
dvc pull
```

### Current Versions
| Version | Images | Notes |
|---------|--------|-------|
| v1 | 1,871 | Initial capture, near-duplicates, single session |

---

## Blog (Quarto)

### Purpose
Document the project journey in a personal, relatable way. Convert working notebooks into polished posts.

### Structure
- `blog/posts/` - One folder per post (contains .qmd file)
- `blog/figures/` - Centralized figures with ordered naming
- `blog/_quarto.yml` - Site configuration

### Figure Naming Convention
```
{post_number}-{description}.png
Examples:
  01-class-distribution.png
  02-training-curves.png
  02-confusion-matrix.png
```

### Workflow
1. Explore in `notebooks/` (working notebooks)
2. Extract insights and figures
3. Create polished post in `blog/posts/`
4. Save figures to `blog/figures/`

### Build & Preview
```powershell
cd blog
quarto preview    # Local preview
quarto render     # Build static site
```

---

## Core Modules

### `src/cv_model/` - Letters Pipeline (Images)
**Purpose**: Letter gesture recognition from images/landmarks

| File | Description |
|------|-------------|
| `preprocessing.py` | Image Dataset class, transforms, augmentation |
| `landmarks_preprocessing.py` | Landmark feature extraction (21 landmarks, X/Y + angles) |
| `landmarks_model.py` | Sequence models: GRU, BiGRU, LSTM, GRU+Attention |
| `train.py` | Image model training with MLflow |
| `landmarks_train.py` | Landmark model training with feature modes |
| `inference.py` | Real-time letter prediction with MediaPipe |
| `landmarks_inference.py` | Landmark-based inference, loads feature_mode from checkpoint |

**Feature modes (letters - 21 landmarks)**:
- `xy`: 42 features (X, Y coordinates)
- `xyz`: 63 features (X, Y, Z - original)
- `xy_angles`: 56 features (X, Y + 14 finger angles) - DEFAULT
- `xy_angles_distances`: 66 features (above + 10 key distances)
- `full`: 108 features (above + 42 velocity features)

### `src/cv_model/` - Words Pipeline (Video Sequences)
**Purpose**: Word-level sign language recognition from holistic landmarks

| File | Description |
|------|-------------|
| `words_preprocessing.py` | Holistic landmarks (51 = 9 pose + 21 left + 21 right) |
| `words_model.py` | Sequence models: GRU, BiGRU, LSTM, GRU+Attention, Transformer |
| `words_train.py` | Training with MLflow, label smoothing, top-5 accuracy |
| `words_inference.py` | Word prediction from video file or webcam |

**Feature modes (words - 51 landmarks)**:
- `xy`: 102 features (X, Y coordinates)
- `xyz`: 153 features (X, Y, Z)
- `xy_angles`: 130 features (X, Y + 28 angles, 14 per hand) - DEFAULT
- `full`: 232 features (above + 102 velocities)

**Key differences from letters**:
- Uses holistic MediaPipe (pose + both hands) instead of single hand
- Variable sequence length (max 90 frames vs fixed 20)
- Larger vocabulary (~1251 words vs 27 letters)
- Normalization centered on shoulders

### `src/llm/`
**Purpose**: LLM-based Spanish word correction

| File | Description |
|------|-------------|
| `corrector.py` | SignLanguageCorrector class with Groq/Ollama backends |

**Key functions**:
- `SignLanguageCorrector(backend="groq")` - Fast cloud API (~100ms)
- `SignLanguageCorrector(backend="ollama")` - Local model (~5s)
- `correct_sequence(letters)` - Returns corrected Spanish word

### `scripts/`
**Purpose**: Data collection and processing utilities

| Script | Description |
|--------|-------------|
| `capture_data.py` | Capture letter images from webcam |
| `video_to_landmarks.py` | Extract landmarks from videos (hand or holistic) |
| `combine_word_datasets.py` | Combine INSOR + YouTube datasets via symlinks |
| `process_youtube_dataset.py` | Process YouTube videos (cut first 2.5s, rename) |

**video_to_landmarks.py usage**:
```bash
# Words (default - holistic landmarks)
python scripts/video_to_landmarks.py data/webscrapping/combined_words/ --skip-existing --workers 4

# Letters (hand only)
python scripts/video_to_landmarks.py videos/ --letters
```

**combine_word_datasets.py usage**:
```bash
python scripts/combine_word_datasets.py --stats     # show statistics
python scripts/combine_word_datasets.py --dry-run   # preview
python scripts/combine_word_datasets.py             # execute
```

---

## Configuration Files

### DVC
- `.dvc/config` - Remote storage configuration
- `.dvcignore` - Files to exclude from DVC tracking
- `data/raw.dvc` - Pointer to data version

### MLflow
- `models/mlruns/` - Experiment tracking data (gitignored)
- Access via: `mlflow ui` (run from project root)

### Docker
- `docker/Dockerfile.api` - API container image
- `docker/Dockerfile.frontend` - Frontend container image
- `docker-compose.yml` - Local orchestration

---

## Data Flow

### Letters Pipeline (Static Images)
```
1. DATA COLLECTION
   Webcam → capture_data.py → data/raw/{letter}/*.jpg

2. TRAINING (images)
   data/raw/ → preprocessing.py → train.py → models/*.pth

3. TRAINING (landmarks)
   data/raw_landmarks/ → landmarks_preprocessing.py → landmarks_train.py → models/*.pth

4. INFERENCE
   Webcam → inference.py → MediaPipe → Model → Letter Predictions
```

### Words Pipeline (Video Sequences)
```
1. DATA COLLECTION
   INSOR videos + YouTube videos
       ↓
   process_youtube_dataset.py (cut first 2.5s)
       ↓
   combine_word_datasets.py (symlinks to combined_words/)
       ↓
   video_to_landmarks.py (extract holistic landmarks)
       ↓
   data/raw_words/{word}/*.npy  (seq_len, 51, 3)

2. TRAINING
   data/raw_words/ → words_preprocessing.py → words_train.py → models/*.pth
                                                             → models/mlruns/

3. INFERENCE
   Video/Webcam → words_inference.py → MediaPipe Holistic → Model → Word Predictions
```

---

## Common Tasks

### Capture New Training Data
```powershell
cd scripts
python capture_data.py
# Press a-z to capture, ESC to quit
```

### Train Model (Letters)
```bash
# Image-based
python -m src.cv_model.train --model mobilenet_v2 --epochs 30

# Landmark-based
python -m src.cv_model.landmarks_train --model bigru --features xy_angles --epochs 100
```

### Train Model (Words)
```bash
# All words (1251 classes)
python -m src.cv_model.words_train --model bigru --epochs 100

# Only words with 2+ samples (recommended)
python -m src.cv_model.words_train --model bigru --min-samples 2 --epochs 100

# Transformer for longer sequences
python -m src.cv_model.words_train --model transformer --min-samples 2
```

### Run Inference (Letters)
```bash
python -m src.cv_model.inference --experiment V3_landmarks
```
Controls: SPACE (capture) | BACKSPACE (delete) | C (clear) | ENTER (correct) | ESC (quit)

### Run Inference (Words)
```bash
python -m src.cv_model.words_inference --video path/to/video.mp4
python -m src.cv_model.words_inference --webcam
```

### Update Data Version
```powershell
dvc add data/raw
dvc push
git add data/raw.dvc
git commit -m "Description of data changes"
```

### View MLflow Experiments
```powershell
mlflow ui --backend-store-uri models/mlruns
# Open http://localhost:5000
```

---

## File Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Python files | `snake_case.py` | `capture_data.py` |
| Classes | `PascalCase` | `LSCDataset` |
| Functions | `snake_case()` | `get_model()` |
| Constants | `UPPER_SNAKE_CASE` | `IMAGENET_MEAN` |
| Model files | `{arch}_v{ver}_{metric}.pth` | `mobilenet_v2_v1_acc0.95.pth` |
| Blog figures | `{num}-{desc}.png` | `02-training-curves.png` |
| Data images | `{letter}_{seq}_{timestamp}.jpg` | `a_0001_20251127_181813.jpg` |

---

## What's Tracked Where

| Content | Git | DVC | Notes |
|---------|-----|-----|-------|
| Source code | ✅ | - | `src/`, `scripts/` |
| Config files | ✅ | - | `.yaml`, `.json`, `.toml` |
| `.dvc` pointer files | ✅ | - | `data/raw.dvc` |
| Raw images | - | ✅ | `data/raw/` |
| Model weights | - | ✅ | `models/*.pth` (optional) |
| MLflow runs | - | - | Gitignored, local only |
| Blog content | ✅ | - | `blog/` |

---

**Last updated**: 2025-12-31
**Maintainer**: Update when structure changes
