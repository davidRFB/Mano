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
│   │   └── ...
│   └── figures/            # All blog figures (ordered naming)
│       ├── 01-class-distribution.png
│       ├── 02-training-curves.png
│       └── ...
├── data/
│   ├── raw/                # Original gesture images (DVC tracked)
│   ├── raw.dvc             # DVC pointer file (Git tracked)
│   ├── processed/          # Preprocessed data
│   └── splits/             # Train/val/test indices
├── models/                 # Trained model checkpoints
│   ├── mlruns/             # MLflow experiment tracking
│   └── *.pth               # Model weights (gitignored)
├── notebooks/              # Working Jupyter notebooks (exploratory)
│   ├── 01_data_analysis.ipynb
│   └── 02_model_analysis.ipynb
├── scripts/                # Utility scripts
│   └── capture_data.py     # Data collection tool
├── src/
│   ├── cv_model/           # Computer vision model code
│   │   ├── __init__.py
│   │   ├── preprocessing.py # Data transforms, Dataset class
│   │   ├── train.py        # Training loop with MLflow
│   │   └── inference.py    # Real-time prediction
│   ├── api/                # FastAPI application (planned)
│   ├── llm/                # LLM integration (planned)
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

### `src/cv_model/`
**Purpose**: Gesture recognition CNN

| File | Description |
|------|-------------|
| `preprocessing.py` | Dataset class, transforms, augmentation, DataLoaders |
| `train.py` | Training script with MLflow, early stopping, checkpoints |
| `inference.py` | Real-time webcam prediction with MediaPipe |

**Key functions**:
- `preprocessing.create_dataloaders()` - Create train/val/test DataLoaders
- `preprocessing.LSCDataset` - PyTorch Dataset for gesture images
- `train.train()` - Main training entry point
- `train.get_model()` - Get pretrained torchvision model
- `inference.main()` - Run real-time inference

### `scripts/capture_data.py`
**Purpose**: Data collection tool

- Hand detection with MediaPipe
- Auto-crop to hand region
- Press a-z to capture, ESC to quit
- Run from `scripts/` directory

---

## Configuration Files

### DVC
- `.dvc/config` - Remote storage configuration
- `.dvcignore` - Files to exclude from DVC tracking
- `data/raw.dvc` - Pointer to data version

### MLflow
- `models/mlruns/` - Experiment tracking data (gitignored)
- Access via: `mlflow ui` (run from project root)

### Docker (planned)
- `docker/Dockerfile.api` - API container image
- `docker/Dockerfile.frontend` - Frontend container image
- `docker-compose.yml` - Local orchestration

---

## Data Flow

```
1. DATA COLLECTION
   Webcam → capture_data.py → data/raw/{letter}/*.jpg
   
2. TRAINING
   data/raw/ → preprocessing.py → train.py → models/*.pth
                                           → models/mlruns/

3. INFERENCE
   Webcam → inference.py → MediaPipe → Model → Predictions
```

---

## Common Tasks

### Capture New Training Data
```powershell
cd scripts
python capture_data.py
# Press a-z to capture, ESC to quit
```

### Train Model
```powershell
python -m src.cv_model.train --model mobilenet_v2 --epochs 30
```

### Run Inference
```powershell
python -m src.cv_model.inference
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

**Last updated**: 2025-11-27  
**Maintainer**: Update when structure changes
