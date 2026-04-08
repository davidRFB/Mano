# STRUCTURE.md - Project Organization

Quick reference for file locations and purposes. **Update this when adding new files/directories.**

---

## Directory Tree

```
Mano/
├── api/                        # FastAPI application
│   ├── main.py                 # API endpoints (landmark-based)
│   └── model.pth               # Trained model artifact (236KB)
├── blog/                       # Project documentation/blog
│   ├── figures/                # Generated figures
│   └── PROGRESS.md             # Progress notes
├── data/
│   ├── raw/                    # Original gesture images (DVC tracked)
│   ├── raw_landmarks/          # Letter landmarks (.npy, 21 landmarks)
│   ├── raw_words/              # Word landmarks (.npy, 51 landmarks)
│   └── webscrapping/           # Video datasets
├── docs/
│   └── index.html              # Web frontend (GitHub Pages)
├── models/
│   ├── checkpoints/            # Trained model checkpoints
│   └── mlruns/                 # MLflow experiment tracking
├── notebooks/                  # Jupyter notebooks (exploration)
│   ├── model_comparison.py
│   └── test.ipynb
├── scripts/                    # Main entry point scripts
│   ├── 01_capture_static.py    # Capture static landmarks from webcam
│   ├── 02_capture_dynamic.py   # Capture dynamic sequences from webcam
│   ├── 03_capture_words.py     # Capture word gestures (Holistic)
│   ├── 04_train.py             # Train models
│   ├── 05_evaluate.py          # Evaluate and generate metrics
│   ├── 06_demo.py              # Real-time webcam demo
│   └── utils/                  # Data processing utilities
│       ├── add_landmarks_to_dataset.py
│       ├── video_to_landmarks.py
│       ├── combine_word_datasets.py
│       └── process_youtube_dataset.py
├── src/
│   ├── preprocessing/          # Data handling
│   │   ├── __init__.py
│   │   ├── preprocessing.py    # Feature extraction, normalization
│   │   └── dataset.py          # PyTorch Dataset classes
│   ├── models/                 # Neural network models
│   │   ├── __init__.py
│   │   ├── static.py           # MLP for static gestures
│   │   └── dynamic.py          # RNN for dynamic gestures
│   ├── training/               # Training utilities
│   │   ├── __init__.py
│   │   ├── trainer.py          # Training loop
│   │   ├── metrics.py          # Evaluation metrics
│   │   └── mlflow_utils.py     # MLflow helpers
│   ├── inference/              # Prediction
│   │   ├── __init__.py
│   │   └── predictor.py        # Unified model wrapper
│   ├── llm/                    # LLM word correction
│   │   ├── __init__.py
│   │   ├── corrector.py        # Groq/Ollama backends
│   │   └── autocomplete.py     # Fast word completion without LLM
│   └── cv_model/               # [LEGACY] Old RGB-based approach
│       └── notebooks/          # Archived exploration notebooks
├── Dockerfile                  # Container image (HF Spaces)
├── requirements.txt            # Python dependencies
├── requirements-api.txt        # API-only dependencies
├── CLAUDE.md                   # AI assistant guidelines
├── CHANGELOG.md                # Project changes log
├── STRUCTURE.md                # This file
└── README.md                   # User documentation (Spanish)
```

---

## Core Modules

### `src/preprocessing/` - Data Handling

| File | Description |
|------|-------------|
| `preprocessing.py` | Feature extraction from landmarks (xy, angles, distances) |
| `dataset.py` | PyTorch Dataset for landmarks, train/val splitting |

**Feature modes:**
- `xy`: 42 features (21 landmarks x 2 coords)
- `xy_angles`: 56 features (xy + 14 finger angles) - DEFAULT
- `xy_angles_distances`: 66 features (above + 10 distances)
- `full`: 108 features (above + velocities)

### `src/models/` - Neural Networks

| File | Description |
|------|-------------|
| `static.py` | MLP for single-frame classification |
| `dynamic.py` | GRU/BiGRU/LSTM for sequence classification |

**Model types:**
- `static`: MLP for letters without movement
- `gru`: Basic GRU
- `bigru`: Bidirectional GRU (recommended)
- `lstm`: Bidirectional LSTM

### `src/training/` - Training Utilities

| File | Description |
|------|-------------|
| `trainer.py` | Training loop with early stopping |
| `metrics.py` | Accuracy, confusion matrix, classification report |
| `mlflow_utils.py` | MLflow experiment tracking helpers |

### `src/inference/` - Prediction

| File | Description |
|------|-------------|
| `predictor.py` | Unified predictor for static/dynamic models |

### `src/llm/` - LLM Correction (Optional)

| File | Description |
|------|-------------|
| `corrector.py` | Spanish word correction with Groq/Ollama |
| `autocomplete.py` | Fast word completion without LLM |

---

## Scripts (Entry Points)

| Script | Description |
|--------|-------------|
| `01_capture_static.py` | Capture static hand landmarks from webcam |
| `02_capture_dynamic.py` | Capture dynamic sequences from webcam |
| `03_capture_words.py` | Capture word gestures with MediaPipe Holistic |
| `04_train.py` | Train gesture recognition models |
| `05_evaluate.py` | Evaluate model and generate figures |
| `06_demo.py` | Real-time webcam demo |

### Usage Examples

```bash
# Capture training data (static letters)
python scripts/01_capture_static.py --mode landmarks

# Train a model on all letters
python scripts/04_train.py --model bigru --epochs 100

# Train only on dynamic letters
python scripts/04_train.py --model bigru --letters j,h,z,nn,s

# Evaluate model
python scripts/05_evaluate.py --checkpoint models/checkpoints/<run>/best.pth

# Run demo
python scripts/06_demo.py --checkpoint models/checkpoints/<run>/best.pth
```

---

## API

FastAPI service for landmark-based prediction, deployed to Hugging Face Spaces.

```bash
# Run locally
uvicorn api.main:app --reload --port 8000

# Build Docker
docker build -t mano-api .

# Run Docker
docker run -p 7860:7860 mano-api
```

Endpoints:
- `GET /` - API info
- `GET /health` - Health check
- `POST /predict` - Predict from landmarks

---

## Data Formats

### Landmarks (Letters)

```
data/raw_landmarks/{letter}/*.npy
Shape: (20, 21, 3) = 20 frames x 21 landmarks x 3 coords
```

### Landmarks (Words)

```
data/raw_words/{word}/*.npy
Shape: (seq_len, 51, 3) = variable frames x 51 landmarks x 3 coords
```

51 landmarks = 9 pose + 21 left hand + 21 right hand

---

## Gesture Classification

### Static Letters (22)
Letters that don't require movement: A-I, K-R, T-Y

### Dynamic Letters (5)
Letters that require movement: J, H, Z, Ñ, S

---

## MLflow Experiments

```bash
# View experiments
mlflow ui --backend-store-uri models/mlruns

# Open http://localhost:5000
```

---

## Legacy Code

The `src/cv_model/` directory contains old code from the RGB-based approach.
It's kept for reference but the active pipeline uses `src/preprocessing/`, `src/models/`, etc.

---

**Last updated**: 2026-04-08
**Maintainer**: Update when structure changes
