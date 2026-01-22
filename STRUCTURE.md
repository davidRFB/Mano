# STRUCTURE.md - Project Organization

Quick reference for file locations and purposes. **Update this when adding new files/directories.**

---

## Directory Tree

```
Mano/
├── api/                        # FastAPI application
│   ├── main.py                 # API endpoints
│   └── Dockerfile              # Container image
├── blog/                       # Project documentation/blog
│   ├── figures/                # Generated figures
│   └── posts/                  # Blog posts
├── data/
│   ├── raw/                    # Original gesture images (DVC tracked)
│   ├── raw_landmarks/          # Letter landmarks (.npy, 21 landmarks)
│   ├── raw_words/              # Word landmarks (.npy, 51 landmarks)
│   └── webscrapping/           # Video datasets
├── models/
│   ├── checkpoints/            # Trained model checkpoints
│   └── mlruns/                 # MLflow experiment tracking
├── notebooks/                  # Jupyter notebooks (exploration)
│   ├── 01_data_analysis.ipynb
│   ├── 02_model_analysis.ipynb
│   └── ...
├── scripts/                    # Main entry point scripts
│   ├── 01_capture_data.py      # Capture landmarks from webcam
│   ├── 03_train.py             # Train models
│   ├── 04_evaluate.py          # Evaluate and generate metrics
│   └── 05_demo.py              # Real-time demo
├── src/
│   ├── data/                   # Data handling
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
│   │   └── metrics.py          # Evaluation metrics
│   ├── inference/              # Prediction code
│   │   ├── __init__.py
│   │   └── predictor.py        # Model wrapper
│   ├── llm/                    # LLM word correction
│   │   ├── __init__.py
│   │   ├── corrector.py        # Groq/Ollama backends
│   │   └── autocomplete.py     # Word completion
│   └── cv_model/               # [LEGACY] Old CV model code
├── tests/                      # Test suite
├── requirements.txt            # Python dependencies
├── requirements-api.txt        # API-only dependencies
├── CLAUDE.md                   # AI assistant guidelines
├── CHANGELOG.md                # Project changes log
├── STRUCTURE.md                # This file
├── REFACTOR_PLAN.md            # Migration plan
└── README.md                   # User documentation
```

---

## Core Modules

### `src/data/` - Data Handling

| File | Description |
|------|-------------|
| `preprocessing.py` | Feature extraction from landmarks (xy, angles, distances) |
| `dataset.py` | PyTorch Dataset for landmarks, train/val splitting |

**Feature modes:**
- `xy`: 42 features (21 landmarks × 2 coords)
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
| `01_capture_data.py` | Capture hand landmarks from webcam |
| `03_train.py` | Train gesture recognition models |
| `04_evaluate.py` | Evaluate model and generate figures |
| `05_demo.py` | Real-time webcam demo |

### Usage Examples

```bash
# Capture training data
python scripts/01_capture_data.py

# Train a model on all letters
python scripts/03_train.py --model bigru --epochs 100

# Train only on dynamic letters
python scripts/03_train.py --model bigru --letters j,h,z,nn,s

# Evaluate model
python scripts/04_evaluate.py --checkpoint models/checkpoints/<run>/best.pth

# Run demo
python scripts/05_demo.py --checkpoint models/checkpoints/<run>/best.pth
```

---

## API

FastAPI service for landmark-based prediction.

```bash
# Run locally
uvicorn api.main:app --reload --port 8000

# Build Docker
docker build -f api/Dockerfile -t lsc-api .

# Run Docker
docker run -p 8000:8000 lsc-api
```

Endpoints:
- `GET /` - API info
- `GET /health` - Health check
- `POST /predict` - Predict from landmark sequence

---

## Data Formats

### Landmarks (Letters)

```
data/raw_landmarks/{letter}/*.npy
Shape: (20, 21, 3) = 20 frames × 21 landmarks × 3 coords
```

### Landmarks (Words)

```
data/raw_words/{word}/*.npy
Shape: (seq_len, 51, 3) = variable frames × 51 landmarks × 3 coords
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
It's kept for reference but the new pipeline uses `src/data/`, `src/models/`, etc.

---

**Last updated**: 2025-01-22
**Maintainer**: Update when structure changes
