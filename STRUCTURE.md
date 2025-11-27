# STRUCTURE.md - Project Organization

Quick reference for file locations and purposes. **Update this when adding new files/directories.**

---

## Directory Tree

```
Mano/
├── .github/
│   └── workflows/          # CI/CD pipelines
│       ├── ci.yml          # Testing, linting, docker build
│       └── cd.yml          # Deployment to cloud
├── data/
│   ├── raw/                # Original gesture images (DVC tracked)
│   ├── processed/          # Preprocessed data (DVC tracked)
│   └── splits/             # Train/val/test indices
├── models/                 # Trained model checkpoints (DVC tracked)
├── notebooks/              # Jupyter notebooks for exploration
│   ├── 01_data_analysis.ipynb  # Dataset stats, normalization, input range
│   └── 02_model_eval.ipynb
├── src/
│   ├── cv_model/           # Computer vision model code
│   │   ├── model.py        # Model architecture
│   │   ├── train.py        # Training loop
│   │   ├── inference.py    # Prediction logic
│   │   └── preprocessing.py # Data transforms
│   ├── api/                # FastAPI application
│   │   ├── main.py         # API endpoints
│   │   ├── models.py       # Pydantic schemas
│   │   └── utils.py        # Helper functions
│   ├── llm/                # LLM integration
│   │   └── corrector.py    # Text correction logic
│   └── frontend/           # Streamlit interface
│       └── app.py          # UI application
├── scripts/                # Utility scripts
│   └── capture_data.py     # Data collection tool
├── tests/                  # Test suite
│   ├── test_api.py
│   ├── test_model.py
│   └── fixtures/           # Test data
├── docker/                 # Dockerfiles
│   ├── Dockerfile.api
│   └── Dockerfile.frontend
├── mlruns/                 # MLflow experiments (gitignored)
├── .dvc/                   # DVC configuration
├── .env.example            # Environment variable template
├── .gitignore
├── .pre-commit-config.yaml
├── docker-compose.yml      # Local orchestration
├── requirements.txt        # Python dependencies
├── requirements-dev.txt    # Development dependencies
├── CLAUDE.md               # AI assistant guidelines
├── CHANGELOG.md            # Project changes log
├── STRUCTURE.md            # This file
└── README.md               # User documentation
```

---

## Core Modules

### `src/cv_model/`
**Purpose**: Gesture recognition CNN

- `preprocessing.py`: Dataset class, transforms, augmentation, DataLoaders
- `train.py`: Training script with early stopping, checkpoint saving
- `model.py`: (planned) Custom model definitions
- `inference.py`: (planned) Load model and predict from image

**Key functions**:
- `preprocessing.py::create_dataloaders()` - Create train/val/test DataLoaders
- `preprocessing.py::LSCDataset` - PyTorch Dataset for gesture images
- `train.py::train()` - Main training entry point
- `train.py::get_model()` - Get pretrained torchvision model

---

### `src/api/`
**Purpose**: FastAPI REST API for serving predictions

- `main.py`: FastAPI app with endpoints
- `models.py`: Pydantic request/response schemas
- `utils.py`: Helper functions (image processing, model loading)

**Endpoints**:
- `GET /` - API info
- `GET /health` - Health check
- `POST /predict` - Gesture prediction from image
- `POST /correct_word` - LLM text correction

---

### `src/llm/`
**Purpose**: LLM-based text correction

- `corrector.py`: SignLanguageCorrector class using Claude/GPT

**Key methods**:
- `correct_sequence(letters: list[str]) -> dict` - Correct letter sequence to Spanish word

---

### `src/frontend/`
**Purpose**: Streamlit web interface

- `app.py`: Main UI with webcam capture and display

**Features**:
- Real-time gesture capture
- Letter buffer display
- Word correction trigger

---

## Configuration Files

### Docker
- `docker/Dockerfile.api` - API container image
- `docker/Dockerfile.frontend` - Frontend container image
- `docker-compose.yml` - Run all services locally

### CI/CD
- `.github/workflows/ci.yml` - Run tests on push
- `.github/workflows/cd.yml` - Deploy to cloud on merge

### MLOps
- `.dvc/config` - DVC remote storage config
- `mlruns/` - MLflow experiment tracking data

### Python
- `requirements.txt` - Production dependencies
- `requirements-dev.txt` - Dev tools (pytest, black, flake8)
- `.pre-commit-config.yaml` - Git hooks for code quality

---

## Data Flow

```
Webcam → Frontend (Streamlit)
    ↓
POST /predict → API (FastAPI)
    ↓
CV Model (PyTorch) → Letter prediction
    ↓
Letter buffer accumulation
    ↓
POST /correct_word → API (FastAPI)
    ↓
LLM Corrector → Corrected Spanish text
    ↓
Display in Frontend
```

---

## External Dependencies

### Storage
- **GCS/S3**: Raw data, processed data, model checkpoints
- **Artifact Registry**: Docker images

### Services
- **MLflow**: Experiment tracking (local or cloud)
- **Cloud Run/App Runner**: API hosting
- **Streamlit Cloud**: Frontend hosting (alternative)

### APIs
- **Anthropic/OpenAI**: LLM text correction

---

## Common Tasks

### Add new API endpoint
1. Define Pydantic model in `src/api/models.py`
2. Add endpoint in `src/api/main.py`
3. Add test in `tests/test_api.py`
4. Update this file's Endpoints section

### Add new model architecture
1. Define in `src/cv_model/model.py`
2. Update `train.py` to support new arch
3. Log experiment with MLflow
4. Update CHANGELOG.md

### Add new dependency
1. Add to `requirements.txt` or `requirements-dev.txt`
2. Rebuild Docker images
3. Update CHANGELOG.md

---

## File Naming Conventions

- **Python files**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions**: `snake_case()`
- **Constants**: `UPPER_SNAKE_CASE`
- **Model files**: `{arch}_v{version}_{metric}.pth`
- **Docker images**: `lsc-connect-{service}:v{version}`

---

## Testing

- `tests/test_*.py` - Unit tests
- `tests/fixtures/` - Test data (sample images, mock responses)
- Run with: `pytest tests/`

---

## Notes

- **DVC tracks**: `data/`, `models/`
- **Git tracks**: Code, configs, `.dvc` files
- **Gitignored**: `mlruns/`, `venv/`, `*.pth`, `.env`

---

**Last updated**: 2025-11-27  
**Maintainer**: Update when structure changes