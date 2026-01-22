# GEMINI.md - Project Context & Guidelines

## Project Overview
**MANO** is a Colombian Sign Language (LSC) translation system. It utilizes a pipeline of Computer Vision (MediaPipe), Deep Learning (PyTorch), and Large Language Models (LLM) to translate static gestures (letters) and dynamic sequences (words) into text.

The project emphasizes a clean, structured workflow from data capture to deployment, supporting both static frames (letters) and movement-based gestures (dynamic letters, whole words).

## Core Mandates
- **Simplicity & Cleanliness**: Prioritize simple, readable code over complex abstractions. Maintain clean pipelines.
- **Idiomatic Python**: Follow PEP 8, use type hints, and adhere to the project's existing style (Black formatter).
- **Ask Before Assuming**: If requirements for a pipeline step are ambiguous, ask for clarification.
- **Documentation**: Update `CHANGELOG.md` and `STRUCTURE.md` when features are completed and explicitly finalized by the user.

## Architecture & Workflows

### Script Pipeline (The "Golden Path")
The project follows a strict numbered workflow for reproducibility. Always adhere to this sequence:

1.  **`scripts/01_capture_static.py`**: Capture single-frame data (Static Letters).
    *   Modes: `photo`, `photo_landmarks`, `landmarks`.
2.  **`scripts/02_capture_dynamic.py`**: Capture sequence data (Dynamic Letters).
    *   Modes: `landmarks`, `video`, `both`.
3.  **`scripts/03_capture_words.py`**: Capture word gestures (Holistic).
    *   Uses MediaPipe Holistic (51 landmarks: 9 pose + 21 L-hand + 21 R-hand).
4.  **`scripts/04_train.py`**: Train models.
    *   Supports various architectures (GRU, BiGRU, LSTM) and feature modes.
    *   Tracks experiments via MLflow.
5.  **`scripts/05_evaluate.py`**: Evaluate models and generate metrics/plots.
6.  **`scripts/06_demo.py`**: Real-time demonstration/inference.

### Directory Structure
- **`src/`**: Core library code.
    - `data/`: Preprocessing, Dataset classes, feature extraction.
    - `models/`: PyTorch model architectures (Static MLP, Dynamic RNNs).
    - `training/`: Training loops, trainers, and metrics.
    - `inference/`: Real-time prediction logic and predictors.
    - `llm/`: Word correction (Groq/Ollama) and autocompletion.
- **`data/`**: Storage for raw images, landmarks (`.npy`), and DVC files.
- **`models/`**: Checkpoints (`checkpoints/`) and MLflow runs (`mlruns/`).
- **`api/`**: FastAPI implementation for deployment.
- **`blog/`**: Project documentation and experiment logging (Quarto).

## Coding Standards

### Python
- **Formatter**: Black (line length 88).
- **Linter**: Flake8.
- **Type Hints**: Mandatory for function signatures.
- **Docstrings**: Google style for public interfaces.
- **Imports**: Absolute imports preferred.

### MLOps
- **Experiment Tracking**: MLflow for metrics, parameters, and artifacts.
- **Data Versioning**: DVC for `data/raw` and other large datasets.
- **Environment**: `requirements.txt` or Micromamba environment `Mano`.

## Key Technologies
- **Computer Vision**: MediaPipe (Hands & Holistic).
- **Deep Learning**: PyTorch (MobileNet, ResNet for images; GRU/LSTM for landmarks).
- **Deployment**: FastAPI, Docker.
- **LLM**: Groq / Ollama for post-processing corrections.

## Operational Guide
- **Data Flow**: `Capture` -> `Preprocess (.npy)` -> `Train` -> `Evaluate` -> `Deploy`.
- **Feature Modes**: Be aware of different feature sets for landmarks (e.g., `xy`, `xy_angles`, `full`).
- **Blog**: Documentation of the journey lives in `blog/`. Update `.qmd` files if requested to document findings or add figures.
