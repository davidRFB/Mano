# Mano: Colombian Sign Language Translator
## Complete Project Plan & Implementation Guide

---

## 🎯 Project Overview

**Concept**: A two-model AI pipeline that translates Colombian Sign Language (LSC) gestures into coherent Spanish text.

**Architecture**:
- **Model 1 (Vision)**: CNN-based gesture recognition → Individual letters
- **Model 2 (Language)**: LLM-based correction → Coherent words/phrases

**Learning Focus**: Model serving, FastAPI, Docker, CI/CD, and cloud deployment

---

## 📚 Technology Stack

### Core ML & Development
| Component | Technology | Purpose |
|-----------|-----------|---------|
| CV Framework | PyTorch 2.x | Model training & inference |
| Model Architecture | EfficientNet-B0 or MobileNetV2 | Lightweight, accurate CNN |
| API Framework | FastAPI 0.104+ | High-performance REST API |
| LLM Integration | OpenAI API / Anthropic Claude API | Text correction & reasoning |
| Frontend | Streamlit | Quick webcam interface |

### MLOps & Versioning
| Component | Technology | Purpose |
|-----------|-----------|---------|
| Experiment Tracking | MLflow | Log metrics, parameters, models |
| Data Versioning | DVC (Data Version Control) | Version datasets & model files |
| Model Registry | MLflow Model Registry | Track model versions & stages |

### DevOps & Deployment
| Component | Technology | Purpose |
|-----------|-----------|---------|
| Containerization | Docker | Package application |
| Orchestration | Docker Compose | Local multi-container setup |
| CI/CD | GitHub Actions | Automated testing & deployment |
| Cloud Platform | GCP or AWS | Production hosting |
| Storage | GCS / S3 | Dataset & artifact storage |
| Deployment | Cloud Run / App Runner | Serverless container hosting |
| Monitoring | Cloud Logging + Prometheus (optional) | Track API performance |

---

## 🗓️ Phase-by-Phase Implementation Plan

---

## **PHASE 1: Foundation & Local CV Model** (Weeks 1-4)

### Week 1: Environment Setup & Data Collection

**Goal**: Set up development environment and acquire/create initial dataset

#### Actions:
1. **Repository Setup**
   ```bash
   # Create project structure
   mkdir lsc-connect
   cd lsc-connect
   git init
   
   # Create directory structure
   mkdir -p {data/{raw,processed},models,notebooks,src/{cv_model,api,llm},tests,docker}
   ```

2. **Python Environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install initial dependencies
   pip install torch torchvision pytorch-lightning
   pip install numpy pandas opencv-python pillow
   pip install jupyter notebook ipywidgets
   ```

3. **Data Collection Strategy**
   - **Option A**: Search for existing LSC alphabet datasets (Kaggle, research papers)
   - **Option B**: Create custom dataset
     - Start with 10-15 letters + 1 "space" gesture (closed fist)
     - Capture 100-200 images per gesture
     - Use your webcam with varied backgrounds, lighting, hand positions
     - Tool: Create a simple OpenCV script to capture frames

4. **Data Organization**
   ```
   data/raw/
   ├── A/
   │   ├── img_001.jpg
   │   ├── img_002.jpg
   ├── B/
   ├── ...
   └── SPACE/
   ```

#### Deliverables:
- ✅ GitHub repository initialized
- ✅ 500-1500 images collected (10-15 classes)
- ✅ Development environment configured

---

### Week 2: Model Training v0.1

**Goal**: Train your first working gesture recognition model

#### Actions:
1. **Data Preprocessing Script** (`src/cv_model/preprocessing.py`)
   ```python
   # Key features:
   # - Resize images to 224x224
   # - Normalize (ImageNet stats)
   # - Data augmentation: rotation, flip, brightness
   # - Train/val/test split (70/15/15)
   ```

2. **Model Architecture** (`src/cv_model/model.py`)
   ```python
   # Start simple:
   import torchvision.models as models
   
   model = models.mobilenet_v2(pretrained=True)
   # Freeze early layers
   # Replace classifier for your n_classes
   ```

3. **Training Script** (`src/cv_model/train.py`)
   ```python
   # Basic training loop with:
   # - CrossEntropyLoss
   # - Adam optimizer (lr=0.001)
   # - 20-30 epochs
   # - Early stopping (patience=5)
   # - Save best model checkpoint
   ```

4. **Evaluation Notebook** (`notebooks/01_model_evaluation.ipynb`)
   - Confusion matrix
   - Per-class accuracy
   - Visualize misclassifications

#### Deliverables:
- ✅ Training pipeline working
- ✅ Model achieving >80% validation accuracy
- ✅ Best model saved as `models/gesture_model_v0.1.pth`

---

### Week 3: MLOps Integration (MLflow + DVC)

**Goal**: Add experiment tracking and data versioning

#### Actions:
1. **MLflow Setup**
   ```bash
   pip install mlflow
   
   # Start MLflow server locally
   mlflow ui --host 0.0.0.0 --port 5000
   ```

2. **Integrate MLflow in Training** (`src/cv_model/train.py`)
   ```python
   import mlflow
   import mlflow.pytorch
   
   with mlflow.start_run():
       # Log hyperparameters
       mlflow.log_param("learning_rate", 0.001)
       mlflow.log_param("batch_size", 32)
       mlflow.log_param("model_arch", "mobilenet_v2")
       
       # Training loop
       for epoch in range(epochs):
           # ... training code ...
           mlflow.log_metric("train_loss", train_loss, step=epoch)
           mlflow.log_metric("val_accuracy", val_acc, step=epoch)
       
       # Log model
       mlflow.pytorch.log_model(model, "model")
   ```

3. **DVC Setup**
   ```bash
   pip install dvc dvc-gs  # or dvc-s3 for AWS
   dvc init
   
   # Add data to DVC
   dvc add data/raw
   git add data/raw.dvc .gitignore
   git commit -m "Add raw data to DVC"
   
   # Configure remote storage (GCS example)
   dvc remote add -d storage gs://lsc-connect-data/dvc-storage
   dvc push
   ```

4. **Version Control Best Practices**
   ```bash
   # .gitignore
   data/raw/
   data/processed/
   models/*.pth
   venv/
   __pycache__/
   mlruns/
   ```

#### Deliverables:
- ✅ MLflow tracking experiments (3+ runs logged)
- ✅ Data versioned with DVC and pushed to cloud storage
- ✅ Model checkpoints versioned

---

### Week 4: Initial Model Refinement

**Goal**: Improve model performance and prepare for API integration

#### Actions:
1. **Hyperparameter Tuning**
   - Test different learning rates [0.0001, 0.001, 0.01]
   - Try different architectures (EfficientNet-B0 vs MobileNetV2)
   - Experiment with data augmentation intensity
   - Log all experiments with MLflow

2. **Model Optimization**
   ```python
   # Add techniques:
   # - Learning rate scheduling
   # - Class weight balancing (if imbalanced)
   # - Mixup/CutMix augmentation
   ```

3. **Inference Testing**
   ```python
   # Create inference script (src/cv_model/inference.py)
   def predict_gesture(image_path, model_path):
       model = load_model(model_path)
       image = preprocess_image(image_path)
       prediction = model(image)
       return prediction.argmax()
   ```

4. **Real-time Testing**
   - Create notebook to test with webcam
   - Measure inference speed (should be <100ms per frame)

#### Deliverables:
- ✅ Model accuracy >85% on validation set
- ✅ Inference script working
- ✅ Ready for API integration

---


### LLM Integration

**Goal**: Add text correction capabilities

#### Actions:
1. **LLM Setup**
   ```bash
   pip install openai anthropic langchain
   ```

2. **LLM Service** (`src/llm/corrector.py`)
   ```python
   from anthropic import Anthropic
   import os
   
   class SignLanguageCorrector:
       def __init__(self):
           self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
       
       def correct_sequence(self, letters: list[str]) -> dict:
           """
           Correct a sequence of letters to form Spanish words
           """
           prompt = f"""You are an expert in Spanish language and Colombian Sign Language.
   
   The following sequence of letters was recognized from sign language gestures:
   {' '.join(letters)}
   
   Task:
   1. Correct any spelling errors
   2. Form coherent Spanish word(s)
   3. Consider common LSC recognition errors
   
   Return only the corrected word(s) in Spanish."""
   
           message = self.client.messages.create(
               model="claude-sonnet-4-20250514",
               max_tokens=100,
               messages=[
                   {"role": "user", "content": prompt}
               ]
           )
           
           corrected_text = message.content[0].text.strip()
           
           return {
               "original_sequence": letters,
               "corrected_text": corrected_text,
               "confidence": "high"  # Could add logic here
           }
   ```


4. **Test LLM Integration**
   ```python
   # Test cases
   test_sequences = [
       ["H", "O", "L", "A"],         # Easy case
       ["C", "A", "S", "S", "A"],    # Common word
       ["M", "U", "N", "D", "O"],    # Another test
   ]
   ```

#### Deliverables:
- ✅ LLM integration working

---
## **PHASE 2: API Development & Containerization** (Weeks 5-8)

### Week 5: FastAPI Development

**Goal**: Create a REST API to serve your model

#### Actions:
1. **FastAPI Setup**
   ```bash
   pip install fastapi uvicorn python-multipart pillow
   ```

2. **Basic API Structure** (`src/api/main.py`)
   ```python
   from fastapi import FastAPI, File, UploadFile
   from fastapi.middleware.cors import CORSMiddleware
   import torch
   from PIL import Image
   import io
   
   app = FastAPI(title="LSC-Connect API", version="1.0.0")
   
   # Add CORS middleware
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["*"],
       allow_methods=["*"],
       allow_headers=["*"],
   )
   
   # Load model at startup
   @app.on_event("startup")
   def load_model():
       global model, class_names
       model = torch.load("models/gesture_model_v0.1.pth")
       model.eval()
       class_names = ["A", "B", "C", ..., "SPACE"]
   
   @app.get("/")
   def root():
       return {"message": "LSC-Connect API", "status": "running"}
   
   @app.post("/predict")
   async def predict_gesture(file: UploadFile = File(...)):
       # Read image
       image_data = await file.read()
       image = Image.open(io.BytesIO(image_data))
       
       # Preprocess
       image_tensor = preprocess(image)
       
       # Predict
       with torch.no_grad():
           output = model(image_tensor.unsqueeze(0))
           prediction = output.argmax(dim=1).item()
       
       return {
           "letter": class_names[prediction],
           "confidence": float(output.softmax(dim=1).max())
       }
   
   @app.get("/health")
   def health_check():
       return {"status": "healthy"}
   ```

3. **Test Locally**
   ```bash
   uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
   
   # Test with curl
   curl -X POST "http://localhost:8000/predict" \
        -F "file=@test_image.jpg"
   ```

4. **API Documentation**
   - Access automatic docs at `http://localhost:8000/docs`
   - Test all endpoints interactively

#### Deliverables:
- ✅ FastAPI app serving model predictions
- ✅ `/predict` endpoint working with image uploads
- ✅ API documentation generated

---

### Week 6: Dockerization

**Goal**: Containerize your API for consistent deployment

#### Actions:
1. **Create Dockerfile** (`docker/Dockerfile.api`)
   ```dockerfile
   FROM python:3.10-slim
   
   WORKDIR /app
   
   # Install system dependencies
   RUN apt-get update && apt-get install -y \
       libgl1-mesa-glx \
       libglib2.0-0 \
       && rm -rf /var/lib/apt/lists/*
   
   # Copy requirements
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   
   # Copy application code
   COPY src/ ./src/
   COPY models/ ./models/
   
   # Expose port
   EXPOSE 8000
   
   # Run the application
   CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
   ```

2. **Requirements File** (`requirements.txt`)
   ```txt
   fastapi==0.104.1
   uvicorn[standard]==0.24.0
   torch==2.1.0
   torchvision==0.16.0
   pillow==10.1.0
   python-multipart==0.0.6
   numpy==1.26.0
   opencv-python-headless==4.8.1.78
   ```

3. **Build and Test**
   ```bash
   # Build image
   docker build -f docker/Dockerfile.api -t lsc-connect-api:v1 .
   
   # Run container
   docker run -p 8000:8000 lsc-connect-api:v1
   
   # Test
   curl http://localhost:8000/health
   ```

4. **Docker Compose** (`docker-compose.yml`)
   ```yaml
   version: '3.8'
   
   services:
     api:
       build:
         context: .
         dockerfile: docker/Dockerfile.api
       ports:
         - "8000:8000"
       volumes:
         - ./models:/app/models
       environment:
         - MODEL_PATH=/app/models/gesture_model_v0.1.pth
   
     mlflow:
       image: ghcr.io/mlflow/mlflow:v2.8.0
       ports:
         - "5000:5000"
       volumes:
         - ./mlruns:/mlflow/mlruns
       command: mlflow server --host 0.0.0.0 --port 5000
   ```

#### Deliverables:
- ✅ Docker image builds successfully
- ✅ Containerized API works locally
- ✅ Docker Compose orchestrates multiple services

---

### Week 7: Testing & Documentation

**Goal**: Add comprehensive tests and documentation

#### Actions:
1. **Unit Tests** (`tests/test_api.py`)
   ```python
   import pytest
   from fastapi.testclient import TestClient
   from src.api.main import app
   
   client = TestClient(app)
   
   def test_root():
       response = client.get("/")
       assert response.status_code == 200
       assert "message" in response.json()
   
   def test_health_check():
       response = client.get("/health")
       assert response.status_code == 200
       assert response.json()["status"] == "healthy"
   
   def test_predict_endpoint():
       # Create test image
       with open("test_data/sample_A.jpg", "rb") as f:
           response = client.post(
               "/predict",
               files={"file": ("test.jpg", f, "image/jpeg")}
           )
       assert response.status_code == 200
       assert "letter" in response.json()
       assert "confidence" in response.json()
   ```

2. **Model Tests** (`tests/test_model.py`)
   ```python
   def test_model_loads():
       model = load_model("models/gesture_model_v0.1.pth")
       assert model is not None
   
   def test_inference_shape():
       # Test output shape is correct
       pass
   ```

3. **Integration Tests**
   ```python
   # Test full pipeline: image → API → prediction
   ```

4. **Documentation**
   - Update README.md with:
     - Project overview
     - Installation instructions
     - API usage examples
     - Architecture diagram

#### Deliverables:
- ✅ Test suite with >80% coverage
- ✅ All tests passing
- ✅ Comprehensive README

---

### Week 8: Performance Optimization

**Goal**: Optimize API for production

#### Actions:
1. **Model Optimization**
   ```python
   # Quantize model for faster inference
   quantized_model = torch.quantization.quantize_dynamic(
       model, {torch.nn.Linear}, dtype=torch.qint8
   )
   ```

2. **API Improvements**
   - Add request validation with Pydantic models
   - Implement rate limiting
   - Add logging
   ```python
   import logging
   from fastapi import HTTPException
   from slowapi import Limiter
   from slowapi.util import get_remote_address
   
   limiter = Limiter(key_func=get_remote_address)
   app.state.limiter = limiter
   
   @app.post("/predict")
   @limiter.limit("60/minute")
   async def predict_gesture(...):
       # Existing code
   ```

3. **Caching Strategy**
   - Cache model in memory (already done at startup)
   - Consider Redis for result caching if needed

4. **Load Testing**
   ```bash
   pip install locust
   # Create load test script
   # Test with 100 concurrent users
   ```

#### Deliverables:
- ✅ API response time <200ms
- ✅ Can handle 100 req/s
- ✅ Optimized Docker image (<1GB)

---

## **PHASE 3: LLM Integration & Frontend** (Weeks 9-12)



### Week 10: Streamlit Frontend

**Goal**: Create user interface for webcam interaction

#### Actions:
1. **Streamlit Setup**
   ```bash
   pip install streamlit streamlit-webrtc opencv-python-headless
   ```

2. **Frontend Application** (`src/frontend/app.py`)
   ```python
   import streamlit as st
   import requests
   from streamlit_webrtc import webrtc_streamer
   import cv2
   import numpy as np
   from PIL import Image
   import io
   
   st.title("🤟 LSC-Connect: Sign Language Translator")
   st.markdown("Make sign language gestures to spell words")
   
   # Configuration
   API_URL = st.sidebar.text_input(
       "API URL", 
       value="http://localhost:8000"
   )
   
   # Session state for letter buffer
   if 'letter_buffer' not in st.session_state:
       st.session_state.letter_buffer = []
   
   # UI Layout
   col1, col2 = st.columns([2, 1])
   
   with col1:
       st.subheader("📹 Webcam Feed")
       
       # Webcam component
       webrtc_ctx = webrtc_streamer(
           key="sign-language",
           video_frame_callback=process_frame,
           rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
       )
   
   with col2:
       st.subheader("📝 Recognized Letters")
       st.text_area(
           "Buffer", 
           value=' '.join(st.session_state.letter_buffer),
           height=100
       )
       
       col_a, col_b = st.columns(2)
       with col_a:
           if st.button("✨ Correct Word"):
               correct_word()
       with col_b:
           if st.button("🗑️ Clear"):
               st.session_state.letter_buffer = []
               st.rerun()
       
       st.subheader("✅ Corrected Text")
       if 'corrected_text' in st.session_state:
           st.success(st.session_state.corrected_text)
   
   def process_frame(frame):
       """Process each video frame"""
       img = frame.to_ndarray(format="bgr24")
       
       # Convert to PIL Image
       pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
       
       # Call API every N frames (e.g., every 15 frames = 0.5s at 30fps)
       if frame.time_counter % 15 == 0:
           predict_gesture(pil_img)
       
       return img
   
   def predict_gesture(image):
       """Call prediction API"""
       try:
           # Convert PIL to bytes
           buf = io.BytesIO()
           image.save(buf, format='JPEG')
           buf.seek(0)
           
           # API call
           response = requests.post(
               f"{API_URL}/predict",
               files={"file": ("frame.jpg", buf, "image/jpeg")},
               timeout=2
           )
           
           if response.status_code == 200:
               result = response.json()
               letter = result['letter']
               confidence = result['confidence']
               
               # Add to buffer if confident enough
               if confidence > 0.8 and letter != "SPACE":
                   st.session_state.letter_buffer.append(letter)
               elif letter == "SPACE":
                   correct_word()
                   
       except Exception as e:
           st.error(f"Prediction error: {e}")
   
   def correct_word():
       """Call correction API"""
       if not st.session_state.letter_buffer:
           return
       
       try:
           response = requests.post(
               f"{API_URL}/correct_word",
               json={"letters": st.session_state.letter_buffer},
               timeout=5
           )
           
           if response.status_code == 200:
               result = response.json()
               st.session_state.corrected_text = result['corrected_text']
               st.session_state.letter_buffer = []
               st.rerun()
               
       except Exception as e:
           st.error(f"Correction error: {e}")
   ```

3. **Run Frontend**
   ```bash
   streamlit run src/frontend/app.py
   ```

4. **Docker for Frontend** (`docker/Dockerfile.frontend`)
   ```dockerfile
   FROM python:3.10-slim
   
   WORKDIR /app
   
   RUN apt-get update && apt-get install -y \
       libgl1-mesa-glx \
       && rm -rf /var/lib/apt/lists/*
   
   COPY requirements-frontend.txt .
   RUN pip install --no-cache-dir -r requirements-frontend.txt
   
   COPY src/frontend/ ./src/frontend/
   
   EXPOSE 8501
   
   CMD ["streamlit", "run", "src/frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

#### Deliverables:
- ✅ Working Streamlit interface
- ✅ Real-time gesture recognition
- ✅ Word correction integration

---

### Week 11-12: System Integration & Polish

**Goal**: Connect all components and refine user experience

#### Actions:
1. **Update Docker Compose** (add frontend service)
   ```yaml
   services:
     api:
       # ... existing config ...
     
     frontend:
       build:
         context: .
         dockerfile: docker/Dockerfile.frontend
       ports:
         - "8501:8501"
       environment:
         - API_URL=http://api:8000
       depends_on:
         - api
     
     mlflow:
       # ... existing config ...
   ```

2. **End-to-End Testing**
   - Test full pipeline with different words
   - Test error cases (bad images, network failures)
   - Measure latency: webcam → API → LLM → display

3. **UX Improvements**
   - Add visual feedback for predictions
   - Show confidence scores
   - Add gesture guide/hints
   - Implement word history

4. **Performance Monitoring**
   ```python
   # Add metrics to API
   from prometheus_client import Counter, Histogram
   
   prediction_counter = Counter('predictions_total', 'Total predictions')
   prediction_duration = Histogram('prediction_duration_seconds', 'Prediction duration')
   ```

#### Deliverables:
- ✅ All services running via Docker Compose
- ✅ End-to-end demo working
- ✅ Polished user experience

---

## **PHASE 4: CI/CD & Cloud Deployment** (Weeks 13-16)

### Week 13: GitHub Actions CI Pipeline

**Goal**: Automate testing and validation

#### Actions:
1. **GitHub Actions Workflow** (`.github/workflows/ci.yml`)
   ```yaml
   name: CI Pipeline
   
   on:
     push:
       branches: [ main, develop ]
     pull_request:
       branches: [ main ]
   
   jobs:
     test:
       runs-on: ubuntu-latest
       
       steps:
       - uses: actions/checkout@v3
       
       - name: Set up Python
         uses: actions/setup-python@v4
         with:
           python-version: '3.10'
       
       - name: Install dependencies
         run: |
           python -m pip install --upgrade pip
           pip install -r requirements.txt
           pip install pytest pytest-cov
       
       - name: Run tests
         run: |
           pytest tests/ --cov=src --cov-report=xml
       
       - name: Upload coverage
         uses: codecov/codecov-action@v3
         with:
           file: ./coverage.xml
     
     lint:
       runs-on: ubuntu-latest
       
       steps:
       - uses: actions/checkout@v3
       
       - name: Set up Python
         uses: actions/setup-python@v4
         with:
           python-version: '3.10'
       
       - name: Install linters
         run: |
           pip install black flake8 mypy
       
       - name: Run Black
         run: black --check src/
       
       - name: Run Flake8
         run: flake8 src/
       
       - name: Run MyPy
         run: mypy src/
     
     docker-build:
       runs-on: ubuntu-latest
       needs: [test, lint]
       
       steps:
       - uses: actions/checkout@v3
       
       - name: Build Docker image
         run: |
           docker build -f docker/Dockerfile.api -t lsc-connect-api:ci .
       
       - name: Test Docker image
         run: |
           docker run -d -p 8000:8000 --name test-api lsc-connect-api:ci
           sleep 10
           curl -f http://localhost:8000/health || exit 1
           docker stop test-api
   ```

2. **Pre-commit Hooks** (`.pre-commit-config.yaml`)
   ```yaml
   repos:
     - repo: https://github.com/psf/black
       rev: 23.10.0
       hooks:
         - id: black
     
     - repo: https://github.com/pycqa/flake8
       rev: 6.1.0
       hooks:
         - id: flake8
     
     - repo: https://github.com/pre-commit/mirrors-mypy
       rev: v1.6.0
       hooks:
         - id: mypy
   ```

3. **Configure Branch Protection**
   - Require CI to pass before merging
   - Require code review
   - Enforce linear history

#### Deliverables:
- ✅ CI pipeline running on every push
- ✅ All checks passing
- ✅ Code quality gates enforced

---

### Week 14: Cloud Setup (GCP)

**Goal**: Prepare cloud infrastructure

#### Actions:
1. **GCP Project Setup**
   ```bash
   # Install gcloud CLI
   # Create project
   gcloud projects create lsc-connect --name="LSC Connect"
   gcloud config set project lsc-connect
   
   # Enable required APIs
   gcloud services enable \
       cloudbuild.googleapis.com \
       run.googleapis.com \
       artifactregistry.googleapis.com \
       storage.googleapis.com
   ```

2. **Artifact Registry**
   ```bash
   # Create Docker repository
   gcloud artifacts repositories create lsc-connect-repo \
       --repository-format=docker \
       --location=us-central1 \
       --description="LSC Connect Docker images"
   
   # Configure Docker authentication
   gcloud auth configure-docker us-central1-docker.pkg.dev
   ```

3. **Cloud Storage**
   ```bash
   # Create bucket for models and data
   gsutil mb -l us-central1 gs://lsc-connect-models
   gsutil mb -l us-central1 gs://lsc-connect-data
   
   # Upload model
   gsutil cp models/gesture_model_v0.1.pth gs://lsc-connect-models/
   ```

4. **Service Account**
   ```bash
   # Create service account for Cloud Run
   gcloud iam service-accounts create lsc-connect-sa \
       --display-name="LSC Connect Service Account"
   
   # Grant permissions
   gcloud projects add-iam-policy-binding lsc-connect \
       --member="serviceAccount:lsc-connect-sa@lsc-connect.iam.gserviceaccount.com" \
       --role="roles/storage.objectViewer"
   ```

#### Deliverables:
- ✅ GCP project configured
- ✅ Artifact Registry ready
- ✅ Cloud Storage buckets created

---

### Week 15: Cloud Run Deployment

**Goal**: Deploy API to production

#### Actions:
1. **Build and Push Docker Image**
   ```bash
   # Tag image
   docker tag lsc-connect-api:v1 \
       us-central1-docker.pkg.dev/lsc-connect/lsc-connect-repo/api:v1
   
   # Push to Artifact Registry
   docker push us-central1-docker.pkg.dev/lsc-connect/lsc-connect-repo/api:v1
   ```

2. **Deploy to Cloud Run**
   ```bash
   gcloud run deploy lsc-connect-api \
       --image us-central1-docker.pkg.dev/lsc-connect/lsc-connect-repo/api:v1 \
       --platform managed \
       --region us-central1 \
       --allow-unauthenticated \
       --memory 2Gi \
       --cpu 2 \
       --timeout 60 \
       --max-instances 10 \
       --service-account lsc-connect-sa@lsc-connect.iam.gserviceaccount.com \
       --set-env-vars MODEL_PATH=/app/models/gesture_model_v0.1.pth
   ```

3. **Configure Custom Domain (Optional)**
   ```bash
   gcloud run domain-mappings create \
       --service lsc-connect-api \
       --domain api.lsc-connect.com \
       --region us-central1
   ```

4. **CD Pipeline** (`.github/workflows/cd.yml`)
   ```yaml
   name: CD Pipeline
   
   on:
     push:
       branches: [ main ]
       tags: [ 'v*' ]
   
   jobs:
     deploy:
       runs-on: ubuntu-latest
       
       steps:
       - uses: actions/checkout@v3
       
       - name: Authenticate to Google Cloud
         uses: google-github-actions/auth@v1
         with:
           credentials_json: ${{ secrets.GCP_SA_KEY }}
       
       - name: Set up Cloud SDK
         uses: google-github-actions/setup-gcloud@v1
       
       - name: Configure Docker
         run: gcloud auth configure-docker us-central1-docker.pkg.dev
       
       - name: Build and Push
         run: |
           docker build -f docker/Dockerfile.api -t api:${{ github.sha }} .
           docker tag api:${{ github.sha }} \
               us-central1-docker.pkg.dev/lsc-connect/lsc-connect-repo/api:${{ github.sha }}
           docker push us-central1-docker.pkg.dev/lsc-connect/lsc-connect-repo/api:${{ github.sha }}
       
       - name: Deploy to Cloud Run
         run: |
           gcloud run deploy lsc-connect-api \
               --image us-central1-docker.pkg.dev/lsc-connect/lsc-connect-repo/api:${{ github.sha }} \
               --region us-central1 \
               --platform managed
   ```

#### Deliverables:
- ✅ API deployed to Cloud Run
- ✅ Public HTTPS URL available
- ✅ Auto-scaling configured

---

### Week 16: Frontend Deployment & Monitoring

**Goal**: Deploy frontend and set up monitoring

#### Actions:
1. **Deploy Frontend to Streamlit Cloud**
   - Push code to GitHub
   - Connect repository to Streamlit Cloud
   - Configure environment variables (API_URL)
   - Deploy

2. **Alternative: Cloud Run for Frontend**
   ```bash
   # Build and push frontend image
   docker build -f docker/Dockerfile.frontend -t frontend:v1 .
   docker tag frontend:v1 \
       us-central1-docker.pkg.dev/lsc-connect/lsc-connect-repo/frontend:v1
   docker push us-central1-docker.pkg.dev/lsc-connect/lsc-connect-repo/frontend:v1
   
   # Deploy
   gcloud run deploy lsc-connect-frontend \
       --image us-central1-docker.pkg.dev/lsc-connect/lsc-connect-repo/frontend:v1 \
       --region us-central1 \
       --allow-unauthenticated
   ```

3. **Cloud Monitoring Setup**
   ```bash
   # Enable Cloud Monitoring API
   gcloud services enable monitoring.googleapis.com
   
   # Create custom dashboard
   # Monitor: latency, error rate, request count
   ```

4. **Logging**
   ```python
   # Update API with structured logging
   import logging
   from google.cloud import logging as cloud_logging
   
   # Set up Cloud Logging
   client = cloud_logging.Client()
   client.setup_logging()
   
   logger = logging.getLogger(__name__)
   
   @app.post("/predict")
   async def predict_gesture(...):
       logger.info("Prediction request received", extra={
           "confidence": confidence,
           "predicted_letter": letter
       })
   ```

5. **Alerts**
   - Set up alerts for high error rates
   - Alert on latency > 1s
   - Alert on cost thresholds

#### Deliverables:
- ✅ Full application deployed and accessible
- ✅ Monitoring dashboard configured
- ✅ Alerts set up

---

## 📊 Success Metrics

### Technical Metrics
- Model accuracy: >85% on test set
- API latency: <200ms (95th percentile)
- System uptime: >99%
- Test coverage: >80%

### Learning Outcomes
- ✅ PyTorch model training & optimization
- ✅ FastAPI development & best practices
- ✅ Docker containerization & orchestration
- ✅ MLOps with MLflow & DVC
- ✅ CI/CD with GitHub Actions
- ✅ Cloud deployment (GCP/AWS)
- ✅ LLM integration & prompt engineering

---

## 🚀 Bonus Extensions (Post-Week 16)

Once core project is complete, consider:

1. **Advanced Features**
   - Word-level LSC support (beyond alphabet)
   - Sentence completion
   - Multiple language support

2. **MLOps Maturity**
   - Model A/B testing
   - Automated retraining pipeline
   - Feature store integration

3. **Architecture**
   - Kubernetes deployment
   - GraphQL API
   - Real-time WebSocket communication

4. **Mobile App**
   - React Native or Flutter app
   - On-device inference with TensorFlow Lite

---

## 📚 Learning Resources

### FastAPI
- Official docs: https://fastapi.tiangolo.com/
- "Building Python Microservices with FastAPI" (book)

### Docker & Containers
- Docker docs: https://docs.docker.com/
- "Docker Deep Dive" by Nigel Poulton

### MLOps
- MLflow docs: https://mlflow.org/docs/
- DVC docs: https://dvc.org/doc
- "Introducing MLOps" by O'Reilly

### Cloud (GCP)
- Google Cloud Skills Boost
- "Google Cloud Platform in Action" (book)

### CI/CD
- GitHub Actions docs: https://docs.github.com/en/actions
- "Continuous Delivery" by Jez Humble

---

## 🎯 Portfolio Impact

This project demonstrates:

1. **Full-Stack ML Engineering**: Not just model training, but production deployment
2. **Modern MLOps Practices**: Experiment tracking, versioning, CI/CD
3. **Cloud-Native Architecture**: Containerization, serverless deployment, scalability
4. **System Design**: Two-model pipeline shows architectural thinking
5. **Social Impact**: Accessibility technology with real-world application

**Resume Bullet Points**:
- "Architected and deployed a cloud-native sign language translation system combining CNN-based gesture recognition with LLM-powered text correction"
- "Built production ML pipeline with MLflow experiment tracking, DVC versioning, and automated CI/CD via GitHub Actions"
- "Developed and containerized FastAPI microservice serving PyTorch models with <200ms latency on Google Cloud Run"

---

## 📝 Project Checklist

- [ ] Phase 1 Complete: CV model trained and optimized
- [ ] Phase 2 Complete: API containerized and tested
- [ ] Phase 3 Complete: LLM integrated, frontend functional
- [ ] Phase 4 Complete: CI/CD pipeline working, deployed to cloud
- [ ] Documentation complete (README, API docs, architecture diagram)
- [ ] GitHub repository public and polished
- [ ] Demo video recorded
- [ ] Blog post written (optional but recommended)

---

**Good luck with your LSC-Connect project! 🚀**