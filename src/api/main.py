"""
LSC-Connect API - Minimal Version

Run with:
    cd your-project-root
    uvicorn src.api.main:app --reload

Test with:
    curl http://localhost:8000/health
    curl -X POST http://localhost:8000/predict -F "file=@data/raw/A/img_001.jpg"

Docs at:
    http://localhost:8000/docs
"""

from io import BytesIO

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms
from pathlib import Path
import mlflow

# Import model factory (lightweight)
from src.cv_model.model_factory import get_model

# =============================================================================
# Configuration
# =============================================================================

MODEL_RUN_ID = "587ca0fd066a4a1fbf1a5a26971c3284"
IMAGE_SIZE = 224

# ImageNet normalization (same as your preprocessing.py)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# Global state (loaded once at startup)
# =============================================================================

model = None
classes = []

# Transform (same as your get_val_transforms)
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(title="LSC-Connect API", version="0.1.0")

# Allow frontend to call API (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_model_safe(run_id: str):
    """Load model without heavy dependencies (mediapipe/cv2)."""
    mlruns_dir = Path("models/mlruns")
    mlflow.set_tracking_uri(f"file:///{mlruns_dir.absolute()}")
    
    # Get run info
    try:
        run = mlflow.get_run(run_id)
        params = run.data.params
        model_name = params.get("model_name", "mobilenet_v2")
        num_classes = int(params.get("num_classes", 26))
        classes_list = params.get("classes", "").split(",")
    except Exception as e:
        print(f"Error reading MLflow run: {e}")
        # Fallback defaults
        model_name = "mobilenet_v2"
        num_classes = 26
        classes_list = [chr(i) for i in range(65, 91)]

    # Find checkpoint
    checkpoint_path = None
    for exp_dir in mlruns_dir.iterdir():
        if exp_dir.is_dir() and exp_dir.name not in ["models", ".trash"]:
            run_dir = exp_dir / run_id / "artifacts" / "checkpoints"
            if run_dir.exists():
                checkpoint_files = list(run_dir.glob("*.pth"))
                if checkpoint_files:
                    checkpoint_path = sorted(checkpoint_files)[-1]
                    break
    
    if not checkpoint_path:
        raise FileNotFoundError(f"No checkpoint found for {run_id} in {mlruns_dir}")
        
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    
    model = get_model(model_name, num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()
    
    return model, classes_list


@app.on_event("startup")
def startup():
    """Load model when server starts."""
    global model, classes
    print("=" * 50)
    print("Loading model...")
    print("=" * 50)
    try:
        model, classes = load_model_safe(MODEL_RUN_ID)
        print(f"✓ Loaded {len(classes)} classes")
        print("✓ API ready!")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        # Don't crash, just log. Predict will fail.
    print("=" * 50)


@app.get("/")
def root():
    """API info."""
    return {"name": "LSC-Connect API", "status": "running"}


@app.get("/health")
def health():
    """Health check."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(DEVICE),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict sign language gesture from image.
    
    Upload a cropped hand image (JPG/PNG) and get the predicted letter.
    """
    # Validate file type
    #if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
    #    raise HTTPException(status_code=400, detail="Only JPG/PNG images allowed")
    
    # Read image
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    
    # Preprocess
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    # Inference
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
    
    # Get result
    confidence, idx = torch.max(probs, dim=0)
    letter = classes[idx.item()]
    
    return {
        "letter": letter,
        "confidence": round(confidence.item(), 4),
    }


# =============================================================================
# Run directly (optional)
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)