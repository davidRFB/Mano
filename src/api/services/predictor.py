"""
Predictor service for gesture recognition.

This module handles:
    - Model loading from MLflow (reusing existing inference code)
    - Image preprocessing
    - Inference
    - Result formatting

The GesturePredictor class is designed to be instantiated once at API startup
and reused for all predictions (singleton pattern via FastAPI dependency).
"""

import time
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.api.config import Settings

# =============================================================================
# Constants (matching your training/preprocessing)
# =============================================================================

# ImageNet normalization (must match training - from preprocessing.py)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class GesturePredictor:
    """
    Gesture recognition predictor service.

    Loads model once at initialization and provides prediction method.
    """

    def __init__(self, settings: Settings):
        """
        Initialize predictor with settings.

        Args:
            settings: Application settings containing model_run_id, etc.
        """
        self.settings = settings
        self.model: torch.nn.Module | None = None
        self.class_names: list[str] = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model_name: str | None = None

        # Validation transforms (same as preprocessing.py get_val_transforms)
        # For API: we receive bytes → PIL Image, so no ToPILImage needed
        self.transform = transforms.Compose(
            [
                transforms.Resize((settings.image_size, settings.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def load_model(self) -> None:
        """
        Load model from MLflow.

        This reuses the logic from your src/cv_model/inference.py
        but adapted for the API context.

        Raises:
            FileNotFoundError: If model checkpoint not found
        """
        import mlflow

        # Import get_model from your existing training code
        from src.cv_model.train import get_model

        mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)

        run_id = self.settings.model_run_id
        print(f"Loading model from MLflow run: {run_id}")

        # Get run info
        run = mlflow.get_run(run_id)
        params = run.data.params

        model_name = params.get("model_name", "mobilenet_v2")
        num_classes = int(params.get("num_classes", 26))
        self.class_names = params.get("classes", "").split(",")
        self._model_name = model_name

        # Find checkpoint in mlruns directory
        # (same logic as your inference.py load_model_from_mlflow)
        mlruns_dir = Path(self.settings.mlflow_tracking_uri)
        checkpoint_path = self._find_checkpoint(mlruns_dir, run_id)

        if checkpoint_path is None:
            raise FileNotFoundError(
                f"No checkpoint found for run {run_id}. "
                f"Searched in {mlruns_dir}"
            )

        # Load checkpoint
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )

        # Build model using YOUR get_model function (ensures architecture matches)
        self.model = get_model(model_name, num_classes, pretrained=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

        val_acc = checkpoint.get("val_acc", "N/A")

        print(f"✓ Loaded {model_name} with {num_classes} classes")
        print(f"✓ Checkpoint: {checkpoint_path.name}")
        print(f"✓ Classes: {self.class_names}")
        print(f"✓ Validation accuracy: {val_acc}")
        print(f"✓ Device: {self.device}")

    def _find_checkpoint(self, mlruns_dir: Path, run_id: str) -> Path | None:
        """
        Find checkpoint file for a given run ID.

        Same logic as your inference.py load_model_from_mlflow.
        """
        for exp_dir in mlruns_dir.iterdir():
            if exp_dir.is_dir() and exp_dir.name not in ["0", "models", ".trash"]:
                run_dir = exp_dir / run_id / "artifacts" / "checkpoints"
                if run_dir.exists():
                    checkpoint_files = list(run_dir.glob("*.pth"))
                    if checkpoint_files:
                        # Return the last checkpoint (sorted alphabetically)
                        return sorted(checkpoint_files)[-1]
        return None

    def predict(self, image_bytes: bytes) -> dict:
        """
        Run prediction on image bytes.

        Args:
            image_bytes: Raw image bytes (JPG, PNG, etc.)

        Returns:
            dict with prediction results:
                - letter: predicted class
                - confidence: confidence score
                - all_probabilities: dict of all class probabilities
                - inference_time_ms: time taken
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        start_time = time.perf_counter()

        # Load image from bytes and convert to RGB
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Apply transforms and add batch dimension
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Run inference (no_grad for efficiency)
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]

        # Get top prediction
        confidence, predicted_idx = torch.max(probabilities, dim=0)
        predicted_letter = self.class_names[predicted_idx.item()]

        # Build probability dict for all classes
        all_probs = {
            self.class_names[i]: round(probabilities[i].item(), 4)
            for i in range(len(self.class_names))
        }

        inference_time_ms = (time.perf_counter() - start_time) * 1000

        return {
            "letter": predicted_letter,
            "confidence": round(confidence.item(), 4),
            "all_probabilities": all_probs,
            "inference_time_ms": round(inference_time_ms, 2),
        }

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None

    @property
    def model_name(self) -> str | None:
        """Return the loaded model architecture name."""
        return self._model_name


# =============================================================================
# Singleton instance (to be initialized at startup)
# =============================================================================

_predictor: GesturePredictor | None = None


def get_predictor() -> GesturePredictor:
    """
    Get the global predictor instance.

    This is used as a FastAPI dependency.
    Raises RuntimeError if predictor not initialized.
    """
    if _predictor is None:
        raise RuntimeError("Predictor not initialized. Check API startup.")
    return _predictor


def init_predictor(settings: Settings) -> GesturePredictor:
    """
    Initialize the global predictor instance.

    Called once at API startup.
    """
    global _predictor
    _predictor = GesturePredictor(settings)
    _predictor.load_model()
    return _predictor