"""
Real-time inference script for LSC gesture recognition.

Usage:
    python -m src.cv_model.inference --list-runs          # List available runs
    python -m src.cv_model.inference --run-id <RUN_ID>    # Use specific run
    python -m src.cv_model.inference                       # Use best run from default experiment

Controls:
    - SPACE: Manually capture current letter to buffer
    - BACKSPACE: Delete last letter from buffer
    - C: Clear entire buffer
    - ENTER: Send buffer to LLM for correction
    - ESC or 'q': Quit

Letter Capture:
    - Manual: Press SPACE to add current prediction
    - Auto: Letters are auto-captured when stable for ~0.5s (15 frames)
"""

import argparse
import logging
from pathlib import Path

import cv2
import mediapipe as mp
import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from src.cv_model.train import MODELS_DIR, MLFLOW_TRACKING_URI
from src.cv_model.model_factory import get_model

# LLM corrector (optional - graceful fallback if not available)
try:
    from src.llm.corrector import SignLanguageCorrector
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    SignLanguageCorrector = None

# Fast autocomplete (no LLM)
try:
    from src.llm.autocomplete import get_autocomplete, deduplicate_letters
    AUTOCOMPLETE_AVAILABLE = True
except ImportError:
    AUTOCOMPLETE_AVAILABLE = False
    get_autocomplete = None
    deduplicate_letters = lambda x: x

logger = logging.getLogger(__name__)


# Configuration
WINDOW_NAME = "LSC Gesture Recognition"
IMAGE_SIZE = 224
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.5
CROP_PADDING = 50

# Letter capture configuration
STABILITY_THRESHOLD = 3  # Frames before auto-capture (~0.5s at 30fps)
MIN_CAPTURE_CONFIDENCE = 0.25  # Minimum confidence for capture

# LLM configuration
DEFAULT_LLM_MODEL = "llama3.2"

# ImageNet normalization (must match training)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def list_runs(experiment_name: str | None = None) -> None:
    """List all MLflow runs with their metrics."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    print("=" * 80)
    print("Available MLflow Runs")
    print("=" * 80)

    # Get all experiments
    experiments = mlflow.search_experiments()

    if not experiments:
        print("No experiments found. Train a model first.")
        return

    for exp in experiments:
        if exp.name == "Default":
            continue

        if experiment_name and exp.name != experiment_name:
            continue

        print(f"\n📁 Experiment: {exp.name}")
        print("-" * 80)

        # Get runs for this experiment
        runs = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["metrics.test_acc DESC"],
        )

        if runs.empty:
            print("  No runs found.")
            continue

        # Display table header
        print(
            f"  {'Run ID':<36} {'Model':<18} {'Val Acc':>8} {'Test Acc':>9} {'Status':<10}"
        )
        print("  " + "-" * 83)

        for _, run in runs.iterrows():
            run_id = run["run_id"][:12] + "..."  # Truncate for display
            full_run_id = run["run_id"]
            model_name = run.get("params.model_name", "N/A")
            val_acc = run.get("metrics.best_val_acc", 0)
            test_acc = run.get("metrics.test_acc", 0)
            status = run.get("status", "N/A")

            val_acc_str = f"{val_acc:.2%}" if val_acc else "N/A"
            test_acc_str = f"{test_acc:.2%}" if test_acc else "N/A"

            print(
                f"  {run_id:<36} {model_name:<18} {val_acc_str:>8} {test_acc_str:>9} {status:<10}"
            )
            print(f"     Full ID: {full_run_id}")

    print("\n" + "=" * 80)
    print("To use a run: python -m src.cv_model.inference --run-id <FULL_RUN_ID>")
    print("=" * 80)


def get_best_run(experiment_name: str = "lsc_gesture_recognition") -> str | None:
    """Get the run ID with the best test accuracy."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            return None

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.test_acc DESC"],
            max_results=1,
        )

        if runs.empty:
            return None

        return runs.iloc[0]["run_id"]
    except Exception:
        return None


def load_model_from_mlflow(run_id: str) -> tuple[torch.nn.Module, list[str]]:
    """
    Load trained model from MLflow run.

    Args:
        run_id: MLflow run ID

    Returns:
        Tuple of (model, class_names)
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    print(f"Loading model from MLflow run: {run_id}")

    # Get run info
    run = mlflow.get_run(run_id)
    params = run.data.params

    model_name = params.get("model_name", "mobilenet_v2")
    num_classes = int(params.get("num_classes", 26))
    classes = params.get("classes", "").split(",")

    # Find checkpoint by directly accessing local mlruns directory
    # (more reliable than mlflow.artifacts.download_artifacts for local file store)
    mlruns_dir = MODELS_DIR / "mlruns"
    checkpoint_path = None

    for exp_dir in mlruns_dir.iterdir():
        if exp_dir.is_dir() and exp_dir.name not in ["0", "models", ".trash"]:
            run_dir = exp_dir / run_id / "artifacts" / "checkpoints"
            if run_dir.exists():
                checkpoint_files = list(run_dir.glob("*.pth"))
                if checkpoint_files:
                    checkpoint_path = sorted(checkpoint_files)[-1]
                    break

    if checkpoint_path is None:
        raise FileNotFoundError(
            f"No checkpoint found for run {run_id}. "
            f"Searched in {mlruns_dir}"
        )

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    model = get_model(model_name, num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    print(f"Loaded {model_name} with {num_classes} classes")
    print(f"Checkpoint: {checkpoint_path.name}")
    print(f"Classes: {classes}")
    print(f"Validation accuracy: {checkpoint.get('val_acc', 'N/A')}")

    return model, classes


def load_model(checkpoint_path: Path) -> tuple[torch.nn.Module, list[str]]:
    """
    Load trained model from checkpoint file (legacy support).

    Args:
        checkpoint_path: Path to .pth checkpoint file

    Returns:
        Tuple of (model, class_names)
    """
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    model_name = checkpoint["model_name"]
    classes = checkpoint["classes"]
    num_classes = checkpoint["num_classes"]

    model = get_model(model_name, num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    print(f"Loaded {model_name} with {num_classes} classes")
    print(f"Classes: {classes}")

    return model, classes


def get_inference_transform() -> transforms.Compose:
    """Get transforms for inference (same as validation)."""
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_hand_bbox(
    hand_landmarks, frame_width: int, frame_height: int
) -> tuple[int, int, int, int]:
    """Extract bounding box from hand landmarks with padding."""
    x_coords = [lm.x * frame_width for lm in hand_landmarks.landmark]
    y_coords = [lm.y * frame_height for lm in hand_landmarks.landmark]

    x_min = int(max(0, min(x_coords) - CROP_PADDING))
    x_max = int(min(frame_width, max(x_coords) + CROP_PADDING))
    y_min = int(max(0, min(y_coords) - CROP_PADDING))
    y_max = int(min(frame_height, max(y_coords) + CROP_PADDING))

    return x_min, y_min, x_max, y_max


def crop_hand(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Crop hand region from frame."""
    x_min, y_min, x_max, y_max = bbox
    return frame[y_min:y_max, x_min:x_max].copy()


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    image: np.ndarray,
    transform: transforms.Compose,
    classes: list[str],
) -> tuple[str, float]:
    """
    Run inference on a cropped hand image.

    Args:
        model: Trained PyTorch model
        image: BGR image (cropped hand region)
        transform: Preprocessing transforms
        classes: List of class names

    Returns:
        Tuple of (predicted_letter, confidence)
    """
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply transforms and add batch dimension
    tensor = transform(rgb_image).unsqueeze(0).to(DEVICE)

    # Forward pass
    outputs = model(tensor)
    probabilities = F.softmax(outputs, dim=1)

    # Get prediction
    confidence, predicted_idx = probabilities.max(1)
    predicted_letter = classes[predicted_idx.item()]

    return predicted_letter, confidence.item()


def draw_prediction(
    frame: np.ndarray,
    letter: str,
    confidence: float,
    bbox: tuple[int, int, int, int],
) -> None:
    """Draw prediction overlay on frame."""
    x_min, y_min, x_max, y_max = bbox

    # Draw bounding box
    color = (0, 255, 0) if confidence > 0.8 else (0, 255, 255)
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

    # Draw letter above bbox
    label = f"{letter.upper()} ({confidence:.0%})"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 3

    # Get text size for background
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    # Draw background rectangle
    cv2.rectangle(
        frame,
        (x_min, y_min - text_h - 15),
        (x_min + text_w + 10, y_min),
        color,
        -1,
    )

    # Draw text
    cv2.putText(
        frame,
        label,
        (x_min + 5, y_min - 10),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
    )


def draw_overlay(
    frame: np.ndarray,
    hand_detected: bool,
    prediction: str | None,
    confidence: float | None,
    letter_buffer: list[str] | None = None,
    deduped_word: str | None = None,
    suggestions: list[str] | None = None,
    corrected_text: str | None = None,
    stability_progress: float = 0.0,
) -> None:
    """Draw status overlay on frame."""
    h, w = frame.shape[:2]

    # Top status bar background
    cv2.rectangle(frame, (0, 0), (w, 80), (40, 40, 40), -1)

    # Title
    cv2.putText(
        frame,
        "LSC Gesture Recognition",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    # Instructions
    cv2.putText(
        frame,
        "SPACE:add | BACKSPACE:del | C:clear | ENTER:correct | ESC:quit",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (180, 180, 180),
        1,
    )

    # Hand detection status
    status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
    status_text = "Hand: DETECTED" if hand_detected else "Hand: NOT FOUND"
    cv2.putText(
        frame,
        status_text,
        (10, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        status_color,
        1,
    )

    # Current prediction (large display)
    if prediction and confidence:
        # Large letter display on right side
        display_text = prediction.upper()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 4
        thickness = 8

        (text_w, text_h), _ = cv2.getTextSize(display_text, font, font_scale, thickness)

        x_pos = w - text_w - 30
        y_pos = 70

        # Background
        cv2.rectangle(
            frame,
            (x_pos - 10, 10),
            (w - 10, y_pos + 10),
            (0, 255, 0) if confidence > 0.8 else (0, 255, 255),
            -1,
        )

        # Letter
        cv2.putText(
            frame,
            display_text,
            (x_pos, y_pos),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
        )

        # Stability progress bar (below prediction)
        if stability_progress > 0:
            bar_width = text_w + 20
            bar_height = 8
            bar_x = x_pos - 10
            bar_y = y_pos + 15

            # Background bar
            cv2.rectangle(
                frame,
                (bar_x, bar_y),
                (bar_x + bar_width, bar_y + bar_height),
                (100, 100, 100),
                -1,
            )

            # Progress bar
            progress_width = int(bar_width * stability_progress)
            cv2.rectangle(
                frame,
                (bar_x, bar_y),
                (bar_x + progress_width, bar_y + bar_height),
                (0, 200, 255),
                -1,
            )

    # Bottom panel for letter buffer, suggestions, and corrected text
    bottom_panel_y = h - 120
    cv2.rectangle(frame, (0, bottom_panel_y), (w, h), (40, 40, 40), -1)

    # Word display (deduped)
    if deduped_word:
        word_text = f"Word: {deduped_word.upper()}"
    elif letter_buffer:
        word_text = "Word: " + "".join([l.upper() for l in letter_buffer])
    else:
        word_text = "Word: (show gestures to spell)"

    cv2.putText(
        frame,
        word_text,
        (10, bottom_panel_y + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
    )

    # Suggestions display (instant autocomplete)
    if suggestions:
        suggestions_text = "Suggestions: " + " | ".join(suggestions[:5])
        cv2.putText(
            frame,
            suggestions_text,
            (10, bottom_panel_y + 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
    else:
        cv2.putText(
            frame,
            "Suggestions: (type more letters)",
            (10, bottom_panel_y + 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (150, 150, 150),
            1,
        )

    # Corrected text display (from LLM)
    if corrected_text:
        cv2.putText(
            frame,
            f"LLM: {corrected_text}",
            (10, bottom_panel_y + 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
    else:
        cv2.putText(
            frame,
            "ENTER: LLM correct | 1-5: pick suggestion",
            (10, bottom_panel_y + 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (150, 150, 150),
            1,
        )


def main(
    run_id: str | None = None,
    model_path: str | None = None,
    experiment_name: str = "lsc_gesture_recognition",
    llm_model: str | None = None,
    disable_llm: bool = False,
) -> None:
    """Main inference loop."""
    print("=" * 60)
    print("LSC Gesture Recognition - Real-time Inference")
    print("=" * 60)

    # Initialize LLM corrector (optional)
    # Try Groq first (fast ~100ms), fallback to Ollama (slow ~5s)
    corrector = None
    if not disable_llm and LLM_AVAILABLE:
        # Try Groq first (requires GROQ_API_KEY env var)
        try:
            corrector = SignLanguageCorrector(backend="groq")
            if corrector.check_connection():
                print(f"LLM corrector: Groq ({corrector.model}) - fast mode")
            else:
                corrector = None
        except Exception:
            corrector = None

        # Fallback to Ollama
        if corrector is None:
            try:
                model_name = llm_model or DEFAULT_LLM_MODEL
                corrector = SignLanguageCorrector(backend="ollama", model=model_name)
                if corrector.check_connection():
                    print(f"LLM corrector: Ollama ({model_name}) - local mode")
                else:
                    print(f"Ollama model '{model_name}' not available. Run: ollama pull {model_name}")
                    corrector = None
            except Exception as e:
                logger.warning(f"Failed to initialize LLM corrector: {e}")
                corrector = None

        if corrector is None:
            print("No LLM backend available. Set GROQ_API_KEY or start Ollama.")
    elif not LLM_AVAILABLE:
        print("LLM corrector not available (install groq or ollama)")
    elif disable_llm:
        print("LLM corrector disabled")

    # Determine how to load the model
    if model_path:
        # Legacy: load from file path
        print(f"Loading from file: {model_path}")
        model, classes = load_model(Path(model_path))
    elif run_id:
        # Load from specific MLflow run
        print(f"Loading from MLflow run: {run_id}")
        model, classes = load_model_from_mlflow(run_id)
    else:
        # Find best run from MLflow
        best_run_id = get_best_run(experiment_name)
        if best_run_id:
            print(f"Using best run from experiment '{experiment_name}'")
            model, classes = load_model_from_mlflow(best_run_id)
        else:
            print("Error: No MLflow runs found.")
            print("Train a model first with: python -m src.cv_model.train")
            print("Or list available runs: python -m src.cv_model.inference --list-runs")
            return

    print(f"Device: {DEVICE}")
    transform = get_inference_transform()

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )

    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\nCamera ready! Show your hand to see predictions.")
    print("Controls:")
    print("  SPACE     - Manually capture current letter")
    print("  BACKSPACE - Delete last letter")
    print("  C         - Clear entire buffer")
    print("  ENTER     - Send buffer to LLM for correction")
    print("  ESC/q     - Quit")
    print("=" * 60)

    # Prediction state
    current_prediction = None
    current_confidence = None

    # Letter buffer state
    letter_buffer: list[str] = []
    corrected_text: str | None = None
    suggestions: list[str] = []

    # Initialize autocomplete
    autocomplete = None
    if AUTOCOMPLETE_AVAILABLE:
        autocomplete = get_autocomplete()
        print(f"Autocomplete loaded: {autocomplete.word_count} words")

    # Stability tracking for auto-capture
    stability_counter = 0
    last_stable_letter: str | None = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame")
            break

        # Mirror the frame
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        hand_detected = False
        current_bbox = None

        # Process hand landmarks
        if results.multi_hand_landmarks:
            hand_detected = True
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

                # Get bounding box
                current_bbox = get_hand_bbox(hand_landmarks, w, h)

                # Crop hand and run inference
                hand_crop = crop_hand(frame, current_bbox)

                # Ensure crop is valid
                if hand_crop.size > 0 and hand_crop.shape[0] > 10 and hand_crop.shape[1] > 10:
                    current_prediction, current_confidence = predict(
                        model, hand_crop, transform, classes
                    )

                    # Draw prediction on frame
                    draw_prediction(frame, current_prediction, current_confidence, current_bbox)

                    # Stability-based auto-capture
                    if current_confidence and current_confidence >= MIN_CAPTURE_CONFIDENCE:
                        if current_prediction == last_stable_letter:
                            stability_counter += 1
                            # Auto-capture when stable for STABILITY_THRESHOLD frames
                            if stability_counter >= STABILITY_THRESHOLD:
                                # Capture the letter
                                letter_buffer.append(current_prediction)
                                corrected_text = None  # Clear old correction
                                print(f"[AUTO] Captured: {current_prediction.upper()} | Buffer: {' '.join([l.upper() for l in letter_buffer])}")
                                # Reset stability counter to prevent rapid capture
                                stability_counter = 0
                        else:
                            # Different letter, reset stability
                            last_stable_letter = current_prediction
                            stability_counter = 1
                    else:
                        # Low confidence, reset stability
                        stability_counter = 0
                        last_stable_letter = None
        else:
            current_prediction = None
            current_confidence = None
            # Reset stability when hand is not detected
            stability_counter = 0
            last_stable_letter = None

        # Calculate stability progress for UI
        stability_progress = min(stability_counter / STABILITY_THRESHOLD, 1.0)

        # Draw overlay with buffer
        # Compute deduped word and suggestions
        deduped_word = ""
        if letter_buffer:
            deduped_letters = deduplicate_letters(letter_buffer)
            deduped_word = "".join(deduped_letters)
            if autocomplete and deduped_word:
                suggestions = autocomplete.suggest(deduped_word, max_results=5)

        draw_overlay(
            frame,
            hand_detected,
            current_prediction,
            current_confidence,
            letter_buffer,
            deduped_word,
            suggestions,
            corrected_text,
            stability_progress,
        )

        # Show frame
        cv2.imshow(WINDOW_NAME, frame)

        # Handle key press
        key = cv2.waitKey(1) & 0xFF

        if key == 27 or key == ord("q"):
            print("\nExiting...")
            break

        elif key == ord(" "):  # SPACE - manual capture
            if current_prediction and current_confidence:
                if current_confidence >= MIN_CAPTURE_CONFIDENCE:
                    letter_buffer.append(current_prediction)
                    corrected_text = None  # Clear old correction
                    print(f"[MANUAL] Captured: {current_prediction.upper()} | Buffer: {' '.join([l.upper() for l in letter_buffer])}")
                    # Reset stability to prevent immediate auto-capture
                    stability_counter = 0
                else:
                    print(f"[SKIP] Confidence too low ({current_confidence:.0%}), need >= {MIN_CAPTURE_CONFIDENCE:.0%}")

        elif key == 8:  # BACKSPACE - delete last letter
            if letter_buffer:
                removed = letter_buffer.pop()
                corrected_text = None  # Clear old correction
                print(f"[DELETE] Removed: {removed.upper()} | Buffer: {' '.join([l.upper() for l in letter_buffer]) if letter_buffer else '(empty)'}")

        elif key == ord("c") or key == ord("C"):  # C - clear buffer
            if letter_buffer:
                letter_buffer.clear()
                corrected_text = None
                print("[CLEAR] Buffer cleared")

        elif key == 13:  # ENTER - trigger LLM correction
            if letter_buffer:
                deduped = deduplicate_letters(letter_buffer)
                print(f"[CORRECT] Sending to LLM: {''.join(deduped).upper()}")
                if corrector:
                    try:
                        result = corrector.correct_sequence(deduped)
                        corrected_text = result.corrected
                        print(f"[RESULT] {corrected_text} (confidence: {result.confidence})")
                    except Exception as e:
                        logger.error(f"LLM correction failed: {e}")
                        corrected_text = f"(Error: {str(e)[:30]})"
                else:
                    corrected_text = "".join(deduped)  # Fallback: just join letters
                    print(f"[FALLBACK] No LLM available: {corrected_text}")
            else:
                print("[SKIP] Buffer is empty, nothing to correct")

        # Keys 1-5: Pick suggestion
        elif key in [ord("1"), ord("2"), ord("3"), ord("4"), ord("5")]:
            idx = key - ord("1")  # 0-4
            if suggestions and idx < len(suggestions):
                corrected_text = suggestions[idx]
                print(f"[PICK] Selected suggestion: {corrected_text}")
                letter_buffer.clear()
                suggestions = []

    # Cleanup
    hands.close()
    cap.release()
    cv2.destroyAllWindows()

    print("Inference session ended.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real-time LSC gesture recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.cv_model.inference --list-runs
  python -m src.cv_model.inference --run-id abc123def456
  python -m src.cv_model.inference --experiment my_experiment
  python -m src.cv_model.inference --model path/to/checkpoint.pth
  python -m src.cv_model.inference --llm-model mistral
  python -m src.cv_model.inference --no-llm
        """,
    )
    parser.add_argument(
        "--list-runs",
        action="store_true",
        help="List all available MLflow runs with metrics",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="MLflow run ID to load model from",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="lsc_gesture_recognition",
        help="MLflow experiment name (default: lsc_gesture_recognition)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to model checkpoint file (legacy, prefer --run-id)",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help=f"LLM model for text correction (default: {DEFAULT_LLM_MODEL})",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM correction",
    )

    args = parser.parse_args()

    if args.list_runs:
        list_runs(args.experiment if args.experiment != "lsc_gesture_recognition" else None)
    else:
        main(
            run_id=args.run_id,
            model_path=args.model,
            experiment_name=args.experiment,
            llm_model=args.llm_model,
            disable_llm=args.no_llm,
        )

