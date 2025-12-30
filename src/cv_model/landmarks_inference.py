"""
Real-time inference for landmark-based LSC gesture recognition.

Uses sliding window approach: maintains buffer of 20 landmark frames,
runs prediction on window, smooths results with majority voting.

Usage:
    python -m src.cv_model.landmarks_inference --list-runs
    python -m src.cv_model.landmarks_inference --run-id <RUN_ID>
    python -m src.cv_model.landmarks_inference

Controls:
    - SPACE: Clear word buffer
    - BACKSPACE: Delete last letter from buffer
    - ENTER: Send buffer to LLM for correction
    - ESC or 'q': Quit
"""

import argparse
import logging
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import mlflow
import numpy as np
import torch
import torch.nn.functional as F

from src.cv_model.landmarks_train import MODELS_DIR, MLFLOW_TRACKING_URI
from src.cv_model.landmarks_model import get_landmarks_model
from src.cv_model.landmarks_preprocessing import (
    extract_features,
    FEATURE_MODES,
    DEFAULT_FEATURE_MODE,
)

# LLM corrector (optional)
try:
    from src.llm.corrector import SignLanguageCorrector
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    SignLanguageCorrector = None

# Fast autocomplete
try:
    from src.llm.autocomplete import get_autocomplete, deduplicate_letters
    AUTOCOMPLETE_AVAILABLE = True
except ImportError:
    AUTOCOMPLETE_AVAILABLE = False
    get_autocomplete = None
    deduplicate_letters = lambda x: x

logger = logging.getLogger(__name__)


# Configuration
WINDOW_NAME = "LSC Landmarks Recognition"
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.5
CROP_PADDING = 50

# Sliding window configuration
SEQUENCE_LENGTH = 10  # frames in sliding window
PREDICT_EVERY_N_FRAMES = 5  # run model every N frames
SMOOTHING_WINDOW = 5  # number of predictions to smooth over

# Capture configuration
STABILITY_THRESHOLD = 3  # stable predictions before auto-capture
MIN_CAPTURE_CONFIDENCE = 0.3

# LLM configuration
DEFAULT_LLM_MODEL = "llama3.2"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def list_runs(experiment_name: str | None = None) -> None:
    """List all MLflow runs for landmarks models."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    print("=" * 80)
    print("Available MLflow Runs (Landmarks)")
    print("=" * 80)

    experiments = mlflow.search_experiments()

    if not experiments:
        print("No experiments found. Train a model first.")
        return

    for exp in experiments:
        if exp.name == "Default":
            continue
        if "landmarks" not in exp.name.lower():
            continue
        if experiment_name and exp.name != experiment_name:
            continue

        print(f"\n Experiment: {exp.name}")
        print("-" * 80)

        runs = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["metrics.test_acc DESC"],
        )

        if runs.empty:
            print("  No runs found.")
            continue

        print(
            f"  {'Run ID':<36} {'Model':<12} {'Val Acc':>8} {'Test Acc':>9} {'Status':<10}"
        )
        print("  " + "-" * 77)

        for _, run in runs.iterrows():
            run_id = run["run_id"][:12] + "..."
            full_run_id = run["run_id"]
            model_name = run.get("params.model_name", "N/A")
            val_acc = run.get("metrics.best_val_acc", 0)
            test_acc = run.get("metrics.test_acc", 0)
            status = run.get("status", "N/A")

            val_acc_str = f"{val_acc:.2%}" if val_acc else "N/A"
            test_acc_str = f"{test_acc:.2%}" if test_acc else "N/A"

            print(
                f"  {run_id:<36} {model_name:<12} {val_acc_str:>8} {test_acc_str:>9} {status:<10}"
            )
            print(f"     Full ID: {full_run_id}")

    print("\n" + "=" * 80)
    print("To use: python -m src.cv_model.landmarks_inference --run-id <FULL_RUN_ID>")
    print("=" * 80)


def get_best_run(experiment_name: str = "lsc_landmarks_recognition") -> str | None:
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


def load_model_from_mlflow(run_id: str) -> tuple[torch.nn.Module, list[str], dict]:
    """
    Load trained landmarks model from MLflow run.

    Returns:
        Tuple of (model, class_names, config_dict)
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    print(f"Loading model from MLflow run: {run_id}")

    run = mlflow.get_run(run_id)
    params = run.data.params

    model_name = params.get("model_name", "bigru")
    num_classes = int(params.get("num_classes", 27))
    hidden_dim = int(params.get("hidden_dim", 128))
    num_layers = int(params.get("num_layers", 2))
    dropout = float(params.get("dropout", 0.3))
    feature_mode = params.get("feature_mode", "xyz")  # Default to old mode for compatibility
    feature_dim = int(params.get("feature_dim", 63))
    classes = params.get("classes", "").split(",")

    # Find checkpoint
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
        raise FileNotFoundError(f"No checkpoint found for run {run_id}")

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    # Get feature_mode from checkpoint if available (overrides params)
    feature_mode = checkpoint.get("feature_mode", feature_mode)
    feature_dim = checkpoint.get("feature_dim", feature_dim)

    model = get_landmarks_model(
        model_name=model_name,
        num_classes=num_classes,
        input_dim=feature_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    config = {
        "model_name": model_name,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "feature_mode": feature_mode,
        "feature_dim": feature_dim,
    }

    print(f"Loaded {model_name} with {num_classes} classes")
    print(f"Feature mode: {feature_mode} ({feature_dim} features)")
    print(f"Checkpoint: {checkpoint_path.name}")
    print(f"Classes: {classes}")

    return model, classes, config


def extract_landmarks(hand_landmarks) -> np.ndarray:
    """Extract 21 landmarks as numpy array (21 x 3)."""
    landmarks = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float32
    )
    return landmarks


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Normalize landmarks relative to wrist and scale.

    Input: (seq_len, 21, 3)
    Output: (seq_len, 21, 3) normalized

    Uses only X, Y for distance calculation.
    """
    # Wrist is landmark 0
    wrist = landmarks[:, 0:1, :]  # (seq_len, 1, 3)

    # Center relative to wrist
    centered = landmarks - wrist

    # Scale by max distance from wrist using X, Y only (per frame)
    distances_xy = np.linalg.norm(centered[:, :, :2], axis=2)  # (seq_len, 21)
    max_dist = distances_xy.max(axis=1, keepdims=True)  # (seq_len, 1)
    max_dist = np.maximum(max_dist, 1e-6)

    normalized = centered / max_dist[:, :, np.newaxis]

    return normalized


class SlidingWindowBuffer:
    """Circular buffer for landmark sequences."""

    def __init__(self, window_size: int = SEQUENCE_LENGTH):
        self.window_size = window_size
        self.buffer: deque = deque(maxlen=window_size)

    def add(self, landmarks: np.ndarray) -> None:
        """Add a frame of landmarks (21 x 3)."""
        self.buffer.append(landmarks)

    def is_full(self) -> bool:
        """Check if buffer has enough frames."""
        return len(self.buffer) >= self.window_size

    def get_sequence(self) -> np.ndarray:
        """Get current window as numpy array (seq_len, 21, 3)."""
        return np.stack(list(self.buffer), axis=0)

    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer.clear()


class PredictionSmoother:
    """Smooth predictions using majority voting."""

    def __init__(self, window_size: int = SMOOTHING_WINDOW):
        self.window_size = window_size
        self.predictions: deque = deque(maxlen=window_size)
        self.confidences: deque = deque(maxlen=window_size)

    def add(self, prediction: str, confidence: float) -> None:
        """Add a prediction."""
        self.predictions.append(prediction)
        self.confidences.append(confidence)

    def get_smoothed(self) -> tuple[str | None, float]:
        """Get majority vote prediction and average confidence."""
        if not self.predictions:
            return None, 0.0

        # Count predictions
        counts: dict[str, int] = {}
        conf_sums: dict[str, float] = {}

        for pred, conf in zip(self.predictions, self.confidences):
            counts[pred] = counts.get(pred, 0) + 1
            conf_sums[pred] = conf_sums.get(pred, 0.0) + conf

        # Get majority
        majority = max(counts, key=counts.get)
        avg_conf = conf_sums[majority] / counts[majority]

        return majority, avg_conf

    def clear(self) -> None:
        """Clear predictions."""
        self.predictions.clear()
        self.confidences.clear()


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    landmarks_seq: np.ndarray,
    classes: list[str],
    feature_mode: str = DEFAULT_FEATURE_MODE,
) -> tuple[str, float]:
    """
    Run inference on a landmark sequence.

    Args:
        model: Trained PyTorch model
        landmarks_seq: (seq_len, 21, 3) landmark sequence
        classes: List of class names
        feature_mode: Feature extraction mode

    Returns:
        Tuple of (predicted_letter, confidence)
    """
    # Normalize first
    normalized = normalize_landmarks(landmarks_seq)

    # Extract features based on mode (handles Y flip, drops Z, adds angles/distances)
    features = extract_features(normalized, mode=feature_mode)

    # Convert to tensor and add batch dimension
    tensor = torch.from_numpy(features).float().unsqueeze(0).to(DEVICE)

    # Forward pass
    outputs = model(tensor)
    probabilities = F.softmax(outputs, dim=1)

    # Get prediction
    confidence, predicted_idx = probabilities.max(1)
    predicted_letter = classes[predicted_idx.item()]

    return predicted_letter, confidence.item()


def draw_overlay(
    frame: np.ndarray,
    hand_detected: bool,
    prediction: str | None,
    confidence: float | None,
    buffer_fill: int,
    buffer_size: int,
    letter_buffer: list[str] | None = None,
    deduped_word: str | None = None,
    suggestions: list[str] | None = None,
    corrected_text: str | None = None,
    stability_progress: float = 0.0,
) -> None:
    """Draw status overlay on frame."""
    h, w = frame.shape[:2]

    # Top status bar
    cv2.rectangle(frame, (0, 0), (w, 90), (40, 40, 40), -1)

    # Title
    cv2.putText(
        frame,
        "LSC Landmarks Recognition (Sliding Window)",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    # Instructions
    cv2.putText(
        frame,
        "SPACE:clear | BACKSPACE:del | ENTER:correct | ESC:quit",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (180, 180, 180),
        1,
    )

    # Hand detection and buffer status
    status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
    status_text = f"Hand: {'OK' if hand_detected else 'NO'} | Buffer: {buffer_fill}/{buffer_size}"
    cv2.putText(
        frame,
        status_text,
        (10, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        status_color,
        1,
    )

    # Buffer fill bar
    bar_x = 250
    bar_width = 150
    bar_height = 10
    fill_ratio = buffer_fill / buffer_size
    cv2.rectangle(frame, (bar_x, 68), (bar_x + bar_width, 68 + bar_height), (100, 100, 100), -1)
    cv2.rectangle(frame, (bar_x, 68), (bar_x + int(bar_width * fill_ratio), 68 + bar_height), (0, 255, 0), -1)

    # Current prediction (large display)
    if prediction and confidence:
        display_text = prediction.upper()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 4
        thickness = 8

        (text_w, text_h), _ = cv2.getTextSize(display_text, font, font_scale, thickness)

        x_pos = w - text_w - 30
        y_pos = 80

        # Background
        color = (0, 255, 0) if confidence > 0.5 else (0, 255, 255)
        cv2.rectangle(frame, (x_pos - 10, 10), (w - 10, y_pos + 10), color, -1)

        # Letter
        cv2.putText(frame, display_text, (x_pos, y_pos), font, font_scale, (0, 0, 0), thickness)

        # Confidence
        conf_text = f"{confidence:.0%}"
        cv2.putText(frame, conf_text, (x_pos, y_pos + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Stability progress bar
        if stability_progress > 0:
            bar_width = text_w + 20
            bar_height = 8
            bar_x = x_pos - 10
            bar_y = y_pos + 40

            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
            progress_width = int(bar_width * stability_progress)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), (0, 200, 255), -1)

    # Bottom panel
    bottom_panel_y = h - 120
    cv2.rectangle(frame, (0, bottom_panel_y), (w, h), (40, 40, 40), -1)

    # Word display
    if deduped_word:
        word_text = f"Word: {deduped_word.upper()}"
    elif letter_buffer:
        word_text = "Word: " + "".join([l.upper() for l in letter_buffer])
    else:
        word_text = "Word: (show gestures to spell)"

    cv2.putText(
        frame, word_text, (10, bottom_panel_y + 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
    )

    # Suggestions
    if suggestions:
        suggestions_text = "Suggestions: " + " | ".join(suggestions[:5])
        cv2.putText(
            frame, suggestions_text, (10, bottom_panel_y + 55),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
        )
    else:
        cv2.putText(
            frame, "Suggestions: (type more letters)", (10, bottom_panel_y + 55),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1
        )

    # Corrected text
    if corrected_text:
        cv2.putText(
            frame, f"LLM: {corrected_text}", (10, bottom_panel_y + 90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
    else:
        cv2.putText(
            frame, "ENTER: LLM correct | 1-5: pick suggestion", (10, bottom_panel_y + 90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1
        )


def main(
    run_id: str | None = None,
    experiment_name: str = "lsc_landmarks_recognition",
    llm_model: str | None = None,
    disable_llm: bool = False,
) -> None:
    """Main inference loop with sliding window."""
    print("=" * 60)
    print("LSC Landmarks Recognition - Sliding Window Inference")
    print("=" * 60)

    # Initialize LLM corrector (optional)
    corrector = None
    if not disable_llm and LLM_AVAILABLE:
        try:
            corrector = SignLanguageCorrector(backend="groq")
            if corrector.check_connection():
                print(f"LLM corrector: Groq ({corrector.model})")
            else:
                corrector = None
        except Exception:
            corrector = None

        if corrector is None:
            try:
                model_name = llm_model or DEFAULT_LLM_MODEL
                corrector = SignLanguageCorrector(backend="ollama", model=model_name)
                if corrector.check_connection():
                    print(f"LLM corrector: Ollama ({model_name})")
                else:
                    corrector = None
            except Exception:
                corrector = None

        if corrector is None:
            print("No LLM backend available")
    elif disable_llm:
        print("LLM corrector disabled")

    # Load model
    if run_id:
        model, classes, config = load_model_from_mlflow(run_id)
    else:
        best_run_id = get_best_run(experiment_name)
        if best_run_id:
            print(f"Using best run from '{experiment_name}'")
            model, classes, config = load_model_from_mlflow(best_run_id)
        else:
            print("Error: No MLflow runs found.")
            print("Train a model first: python -m src.cv_model.landmarks_train")
            return

    # Get feature mode from config
    feature_mode = config.get("feature_mode", DEFAULT_FEATURE_MODE)

    print(f"Device: {DEVICE}")
    print(f"Feature mode: {feature_mode}")
    print(f"Sliding window: {SEQUENCE_LENGTH} frames")
    print(f"Predict every: {PREDICT_EVERY_N_FRAMES} frames")

    # Initialize MediaPipe
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

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\nCamera ready!")
    print("=" * 60)

    # State
    landmark_buffer = SlidingWindowBuffer(SEQUENCE_LENGTH)
    prediction_smoother = PredictionSmoother(SMOOTHING_WINDOW)

    current_prediction = None
    current_confidence = None
    frame_counter = 0

    letter_buffer: list[str] = []
    corrected_text: str | None = None
    suggestions: list[str] = []

    # Stability tracking
    stability_counter = 0
    last_stable_letter: str | None = None

    # Autocomplete
    autocomplete = None
    if AUTOCOMPLETE_AVAILABLE:
        autocomplete = get_autocomplete()
        print(f"Autocomplete: {autocomplete.word_count} words")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        hand_detected = False

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

                # Extract and buffer landmarks
                landmarks = extract_landmarks(hand_landmarks)
                landmark_buffer.add(landmarks)

                # Run prediction every N frames when buffer is full
                frame_counter += 1
                if landmark_buffer.is_full() and frame_counter >= PREDICT_EVERY_N_FRAMES:
                    frame_counter = 0

                    # Get sequence and predict
                    sequence = landmark_buffer.get_sequence()
                    pred, conf = predict(model, sequence, classes, feature_mode)

                    # Add to smoother
                    prediction_smoother.add(pred, conf)

                    # Get smoothed prediction
                    current_prediction, current_confidence = prediction_smoother.get_smoothed()

                    # Stability tracking for auto-capture
                    if current_confidence and current_confidence >= MIN_CAPTURE_CONFIDENCE:
                        if current_prediction == last_stable_letter:
                            stability_counter += 1
                            if stability_counter >= STABILITY_THRESHOLD:
                                letter_buffer.append(current_prediction)
                                corrected_text = None
                                print(f"[AUTO] Captured: {current_prediction.upper()} | Buffer: {' '.join([l.upper() for l in letter_buffer])}")
                                stability_counter = 0
                        else:
                            last_stable_letter = current_prediction
                            stability_counter = 1
                    else:
                        stability_counter = 0
                        last_stable_letter = None
        else:
            # No hand - clear buffer for fresh start
            landmark_buffer.clear()
            prediction_smoother.clear()
            current_prediction = None
            current_confidence = None
            stability_counter = 0
            last_stable_letter = None

        # Calculate stability progress
        stability_progress = min(stability_counter / STABILITY_THRESHOLD, 1.0) if STABILITY_THRESHOLD > 0 else 0.0

        # Update suggestions
        deduped_word = ""
        if letter_buffer:
            deduped_letters = deduplicate_letters(letter_buffer)
            deduped_word = "".join(deduped_letters)
            if autocomplete and deduped_word:
                suggestions = autocomplete.suggest(deduped_word, max_results=5)

        # Draw overlay
        draw_overlay(
            frame,
            hand_detected,
            current_prediction,
            current_confidence,
            len(landmark_buffer.buffer),
            SEQUENCE_LENGTH,
            letter_buffer,
            deduped_word,
            suggestions,
            corrected_text,
            stability_progress,
        )

        cv2.imshow(WINDOW_NAME, frame)

        # Handle keys
        key = cv2.waitKey(1) & 0xFF

        if key == 27 or key == ord("q"):
            print("\nExiting...")
            break

        elif key == ord(" "):  # SPACE - clear buffer
            if letter_buffer:
                letter_buffer.clear()
                corrected_text = None
                suggestions = []
                print("[CLEAR] Buffer cleared")

        elif key == 8:  # BACKSPACE
            if letter_buffer:
                removed = letter_buffer.pop()
                corrected_text = None
                print(f"[DELETE] Removed: {removed.upper()}")

        elif key == 13:  # ENTER
            if letter_buffer:
                deduped = deduplicate_letters(letter_buffer)
                print(f"[CORRECT] Sending: {''.join(deduped).upper()}")
                if corrector:
                    try:
                        result = corrector.correct_sequence(deduped)
                        corrected_text = result.corrected
                        print(f"[RESULT] {corrected_text}")
                    except Exception as e:
                        corrected_text = f"(Error: {str(e)[:30]})"
                else:
                    corrected_text = "".join(deduped)

        elif key in [ord("1"), ord("2"), ord("3"), ord("4"), ord("5")]:
            idx = key - ord("1")
            if suggestions and idx < len(suggestions):
                corrected_text = suggestions[idx]
                print(f"[PICK] Selected: {corrected_text}")
                letter_buffer.clear()
                suggestions = []

    hands.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Session ended.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real-time landmarks-based LSC gesture recognition"
    )
    parser.add_argument(
        "--list-runs",
        action="store_true",
        help="List all available MLflow runs",
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
        default="lsc_landmarks_recognition",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help=f"LLM model for correction (default: {DEFAULT_LLM_MODEL})",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM correction",
    )

    args = parser.parse_args()

    if args.list_runs:
        list_runs(args.experiment if args.experiment != "lsc_landmarks_recognition" else None)
    else:
        main(
            run_id=args.run_id,
            experiment_name=args.experiment,
            llm_model=args.llm_model,
            disable_llm=args.no_llm,
        )
