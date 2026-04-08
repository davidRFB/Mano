#!/usr/bin/env python3
"""
Real-time gesture recognition demo with letter capture and LLM correction.

Usage:
    python scripts/06_demo.py --run-id <mlflow_run_id>
    python scripts/06_demo.py --run-name <mlflow_run_name>
    python scripts/06_demo.py --list                    # List runs from default experiment
    python scripts/06_demo.py --list-all                # List runs from ALL experiments

Controls:
    - SPACE: Manually capture current letter to buffer
    - BACKSPACE: Delete last letter from buffer
    - C: Clear entire buffer
    - ENTER: Send buffer to LLM for correction
    - 1-5: Pick suggestion from autocomplete
    - ESC or Q: Quit

Letter Capture:
    - Manual: Press SPACE to add current prediction
    - Auto: Letters are auto-captured when stable for ~0.5s
"""

import argparse
import sys
from collections import deque
from pathlib import Path

# Add project root to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.preprocessing import (
    normalize_landmarks,
    extract_features,
    DEFAULT_FEATURE_MODE,
)
from src.training import load_model_from_mlflow, list_runs, list_all_runs, list_experiments

# MobileNet image preprocessing (same as training)
MOBILENET_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),  # Direct resize, no crop
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# LLM corrector (optional)
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
    def deduplicate_letters(x):
        return x

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW_NAME = "LSC Gesture Recognition"
SEQUENCE_LENGTH = 20
PREDICT_EVERY = 5

# Stability-based auto-capture
STABILITY_THRESHOLD = 15  # Frames before auto-capture (~0.5s at 30fps)
MIN_CAPTURE_CONFIDENCE = 0.5

# LLM configuration
DEFAULT_LLM_MODEL = "llama3.2"


def extract_mp_landmarks(hand_landmarks) -> np.ndarray:
    """Extract landmarks from MediaPipe result."""
    return np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float32
    )


@torch.no_grad()
def predict_sequence(
    model: torch.nn.Module,
    buffer: deque,
    classes: list[str],
    feature_mode: str,
) -> tuple[str, float]:
    """Run prediction on buffered landmark sequence (for RNN models)."""
    sequence = np.stack(list(buffer), axis=0)
    normalized = normalize_landmarks(sequence)
    features = extract_features(normalized, mode=feature_mode)

    tensor = torch.from_numpy(features).float().unsqueeze(0).to(DEVICE)
    outputs = model(tensor)
    probs = F.softmax(outputs, dim=1)
    conf, pred_idx = probs.max(1)

    return classes[pred_idx.item()], conf.item()


@torch.no_grad()
def predict_static_landmarks(
    model: torch.nn.Module,
    landmarks: np.ndarray,
    classes: list[str],
    feature_mode: str,
) -> tuple[str, float]:
    """Run prediction on single frame landmarks (for static MLP model)."""
    # Normalize single frame
    normalized = normalize_landmarks(landmarks)
    # Extract features - add temporal dim, extract, take first frame
    features = extract_features(normalized[np.newaxis, :, :], mode=feature_mode)[0]

    tensor = torch.from_numpy(features).float().unsqueeze(0).to(DEVICE)
    outputs = model(tensor)
    probs = F.softmax(outputs, dim=1)
    conf, pred_idx = probs.max(1)

    return classes[pred_idx.item()], conf.item()


@torch.no_grad()
def predict_image(
    model: torch.nn.Module,
    frame: np.ndarray,
    classes: list[str],
) -> tuple[str, float]:
    """Run prediction on image frame (for MobileNet)."""
    # Convert BGR (OpenCV) to RGB (PIL)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)

    # Apply transforms
    tensor = MOBILENET_TRANSFORM(pil_image).unsqueeze(0).to(DEVICE)

    outputs = model(tensor)
    probs = F.softmax(outputs, dim=1)
    conf, pred_idx = probs.max(1)

    return classes[pred_idx.item()], conf.item()


def fmt_acc(val: float | None) -> str:
    return f"{val:.1%}" if val is not None else "  -  "


def print_run(run: dict, show_experiment: bool = False) -> None:
    train = fmt_acc(run.get("train_acc"))
    val = fmt_acc(run.get("val_acc"))
    test = fmt_acc(run.get("test_acc"))
    name = run.get("run_name") or "unnamed"
    exp = f"{run['experiment']:<15}  " if show_experiment else ""
    print(f"  {exp}{name:<25}  train={train}  val={val}  test={test}  {run['status']:<10}  {run['run_id']}")


def draw_overlay(
    frame: np.ndarray,
    hand_detected: bool,
    prediction: str | None,
    confidence: float | None,
    buffer_len: int,
    buffer_max: int,
    letter_buffer: list[str],
    deduped_word: str,
    suggestions: list[str],
    corrected_text: str | None,
    stability_progress: float,
) -> None:
    """Draw full UI overlay on frame."""
    h, w = frame.shape[:2]

    # Top status bar
    cv2.rectangle(frame, (0, 0), (w, 90), (40, 40, 40), -1)

    # Title
    cv2.putText(frame, "LSC Gesture Recognition", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Instructions
    cv2.putText(frame, "SPACE:add | BACKSPACE:del | C:clear | ENTER:correct | ESC:quit",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    # Hand status and buffer
    status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
    if buffer_max > 0:
        status_text = f"Hand: {'OK' if hand_detected else '--'} | Buffer: {buffer_len}/{buffer_max}"
    else:
        status_text = f"Hand: {'OK' if hand_detected else '--'} | Mode: Image"
    cv2.putText(frame, status_text, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)

    # Current prediction (large display on right)
    if prediction and confidence:
        display = "NN" if prediction == "nn" else prediction.upper()
        font_scale = 4
        thickness = 8

        (tw, th), _ = cv2.getTextSize(display, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        x = w - tw - 30
        y = 70

        # Background color based on confidence
        pred_color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255) if confidence > 0.4 else (0, 165, 255)
        cv2.rectangle(frame, (x - 10, 10), (w - 10, y + 10), pred_color, -1)
        cv2.putText(frame, display, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
        cv2.putText(frame, f"{confidence:.0%}", (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Stability progress bar
        if stability_progress > 0:
            bar_width = tw + 20
            bar_height = 8
            bar_x = x - 10
            bar_y = y + 40

            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
            progress_width = int(bar_width * stability_progress)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), (0, 200, 255), -1)

    # Bottom panel
    bottom_y = h - 120
    cv2.rectangle(frame, (0, bottom_y), (w, h), (40, 40, 40), -1)

    # Word display
    if deduped_word:
        word_text = f"Word: {deduped_word.upper()}"
    elif letter_buffer:
        word_text = "Word: " + "".join([l.upper() for l in letter_buffer])
    else:
        word_text = "Word: (show gestures to spell)"

    cv2.putText(frame, word_text, (10, bottom_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Suggestions
    if suggestions:
        suggestions_text = "Suggestions: " + " | ".join(suggestions[:5])
        cv2.putText(frame, suggestions_text, (10, bottom_y + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    else:
        cv2.putText(frame, "Suggestions: (type more letters)", (10, bottom_y + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    # Corrected text / instructions
    if corrected_text:
        cv2.putText(frame, f"LLM: {corrected_text}", (10, bottom_y + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "ENTER: LLM correct | 1-5: pick suggestion", (10, bottom_y + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)


def main():
    parser = argparse.ArgumentParser(description="Real-time LSC demo")
    parser.add_argument("--run-id", type=str, help="MLflow run ID")
    parser.add_argument("--run-name", type=str, help="MLflow run name")
    parser.add_argument("--list", action="store_true", help="List runs from experiment")
    parser.add_argument("--list-all", action="store_true", help="List runs from ALL experiments")
    parser.add_argument("--experiment", type=str, default="lsc_letters", help="Experiment name")
    parser.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH)
    parser.add_argument("--llm-model", type=str, default=None, help="LLM model for correction")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM correction")
    args = parser.parse_args()

    # List all runs mode
    if args.list_all:
        print("All MLflow runs:")
        experiments = list_experiments()
        print(f"Experiments: {experiments}\n")
        print(f"  {'experiment':<15}  {'run_name':<25}  {'train':<10}  {'val':<10}  {'test':<10}  {'status':<10}  run_id")
        print("-" * 130)
        for run in list_all_runs(limit=20):
            print_run(run, show_experiment=True)
        return

    # List runs mode
    if args.list:
        print(f"MLflow runs (experiment: {args.experiment}):")
        print(f"  {'run_name':<25}  {'train':<10}  {'val':<10}  {'test':<10}  {'status':<10}  run_id")
        print("-" * 115)
        runs = list_runs(experiment_name=args.experiment, limit=10)
        if not runs:
            print(f"  No runs found. Available experiments: {list_experiments()}")
        for run in runs:
            print_run(run, show_experiment=False)
        return

    if not args.run_id and not args.run_name:
        print("Error: Must provide --run-id or --run-name (or use --list / --list-all)")
        return

    print("=" * 60)
    print("LSC Real-time Demo")
    print("=" * 60)

    # Initialize LLM corrector (optional)
    corrector = None
    if not args.no_llm and LLM_AVAILABLE:
        try:
            corrector = SignLanguageCorrector(backend="groq")
            if corrector.check_connection():
                print(f"LLM: Groq ({corrector.model}) - fast mode")
            else:
                corrector = None
        except Exception:
            corrector = None

        if corrector is None:
            try:
                model_name = args.llm_model or DEFAULT_LLM_MODEL
                corrector = SignLanguageCorrector(backend="ollama", model=model_name)
                if corrector.check_connection():
                    print(f"LLM: Ollama ({model_name}) - local mode")
                else:
                    corrector = None
            except Exception:
                corrector = None

        if corrector is None:
            print("LLM: Not available (set GROQ_API_KEY or start Ollama)")
    elif args.no_llm:
        print("LLM: Disabled")
    else:
        print("LLM: Not installed")

    # Initialize autocomplete
    autocomplete = None
    if AUTOCOMPLETE_AVAILABLE:
        autocomplete = get_autocomplete()
        print(f"Autocomplete: {autocomplete.word_count} words loaded")

    # Load model from MLflow
    try:
        model, config = load_model_from_mlflow(
            run_id=args.run_id,
            run_name=args.run_name,
            device=DEVICE,
        )
    except ValueError as e:
        print(f"Error: {e}")
        return

    classes = config.get("classes", [])
    feature_mode = config.get("feature_mode", DEFAULT_FEATURE_MODE)
    model_type = config.get("model", "bigru")
    is_mobilenet = model_type in ("mobilenet", "mobilenet_v2")
    is_static_landmarks = model_type == "static"
    is_sequence = model_type in ("gru", "bigru", "lstm")

    print(f"Model: {model_type}")
    print(f"Classes: {classes}")
    print(f"Features: {'images' if is_mobilenet else feature_mode}")
    print(f"Device: {DEVICE}")

    # MediaPipe setup
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n" + "=" * 60)
    print("Camera ready! Show hand gestures.")
    print("Controls:")
    print("  SPACE     - Manually capture current letter")
    print("  BACKSPACE - Delete last letter")
    print("  C         - Clear entire buffer")
    print("  ENTER     - Send buffer to LLM for correction")
    print("  1-5       - Pick suggestion")
    print("  ESC/q     - Quit")
    print("=" * 60)

    # Prediction state
    landmark_buffer: deque = deque(maxlen=args.sequence_length)
    frame_count = 0
    current_pred = None
    current_conf = 0.0

    # Letter capture state
    letter_buffer: list[str] = []
    corrected_text: str | None = None
    suggestions: list[str] = []

    # Stability tracking
    stability_counter = 0
    last_stable_letter: str | None = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        hand_detected = False

        if results.multi_hand_landmarks:
            hand_detected = True
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks (visual feedback)
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

                frame_count += 1
                landmarks = extract_mp_landmarks(hand_landmarks)

                if is_mobilenet:
                    # MobileNet: predict directly from frame
                    if frame_count >= PREDICT_EVERY:
                        frame_count = 0
                        current_pred, current_conf = predict_image(
                            model, frame, classes
                        )
                elif is_static_landmarks:
                    # Static MLP: predict from single frame landmarks
                    if frame_count >= PREDICT_EVERY:
                        frame_count = 0
                        current_pred, current_conf = predict_static_landmarks(
                            model, landmarks, classes, feature_mode
                        )
                else:
                    # Sequence models (GRU/BiGRU/LSTM): buffer and predict
                    landmark_buffer.append(landmarks)

                    if len(landmark_buffer) >= args.sequence_length and frame_count >= PREDICT_EVERY:
                        frame_count = 0
                        current_pred, current_conf = predict_sequence(
                            model, landmark_buffer, classes, feature_mode
                        )

                # Stability-based auto-capture (works for both model types)
                if current_pred and current_conf >= MIN_CAPTURE_CONFIDENCE:
                    if current_pred == last_stable_letter:
                        stability_counter += 1
                        if stability_counter >= STABILITY_THRESHOLD:
                            letter_buffer.append(current_pred)
                            corrected_text = None
                            print(f"[AUTO] Captured: {current_pred.upper()} | Buffer: {' '.join([l.upper() for l in letter_buffer])}")
                            stability_counter = 0
                    else:
                        last_stable_letter = current_pred
                        stability_counter = 1
                elif current_pred:
                    stability_counter = 0
                    last_stable_letter = None
        else:
            if is_sequence:
                landmark_buffer.clear()
            current_pred = None
            current_conf = 0.0
            stability_counter = 0
            last_stable_letter = None

        # Compute deduped word and suggestions
        deduped_word = ""
        if letter_buffer:
            deduped_letters = deduplicate_letters(letter_buffer)
            deduped_word = "".join(deduped_letters)
            if autocomplete and deduped_word:
                suggestions = autocomplete.suggest(deduped_word, max_results=5)

        # Stability progress
        stability_progress = min(stability_counter / STABILITY_THRESHOLD, 1.0)

        # Draw UI
        buffer_len = len(landmark_buffer) if is_sequence else 0
        buffer_max = args.sequence_length if is_sequence else 0
        draw_overlay(
            frame,
            hand_detected,
            current_pred,
            current_conf,
            buffer_len,
            buffer_max,
            letter_buffer,
            deduped_word,
            suggestions,
            corrected_text,
            stability_progress,
        )

        cv2.imshow(WINDOW_NAME, frame)

        # Handle key press
        key = cv2.waitKey(1) & 0xFF

        if key == 27 or key == ord("q"):
            print("\nExiting...")
            break

        elif key == ord(" "):  # SPACE - manual capture
            if current_pred and current_conf >= MIN_CAPTURE_CONFIDENCE:
                letter_buffer.append(current_pred)
                corrected_text = None
                print(f"[MANUAL] Captured: {current_pred.upper()} | Buffer: {' '.join([l.upper() for l in letter_buffer])}")
                stability_counter = 0
            elif current_pred:
                print(f"[SKIP] Confidence too low ({current_conf:.0%}), need >= {MIN_CAPTURE_CONFIDENCE:.0%}")

        elif key == 8:  # BACKSPACE
            if letter_buffer:
                removed = letter_buffer.pop()
                corrected_text = None
                print(f"[DELETE] Removed: {removed.upper()} | Buffer: {' '.join([l.upper() for l in letter_buffer]) if letter_buffer else '(empty)'}")

        elif key == ord("c") or key == ord("C"):  # Clear
            if letter_buffer:
                letter_buffer.clear()
                corrected_text = None
                suggestions = []
                print("[CLEAR] Buffer cleared")

        elif key == 13:  # ENTER - LLM correction
            if letter_buffer:
                deduped = deduplicate_letters(letter_buffer)
                print(f"[CORRECT] Sending to LLM: {''.join(deduped).upper()}")
                if corrector:
                    try:
                        result = corrector.correct_sequence(deduped)
                        corrected_text = result.corrected
                        print(f"[RESULT] {corrected_text} (confidence: {result.confidence})")
                    except Exception as e:
                        corrected_text = f"(Error: {str(e)[:30]})"
                else:
                    corrected_text = "".join(deduped)
                    print(f"[FALLBACK] No LLM: {corrected_text}")

        # Keys 1-5: Pick suggestion
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
    print("\nDemo ended.")


if __name__ == "__main__":
    main()
