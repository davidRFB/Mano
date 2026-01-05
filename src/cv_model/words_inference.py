"""
Inference script for word-level sign language recognition.

Usage:
    # From video file
    python -m src.cv_model.words_inference --video path/to/video.mp4

    # From webcam (record gesture, then predict)
    python -m src.cv_model.words_inference --webcam

    # Specify model checkpoint
    python -m src.cv_model.words_inference --video video.mp4 --checkpoint path/to/model.pth
"""

import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch

from src.cv_model.words_model import get_words_model
from src.cv_model.words_preprocessing import (
    extract_features,
    pad_or_truncate,
    MAX_SEQUENCE_LENGTH,
    DEFAULT_FEATURE_MODE,
    FEATURE_MODES,
    NUM_POSE_LANDMARKS,
    NUM_HAND_LANDMARKS,
)

# Upper body pose indices (same as video_to_landmarks.py)
UPPER_BODY_INDICES = [0, 11, 12, 13, 14, 15, 16, 23, 24]


class WordsPredictor:
    """Real-time word prediction from video/webcam."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = torch.device(device)
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model config from checkpoint
        self.classes = self.checkpoint["classes"]
        self.num_classes = self.checkpoint["num_classes"]
        self.feature_mode = self.checkpoint.get("feature_mode", DEFAULT_FEATURE_MODE)
        self.feature_dim = self.checkpoint.get("feature_dim", FEATURE_MODES[self.feature_mode])
        self.max_seq_len = self.checkpoint.get("max_seq_len", MAX_SEQUENCE_LENGTH)
        self.model_name = self.checkpoint.get("model_name", "bigru")
        self.hidden_dim = self.checkpoint.get("hidden_dim", 256)
        self.num_layers = self.checkpoint.get("num_layers", 2)

        # Initialize model
        self.model = get_words_model(
            model_name=self.model_name,
            num_classes=self.num_classes,
            input_dim=self.feature_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=0.0,  # No dropout at inference
            max_seq_len=self.max_seq_len,
        )
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Initialize MediaPipe Holistic
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        print(f"Loaded model: {self.model_name}")
        print(f"Vocabulary: {self.num_classes} words")
        print(f"Feature mode: {self.feature_mode} ({self.feature_dim} features)")

    def extract_frame_landmarks(self, frame: np.ndarray) -> np.ndarray | None:
        """Extract holistic landmarks from a single frame."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb)

        # Extract pose (upper body only)
        if results.pose_landmarks:
            pose = np.array(
                [
                    [results.pose_landmarks.landmark[i].x,
                     results.pose_landmarks.landmark[i].y,
                     results.pose_landmarks.landmark[i].z]
                    for i in UPPER_BODY_INDICES
                ],
                dtype=np.float32,
            )
        else:
            pose = np.zeros((NUM_POSE_LANDMARKS, 3), dtype=np.float32)

        # Extract left hand
        if results.left_hand_landmarks:
            left = np.array(
                [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark],
                dtype=np.float32,
            )
        else:
            left = np.zeros((NUM_HAND_LANDMARKS, 3), dtype=np.float32)

        # Extract right hand
        if results.right_hand_landmarks:
            right = np.array(
                [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark],
                dtype=np.float32,
            )
        else:
            right = np.zeros((NUM_HAND_LANDMARKS, 3), dtype=np.float32)

        # Combine: (51, 3)
        landmarks = np.concatenate([pose, left, right], axis=0)
        return landmarks

    def process_video(self, video_path: str) -> list[np.ndarray]:
        """Extract landmarks from video file."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        landmarks_sequence = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            landmarks = self.extract_frame_landmarks(frame)
            if landmarks is not None:
                landmarks_sequence.append(landmarks)

        cap.release()
        return landmarks_sequence

    def normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """Normalize holistic landmarks (same as preprocessing)."""
        # Use shoulder midpoint as reference
        left_shoulder = landmarks[:, 1:2, :]
        right_shoulder = landmarks[:, 2:3, :]
        center = (left_shoulder + right_shoulder) / 2

        centered = landmarks - center

        # Scale by shoulder width
        shoulder_dist = np.linalg.norm(
            landmarks[:, 1, :2] - landmarks[:, 2, :2], axis=1, keepdims=True
        )
        shoulder_dist = np.maximum(shoulder_dist, 1e-6)

        normalized = centered.copy()
        normalized[:, :, :2] = centered[:, :, :2] / shoulder_dist[:, :, np.newaxis]

        return normalized

    @torch.no_grad()
    def predict(self, landmarks_sequence: list[np.ndarray], top_k: int = 5) -> list[tuple[str, float]]:
        """Predict word from landmark sequence."""
        if len(landmarks_sequence) < 5:
            return [("insufficient_frames", 0.0)]

        # Stack and normalize
        landmarks = np.stack(landmarks_sequence, axis=0)  # (seq_len, 51, 3)
        landmarks = self.normalize_landmarks(landmarks)
        landmarks = pad_or_truncate(landmarks, self.max_seq_len)

        # Extract features
        features = extract_features(landmarks, mode=self.feature_mode)
        features = torch.from_numpy(features).float().unsqueeze(0).to(self.device)

        # Predict
        logits = self.model(features)
        probs = torch.softmax(logits, dim=1)

        # Get top-k predictions
        top_probs, top_indices = probs.topk(top_k, dim=1)
        top_probs = top_probs[0].cpu().numpy()
        top_indices = top_indices[0].cpu().numpy()

        predictions = [(self.classes[idx], float(prob)) for idx, prob in zip(top_indices, top_probs)]
        return predictions

    def predict_from_video(self, video_path: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Predict word from video file."""
        landmarks_sequence = self.process_video(video_path)
        return self.predict(landmarks_sequence, top_k)

    def run_webcam(self, record_key: str = "r", quit_key: str = "q"):
        """Run interactive webcam prediction."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise ValueError("Cannot open webcam")

        print("\nWebcam mode:")
        print(f"  Press '{record_key}' to start/stop recording")
        print(f"  Press '{quit_key}' to quit")

        recording = False
        landmarks_buffer = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Mirror for natural interaction
            frame = cv2.flip(frame, 1)

            # Extract landmarks (for visualization)
            landmarks = self.extract_frame_landmarks(frame)

            if recording:
                landmarks_buffer.append(landmarks)
                cv2.putText(
                    frame,
                    f"RECORDING... ({len(landmarks_buffer)} frames)",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
            else:
                cv2.putText(
                    frame,
                    "Press 'r' to record",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("Word Recognition", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(record_key):
                if recording:
                    # Stop recording and predict
                    recording = False
                    if len(landmarks_buffer) > 10:
                        predictions = self.predict(landmarks_buffer)
                        print("\nPredictions:")
                        for word, prob in predictions:
                            print(f"  {word}: {prob:.4f}")
                    landmarks_buffer = []
                else:
                    # Start recording
                    recording = True
                    landmarks_buffer = []

            elif key == ord(quit_key):
                break

        cap.release()
        cv2.destroyAllWindows()

    def close(self):
        """Clean up resources."""
        self.holistic.close()


def find_latest_checkpoint(models_dir: Path = Path("models")) -> Path | None:
    """Find most recent words checkpoint in MLflow artifacts."""
    mlruns = models_dir / "mlruns"
    if not mlruns.exists():
        return None

    # Search for words checkpoints
    checkpoints = list(mlruns.rglob("words_*.pth"))
    if not checkpoints:
        return None

    # Return most recent
    return max(checkpoints, key=lambda p: p.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(description="Word-level sign language inference")
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--webcam", action="store_true", help="Use webcam")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top predictions")

    args = parser.parse_args()

    # Find checkpoint
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = find_latest_checkpoint()
        if checkpoint_path is None:
            print("No checkpoint found. Train a model first:")
            print("  python -m src.cv_model.words_train")
            return
        print(f"Using checkpoint: {checkpoint_path}")

    predictor = WordsPredictor(str(checkpoint_path))

    try:
        if args.webcam:
            predictor.run_webcam()
        elif args.video:
            predictions = predictor.predict_from_video(args.video, args.top_k)
            print("\nPredictions:")
            for word, prob in predictions:
                print(f"  {word}: {prob:.4f}")
        else:
            print("Specify --video or --webcam")
    finally:
        predictor.close()


if __name__ == "__main__":
    main()
