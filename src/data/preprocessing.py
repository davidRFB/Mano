"""
Feature extraction and normalization for hand landmarks.

Converts raw MediaPipe landmarks (21 points × 3 coords) into features
suitable for machine learning models.

Feature modes:
- "xy": X, Y coordinates only (42 features)
- "xy_angles": X, Y + finger joint angles (56 features) - RECOMMENDED
- "xy_angles_distances": X, Y + angles + key distances (66 features)
- "full": All above + velocities (108 features)
"""

import numpy as np

# Feature dimensions per mode
FEATURE_MODES = {
    "xy": 42,                    # 21 landmarks × 2 coords
    "xy_angles": 56,             # 42 + 14 finger angles
    "xy_angles_distances": 66,   # 56 + 10 key distances
    "full": 108,                 # 66 + 42 velocities
}
DEFAULT_FEATURE_MODE = "xy_angles"

# MediaPipe hand landmark indices
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

# Joint triplets for angle calculation (middle point is the joint)
FINGER_ANGLES = [
    # Thumb (2 angles)
    (THUMB_CMC, THUMB_MCP, THUMB_IP),
    (THUMB_MCP, THUMB_IP, THUMB_TIP),
    # Index (3 angles)
    (WRIST, INDEX_MCP, INDEX_PIP),
    (INDEX_MCP, INDEX_PIP, INDEX_DIP),
    (INDEX_PIP, INDEX_DIP, INDEX_TIP),
    # Middle (3 angles)
    (WRIST, MIDDLE_MCP, MIDDLE_PIP),
    (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP),
    (MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    # Ring (3 angles)
    (WRIST, RING_MCP, RING_PIP),
    (RING_MCP, RING_PIP, RING_DIP),
    (RING_PIP, RING_DIP, RING_TIP),
    # Pinky (3 angles)
    (WRIST, PINKY_MCP, PINKY_PIP),
    (PINKY_MCP, PINKY_PIP, PINKY_DIP),
    (PINKY_PIP, PINKY_DIP, PINKY_TIP),
]

# Key distances (fingertip spreads, pinch distances)
KEY_DISTANCES = [
    (THUMB_TIP, INDEX_TIP),
    (THUMB_TIP, MIDDLE_TIP),
    (THUMB_TIP, RING_TIP),
    (THUMB_TIP, PINKY_TIP),
    (INDEX_TIP, MIDDLE_TIP),
    (MIDDLE_TIP, RING_TIP),
    (RING_TIP, PINKY_TIP),
    (INDEX_MCP, PINKY_MCP),  # Palm width
    (WRIST, MIDDLE_TIP),
    (WRIST, THUMB_TIP),
]


def compute_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Compute angle at p2 formed by vectors p1->p2 and p2->p3."""
    v1 = p1 - p2
    v2 = p3 - p2
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cos_angle = np.clip(cos_angle, -1, 1)
    angle = np.arccos(cos_angle)
    # Normalize: 0 = straight (180°), 1 = fully bent (0°)
    return 1.0 - (angle / np.pi)


def extract_angles(landmarks: np.ndarray) -> np.ndarray:
    """Extract 14 finger joint angles from landmarks (21, 2) or (21, 3)."""
    return np.array([
        compute_angle(landmarks[p1, :2], landmarks[p2, :2], landmarks[p3, :2])
        for p1, p2, p3 in FINGER_ANGLES
    ], dtype=np.float32)


def extract_distances(landmarks: np.ndarray) -> np.ndarray:
    """Extract 10 key distances between landmarks."""
    return np.array([
        np.linalg.norm(landmarks[p1, :2] - landmarks[p2, :2])
        for p1, p2 in KEY_DISTANCES
    ], dtype=np.float32)


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Normalize landmarks relative to wrist and scale by hand size.

    Args:
        landmarks: (seq_len, 21, 3) or (21, 3) raw landmarks

    Returns:
        Same shape, normalized
    """
    single_frame = landmarks.ndim == 2
    if single_frame:
        landmarks = landmarks[np.newaxis, :, :]

    # Center on wrist
    wrist = landmarks[:, 0:1, :]
    centered = landmarks - wrist

    # Scale by max distance from wrist (using X, Y only)
    distances = np.linalg.norm(centered[:, :, :2], axis=2)
    max_dist = np.maximum(distances.max(axis=1, keepdims=True), 1e-6)
    normalized = centered / max_dist[:, :, np.newaxis]

    return normalized[0] if single_frame else normalized


def extract_features(sequence: np.ndarray, mode: str = DEFAULT_FEATURE_MODE) -> np.ndarray:
    """
    Extract features from landmark sequence.

    Args:
        sequence: (seq_len, 21, 3) normalized landmarks
        mode: Feature mode (xy, xy_angles, xy_angles_distances, full)

    Returns:
        (seq_len, feature_dim) feature array
    """
    seq_len = sequence.shape[0]

    # Flip Y axis (MediaPipe Y goes down, we want Y up)
    sequence = sequence.copy()
    sequence[:, :, 1] = -sequence[:, :, 1]

    # Base: X, Y coordinates (drop Z which is noisy)
    xy = sequence[:, :, :2]
    features = [xy.reshape(seq_len, -1)]  # 42 features

    if mode == "xy":
        return features[0]

    # Add angles
    if mode in ["xy_angles", "xy_angles_distances", "full"]:
        angles = np.array([extract_angles(xy[i]) for i in range(seq_len)])
        features.append(angles)  # +14 features

    # Add distances
    if mode in ["xy_angles_distances", "full"]:
        distances = np.array([extract_distances(xy[i]) for i in range(seq_len)])
        features.append(distances)  # +10 features

    # Add velocities
    if mode == "full":
        xy_flat = xy.reshape(seq_len, -1)
        velocities = np.zeros_like(xy_flat)
        velocities[1:] = xy_flat[1:] - xy_flat[:-1]
        features.append(velocities)  # +42 features

    return np.concatenate(features, axis=1).astype(np.float32)


def extract_single_frame_features(landmarks: np.ndarray, mode: str = DEFAULT_FEATURE_MODE) -> np.ndarray:
    """
    Extract features from a single frame of landmarks.

    Args:
        landmarks: (21, 3) single frame landmarks (normalized)
        mode: Feature mode

    Returns:
        (feature_dim,) feature vector
    """
    # Add temporal dimension, extract, remove temporal dimension
    sequence = landmarks[np.newaxis, :, :]
    features = extract_features(sequence, mode)
    return features[0]
