# %% [markdown]
# # Model Comparison: Landmark MLP vs MobileNetV2
#
# Comparing two models:
# - **inquisitive-robin-2**: Landmark-only MLP model (Landmark_only experiment)
# - **mobilenetv2_png_landmark**: CNN with raw images (CNN_v2 experiment)

# %% Imports
import pandas as pd
import numpy as np
import logging
import os
import warnings
import seaborn as sns
import sys
from pathlib import Path
# Suppress MLflow logs BEFORE importing mlflow
os.environ["MLFLOW_TRACKING_SILENCE_DEPRECATION_WARNINGS"] = "true"
import mlflow
import numpy as np
import torch
from mlflow.tracking import MlflowClient
logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.getLogger("alembic").setLevel(logging.ERROR)
logging.getLogger("mlflow.store.db.utils").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*filesystem tracking backend.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow")


# Add project root to path


# %% MLflow Helper Functions
def get_mlflow_client() -> MlflowClient:
    """Get configured MLflow client."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    return MlflowClient()


def list_experiments_summary() -> None:
    """Print summary table of all experiments and runs."""
    client = get_mlflow_client()
    experiments = client.search_experiments()

    print(f"{'Experiment':<25} {'Runs':>6} {'Finished':>10} {'Best Test Acc':>15}")
    print("-" * 60)

    for exp in experiments:
        if exp.name == "Default":
            continue

        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            max_results=100,
        )

        n_runs = len(runs)
        n_finished = sum(1 for r in runs if r.info.status == "FINISHED")

        # Best test accuracy
        best_test = max(
            (r.data.metrics.get("test_acc", 0) for r in runs if r.info.status == "FINISHED"),
            default=0,
        )

        best_str = f"{best_test:.1%}" if best_test > 0 else "-"
        print(f"{exp.name:<25} {n_runs:>6} {n_finished:>10} {best_str:>15}")


def get_run_info(run_id: str) -> dict:
    """Get full run information from MLflow."""
    client = get_mlflow_client()
    run = client.get_run(run_id)

    return {
        "run_id": run.info.run_id,
        "run_name": run.info.run_name,
        "status": run.info.status,
        "start_time": run.info.start_time,
        "end_time": run.info.end_time,
        "params": dict(run.data.params),
        "metrics": dict(run.data.metrics),
        "tags": dict(run.data.tags),
    }


def get_run_metrics(run_id: str) -> dict:
    """Get all metrics for a run."""
    client = get_mlflow_client()
    run = client.get_run(run_id)
    return dict(run.data.metrics)


def get_run_params(run_id: str) -> dict:
    """Get all parameters for a run."""
    client = get_mlflow_client()
    run = client.get_run(run_id)
    return dict(run.data.params)


def list_artifacts(run_id: str) -> list[str]:
    """List all artifacts for a run."""
    client = get_mlflow_client()
    artifacts = client.list_artifacts(run_id)
    return [a.path for a in artifacts]


def download_artifact(run_id: str, artifact_path: str) -> str:
    """Download artifact and return local path."""
    client = get_mlflow_client()
    return client.download_artifacts(run_id, artifact_path)


# %% Model Comparison Table
def print_model_comparison_table() -> pd.DataFrame:
    """Print a formatted comparison table for models in MODELS_CONFIG."""
    client = get_mlflow_client()

    # Collect data for each model
    rows = []
    for key, cfg in MODELS_CONFIG.items():
        data = {}
        run = client.get_run(cfg["run_id"])
        metrics = run.data.metrics
        params = run.data.params
        data["Model"] = cfg["run_name"]
        data.update(params)
        data.update(metrics)
        rows.append(data)
    # Print table header
    return pd.DataFrame(rows)


# %% Model Loading Functions
def load_model_checkpoint(run_id: str, artifact_name: str = "best.pth") -> dict:
    """Load model checkpoint from MLflow artifacts.

    Args:
        run_id: MLflow run ID
        artifact_name: Name of checkpoint file (best.pth or model.pth)

    Returns:
        Checkpoint dict with model_state_dict, config, etc.
    """
    artifact_path = f"model/{artifact_name}"
    local_path = download_artifact(run_id, artifact_path)
    checkpoint = torch.load(local_path, map_location=DEVICE, weights_only=False)
    return checkpoint


def load_landmark_model(run_id: str, artifact_name: str = "best.pth") -> tuple:
    """Load landmark-based model from MLflow.

    Returns:
        (model, config, checkpoint)
    """
    from src.models import get_model

    checkpoint = load_model_checkpoint(run_id, artifact_name)
    config = checkpoint.get("config", {})

    model = get_model(
        model_type=config.get("model", "bigru"),
        num_classes=config["num_classes"],
        input_dim=config["input_dim"],
        hidden_dim=config.get("hidden_dim", 128),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model, config, checkpoint


def load_image_model(run_id: str, artifact_name: str = "best.pth") -> tuple:
    """Load image-based model (MobileNet) from MLflow.

    Returns:
        (model, config, checkpoint)
    """
    from src.models import get_model

    checkpoint = load_model_checkpoint(run_id, artifact_name)
    config = checkpoint.get("config", {})

    model = get_model(
        model_type=config.get("model", "mobilenet_v2"),
        num_classes=config["num_classes"],
        input_dim=0,  # Not used for image models
        freeze_backbone=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model, config, checkpoint


def load_model_by_config(model_key: str) -> tuple:
    """Load model using MODELS_CONFIG.

    Args:
        model_key: Key from MODELS_CONFIG ('landmark_mlp' or 'mobilenet_v2')

    Returns:
        (model, config, checkpoint)
    """
    cfg = MODELS_CONFIG[model_key]
    run_id = cfg["run_id"]

    if cfg["model_type"] == "landmark":
        return load_landmark_model(run_id)
    elif cfg["model_type"] == "image":
        return load_image_model(run_id)
    else:
        raise ValueError(f"Unknown model type: {cfg['model_type']}")


# %% Data Loading Functions
def get_dataloaders_from_params(run_id: str, seed: int = 42):
    """Reproduce exact dataloaders from training run params.

    Args:
        run_id: MLflow run ID
        seed: Random seed (default 42, same as training)

    Returns:
        (train_loader, val_loader, test_loader, num_classes, classes, feature_dim)
    """
    from src.preprocessing.dataset import create_dataloaders, create_image_dataloaders

    params = get_run_params(run_id)

    # Parse params from MLflow
    data_dir = Path(params.get("data_dir", None))
    assert data_dir is not None , "data_dir is None"
    model_type = params.get("model", "bigru")
    letters_str = params.get("letters", "all")
    letters = None if letters_str == "all" else letters_str.split(",")
    batch_size = int(params.get("batch_size", 32))
    feature_mode = params.get("features", "xy_angles")

    # Split ratios (training uses 70/15/15)
    split_str = params.get("split", "70/15/15")
    if split_str == "70/15/15":
        train_ratio, val_ratio, test_ratio = 0.70, 0.15, 0.15
    elif split_str == "80/20/0":
        train_ratio, val_ratio, test_ratio = 0.80, 0.20, 0.0
    else:
        train_ratio, val_ratio, test_ratio = 0.70, 0.15, 0.15

    is_image_model = model_type in ("mobilenet", "mobilenet_v2")

    if is_image_model:
        return create_image_dataloaders(
            data_dir=data_dir,
            batch_size=batch_size,
            letters=letters,
            seed=seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
    else:
        output_mode = "static" if model_type == "static" else "sequence"
        # Handle "images (N/A)" feature mode from image models
        if "N/A" in feature_mode:
            feature_mode = "xy_angles"
        return create_dataloaders(
            data_dir=data_dir,
            feature_mode=feature_mode,
            output_mode=output_mode,
            batch_size=batch_size,
            letters=letters,
            seed=seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )


def get_dataloaders_by_config(model_key: str, seed: int = 42):
    """Get dataloaders for a model from MODELS_CONFIG.

    Args:
        model_key: Key from MODELS_CONFIG ('landmark_mlp' or 'mobilenet_v2')
        seed: Random seed (default 42)

    Returns:
        (train_loader, val_loader, test_loader, num_classes, classes, feature_dim)
    """
    cfg = MODELS_CONFIG[model_key]
    return get_dataloaders_from_params(cfg["run_id"], seed=seed)


def get_landmark_dataloaders(seed: int = 42):
    """Get dataloaders for landmark-based models (default config)."""
    from src.preprocessing.dataset import create_dataloaders

    return create_dataloaders(seed=seed)


def get_image_dataloaders(seed: int = 42):
    """Get dataloaders for image-based models (default config)."""
    from src.preprocessing.dataset import create_image_dataloaders

    return create_image_dataloaders(seed=seed)

# %% Data Paths

# %% Visualization: Dataset Samples (6x5 grid)
import matplotlib.pyplot as plt
from PIL import Image


def plot_landmarks(ax, landmarks: np.ndarray, title: str = "", normalize: bool = True):
    """Plot hand landmarks on axes.

    Args:
        ax: Matplotlib axes
        landmarks: (21, 3) or (seq_len, 21, 3) landmark array
        normalize: Whether to normalize landmarks to [0, 1] range
    """
    # Use first frame if sequence
    if landmarks.ndim == 3:
        landmarks = landmarks[0]

    x, y = landmarks[:, 0].copy(), landmarks[:, 1].copy()

    # Normalize to [0, 1] range
    if normalize:
        x = x - x.min()
        y = y - y.min()
        scale = max(x.max(), y.max())
        if scale > 0:
            x = x / scale
            y = y / scale

    # MediaPipe hand connections
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
        (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
        (5, 9), (9, 13), (13, 17),  # Palm
    ]

    # Draw connections first
    for i, j in connections:
        ax.plot([x[i], x[j]], [y[i], y[j]], color="royalblue", linewidth=2, alpha=0.8)

    # Draw points on top
    ax.scatter(x, y, c="red", s=40, zorder=5, edgecolors="darkred", linewidths=0.5)

    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(1.15, -0.15)  # Inverted for image-like coordinates
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)


def plot_folder_samples(data_dir: Path, n_cols: int = 6, title: str | None = None, seed: int = 42):
    """Plot one sample per class from a data folder.

    Args:
        data_dir: Path to folder with subfolders per class
        n_cols: Number of columns in grid
        title: Plot title (default: folder name)
        seed: Random seed
    """
    np.random.seed(seed)
    data_dir = Path(data_dir)

    # Get classes
    classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    n_classes = len(classes)
    n_rows = (n_classes + n_cols - 1) // n_cols

    # Detect type: landmarks (.npy) or images
    sample_dir = data_dir / classes[0]
    is_landmarks = bool(list(sample_dir.glob("*.npy")))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    axes = np.array(axes).flatten()

    if title is None:
        title = data_dir.name
    fig.suptitle(title, fontsize=14)

    for idx, letter in enumerate(classes):
        ax = axes[idx]
        class_dir = data_dir / letter

        if is_landmarks:
            files = list(class_dir.glob("*.npy"))
            if files:
                landmarks = np.load(np.random.choice(files))
                plot_landmarks(ax, landmarks)
        else:
            files = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))
            if files:
                img = Image.open(np.random.choice(files))
                ax.imshow(img)

        ax.set_title(letter.upper(), fontsize=11, fontweight="bold")
        ax.axis("off")

    # Hide unused
    for idx in range(len(classes), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    return fig


# %% Dataset Split Statistics
def print_loader_stats(train_loader, val_loader, test_loader, classes: list[str]) -> pd.DataFrame:
    """Print sample counts per loader and return DataFrame with per-class counts."""
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)
    test_size = len(test_loader.dataset) if test_loader else 0
    total = train_size + val_size + test_size

    print(f"{'Split':<10} {'Samples':>10} {'Percent':>10}")
    print("-" * 32)
    print(f"{'Train':<10} {train_size:>10} {train_size/total:>10.1%}")
    print(f"{'Val':<10} {val_size:>10} {val_size/total:>10.1%}")
    if test_size > 0:
        print(f"{'Test':<10} {test_size:>10} {test_size/total:>10.1%}")
    print("-" * 32)
    print(f"{'Total':<10} {total:>10}")

    # Count per class per split
    def count_per_class(loader, split_name: str) -> list[dict]:
        counts = {c: 0 for c in classes}
        for _, labels in loader:
            for label in labels.numpy():
                counts[classes[label]] += 1
        return [{"class": c, "split": split_name, "count": counts[c]} for c in classes]

    rows = []
    rows.extend(count_per_class(train_loader, "train"))
    rows.extend(count_per_class(val_loader, "val"))
    if test_loader and test_size > 0:
        rows.extend(count_per_class(test_loader, "test"))

    return pd.DataFrame(rows)


def plot_split_distribution(df: pd.DataFrame, title: str = "Class Distribution by Split"):
    """Bar plot with class counts, hue by split."""

    # Order splits logically
    split_order = ["train", "val", "test"]
    df["split"] = pd.Categorical(df["split"], categories=split_order, ordered=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=df, x="class", y="count", hue="split", ax=ax, palette="Set2")
    ax.set_xlabel("Class")
    ax.set_ylabel("Sample Count")
    ax.set_title(title)
    ax.legend(title="Split")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig




PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
print(PROJECT_ROOT)

# %% Configuration
# MLflow paths
MLFLOW_URI = f"sqlite:///{PROJECT_ROOT / 'models/mlflow.db'}"
MLFLOW_ARTIFACTS = f"file://{PROJECT_ROOT / 'models/mlartifacts'}"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Models to compare
MODELS_CONFIG = {
    "landmark_mlp": {
        "run_name": "inquisitive-robin-2",
        "run_id": "c6d5eb3a9c544c04a8f6cefca6dc3e72",
        "experiment": "Landmark_only",
        "description": "MLP on hand landmarks (56 features)",
        "model_type": "landmark",  # Uses landmark preprocessing
    },
    "mobilenet_v2": {
        "run_name": "mobilenetv2_png_landmark",
        "run_id": "21e24b44d3f6477c89c6acf2d9c11ff4",
        "experiment": "CNN_v2",
        "description": "MobileNetV2 on raw images (224x224)",
        "model_type": "image",  # Uses image preprocessing
    },
}
DATA_DIR_LANDMARKS = PROJECT_ROOT / "data/raw_landmarks"
DATA_DIR_PHOTOS = PROJECT_ROOT / "data/raw_photo_landmark"

# %% Load dataloaders and show stats for landmark model
print("=" * 50)
print("Landmark MLP Dataset Stats")
print("=" * 50)
train_ld, val_ld, test_ld, num_classes, classes_list, feat_dim = get_dataloaders_by_config("landmark_mlp")
df_stats = print_loader_stats(train_ld, val_ld, test_ld, classes_list)

# %% Plot split distribution
fig_split = plot_split_distribution(df_stats, title="Landmark Dataset: Class Distribution by Split")
#plt.show()

# %% Plot photos
fig_photos = plot_folder_samples(DATA_DIR_PHOTOS, n_cols=6, title="Photos (raw_photo_landmark)")
#plt.savefig(PROJECT_ROOT / "blog/figures/01_samples_photos.png", dpi=150, bbox_inches="tight")
#plt.show()

# %% Plot landmarks
fig_landmarks = plot_folder_samples(DATA_DIR_LANDMARKS, n_cols=6, title="Landmarks (raw_landmarks)")
#plt.savefig(PROJECT_ROOT / "blog/figures/02_samples_landmarks.png", dpi=150, bbox_inches="tight")
#plt.show()

# %%
