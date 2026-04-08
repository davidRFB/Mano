"""MLflow utilities for loading models from artifacts."""

import logging
import warnings
from pathlib import Path

# Suppress MLflow/Alembic logs and warnings BEFORE importing mlflow
logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.getLogger("alembic").setLevel(logging.ERROR)
logging.getLogger("mlflow.store.db.utils").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*filesystem tracking backend.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow")

import mlflow
import torch
from mlflow.tracking import MlflowClient

from src.models import get_model

MLFLOW_URI = f"sqlite:///{Path('models/mlflow.db').absolute()}"
MLFLOW_ARTIFACTS = f"file://{Path('models/mlartifacts').absolute()}"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_run_by_name(run_name: str, experiment_name: str | None = None) -> str | None:
    """Get run ID by run name.

    Args:
        run_name: Name of the MLflow run
        experiment_name: Name of the experiment (None = search all experiments)

    Returns:
        Run ID if found, None otherwise
    """
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()

    if experiment_name:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            return None
        exp_ids = [experiment.experiment_id]
    else:
        # Search all experiments
        experiments = client.search_experiments()
        exp_ids = [exp.experiment_id for exp in experiments]

    if not exp_ids:
        return None

    runs = client.search_runs(
        experiment_ids=exp_ids,
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        max_results=1,
    )

    return runs[0].info.run_id if runs else None


def load_model_from_mlflow(
    run_id: str | None = None,
    run_name: str | None = None,
    artifact_name: str = "best.pth",
    device: torch.device = DEVICE,
) -> tuple[torch.nn.Module, dict]:
    """Load model from MLflow artifacts.

    Args:
        run_id: MLflow run ID (takes precedence over run_name)
        run_name: MLflow run name (used if run_id not provided)
        artifact_name: Name of artifact file (best.pth or model.pth)
        device: Device to load model on

    Returns:
        Tuple of (model, config)

    Raises:
        ValueError: If neither run_id nor run_name provided, or run not found
    """
    mlflow.set_tracking_uri(MLFLOW_URI)

    # Resolve run_id from run_name if needed
    if run_id is None:
        if run_name is None:
            raise ValueError("Must provide either run_id or run_name")
        run_id = get_run_by_name(run_name)
        if run_id is None:
            raise ValueError(f"Run not found with name: {run_name}")

    # Download artifact
    client = MlflowClient()
    artifact_path = f"model/{artifact_name}"

    try:
        local_path = client.download_artifacts(run_id, artifact_path)
    except Exception as e:
        raise ValueError(f"Failed to download artifact '{artifact_path}' from run {run_id}: {e}")

    # Load checkpoint
    checkpoint = torch.load(local_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})

    if "num_classes" not in config:
        raise ValueError(
            f"Checkpoint missing 'config'. Re-train the model with updated scripts. "
            f"Checkpoint keys: {list(checkpoint.keys())}"
        )

    # Build model
    model = get_model(
        model_type=config.get("model", "bigru"),
        num_classes=config["num_classes"],
        input_dim=config["input_dim"],
        hidden_dim=config.get("hidden_dim", 128),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, config


def _extract_run_info(run, experiment_name: str) -> dict:
    """Extract run info with all accuracy metrics."""
    metrics = run.data.metrics
    return {
        "run_id": run.info.run_id,
        "run_name": run.info.run_name,
        "experiment": experiment_name,
        "train_acc": metrics.get("train_acc"),
        "val_acc": metrics.get("best_val_acc"),
        "test_acc": metrics.get("test_acc"),
        "status": run.info.status,
    }


def list_runs(experiment_name: str = "lsc_letters", limit: int = 10) -> list[dict]:
    """List recent MLflow runs from a specific experiment.

    Args:
        experiment_name: Name of the experiment (default: lsc_letters)
        limit: Maximum number of runs to return

    Returns:
        List of run info dicts with keys: run_id, run_name, experiment,
        train_acc, val_acc, test_acc, status
    """
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return []

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=limit,
    )

    return [_extract_run_info(r, experiment_name) for r in runs]


def list_all_runs(limit: int = 20) -> list[dict]:
    """List recent MLflow runs from ALL experiments.

    Args:
        limit: Maximum number of runs to return

    Returns:
        List of run info dicts with keys: run_id, run_name, experiment,
        train_acc, val_acc, test_acc, status
    """
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()

    # Get all experiments
    experiments = client.search_experiments()
    if not experiments:
        return []

    exp_ids = [exp.experiment_id for exp in experiments]
    exp_map = {exp.experiment_id: exp.name for exp in experiments}

    runs = client.search_runs(
        experiment_ids=exp_ids,
        order_by=["start_time DESC"],
        max_results=limit,
    )

    return [_extract_run_info(r, exp_map.get(r.info.experiment_id, "unknown")) for r in runs]


def list_experiments() -> list[str]:
    """List all MLflow experiment names."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()
    experiments = client.search_experiments()
    return [exp.name for exp in experiments if exp.name != "Default"]
