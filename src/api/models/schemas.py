"""
Pydantic models for API request/response schemas.

These define the exact shape of data the API accepts and returns.
FastAPI uses these for:
    - Automatic validation
    - Automatic documentation (OpenAPI/Swagger)
    - Type hints for IDE support
"""

from pydantic import BaseModel, Field


# =============================================================================
# Response Models
# =============================================================================


class PredictionResult(BaseModel):
    """Single gesture prediction result."""

    letter: str = Field(..., description="Predicted letter (A-Z or SPACE)")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score between 0 and 1"
    )


class AllProbabilities(BaseModel):
    """Probability distribution across all classes."""

    # This will be a dict like {"A": 0.94, "B": 0.02, ...}
    # Using dict for flexibility with dynamic class names
    pass


class PredictionMetadata(BaseModel):
    """Metadata about the prediction."""

    model_version: str = Field(..., description="Model run ID from MLflow")
    inference_time_ms: float = Field(
        ..., ge=0, description="Time taken for inference in milliseconds"
    )


class PredictResponse(BaseModel):
    """
    Full response from /predict endpoint.

    Example:
        {
            "prediction": {
                "letter": "A",
                "confidence": 0.94
            },
            "all_probabilities": {"A": 0.94, "B": 0.02, "C": 0.01, ...},
            "metadata": {
                "model_version": "587ca0fd...",
                "inference_time_ms": 45.2
            }
        }
    """

    prediction: PredictionResult
    all_probabilities: dict[str, float] = Field(
        ..., description="Probability for each class"
    )
    metadata: PredictionMetadata

    class Config:
        json_schema_extra = {
            "example": {
                "prediction": {"letter": "A", "confidence": 0.94},
                "all_probabilities": {"A": 0.94, "B": 0.02, "C": 0.01},
                "metadata": {
                    "model_version": "587ca0fd066a4a1fbf1a5a26971c3284",
                    "inference_time_ms": 45.2,
                },
            }
        }


# =============================================================================
# Health Check Models
# =============================================================================


class HealthResponse(BaseModel):
    """Response from /health endpoint."""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_version: str | None = Field(None, description="Loaded model version")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "model_version": "587ca0fd066a4a1fbf1a5a26971c3284",
            }
        }


# =============================================================================
# Error Models
# =============================================================================


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str = Field(..., description="Error type")
    detail: str = Field(..., description="Error details")

    class Config:
        json_schema_extra = {
            "example": {"error": "ValidationError", "detail": "Invalid image format"}
        }