# =============================================================================
# LSC Gesture Recognition API (CPU Only)
# =============================================================================
# Build:  docker build -t mano-api .
# Run:    docker run -p 8000:8000 mano-api
#
# Hugging Face Spaces (port 7860):
#   Automatically handled via PORT env var
# =============================================================================

FROM python:3.11-slim

WORKDIR /app

# Install PyTorch CPU-only (smaller image)
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install API dependencies
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# --- Hugging Face Spaces: non-root user ---
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy only what the API needs
COPY --chown=user src/preprocessing/ src/preprocessing/
COPY --chown=user src/models/ src/models/
COPY --chown=user src/__init__.py src/__init__.py
COPY --chown=user api/main.py api/main.py
COPY --chown=user api/model.pth api/model.pth

EXPOSE 7860

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
