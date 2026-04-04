FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Runtime libs required by opencv-python and common ML deps.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt

COPY . .

EXPOSE 8000

# Override at runtime if needed.
ENV MLFLOW_TRACKING_URI=sqlite:///artifacts/mlflow/mlflow.db

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
