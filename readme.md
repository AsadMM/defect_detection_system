# Defect Detection System (Autoencoder + FastAPI)

An anomaly/defect detection service built on MVTEC-AD style data.

- One model per object class (for localized defect masking/drawing)
- FastAPI inference API with model registry/cache and MLflow integration
- Training pipeline with configurable CLI/YAML settings and MLflow tracking

## Architecture

```text
Client
  -> FastAPI (api/main.py, api/routes.py)
      -> Inference runtime (src/inference/serving.py)
          -> Model cache + MLflow/local fallback
          -> TensorFlow model inference
      -> Response (array output or zipped images)

Training (src/training/train.py)
  -> Data loading + augmentation
  -> Autoencoder training
  -> Threshold estimation
  -> Local artifacts + MLflow run/model registry
```

## Repository Layout

- `api/` HTTP layer (routes, schemas, enums, constants)
- `src/inference/` serving runtime (model registry, inference)
- `src/training/` training pipeline
- `src/data/` data loading + augmentation
- `src/models/` model architecture + threshold logic
- `artifacts/` local model, thresholds, sizes, checkpoints
- `configs/` YAML training configs
- `data/` dataset root (e.g., MVTEC)

## Prerequisites

- Python 3.13
- Docker (optional)
- Dataset available under `data/`
- At least one trained model in `artifacts/models` for API startup

## Install (Local Python)

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run API (Dev)

```bash
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

- Swagger: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

## Run API (Prod-style)

CPU-oriented example:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 2
```

Single-GPU machines (limited VRAM):

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1
```


## Run with Docker

Build:

```bash
docker build -t defect-detector:latest .
```

Run:

```bash
docker run --rm -p 8000:8000 defect-detector:latest
```

Open docs at `http://127.0.0.1:8000/docs`.

## MLflow

Training and serving use:

- Tracking URI: `sqlite:///artifacts/mlflow/mlflow.db`
- Experiment: `autoencoder_anomaly_detection`

Start MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///artifacts/mlflow/mlflow.db --host 127.0.0.1 --port 5000
```

Open: `http://127.0.0.1:5000`

## Training

Short form:

```bash
python3 -m src.training.train --config configs/screw.yaml
```

CLI form:

```bash
python3 -m src.training.train \
  --name screw \
  --epochs 20 \
  --batch_size 8 \
  --img_size 128 \
  --aug_to 2000 \
  --rotate_min -45 \
  --rotate_max 45 \
  --crop_limit 100 \
  --test_size 0.2 \
  --threshold_percentile 99 \
  --filters 32 64 96 \
  --latent_dim 100 \
  --seed 26
```

For full training docs (all options + YAML mapping), see:

- [docs/training.md](docs/training.md)

## API Endpoints

- `POST /predict_array/{model_name}`
- `POST /predict_image/{model_name}`

Both endpoints support:

- `stage` / `version` model selection
- threshold selection (`90.0 <= threshold < 100`)
- output format (`mask` or `redrawn`)

## Notes

- API startup intentionally fails if no models are available in artifacts.
- `requirements.txt` is a slim direct-dependency file.
- `requirements.lock.txt` pins the full environment snapshot.
