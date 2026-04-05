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
- At least one trained model in `artifacts/models` for API startup
- Dataset under `data/` is required only for training (not required for quick inference tests with bundled artifacts).

## Quick Test Without Training

A prebuilt artifact bundle is included at the repo root (timestamped filename):

- `artifacts_bundle_*.tar.gz`

It contains `artifacts/models/`, `artifacts/sizes/`, and `artifacts/thresholds/` so you can run inference/API tests without training first.

Extract it:

```bash
BUNDLE=$(ls -1t artifacts_bundle_*.tar.gz | head -n 1)
rm -rf artifacts/models artifacts/sizes artifacts/thresholds
tar -xzf "$BUNDLE" -C .
```

Sanity-check extracted files:

```bash
ls artifacts/models artifacts/sizes artifacts/thresholds
```

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

Smoke test with the bundled `screw` artifacts:

```bash
python3 scripts/smoke_test_api.py --model screw --output-format mask --stage Production
```

## Run API (Prod-style)

CPU-oriented example:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 2
```

Single-GPU machines (limited VRAM):

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1
```

### Runtime guardrails (env vars)

```bash
export MAX_ARRAY_BATCH_SIZE=32
export MAX_IMAGE_BATCH_SIZE=16
export MODEL_CACHE_MAX_BYTES=1073741824
export MLFLOW_TRACKING_URI=sqlite:///artifacts/mlflow/mlflow.db
```

- `MODEL_CACHE_MAX_BYTES` controls in-memory model cache budget (default: `1073741824` = 1 GiB).
- Cache policy is LRU by model key (`name + stage/version`); oldest cached model is evicted first when capacity is exceeded.


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

Important: keep API and MLflow UI on the same tracking backend. If they use different `MLFLOW_TRACKING_URI` values, stage/version lookups in API can fail even when UI shows models.

Register local artifact models into MLflow (and promote latest version to `Production`):

```bash
python3 scripts/test_register_artifacts.py
```

You can also run it from inside the `scripts/` directory:

```bash
cd scripts
python3 test_register_artifacts.py
```

What this script does:

- scans `artifacts/models/model_{name}.keras` for `bottle`, `hazelnut`, `screw`, `wood` (you can change this to include your own model as well)
- logs/registers each found model in MLflow Model Registry
- transitions the latest registered version for each model to `Production`

Promote a specific model version to `Production` from command line:

```bash
python3 scripts/promote_model_to_production.py --model screw --version 3
```

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

Load/performance test notes and baseline numbers:

- [docs/load-testing.md](docs/load-testing.md)

## API Endpoints

- `POST /predict_array/{model_name}`
- `POST /predict_image/{model_name}`

Both endpoints support:

- `stage` / `version` model selection
- threshold selection (`90.0 <= threshold < 100`)
- output format (`mask` or `redrawn`)

Run inference with your own inputs:

- Browser flow: open Swagger at `http://127.0.0.1:8000/docs`, expand `POST /predict_image/{model_name}` or `POST /predict_array/{model_name}`, click `Try it out`, submit your payload/files.
- Script flow: use `python3 scripts/smoke_test_api.py --model screw --output-format mask --stage Production` for array endpoint smoke tests.
- Script flow for single predict-image request with ZIP download: `python3 scripts/test_predict_image_download_zip.py --base-url http://127.0.0.1:8000 --model screw --output-format redrawn --stage Production --batch-size 2 --out artifacts/predict_image_output.zip`.
- Script flow for image endpoint under concurrency: `python3 scripts/load_test.py --base-url http://127.0.0.1:8000 --model screw --output-format mask --requests 10 --concurrency 2 --batch-size 2 --images-dir artifacts/comparison_images`.

Version vs stage behavior:

- `?version=<n>` loads `models:/<name>/<n>` directly, so it works even if that version is not in `Production`.
- `?stage=Production` loads `models:/<name>/Production`, so at least one version must be promoted to that stage.
- Do not send non-Production stage together with version (`version` + `stage=Staging` is rejected).

## Retraining Strategy (Design)

This system does not perform automatic retraining by default.
In a production setup, retraining would be triggered based on:

- Data drift detection (input distribution shift)
- Performance degradation (metric monitoring)
- Scheduled retraining (time-based fallback)

### Proposed Pipeline

1. Collect inference data + predictions
2. Periodically evaluate against ground truth
3. Trigger retraining job if thresholds violated
4. Log new model to MLflow
5. Promote model via staging → production
6. Deploy via model registry update

### Safeguards

- Manual approval gate before production promotion
- Shadow testing / A/B testing for new models
- Rollback to previous production model if needed

## Notes

- API startup intentionally fails if no models are available in artifacts.
- `requirements.txt` is a slim direct-dependency file.
- `requirements.lock.txt` pins the full environment snapshot.
