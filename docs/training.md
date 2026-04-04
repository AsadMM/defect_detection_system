# Training Guide

This document covers training configuration and execution for `src/training/train.py`.

## Run Training

### Using YAML config

```bash
python3 -m src.training.train --config configs/screw.yaml
```

### Using CLI arguments

```bash
python3 -m src.training.train --name screw --epochs 10 --batch_size 8
```

## CLI Arguments

- `--name` object class name (e.g. `screw`, `bottle`)
- `--epochs` number of epochs
- `--batch_size` training batch size
- `--img_size` image resize dimension
- `--aug_to` total size target after augmentation
- `--threshold_percentile` threshold percentile to pick from validation residual map
- `--rotate_min` minimum augmentation rotation angle
- `--rotate_max` maximum augmentation rotation angle
- `--crop_limit` maximum random crop size
- `--test_size` validation split ratio (`0 < test_size < 1`)
- `--filters` encoder/decoder filter list (space separated)
- `--latent_dim` bottleneck dimension
- `--seed` global random seed
- `--config` YAML config path

## YAML Config

Example (`configs/screw.yaml`):

```yaml
name: screw
epochs: 10
batch_size: 8
img_size: 128
aug_to: 2000
rotate_min: -45
rotate_max: 45
crop_limit: 100
test_size: 0.2
filters: [32, 64, 96]
latent_dim: 100
threshold_percentile: 99
seed: 26
```

Notes:

- YAML values override CLI defaults.
- If both CLI and YAML are provided for the same field, current implementation applies YAML values.

## Artifacts Produced

Training writes:

- `artifacts/models/model_<name>.h5`
- `artifacts/thresholds/thresholds_<name>.pkl`
- `artifacts/sizes/sizes_<name>.pkl`
- checkpoints under `artifacts/checkpoints/<name>/`

### Why we save more than just model files

Serving requires both learned weights and serving metadata:

- `model_<name>.h5`: learned reconstruction model weights.
- `sizes_<name>.pkl`: expected input shape metadata used by the API for resize/reshape and validation.
- `thresholds_<name>.pkl`: percentile-to-threshold map used to convert reconstruction error into anomaly masks.
- checkpoints: intermediate training snapshots for recovery and debugging model-quality regressions.

If only model weights are saved, the service can reconstruct images but cannot reliably apply consistent anomaly decision boundaries or input-shape validation.

## MLflow Tracking

The training script logs:

- parameters (model/data/config)
- per-epoch loss metrics
- selected threshold metric
- local artifacts
- registered model in MLflow registry

Default tracking URI:

```bash
sqlite:///artifacts/mlflow/mlflow.db
```

Start UI:

```bash
mlflow ui --backend-store-uri sqlite:///artifacts/mlflow/mlflow.db --host 127.0.0.1 --port 5000
```

## TensorFlow Model Architecture

The training pipeline uses the `build1(...)` autoencoder from `src/models/autoencoder.py`.

### High-level design

- Input: image tensor `(img_size, img_size, 3)`
- Encoder:
  - Stack of `Conv2D(strides=2, padding="same")` blocks
  - Each block uses `LeakyReLU` activation and `BatchNormalization`
  - Number of blocks is controlled by `--filters`
- Bottleneck:
  - Flattened encoder output
  - Dense latent vector of size `--latent_dim`
- Decoder:
  - Dense projection back to encoder feature volume
  - `Reshape` to decoder start shape
  - Symmetric stack of `Conv2DTranspose(strides=2, padding="same")` blocks
  - Final `Conv2DTranspose` to 3 channels + `sigmoid` activation

### Default training objective

- Loss: Mean Squared Error (`mse`)
- Optimizer: `adam`
- Early stopping on `val_loss` with best-weight restore

### How this ties to anomaly detection

- The model reconstructs normal images.
- At inference/training-eval time, per-pixel reconstruction error is computed.
- A threshold map (percentile-based) is estimated from validation residuals.
- Pixels above threshold are marked as anomalous and returned as mask/redrawn output.
