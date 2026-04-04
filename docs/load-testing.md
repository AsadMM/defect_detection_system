# Load Testing Notes

This document captures local load-test results for the FastAPI inference API.

## Environment

- OS: Linux (WSL2), kernel `6.6.87.2-microsoft-standard-WSL2`
- Python: `3.13.12`
- CPU: AMD Ryzen 5 3550H (6 vCPUs visible)
- RAM: `9.7 GiB`
- GPU: NVIDIA GeForce GTX 1650, `4096 MiB`

## Test Setup

- Endpoint: `POST /predict_image/{model_name}`
- Base URL: `http://127.0.0.1:8000`
- Model: `screw`
- Output format: `mask`
- Stage: `Production`
- Input source: `artifacts/comparison_images/screw`
- Requests: `50`
- Concurrency: `50`
- Batch size: `2`

Command used:

```bash
python3.13 scripts/load_test.py \
  --base-url http://127.0.0.1:8000 \
  --model screw \
  --output-format mask \
  --requests 50 \
  --concurrency 50 \
  --batch-size 2
```

## Results

### Cold-ish run (includes model loading/warmup effects)

- success: `50/50 (100.00%)`
- total time: `39.427 s`
- throughput: `1.268 req/s`
- latency mean: `39.166 s`
- latency median: `39.175 s`
- latency p95: `39.239 s`
- latency p99: `39.292 s`
- response size mean: `1066.1 bytes`

### Warm run (model already loaded in cache)

- success: `50/50 (100.00%)`
- total time: `3.519 s`
- throughput: `14.208 req/s`
- latency mean: `3.038 s`
- latency median: `3.379 s`
- latency p95: `3.414 s`
- latency p99: `3.417 s`
- response size mean: `1066.1 bytes`

## Interpretation

- Large cold/warm gap indicates startup/cache warmup cost is significant.
- Warm-cache numbers better represent steady-state API behavior for serving.
- For production reporting, include both cold-start and warm-cache latency.
