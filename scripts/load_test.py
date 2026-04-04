#!/usr/bin/env python3
import argparse
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple concurrent load test for /predict_image endpoint")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="API base URL")
    parser.add_argument("--model", default="screw", help="Model name path parameter")
    parser.add_argument("--output-format", default="mask", choices=["mask", "redrawn"], help="Response output format")
    parser.add_argument("--threshold", type=float, default=99.0, help="Threshold query parameter")
    parser.add_argument("--stage", default="Production", help="Model stage query parameter")
    parser.add_argument("--requests", type=int, default=40, help="Total number of requests")
    parser.add_argument("--concurrency", type=int, default=4, help="Number of concurrent workers")
    parser.add_argument("--batch-size", type=int, default=2, help="Number of images per request")
    parser.add_argument(
        "--images-dir",
        default="artifacts/comparison_images",
        help="Directory used for image sampling",
    )
    parser.add_argument("--seed", type=int, default=26, help="Random seed for deterministic test inputs")
    return parser.parse_args()


def collect_image_pool(images_dir: Path, model: str) -> list[tuple[str, bytes]]:
    model_dir = images_dir / model
    if not model_dir.exists():
        raise FileNotFoundError(f"Model image directory not found: {model_dir}")

    candidates = sorted(model_dir.glob("original_*.png"))
    if not candidates:
        # fallback to any pngs in this model folder
        candidates = sorted(model_dir.glob("*.png"))

    if not candidates:
        raise RuntimeError(f"No PNG files found in: {model_dir}")

    pool = []
    for p in candidates:
        pool.append((p.name, p.read_bytes()))
    return pool


def run_single_request(
    base_url: str,
    model: str,
    output_format: str,
    threshold: float,
    stage: str,
    batch_size: int,
    image_pool: list[tuple[str, bytes]],
) -> tuple[int, float, int, str]:
    chosen = random.sample(image_pool, k=batch_size)
    files = [("files", (name, content, "image/png")) for name, content in chosen]
    url = f"{base_url}/predict_image/{model}"
    params = {
        "output_format": output_format,
        "threshold": threshold,
        "stage": stage,
    }

    start = time.perf_counter()
    try:
        response = requests.post(url, params=params, files=files, timeout=120)
        latency = time.perf_counter() - start
        return response.status_code, latency, len(response.content), response.headers.get("content-type", "")
    except Exception:
        latency = time.perf_counter() - start
        return 0, latency, 0, ""


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * p
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    frac = index - lower
    return ordered[lower] * (1 - frac) + ordered[upper] * frac


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    if args.batch_size <= 0:
        raise ValueError("batch-size must be > 0")
    if args.concurrency <= 0:
        raise ValueError("concurrency must be > 0")
    if args.requests <= 0:
        raise ValueError("requests must be > 0")

    image_pool = collect_image_pool(Path(args.images_dir), args.model)
    if args.batch_size > len(image_pool):
        raise ValueError(
            f"batch-size ({args.batch_size}) cannot exceed available image pool ({len(image_pool)})"
        )

    started = time.perf_counter()
    statuses: list[int] = []
    latencies: list[float] = []
    payload_sizes: list[int] = []

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(
                run_single_request,
                args.base_url,
                args.model,
                args.output_format,
                args.threshold,
                args.stage,
                args.batch_size,
                image_pool,
            )
            for _ in range(args.requests)
        ]

        for fut in as_completed(futures):
            status, latency, size, content_type = fut.result()
            statuses.append(status)
            latencies.append(latency)
            payload_sizes.append(size)
            if status not in (200,):
                print(f"WARN status={status} latency={latency:.3f}s content_type={content_type}")

    total_time = time.perf_counter() - started
    success = sum(1 for s in statuses if s == 200)
    success_rate = success / len(statuses) if statuses else 0.0

    print("=== Load Test Summary ===")
    print(f"base_url={args.base_url}")
    print(f"model={args.model}")
    print(f"output_format={args.output_format}")
    print(f"threshold={args.threshold}")
    print(f"stage={args.stage}")
    print(f"requests={args.requests}")
    print(f"concurrency={args.concurrency}")
    print(f"batch_size={args.batch_size}")
    print(f"image_pool={len(image_pool)}")
    print(f"success={success}/{len(statuses)} ({success_rate * 100:.2f}%)")
    print(f"total_time_s={total_time:.3f}")
    print(f"throughput_req_per_s={len(statuses) / total_time:.3f}")
    print(f"latency_mean_s={statistics.mean(latencies):.3f}")
    print(f"latency_median_s={statistics.median(latencies):.3f}")
    print(f"latency_p95_s={percentile(latencies, 0.95):.3f}")
    print(f"latency_p99_s={percentile(latencies, 0.99):.3f}")
    print(f"response_size_mean_bytes={statistics.mean(payload_sizes):.1f}")


if __name__ == "__main__":
    main()
