#!/usr/bin/env python3
import argparse
import json
import pickle
import urllib.error
import urllib.request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test for /predict_array using local artifact metadata.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="API base URL")
    parser.add_argument("--model", default="screw", help="Model name path parameter")
    parser.add_argument("--output-format", default="mask", choices=["mask", "redrawn"], help="Response output format")
    parser.add_argument("--stage", default="Production", help="Model stage query parameter")
    parser.add_argument("--batch-size", type=int, default=2, help="Number of array items in request")
    parser.add_argument("--timeout", type=int, default=180, help="Request timeout in seconds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be > 0")

    size_path = f"artifacts/sizes/sizes_{args.model}.pkl"
    with open(size_path, "rb") as file:
        img_size, channels = pickle.load(file)

    flat_size = int(img_size) * int(img_size) * int(channels)
    payload = {"data": [[0] * flat_size for _ in range(args.batch_size)]}
    url = (
        f"{args.base_url}/predict_array/{args.model}"
        f"?output_format={args.output_format}&stage={args.stage}"
    )

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            body = response.read().decode("utf-8")
            print(f"status={response.status}")
            print(f"body_prefix={body[:200]}")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        print(f"status={exc.code}")
        print(f"body_prefix={error_body[:500]}")
        raise


if __name__ == "__main__":
    main()
