#!/usr/bin/env python3
import argparse
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call /predict_image/{model_name} and save the returned ZIP file."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="API base URL")
    parser.add_argument("--model", default="screw", help="Model name path parameter")
    parser.add_argument("--output-format", default="mask", choices=["mask", "redrawn"], help="Response output format")
    parser.add_argument("--threshold", type=float, default=99.0, help="Threshold query parameter")
    parser.add_argument("--stage", default="Production", help="Model stage query parameter")
    parser.add_argument("--version", type=int, default=None, help="Model version query parameter")
    parser.add_argument(
        "--images-dir",
        default="artifacts/comparison_images",
        help="Directory that contains per-model image folders",
    )
    parser.add_argument(
        "--file",
        action="append",
        default=[],
        help="Explicit image file path(s); can be passed multiple times",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="How many images to send when using --images-dir")
    parser.add_argument(
        "--out",
        default="artifacts/predict_image_output.zip",
        help="Path to save the returned ZIP",
    )
    parser.add_argument("--timeout", type=int, default=180, help="Request timeout in seconds")
    return parser.parse_args()


def pick_images(args: argparse.Namespace) -> list[Path]:
    if args.file:
        paths = [Path(p) for p in args.file]
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing input file(s): {', '.join(missing)}")
        return paths

    model_dir = Path(args.images_dir) / args.model
    if not model_dir.exists():
        raise FileNotFoundError(f"Model image directory not found: {model_dir}")

    candidates = sorted(model_dir.glob("original_*.png"))
    if not candidates:
        candidates = sorted(model_dir.glob("*.png"))
    if not candidates:
        raise RuntimeError(f"No PNG files found in: {model_dir}")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be > 0")

    return candidates[: args.batch_size]


def main() -> None:
    args = parse_args()
    images = pick_images(args)

    url = f"{args.base_url}/predict_image/{args.model}"
    params = {
        "output_format": args.output_format,
        "threshold": args.threshold,
        "stage": args.stage,
    }
    if args.version is not None:
        params["version"] = args.version

    files = []
    for path in images:
        files.append(("files", (path.name, path.read_bytes(), "image/png")))

    response = requests.post(url, params=params, files=files, timeout=args.timeout)
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(response.content)

    print(f"status={response.status_code}")
    print(f"content_type={response.headers.get('content-type', '')}")
    print(f"bytes={len(response.content)}")
    print(f"saved_zip={output_path}")
    print(f"images_sent={len(images)}")
    if response.status_code != 200:
        print("body_prefix=" + response.text[:500])
        response.raise_for_status()


if __name__ == "__main__":
    main()
