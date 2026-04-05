#!/usr/bin/env python3
import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote a registered MLflow model version to Production.")
    parser.add_argument("--model", required=True, help="Registered model name")
    parser.add_argument("--version", required=True, type=int, help="Model version number to promote")
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="MLflow tracking URI (defaults to MLFLOW_TRACKING_URI or local artifacts DB)",
    )
    parser.add_argument(
        "--no-archive-existing",
        action="store_true",
        help="Do not archive existing Production versions",
    )
    return parser.parse_args()


def resolve_tracking_uri(cli_value: str | None) -> str:
    if cli_value:
        return cli_value
    env_value = os.getenv("MLFLOW_TRACKING_URI")
    if env_value:
        return env_value
    repo_root = Path(__file__).resolve().parent.parent
    db_path = repo_root / "artifacts" / "mlflow" / "mlflow.db"
    return f"sqlite:///{db_path.as_posix()}"


def main() -> None:
    args = parse_args()

    import mlflow
    from mlflow.tracking import MlflowClient

    tracking_uri = resolve_tracking_uri(args.tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    client.transition_model_version_stage(
        name=args.model,
        version=str(args.version),
        stage="Production",
        archive_existing_versions=not args.no_archive_existing,
    )
    print(f"tracking_uri={tracking_uri}")
    print(f"promoted={args.model} v{args.version} -> Production")


if __name__ == "__main__":
    main()
