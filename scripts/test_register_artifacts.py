import os
from pathlib import Path

import mlflow
import mlflow.keras
import numpy as np
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
MLFLOW_DB_PATH = ARTIFACTS_DIR / "mlflow" / "mlflow.db"
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{MLFLOW_DB_PATH.as_posix()}")
NAMES = ["bottle", "hazelnut", "screw", "wood"]

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("autoencoder_anomaly_detection")
client = MlflowClient(tracking_uri=TRACKING_URI)

print(f"Using MLflow tracking URI: {TRACKING_URI}")

for name in NAMES:
    model_path = ARTIFACTS_DIR / "models" / f"model_{name}.keras"
    if not model_path.is_file():
        print(f"Skip {name}: missing .keras model file")
        continue
    model = mlflow.keras.load_model(str(model_path))
    _, height, width, channels = model.input_shape
    signature_input = np.zeros((1, height, width, channels), dtype=np.float32)
    signature_output = model.predict(signature_input, verbose=0)
    signature = infer_signature(signature_input, signature_output)

    with mlflow.start_run(run_name=f"bootstrap-register-{name}"):
        mlflow.log_param("dataset_name", f"mvtec/{name}")
        mlflow.log_param("bootstrap_from_local_artifact", "keras")
        mlflow.set_tag("model_name", name)
        mlflow.keras.log_model(
            model,
            artifact_path="model",
            registered_model_name=name,
            signature=signature,
            metadata={"model_name": name},
        )
        print(f"Registered {name} from local keras")

    versions = client.search_model_versions(f"name='{name}'")
    latest = max(versions, key=lambda v: int(v.version))
    client.transition_model_version_stage(
        name=name,
        version=latest.version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"Promoted {name} v{latest.version} -> Production")

print("Done.")
