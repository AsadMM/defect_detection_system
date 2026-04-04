import pickle
import mlflow
import mlflow.keras
from mlflow.tracking import MlflowClient
from src.models.autoencoder import build1
import os

TRACKING_URI = "sqlite:///artifacts/mlflow/mlflow.db"
NAMES = ["bottle", "hazelnut", "screw", "wood"]

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("autoencoder_anomaly_detection")
client = MlflowClient(tracking_uri=TRACKING_URI)

for name in NAMES:
    size_path = f"artifacts/sizes/sizes_{name}.pkl"
    model_path = f"artifacts/models/model_{name}.h5"
    if not (os.path.isfile(size_path) and os.path.isfile(model_path)):
        print(f"Skip {name}: missing size/model file")
        continue

    with open(size_path, "rb") as f:
        size = pickle.load(f)

    model = build1(size[0], size[0], size[1])
    model.load_weights(model_path)

    with mlflow.start_run(run_name=f"bootstrap-register-{name}"):
        mlflow.log_param("dataset_name", f"mvtec/{name}")
        mlflow.log_param("bootstrap_from_local_h5", True)
        info = mlflow.keras.log_model(
            model,
            artifact_path="model",
            registered_model_name=name,
        )
        print(f"Registered {name} from local h5")

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