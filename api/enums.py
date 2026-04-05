from enum import Enum

from src.inference.serving import model_registry


def build_model_enum(names: set[str]) -> type[Enum]:
    """Build a string Enum of model names discovered from local metadata."""
    values = sorted(names)
    if not values:
        raise RuntimeError("No model metadata available; cannot build ModelName enum.")
    members = {name: name for name in values}
    return Enum("ModelName", members, type=str)


# Available model names loaded at startup from `artifacts/models` metadata.
ModelName = build_model_enum(model_registry.available_models)


class ArrayOutputFormat(str, Enum):
    """Controls response payload style returned by inference endpoints."""

    # Raw anomaly mask output.
    mask = "mask"
    # Original image with detected defects highlighted in color.
    redrawn = "redrawn"


class AnomalyColor(str, Enum):
    """Color used when `output_format=redrawn`."""

    # Colors are BGR-mapped internally for OpenCV drawing.
    blue = "blue"
    green = "green"
    red = "red"


class ModelStage(str, Enum):
    """MLflow registry stages accepted by the API."""

    production = "Production"
    staging = "Staging"
    archived = "Archived"

    @classmethod
    def _missing_(cls, value):
        """Allow case-insensitive stage query values from clients."""
        if isinstance(value, str):
            normalized = value.strip().lower()
            for member in cls:
                if member.value.lower() == normalized:
                    return member
        return None
