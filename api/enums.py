from enum import Enum

from api.services import AVAILABLE_MODEL_NAMES


def build_model_enum(names: set[str]) -> type[Enum]:
    values = sorted(names)
    if not values:
        raise RuntimeError("No model metadata available; cannot build ModelName enum.")
    members = {name: name for name in values}
    return Enum("ModelName", members, type=str)


ModelName = build_model_enum(AVAILABLE_MODEL_NAMES)


class ArrayOutputFormat(str, Enum):
    mask = "mask"
    redrawn = "redrawn"


class AnomalyColor(str, Enum):
    blue = "blue"
    green = "green"
    red = "red"
