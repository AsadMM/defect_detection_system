class UnknownModelError(KeyError):
    """Raised when a requested model name is not known to the registry."""


class ModelMetadataError(KeyError):
    """Raised when model metadata is missing or inconsistent."""
