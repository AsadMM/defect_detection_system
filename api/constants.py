import os


def _get_positive_int_env(name: str, default: int) -> int:
    """Read a positive integer from env, falling back to default on invalid input."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


MAX_ARRAY_BATCH_SIZE = _get_positive_int_env("MAX_ARRAY_BATCH_SIZE", 32)
MAX_IMAGE_BATCH_SIZE = _get_positive_int_env("MAX_IMAGE_BATCH_SIZE", 16)


# OpenAPI schema override for multipart file uploads (`files: list[UploadFile]`).
MULTI_FILE_OPENAPI_SCHEMA = {
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "files": {
                            "type": "array",
                            "items": {"type": "string", "format": "binary"},
                        }
                    },
                    "required": ["files"],
                }
            }
        },
    }
}
