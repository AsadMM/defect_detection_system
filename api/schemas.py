from typing import List

from pydantic import BaseModel, Field


class ArrayInput(BaseModel):
    """Request body for `/predict_array/{model_name}`."""

    data: List[List[int]] = Field(description="Batch of flattened uint8-like image arrays.")


class ErrorDetail(BaseModel):
    """Structured API error information."""

    code: str = Field(description="Stable machine-readable error code.")
    message: str = Field(description="Human-readable error summary.")
    details: str | None = Field(default=None, description="Optional contextual details.")


class ErrorResponse(BaseModel):
    """Top-level error payload returned by failed requests."""

    detail: ErrorDetail = Field(description="Detailed error object.")


class ArrayOutputResponse(BaseModel):
    """Response body for array inference endpoints."""

    output: List[List[int]] = Field(description="Batch of flattened output arrays.")
