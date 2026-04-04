from typing import List

from pydantic import BaseModel


class ArrayInput(BaseModel):
    data: List[List[int]]


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: str | None = None


class ErrorResponse(BaseModel):
    detail: ErrorDetail


class ArrayOutputResponse(BaseModel):
    output: List[List[int]]
