from typing import List

from pydantic import BaseModel


class ArrayInput(BaseModel):
    data: List[List[int]]
